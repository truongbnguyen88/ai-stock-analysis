"""Agent tools — thin wrappers over existing pipelines.

Each tool takes primitive JSON args (ticker/days/horizon) and fetches what it
needs internally (the disk cache makes repeated fetches free within the TTL), so
the model never has to pass objects between calls. Tools return JSON-serializable
dicts; errors are returned as ``{"error": ...}`` rather than raised, so the agent
can react instead of crashing the loop.

No business logic lives here — tools delegate to data/indicators/news/forecasting.
"""

from __future__ import annotations

from typing import Any

from stock_agent.data.loader import LoadResult, PriceLoader
from stock_agent.data.validation import DataIssue
from stock_agent.features.news_features import build_news_features
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import TextLLM
from stock_agent.llm.news_summarizer import summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.pipelines.forecast import MODEL_REGISTRY, run_forecast
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.settings import Settings


def _warnings(issues: list[DataIssue]) -> list[str]:
    """Format warning/error data-quality issues for tool output (info-level skipped)."""
    return [f"{i.code}: {i.message}" for i in issues if i.severity in ("warning", "error")]


log = get_logger(__name__)

# Wide window so MA200 and forecast samples exist regardless of the news window.
_PRICE_LOOKBACK_DAYS = 420

# Anthropic tool schemas (the contract the model sees).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_price_summary",
        "description": "Recent price statistics for a ticker over a window of calendar days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. NVDA"},
                "days": {
                    "type": "integer",
                    "description": "Lookback window in days",
                    "default": 30,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compute_indicators",
        "description": "Technical indicators: MAs, RSI, MACD, volatility, ATR, drawdown, trend.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Recent news headlines (title, source, date, url) for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "default": 14},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "summarize_news",
        "description": (
            "LLM synthesis of recent news: overview, themes, bullish/bearish/risks/catalysts "
            "with article citations. Qualitative only — contains no probabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "default": 14},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news_sentiment",
        "description": (
            "Aggregate NUMERIC news context: average sentiment, % positive/negative, sentiment "
            "coverage, article count, and event flags (earnings/regulatory/upgrade/downgrade). "
            "Default uses free Alpha Vantage scores; set use_llm=true for fuller Claude coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "default": 14},
                "use_llm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Claude-score all articles for fuller coverage (small cost).",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "run_forecast",
        "description": (
            "Model-generated forward-return scenario probabilities, expected return, VaR, and "
            "confidence interval for a horizon (trading days). Probabilities come from the chosen "
            "statistical/ML model only. Call multiple times with different models to compare."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "description": "Forecast horizon in trading days",
                },
                "model": {
                    "type": "string",
                    "enum": list(MODEL_REGISTRY),
                    "default": "historical_sim",
                    "description": (
                        "Forecast model: 'historical_sim' (empirical baseline, default), "
                        "'monte_carlo_gbm'/'monte_carlo_bootstrap' (simulation), or ML "
                        "('xgboost'/'lightgbm'/'logistic'/'random_forest' — need a trained "
                        "artifact, else fall back to the baseline with a note)."
                    ),
                },
            },
            "required": ["ticker", "horizon_days"],
        },
    },
]


class ToolExecutor:
    """Dispatches tool calls to the underlying pipelines."""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: ProviderRegistry | None = None,
        llm: TextLLM | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry or build_default_registry(settings)
        self._llm = llm  # TextLLM for summarize_news (Role A); may be None

    # ---- data helpers --------------------------------------------------------
    def _load(self, ticker: str, days: int) -> LoadResult:
        """Load prices AND keep the data-quality issues (so tools can warn)."""
        return PriceLoader(self._registry).load_recent(ticker.upper(), days, min_bars=20)

    # ---- dispatch ------------------------------------------------------------
    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name; never raises (errors become {'error': ...})."""
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"error": f"unknown tool '{name}'"}
            result: dict[str, Any] = handler(args)
            return result
        except (ProviderError, ValueError, KeyError) as exc:
            log.warning("agent.tool_failed", tool=name, error=str(exc))
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _tool_get_price_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        days = int(args.get("days", 30))
        loaded = self._load(ticker, days)
        series = loaded.series
        closes = series.closes
        return {
            "ticker": ticker,
            "start": series.dates[0].isoformat(),
            "end": series.dates[-1].isoformat(),
            "n_bars": len(series),
            "first_close": closes[0],
            "last_close": closes[-1],
            "pct_change": closes[-1] / closes[0] - 1.0,
            "period_high": max(closes),
            "period_low": min(closes),
            "data_warnings": _warnings(loaded.issues),
        }

    def _tool_compute_indicators(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        loaded = self._load(ticker, _PRICE_LOOKBACK_DAYS)
        result = compute_snapshot(loaded.series).model_dump(mode="json")
        result["data_warnings"] = _warnings(loaded.issues)
        return result

    def _tool_get_news(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        days = int(args.get("days", 14))
        bundle = NewsFetcher(self._registry).fetch(ticker, lookback_days=days, top_n=10)
        return {
            "ticker": ticker,
            "articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "published": a.published_at.date().isoformat(),
                    "url": str(a.url),
                }
                for a in bundle.articles
            ],
        }

    def _tool_summarize_news(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._llm is None:
            return {"error": "news summarization unavailable (no LLM configured)"}
        ticker = str(args["ticker"]).upper()
        days = int(args.get("days", 14))
        bundle = NewsFetcher(self._registry).fetch(ticker, lookback_days=days, top_n=25)
        return summarize_news(bundle, self._llm).model_dump(mode="json")

    def _tool_get_news_sentiment(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        days = int(args.get("days", 14))
        use_llm = bool(args.get("use_llm", False))
        bundle = NewsFetcher(self._registry).fetch(ticker, lookback_days=days, top_n=25)
        # Default = free Alpha Vantage scores; Claude scoring is opt-in (per cost decision).
        result: dict[str, Any] = dict(
            build_news_features(bundle, llm=self._llm, use_llm_sentiment=use_llm)
        )
        result["ticker"] = ticker
        result["sentiment_source"] = (
            "claude" if (use_llm and self._llm is not None) else "alpha_vantage"
        )
        return result

    def _tool_run_forecast(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args["horizon_days"])
        model = str(args.get("model", "historical_sim"))
        # Route through the forecast pipeline so the agent can use any registered
        # model (baseline / Monte Carlo / pooled ML); ML falls back to the
        # baseline with a note when no trained artifact exists.
        forecast = run_forecast(
            ticker,
            horizon,
            model_name=model,
            settings=self._settings,
            registry=self._registry,
        )
        result = forecast.model_dump(mode="json")
        # VaR confidence levels appear only as field *names* (var_95, var_99) in the
        # model dump — the numbers 95 and 99 never enter the grounding set, so the
        # agent can't say "99% VaR" without a false-positive grounding violation.
        # Add them explicitly so standard statistical labels are always grounded.
        result["var_confidence_levels_pct"] = [90, 95, 99]
        return result
