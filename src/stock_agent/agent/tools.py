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

from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import TextLLM
from stock_agent.llm.news_summarizer import summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import Settings

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
        "name": "run_forecast",
        "description": (
            "Model-generated forward-return scenario probabilities, expected return, VaR, and "
            "confidence interval for a horizon (trading days). Probabilities come from this "
            "statistical model only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "description": "Forecast horizon in trading days",
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
    def _load(self, ticker: str, days: int) -> PriceSeries:
        return PriceLoader(self._registry).load_recent(ticker.upper(), days, min_bars=20).series

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
        series = self._load(ticker, days)
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
        }

    def _tool_compute_indicators(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        series = self._load(ticker, _PRICE_LOOKBACK_DAYS)
        return compute_snapshot(series).model_dump(mode="json")

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

    def _tool_run_forecast(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args["horizon_days"])
        series = self._load(ticker, _PRICE_LOOKBACK_DAYS)
        forecast = HistoricalSimulation().forecast(series, horizon_days=horizon)
        result = forecast.model_dump(mode="json")
        # VaR confidence levels appear only as field *names* (var_95, var_99) in the
        # model dump — the numbers 95 and 99 never enter the grounding set, so the
        # agent can't say "99% VaR" without a false-positive grounding violation.
        # Add them explicitly so standard statistical labels are always grounded.
        result["var_confidence_levels_pct"] = [90, 95, 99]
        return result
