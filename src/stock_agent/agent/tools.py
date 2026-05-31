"""Agent tools — thin wrappers over existing pipelines.

Each tool takes primitive JSON args (ticker/days/horizon) and fetches what it
needs internally (the disk cache makes repeated fetches free within the TTL), so
the model never has to pass objects between calls. Tools return JSON-serializable
dicts; errors are returned as ``{"error": ...}`` rather than raised, so the agent
can react instead of crashing the loop.

No business logic lives here — tools delegate to data/indicators/news/forecasting.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from datetime import date as Date
from typing import Any

from stock_agent.backtesting.calibration import calibration_label
from stock_agent.data.earnings import fetch_earnings_context
from stock_agent.data.loader import LoadResult, PriceLoader
from stock_agent.data.validation import DataIssue
from stock_agent.features.news_features import build_news_features
from stock_agent.forecasting.large_move import large_move_breakdown
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import TextLLM
from stock_agent.llm.news_summarizer import summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.pipelines.forecast import MODEL_NAMES, run_forecast
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.backtest import BacktestResult
from stock_agent.settings import Settings


def _warnings(issues: list[DataIssue]) -> list[str]:
    """Format warning/error data-quality issues for tool output (info-level skipped)."""
    return [f"{i.code}: {i.message}" for i in issues if i.severity in ("warning", "error")]


log = get_logger(__name__)

# Wide window so MA200 and forecast samples exist regardless of the news window.
_PRICE_LOOKBACK_DAYS = 420

# Backtest tool bounds (heavy op → keep the chat responsive). Only fast, offline,
# leakage-safe stateless models are exposed to the agent; ML backtests need a
# per-fold pooled refit (minutes) and stay a CLI-only operation.
_BACKTEST_MODELS: tuple[str, ...] = (
    "historical_sim",
    "monte_carlo_gbm",
    "monte_carlo_bootstrap",
)
_BT_MIN_HORIZON, _BT_MAX_HORIZON = 5, 60
_BT_TIMEOUT_S = 45.0  # wall-clock backstop; argument bounds keep typical runs ~seconds


def _run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``fn`` in a worker thread, bounding the caller's wait.

    CPU-bound numpy work can't be force-killed, so on timeout the orphan thread
    finishes on its own — but the agent gets control back promptly with a clear
    error instead of hanging the chat. Argument bounds are the primary limiter;
    this is a defensive backstop. ``shutdown(wait=False)`` so we never block.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


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
        "name": "get_earnings_context",
        "description": (
            "Next and last earnings dates with proximity: days_to_next_earnings, "
            "days_since_last_earnings, and whether earnings fall WITHIN the horizon. Important "
            "because the price-only forecast cannot see a scheduled earnings event inside the "
            "window — if earnings_in_horizon is true, expect a wider/fatter-tailed outcome."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "default": 20,
                    "description": "Horizon to check earnings against.",
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
                    "enum": list(MODEL_NAMES),
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
    {
        "name": "run_backtest",
        "description": (
            "Out-of-sample WALK-FORWARD track record of a forecast model for a ticker: how "
            "accurate its probabilities have been historically. Returns Brier score, log loss, "
            "ROC AUC and accuracy per return threshold, plus calibration (ECE) and a trust label. "
            "Use for 'how accurate/reliable has the model been?'. Heavier than one forecast — "
            "limited to fast offline models (historical_sim / monte_carlo_gbm / "
            "monte_carlo_bootstrap) and horizon 5-60 trading days. ROC AUC near 0.5 means no "
            "directional edge (expected for the unconditional baselines)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "default": 20,
                    "description": "Horizon in trading days (5–60).",
                },
                "model": {
                    "type": "string",
                    "enum": list(_BACKTEST_MODELS),
                    "default": "historical_sim",
                    "description": "Model to evaluate (fast offline models only).",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_calibration",
        "description": (
            "Is a model's forecast WELL-CALIBRATED for a ticker/horizon? Returns the Expected "
            "Calibration Error (ECE = average gap between predicted probability and realized "
            "frequency; 0 = perfect), a reliability table (predicted vs realized), a plain trust "
            "label, and whether post-hoc recalibration would help. Use this for 'is your "
            "forecast trustworthy / well-calibrated / can I trust these numbers?'. Same fast "
            "offline models and 5-60 day horizon bound as run_backtest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "default": 20,
                    "description": "Horizon in trading days (5–60).",
                },
                "model": {
                    "type": "string",
                    "enum": list(_BACKTEST_MODELS),
                    "default": "historical_sim",
                    "description": "Model whose calibration to assess (fast offline models only).",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_large_move",
        "description": (
            "Probability of a LARGE MOVE (a big up or down move, regardless of small-scale "
            "direction) over the horizon: P(|return| > k), split into P(up > +k) and P(down < -k). "
            "This is where the ML model has genuine, backtested skill — predicting big moves / "
            "volatility — unlike plain direction, which is ~a coin flip. Use for 'chance of a big "
            "move', 'how volatile / could it spike or crash', 'is a large swing likely'. Defaults "
            "to the logistic model; k is 5% or 10%. The large-move total is the most reliable "
            "part; the up/down split shows which tail leans but is less certain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "horizon_days": {
                    "type": "integer",
                    "default": 20,
                    "description": "Horizon in trading days.",
                },
                "threshold_pct": {
                    "type": "integer",
                    "enum": [5, 10],
                    "default": 10,
                    "description": "How big is 'big' — a 5% or 10% move (a bucket edge).",
                },
                "model": {
                    "type": "string",
                    "enum": ["logistic", "lightgbm"],
                    "default": "logistic",
                    "description": (
                        "Big-move model: 'logistic' (default, best for stable names) or "
                        "'lightgbm' (regularized, better for VOLATILE names). Call both to compare."
                    ),
                },
            },
            "required": ["ticker"],
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
        # Per-session memoization of backtests so run_backtest + get_calibration on
        # the same (ticker, horizon, model) reuse one computation instead of two.
        self._backtest_cache: dict[tuple[str, int, str], BacktestResult] = {}

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
        # Reflection (self-critique) depth is config-driven; default 1 pass.
        return summarize_news(
            bundle,
            self._llm,
            reflection_iterations=self._settings.news_reflection_iterations,
        ).model_dump(mode="json")

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

    def _tool_get_earnings_context(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args.get("horizon_days", 20))
        ctx = fetch_earnings_context(
            self._registry, ticker, as_of=Date.today(), horizon_days=horizon
        )
        return ctx.model_dump(mode="json")

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

    def _tool_get_large_move(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args.get("horizon_days", 20))
        threshold_pct = int(args.get("threshold_pct", 10))
        if threshold_pct not in (5, 10):
            return {"error": "threshold_pct must be 5 or 10 (the model's bucket boundaries)"}
        # The two validated big-move models: logistic (default; best for stable names) and
        # regularized lightgbm (better for volatile names). Both fall back to the baseline
        # (with a note) at horizons without a trained artifact.
        model = str(args.get("model", "logistic"))
        if model not in ("logistic", "lightgbm"):
            return {"error": "model must be 'logistic' or 'lightgbm' for the big-move signal"}
        forecast = run_forecast(
            ticker, horizon, model_name=model, settings=self._settings, registry=self._registry
        )
        breakdown = large_move_breakdown(forecast, threshold=threshold_pct / 100.0)
        result = breakdown.model_dump(mode="json")
        result["threshold_pct"] = threshold_pct
        # Honest static trust framing — a precomputed per-ticker skill scorecard
        # (backtested AUC/calibration) is the planned next step.
        result["reliability"] = (
            "The large-move total P(|r|>k) is the model's most reliable signal in backtesting; "
            "the up/down split shows which tail leans but is rarer and noisier. Strongest at "
            "short horizons (<= 20 days). A magnitude signal, not a directional call on the median."
        )
        if forecast.notes:
            result["model_note"] = forecast.notes
        return result

    # ---- backtesting / calibration (Phase 6.5) -------------------------------
    def _backtest_result(self, ticker: str, horizon: int, model: str) -> BacktestResult:
        """Run (or reuse a cached) bounded walk-forward backtest.

        Validates argument bounds and runs under a wall-clock timeout. Raises
        ``ValueError`` on out-of-bounds args / unavailable model (the dispatcher
        turns it into an ``{"error": ...}`` the agent can read).
        """
        if model not in _BACKTEST_MODELS:
            raise ValueError(
                f"model '{model}' is not available for agent backtests; choose one of "
                f"{list(_BACKTEST_MODELS)}. ML-model backtests require a per-fold pooled "
                "retrain and run offline via the CLI: `backtest --model <name>`."
            )
        if not (_BT_MIN_HORIZON <= horizon <= _BT_MAX_HORIZON):
            raise ValueError(
                f"horizon_days must be {_BT_MIN_HORIZON}–{_BT_MAX_HORIZON} for the backtest "
                f"tools (got {horizon})."
            )
        key = (ticker, horizon, model)
        if key not in self._backtest_cache:
            from stock_agent.pipelines.backtest import run_backtest_pipeline

            def _run() -> BacktestResult:
                return run_backtest_pipeline(
                    ticker,
                    horizon,
                    model_names=[model],
                    settings=self._settings,
                    registry=self._registry,
                    log_experiment=False,  # chat backtests shouldn't spam experiment dirs
                )[model]

            self._backtest_cache[key] = _run_with_timeout(_run, _BT_TIMEOUT_S)
        return self._backtest_cache[key]

    def _tool_run_backtest(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args.get("horizon_days", 20))
        model = str(args.get("model", "historical_sim"))
        try:
            r = self._backtest_result(ticker, horizon, model)
        except ValueError as exc:
            return {"error": str(exc)}
        except TimeoutError:
            return {"error": f"backtest exceeded {int(_BT_TIMEOUT_S)}s; try a shorter horizon."}
        c = r.calibration
        return {
            "ticker": r.ticker,
            "horizon_days": r.horizon_days,
            "model": r.model_name,
            "n_oos_forecasts": r.n_predictions,
            "n_folds": r.n_folds,
            "as_of_start": r.as_of_start.isoformat(),
            "as_of_end": r.as_of_end.isoformat(),
            "mean_brier": round(r.mean_brier, 4),
            "mean_log_loss": round(r.mean_log_loss, 4),
            "calibration_ece": round(c.ece, 4),
            "calibration_quality": calibration_label(c.ece),
            "per_threshold": [
                {
                    "threshold_label": f"{m.threshold:+.0%}",
                    "base_rate": round(m.base_rate, 4),
                    "brier": round(m.brier, 4),
                    "log_loss": round(m.log_loss, 4),
                    "accuracy": round(m.accuracy, 4),
                    "roc_auc": round(m.roc_auc, 4) if m.roc_auc is not None else None,
                }
                for m in r.thresholds
            ],
            "methodology": (
                f"Walk-forward out-of-sample, embargo = horizon, {r.n_folds} folds. ROC AUC near "
                "0.5 means no directional skill (expected for unconditional baselines)."
            ),
        }

    def _tool_get_calibration(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args.get("horizon_days", 20))
        model = str(args.get("model", "historical_sim"))
        try:
            r = self._backtest_result(ticker, horizon, model)
        except ValueError as exc:
            return {"error": str(exc)}
        except TimeoutError:
            return {"error": f"backtest exceeded {int(_BT_TIMEOUT_S)}s; try a shorter horizon."}
        c = r.calibration
        post = None
        if c.ece_post is not None and c.ece_pre_holdout is not None:
            post = {
                "method": c.method_post,
                "ece_before": round(c.ece_pre_holdout, 4),
                "ece_after": round(c.ece_post, 4),
            }
        return {
            "ticker": r.ticker,
            "horizon_days": r.horizon_days,
            "model": r.model_name,
            "n_oos_forecasts": r.n_predictions,
            "n_calibration_points": c.n,
            "ece": round(c.ece, 4),
            "mce": round(c.mce, 4),
            "calibration_quality": calibration_label(c.ece),
            "reliability": [
                {
                    "predicted": round(b.mean_pred, 3),
                    "realized": round(b.frequency, 3),
                    "count": b.count,
                }
                for b in c.bins
            ],
            "post_hoc_recalibration": post,  # null when too few points to split honestly
            "interpretation_key": (
                "ECE is the average gap between predicted probability and realized frequency "
                "(0 = perfectly calibrated). In the reliability table, 'realized' below "
                "'predicted' means the model was overconfident in that probability band."
            ),
            "methodology": (
                f"Walk-forward out-of-sample, embargo = horizon, {r.n_folds} folds, "
                f"{c.n_bins}-bin reliability."
            ),
        }
