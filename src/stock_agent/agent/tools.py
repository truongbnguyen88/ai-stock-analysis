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
from datetime import timedelta
from typing import Any

from stock_agent.backtesting.calibration import calibration_label
from stock_agent.data.earnings import fetch_earnings_context
from stock_agent.data.loader import LoadResult, PriceLoader
from stock_agent.data.validation import DataIssue
from stock_agent.features.news_features import build_news_features
from stock_agent.forecasting.large_move import large_move_breakdown
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import LLMError, TextLLM
from stock_agent.llm.news_summarizer import summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher, TopicNewsFetcher
from stock_agent.news.topics import (
    ResolvedTopic,
    gdelt_query_expression,
    list_topics,
    looks_like_sentence_topic,
    resolve_topic,
)
from stock_agent.pipelines.forecast import MODEL_NAMES, run_forecast
from stock_agent.pipelines.research import (
    DEFAULT_NEWS_LOOKBACK_DAYS,
    ResearchPipelineError,
    run_research,
)
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.rag.embeddings import embedding_namespace
from stock_agent.rag.read_path import build_graph_system, build_retrieval_system
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.rag.vector_store import build_vector_store, collection_name_for
from stock_agent.research.agentic import answer_multistep
from stock_agent.research.brief import render_research_brief
from stock_agent.research.synthesis import ResearchGuardError, answer_question
from stock_agent.schemas.agentic import MultiStepAnswer
from stock_agent.schemas.backtest import BacktestResult
from stock_agent.schemas.comparison import (
    ForecastComparison,
    HeadlineRef,
    NewsComparison,
    TickerForecastRow,
    TickerNewsRow,
)
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.research import ResearchMemo
from stock_agent.schemas.retrieval import ChunkFilter
from stock_agent.settings import Settings


def _warnings(issues: list[DataIssue]) -> list[str]:
    """Format warning/error data-quality issues for tool output (info-level skipped)."""
    return [f"{i.code}: {i.message}" for i in issues if i.severity in ("warning", "error")]


log = get_logger(__name__)

# Wide window so MA200 and forecast samples exist regardless of the news window.
_PRICE_LOOKBACK_DAYS = 420

# Backtest tool bounds (heavy op → keep the chat responsive). Only fast, offline,
# leakage-safe stateless models are exposed to the agent; ML backtests need a
# per-fold pooled refit (minutes) and stay a CLI-only operation. The two promoted
# Monte-Carlo methods are bootstrap (best/tied at short horizons) and GARCH (the
# validated winner at h30/h60); gbm is dropped — it lost on Brier everywhere.
_BACKTEST_MODELS: tuple[str, ...] = (
    "historical_sim",
    "monte_carlo_bootstrap",
    "monte_carlo_garch",
)
_BT_MIN_HORIZON, _BT_MAX_HORIZON = 5, 60
_BT_TIMEOUT_S = 45.0  # wall-clock backstop; argument bounds keep typical runs ~seconds

# Multi-ticker batch tools (Enhancement B). Each ticker is N forecasts (ensemble =
# several members) or N news fetches, so cap the list and bound the wall-clock time.
# Beyond the cap, extra tickers are truncated (reported in ``skipped``) rather than
# rejected, so a slightly-too-long request still returns a useful comparison.
_MAX_TICKERS = 6
_COMPARE_TIMEOUT_S = 90.0

# research_summary is the heaviest tool (prices + forecast + retrieval + a news-summary
# call + the memo synthesis call ≈ 2 LLM calls). run_research runs its 3 independent gather
# stages concurrently (≈ their max, not sum), so this is comfortable headroom over the observed
# ~80-95s; it's the wall-clock cap (the LLM client carries its own per-request timeout).
_RESEARCH_TIMEOUT_S = 180.0

# Curated theme slugs surfaced to the model (free-form themes also work).
_KNOWN_TOPICS_DESC = ", ".join(list_topics())


def _parse_tickers(args: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Normalize the ``tickers`` arg -> (kept, skipped) honoring the cap.

    Accepts a JSON list or a comma-separated string; upper-cases, drops blanks,
    de-duplicates preserving order, then truncates to ``_MAX_TICKERS``.
    """
    raw = args.get("tickers", [])
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    seen: set[str] = set()
    tickers: list[str] = []
    for item in items:
        sym = str(item).strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            tickers.append(sym)
    return tickers[:_MAX_TICKERS], tickers[_MAX_TICKERS:]


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
        "description": (
            "Recent news headlines (title, source, date, url) for a ticker, "
            "ordered newest-first over the lookback window."
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
                    "default": "ensemble",
                    "description": (
                        "Forecast model. DEFAULT 'ensemble' — a calibrated probability pool of all "
                        "the others (historical + bootstrap + GARCH + logistic + lightgbm); use it "
                        "unless the user asks for a specific model or a comparison, because at "
                        "inference you can't know which single model is best for this name. "
                        "Individual models: 'historical_sim' (baseline), 'monte_carlo_bootstrap' "
                        "(fat-tail resampling; short horizons), 'monte_carlo_garch' (conditional "
                        "vol; best >=30d), ML 'logistic'/'lightgbm' (need a trained artifact, else "
                        "fall back to the baseline with a note)."
                    ),
                },
            },
            "required": ["ticker", "horizon_days"],
        },
    },
    {
        "name": "compare_forecasts",
        "description": (
            "Compare MODEL FORECASTS across SEVERAL tickers at one horizon in a single call — "
            "per ticker: expected return, P(up)/P(down), VaR95, CI, and P(big move). Use this "
            "(instead of calling run_forecast repeatedly) when the user names 2+ tickers or asks "
            "to compare/rank names by forecast. Numbers come from the chosen model only; you "
            f"narrate the comparison (NON-ADVISORY — no 'buy X over Y'). Up to {_MAX_TICKERS} "
            "tickers; extras are returned in 'skipped'. A ticker with no data comes back with an "
            "'error' field instead of numbers — report it and continue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"2–{_MAX_TICKERS} ticker symbols, e.g. ['NVDA','MSFT','AMD'].",
                },
                "horizon_days": {
                    "type": "integer",
                    "default": 20,
                    "description": "Forecast horizon in trading days (same for every ticker).",
                },
                "model": {
                    "type": "string",
                    "enum": list(MODEL_NAMES),
                    "default": "ensemble",
                    "description": "Forecast model applied to every ticker (DEFAULT 'ensemble').",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "conditional_outlook",
        "description": (
            "LEAKAGE-SAFE HISTORICAL CONDITIONAL linking a market driver to a stock — the honest "
            "way to ask 'how might <news theme> affect <stock>'. Map the news theme to a DRIVER "
            "proxy ticker (oil->USO or XLE, defense->ITA, broad vol/risk-off->^VIX, rates->TLT, "
            "gold->GLD, semis->SMH), then this returns how the TARGET's forward return over the "
            "next horizon historically distributed on days AFTER the driver moved >= shock_pct "
            "over a trailing window, vs its baseline (mean, median, P(up), the lift). DESCRIPTIVE "
            "history, NOT a forecast or causal claim: overlapping windows make events "
            "autocorrelated, so weigh effective_independent_events and low_confidence. Numbers "
            "from price history only; you narrate, non-advisory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Stock to study, e.g. DAL."},
                "driver": {
                    "type": "string",
                    "description": "Proxy ticker for the theme, e.g. USO (oil) / ITA (defense).",
                },
                "shock_pct": {
                    "type": "integer",
                    "default": 5,
                    "description": "Driver move size (percent) that defines an event (1–50).",
                },
                "event_window_days": {
                    "type": "integer",
                    "default": 5,
                    "description": "Trailing trading days the driver move is measured over (1–60).",
                },
                "horizon_days": {
                    "type": "integer",
                    "default": 10,
                    "description": "Target forward-return horizon in trading days (1–60).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["both", "up", "down"],
                    "default": "both",
                    "description": "Driver move direction defining the event.",
                },
            },
            "required": ["target", "driver"],
        },
    },
    {
        "name": "compare_news",
        "description": (
            "Compare NEWS across SEVERAL tickers in a single call — per ticker: numeric sentiment "
            "(average, % positive/negative; from Alpha Vantage scores) and the newest headlines. "
            "Use when the user names 2+ tickers and asks about news/sentiment. You write the "
            "cross-ticker narrative (themes, divergences) — NON-ADVISORY (no 'buy X over Y'); the "
            f"sentiment numbers come from the data, not you. Up to {_MAX_TICKERS} tickers; extras "
            "are returned in 'skipped'. A ticker with no news comes back with an 'error' field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"2–{_MAX_TICKERS} ticker symbols, e.g. ['NVDA','MSFT','AMD'].",
                },
                "days": {
                    "type": "integer",
                    "default": 14,
                    "description": "News lookback window in days (same for every ticker).",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_topic_news",
        "description": (
            "Recent news by THEME/SECTOR rather than by ticker — e.g. 'robotics', 'EVs', "
            "'AI memory', 'semiconductors'. Returns newest-first headlines (title, source, date, "
            "url) plus the resolved keywords so the match is transparent. Route here when the user "
            "names a sector/theme/topic (not a single company). Known themes: "
            f"{_KNOWN_TOPICS_DESC}; any other phrase works as a free-form search (lower precision)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Theme/sector phrase, e.g. 'robotics' or 'AI data centers'.",
                },
                "days": {"type": "integer", "default": 14},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "analyze_topic_news",
        "description": (
            "Qualitative SYNTHESIS of a theme's recent news: overview, key themes, "
            "bullish/bearish drivers, risks, and catalysts (each cited), for a sector/theme rather "
            "than a single ticker. Use for 'analyze <theme> news'. Surfaces the resolved query for "
            "transparency. Qualitative only — contains no probabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Theme/sector phrase, e.g. 'robotics'."},
                "days": {"type": "integer", "default": 14},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Out-of-sample WALK-FORWARD track record of a forecast model for a ticker: how "
            "accurate its probabilities have been historically. Returns Brier score, log loss, "
            "ROC AUC and accuracy per return threshold, plus calibration (ECE) and a trust label. "
            "Use for 'how accurate/reliable has the model been?'. Heavier than one forecast — "
            "limited to fast offline models (historical_sim / monte_carlo_bootstrap / "
            "monte_carlo_garch) and horizon 5-60 trading days. ROC AUC near 0.5 means no "
            "directional edge (expected — only magnitude/volatility is predictable)."
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
            "to the logistic model; k scales with horizon (default 5%/10%/15% at 20/30/60d) — omit "
            "threshold_pct for that default. The large-move total is the most reliable part; the "
            "up/down split shows which tail leans but is less certain."
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
                    "description": (
                        "How big is 'big' — a percent move that must be a bucket edge for the "
                        "horizon (20d: 5 or 10; 30d: 10 or 20; 60d: 15 or 30). OMIT to use the "
                        "horizon's default inner edge (5/10/15 at 20/30/60d)."
                    ),
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
    {
        "name": "search_filings",
        "description": (
            "Answer a SPECIFIC question from a company's SEC FILINGS (10-K / 10-Q / 8-K) using "
            "the ingested filing text — risk factors, business/products, MD&A, management "
            "commentary, accounting, legal proceedings, segment detail. Returns a cited answer "
            "(each claim marked [n], with the source filing+section) grounded ONLY in the "
            "retrieved filings — never general knowledge. Use this for 'what are NVDA's risk "
            "factors / what does management say about X / what's in the latest 10-K'. Qualitative "
            "evidence only — no probabilities. If filings for the ticker haven't been ingested, "
            "it reports insufficient evidence (it does NOT fetch on the fly)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. NVDA"},
                "question": {
                    "type": "string",
                    "description": "The specific filing question to answer from the SEC text.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many filing chunks to ground on (default from settings).",
                },
            },
            "required": ["ticker", "question"],
        },
    },
    {
        "name": "research_summary",
        "description": (
            "INTEGRATED executive summary for one ticker fusing SEC FILINGS + NEWS + the price "
            "FORECAST into one cited brief: executive summary, business drivers, risk factors, "
            "bullish/bearish evidence, uncertainty notes, recent-news themes, the headline model "
            "forecasts (P(up)/E[r]/VaR95 per horizon), and key technical indicators. Use ONLY for "
            "explicit full-picture requests — 'give me the full picture / overview / executive "
            "summary / brief on TICKER'. This is the HEAVIEST tool (forecast + retrieval + a "
            "news-summary call + the memo synthesis ≈ 2 LLM calls, ~30–60s) — reserve it for those "
            "requests, not casual questions. Numbers come from the models; filing claims cited; "
            "NON-ADVISORY (no recommendation). Needs the ticker's filings ingested first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {
                    "type": "integer",
                    "default": 30,
                    "description": "News lookback window in days for the summary.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "research_multistep",
        "description": (
            "MULTI-HOP SEC filing research: answers a filing question whose evidence a SINGLE "
            "retrieval cannot gather, by running a bounded reason-retrieve-observe loop (up to 4 "
            "LLM calls) that collects evidence across a few retrieval steps before answering. Use "
            "when answering requires pulling and CONNECTING evidence from DIFFERENT filings, "
            "sections, periods, or entities, e.g.: (1) COMPARE/CONTRAST two+ companies or segments "
            "('compare NVDA vs AMD risk factors', 'how do their data-center segments differ'); "
            "(2) CHANGE OVER TIME ('what changed in TSLA's risk disclosures from 2023 to 2025'); "
            "(3) BRIDGING / follow-up where a later query depends on an earlier finding ('which of "
            "NVDA's named suppliers flag the same risk', 'who are its customers and are any "
            "concentrated'); (4) COMPOUND / multi-part questions ('what is X's AI strategy AND "
            "which stated risks threaten it'); (5) AGGREGATE across many sections/filings ('pull "
            "together every segment's stated headwinds'); (6) CAUSAL/CONSISTENCY links ('does "
            "management's MD&A optimism square with the risk factors'). For a SINGLE-topic, "
            "single-entity question use search_filings (cheaper). Returns ONE cited answer (each "
            "claim [n], grounded ONLY in retrieved filings) plus the step trace. Reports "
            "insufficient evidence if the relevant filings aren't ingested. Qualitative only — no "
            "probabilities. NON-ADVISORY."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The multi-hop filing question. Name the entities / periods / sub-parts "
                        "explicitly so the loop can scope each retrieval step."
                    ),
                },
            },
            "required": ["question"],
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
        retriever: RetrievalSystem | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry or build_default_registry(settings)
        self._llm = llm  # TextLLM for summarize_news (Role A); may be None
        # Per-session memoization of backtests so run_backtest + get_calibration on
        # the same (ticker, horizon, model) reuse one computation instead of two.
        self._backtest_cache: dict[tuple[str, int, str], BacktestResult] = {}
        # Explicit base-retriever override (tests / advanced wiring): when set, BOTH the single-shot
        # and multi-hop paths reuse it directly — no embedder is loaded, no graph DB opened. This is
        # DISTINCT from the lazy `_retriever` cache below: the override means "use exactly this",
        # whereas a filled cache only means "single-shot already ran this session" and must NOT
        # divert the multi-hop path off graph (the A5.3-promoted routing).
        self._injected_base: RetrievalSystem | None = retriever
        # Lazily-built, session-memoized single-shot RAG retriever (hybrid by default) so repeated
        # filing questions / the research summary don't reload the embedder or re-open the store.
        self._retriever: RetrievalSystem | None = None
        # A5.3: session-memoized GraphRetriever for the multi-hop path (graph traversal ⊕ hybrid).
        self._graph_retriever: RetrievalSystem | None = None

    def _get_retriever(self) -> RetrievalSystem:
        """Build (once) and return the shared SEC retriever for this session.

        Default-OFF rerank: ``build_retrieval_system`` returns the plain dense ``Retriever`` unless
        ``settings.rerank_provider`` is set, in which case it wraps it in a ``RerankingRetriever``.
        """
        if self._injected_base is not None:  # explicit override wins (tests / wiring)
            return self._injected_base
        if self._retriever is None:
            store = build_vector_store(self._settings)
            self._retriever = build_retrieval_system(self._settings, store=store)
            # One-time observability: surface which embedder + collection the agent is querying
            # (e.g. voyage-voyage-4 -> filings-voyage-voyage-4) so the active RAG corpus is visible.
            namespace = embedding_namespace(self._settings)
            try:
                n_chunks: int = store.count()
            except Exception as exc:  # noqa: BLE001 — observability must never break retrieval
                log.warning("agent.rag_retriever_count_failed", error=str(exc))
                n_chunks = -1
            log.info(
                "agent.rag_retriever",
                embedder=namespace,
                collection=collection_name_for(namespace),
                chunks=n_chunks,
            )
        return self._retriever

    def _get_multistep_retriever(self) -> RetrievalSystem:
        """Retriever for the A4 multi-hop loop — the A5.3-promoted GraphRetriever by default.

        Routing policy (measured over 2 seeds): the agentic/multi-hop path uses graph traversal
        (graph-built tickers bridge via stored edges; tickers without edges degrade to the hybrid
        base and rely on the still-ON alias-bridge), while single-shot ``search_filings`` keeps the
        plain hybrid retriever (single-shot graph regressed easy questions). Falls back to hybrid
        if ``graph_multistep_enabled`` is off, a base retriever was injected (tests), or
        graph wiring is unavailable — so graph is a safe enhancement, never a hard dependency.

        Routing keys off ``_injected_base`` (the explicit override), **not** the lazy ``_retriever``
        single-shot cache: a prior ``search_filings`` / ``research_summary`` in the same session
        fills that cache, and reusing it here would silently divert multi-hop OFF graph (the A5.3
        regression this method exists to avoid). So graph activation is now call-order-independent.
        """
        # An explicit base override (tests / wiring) takes precedence — reuse it directly so unit/
        # integration tests never load an embedder or open the graph DB. (The lazy single-shot cache
        # is deliberately NOT consulted here.)
        if self._injected_base is not None:
            return self._injected_base
        if not self._settings.graph_multistep_enabled:
            return self._get_retriever()
        if self._graph_retriever is None:
            try:
                self._graph_retriever = build_graph_system(self._settings)
                log.info("agent.graph_retriever_active")
            except Exception as exc:  # noqa: BLE001 — graph is an enhancement; degrade to hybrid
                log.warning("agent.graph_retriever_unavailable", error=str(exc))
                self._graph_retriever = self._get_retriever()
        return self._graph_retriever

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
        except Exception as exc:  # noqa: BLE001 — contract: execute NEVER raises (see docstring)
            # Final safety net for unexpected error types (IndexError on an empty series, an
            # unwrapped network/timeout error, etc.). Returning an error dict keeps the agent loop
            # and the deterministic UI path alive instead of crashing the turn; logged at error so
            # the underlying bug is still visible in observability.
            log.error(
                "agent.tool_crashed",
                tool=name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
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
        # Newest-first: "pull news" prioritizes the latest headlines (vs relevance ranking,
        # which summarize_news/get_news_sentiment keep for synthesis quality).
        bundle = NewsFetcher(self._registry).fetch(
            ticker, lookback_days=days, top_n=10, order="recency"
        )
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
        model = str(args.get("model", "ensemble"))
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

    # ---- multi-ticker comparison (Enhancement B) -----------------------------
    @staticmethod
    def _large_move_from_forecast(forecast: ScenarioForecast) -> tuple[float | None, int | None]:
        """P(|r|>k) at the horizon's inner positive bucket edge (None if no buckets).

        Reuses the same bucket-boundary logic as ``get_large_move`` so the batch
        magnitude number is consistent with the single-ticker tool.
        """
        valid_pcts = sorted(
            {round(b.lower * 100) for b in forecast.buckets if b.lower is not None and b.lower > 0}
        )
        if not valid_pcts:
            return None, None
        thr_pct = valid_pcts[0]
        breakdown = large_move_breakdown(forecast, threshold=thr_pct / 100.0)
        return breakdown.prob_large_move, thr_pct

    def _forecast_row(self, ticker: str, horizon: int, model: str) -> TickerForecastRow:
        """One ticker's forecast row; a per-ticker failure becomes an ``error`` row."""
        try:
            forecast = run_forecast(
                ticker, horizon, model_name=model, settings=self._settings, registry=self._registry
            )
        except (ProviderError, ValueError, KeyError) as exc:
            return TickerForecastRow(ticker=ticker, error=f"{type(exc).__name__}: {exc}")
        prob_lm, thr_pct = self._large_move_from_forecast(forecast)
        return TickerForecastRow(
            ticker=ticker,
            horizon_days=forecast.horizon_days,
            model_name=forecast.model_name,
            expected_return=forecast.expected_return,
            upside_prob=forecast.upside_prob,
            downside_prob=forecast.downside_prob,
            var_95=forecast.var_95,
            ci_low=forecast.ci_low,
            ci_high=forecast.ci_high,
            prob_large_move=prob_lm,
            large_move_threshold_pct=thr_pct,
            notes=forecast.notes,
        )

    def _tool_compare_forecasts(self, args: dict[str, Any]) -> dict[str, Any]:
        tickers, skipped = _parse_tickers(args)
        if len(tickers) < 2:
            return {
                "error": "compare_forecasts needs at least 2 tickers; use run_forecast for one."
            }
        horizon = int(args.get("horizon_days", 20))
        model = str(args.get("model", "ensemble"))
        if model not in MODEL_NAMES:
            return {"error": f"unknown model '{model}'; choose from {list(MODEL_NAMES)}"}

        def _build() -> list[TickerForecastRow]:
            return [self._forecast_row(t, horizon, model) for t in tickers]

        try:
            rows = _run_with_timeout(_build, _COMPARE_TIMEOUT_S)
        except TimeoutError:
            return {
                "error": (
                    f"comparison exceeded {int(_COMPARE_TIMEOUT_S)}s; "
                    "try fewer tickers or a shorter horizon."
                )
            }
        comparison = ForecastComparison(
            horizon_days=horizon, model=model, rows=rows, skipped=skipped
        )
        return comparison.model_dump(mode="json")

    # ---- conditional event study (news -> price bridge, #7) ------------------
    def _tool_conditional_outlook(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args["target"]).upper()
        driver = str(args["driver"]).upper()
        shock_pct = int(args.get("shock_pct", 5))
        event_window = int(args.get("event_window_days", 5))
        horizon = int(args.get("horizon_days", 10))
        direction = str(args.get("direction", "both"))
        if direction not in ("both", "up", "down"):
            return {"error": "direction must be 'both', 'up', or 'down'"}
        if not (1 <= shock_pct <= 50):
            return {"error": "shock_pct must be 1–50 (percent)"}
        if not (1 <= event_window <= 60):
            return {"error": "event_window_days must be 1–60"}
        if not (1 <= horizon <= 60):
            return {"error": "horizon_days must be 1–60"}
        from stock_agent.pipelines.conditional import run_conditional_study

        study = run_conditional_study(
            target,
            driver,
            shock_pct=shock_pct / 100.0,
            event_window_days=event_window,
            horizon_days=horizon,
            direction=direction,  # type: ignore[arg-type]
            settings=self._settings,
            registry=self._registry,
        )
        return study.model_dump(mode="json")

    def _news_row(self, ticker: str, days: int) -> TickerNewsRow:
        """One ticker's news/sentiment row; empty/failed fetch becomes an ``error`` row."""
        try:
            # Newest-first (like get_news): the comparison prioritizes the latest headlines.
            bundle = NewsFetcher(self._registry).fetch(
                ticker, lookback_days=days, top_n=25, order="recency"
            )
        except (ProviderError, ValueError, KeyError) as exc:
            return TickerNewsRow(ticker=ticker, error=f"{type(exc).__name__}: {exc}")
        if not bundle.articles:
            return TickerNewsRow(ticker=ticker, error="no news found in the window")
        # Numeric sentiment from Alpha Vantage scores (no LLM — keeps the batch cheap).
        feats = build_news_features(bundle, llm=None, use_llm_sentiment=False)
        headlines = [
            HeadlineRef(
                title=a.title,
                source=a.source,
                published=a.published_at.date(),
                url=str(a.url),
            )
            for a in bundle.articles[:5]  # newest-first; cap for a compact comparison
        ]
        return TickerNewsRow(
            ticker=ticker,
            article_count=int(feats["article_count"]),
            avg_sentiment=feats["avg_sentiment"],
            pct_positive=feats["pct_positive"],
            pct_negative=feats["pct_negative"],
            sentiment_source="alpha_vantage",
            top_headlines=headlines,
        )

    def _tool_compare_news(self, args: dict[str, Any]) -> dict[str, Any]:
        tickers, skipped = _parse_tickers(args)
        if len(tickers) < 2:
            return {"error": "compare_news needs at least 2 tickers; use get_news for one."}
        days = int(args.get("days", 14))

        def _build() -> list[TickerNewsRow]:
            return [self._news_row(t, days) for t in tickers]

        try:
            rows = _run_with_timeout(_build, _COMPARE_TIMEOUT_S)
        except TimeoutError:
            return {
                "error": (
                    f"comparison exceeded {int(_COMPARE_TIMEOUT_S)}s; try fewer tickers."
                )
            }
        return NewsComparison(days=days, rows=rows, skipped=skipped).model_dump(mode="json")

    # ---- topic / theme news (Enhancement C) ----------------------------------
    @staticmethod
    def _topic_meta(resolved: ResolvedTopic) -> dict[str, Any]:
        """Transparency block: what the theme phrase actually resolved to."""
        return {
            "topic": resolved.label,
            "known_topic": resolved.known,
            "keywords": list(resolved.keywords),
            "resolved_query": gdelt_query_expression(resolved),
        }

    @staticmethod
    def _reject_sentence_topic(topic: str) -> dict[str, Any] | None:
        """Guard the theme route: reject a pasted sentence/question, else return None.

        The theme tools take a SUBJECT PHRASE ("AI memory"), but the deterministic route feeds them
        the raw chat text — so a full question would become a literal-string query that silently
        matches nothing. When an UNKNOWN topic looks like a sentence, fail loudly with guidance
        instead (curated registry themes are exempt: they resolve regardless of length).
        """
        if resolve_topic(topic).known or not looks_like_sentence_topic(topic):
            return None
        return {
            "error": (
                "That looks like a full question, not a theme. The theme route needs a short "
                "subject phrase (e.g. 'AI memory', 'robotics', 'semiconductors') — type just the "
                "theme here, or switch to Auto (LLM) mode to ask a full question."
            ),
            "topic": topic,
            "known_themes": list(list_topics()),
        }

    @staticmethod
    def _topic_sentiment(articles: list[Any]) -> dict[str, Any] | None:
        """Average article tone over the articles that carry a score (None if none do).

        GDELT DOC ArtList supplies no per-article tone, so this is typically None —
        reported honestly rather than fabricated. Numbers come from the data only.
        """
        scored = [a.sentiment for a in articles if a.sentiment is not None]
        if not scored:
            return None
        return {
            "avg_tone": sum(scored) / len(scored),
            "tone_coverage": len(scored) / len(articles),
            "n_scored": len(scored),
        }

    def _tool_get_topic_news(self, args: dict[str, Any]) -> dict[str, Any]:
        topic = str(args["topic"])
        if (rejected := self._reject_sentence_topic(topic)) is not None:
            return rejected
        days = int(args.get("days", 14))
        resolved, bundle = TopicNewsFetcher(self._registry).fetch(
            topic,
            lookback_days=days,
            top_n=10,
            llm=self._llm,
            expand=self._settings.news_topic_expansion,
        )
        result = self._topic_meta(resolved)
        result["days"] = days
        result["articles"] = [
            {
                "title": a.title,
                "source": a.source,
                "published": a.published_at.date().isoformat(),
                "url": str(a.url),
            }
            for a in bundle.articles
        ]
        return result

    def _tool_analyze_topic_news(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._llm is None:
            return {"error": "topic-news analysis unavailable (no LLM configured)"}
        topic = str(args["topic"])
        if (rejected := self._reject_sentence_topic(topic)) is not None:
            return rejected
        days = int(args.get("days", 14))
        resolved, bundle = TopicNewsFetcher(self._registry).fetch(
            topic,
            lookback_days=days,
            top_n=25,
            llm=self._llm,
            expand=self._settings.news_topic_expansion,
        )
        result = self._topic_meta(resolved)
        result["days"] = days
        result["article_count"] = len(bundle.articles)
        if not bundle.articles:
            result["error"] = "no news found for this theme in the window"
            return result
        summary = summarize_news(
            bundle, self._llm, reflection_iterations=self._settings.news_reflection_iterations
        )
        result.update(summary.model_dump(mode="json"))
        # Topic sentiment, in preference order, all DATA-derived (never the LLM):
        #   1. GDELT ToneChart aggregate tone for the resolved keywords,
        #   2. mean per-article tone if the source carried any, else None.
        end = Date.today()
        start = end - timedelta(days=days)
        tone = self._registry.get_topic_tone(resolved.keywords, start, end)
        result["topic_sentiment"] = (
            tone if tone is not None else self._topic_sentiment(bundle.articles)
        )
        return result

    def _tool_get_large_move(self, args: dict[str, Any]) -> dict[str, Any]:
        ticker = str(args["ticker"]).upper()
        horizon = int(args.get("horizon_days", 20))
        # The two validated big-move models: logistic (default; best for stable names) and
        # regularized lightgbm (better for volatile names). Both fall back to the baseline
        # (with a note) at horizons without a trained artifact.
        model = str(args.get("model", "logistic"))
        if model not in ("logistic", "lightgbm"):
            return {"error": "model must be 'logistic' or 'lightgbm' for the big-move signal"}
        forecast = run_forecast(
            ticker, horizon, model_name=model, settings=self._settings, registry=self._registry
        )
        # Valid thresholds are the forecast's positive bucket boundaries, which scale
        # with horizon (h20 -> 5/10, h30 -> 10/20, h60 -> 15/30). Default to the inner
        # boundary for the horizon so "big" stays informative as the horizon grows.
        valid_pcts = sorted(
            {round(b.lower * 100) for b in forecast.buckets if b.lower is not None and b.lower > 0}
        )
        threshold_pct = int(args.get("threshold_pct", valid_pcts[0] if valid_pcts else 10))
        if threshold_pct not in valid_pcts:
            return {
                "error": (
                    f"threshold_pct must be one of {valid_pcts} at horizon {horizon} "
                    "(bucket boundaries scale with horizon)."
                )
            }
        breakdown = large_move_breakdown(forecast, threshold=threshold_pct / 100.0)
        result = breakdown.model_dump(mode="json")
        result["threshold_pct"] = threshold_pct
        # Honest static trust framing — a precomputed per-ticker skill scorecard
        # (backtested AUC/calibration) is the planned next step.
        result["reliability"] = (
            "The large-move total P(|r|>k) is the model's most reliable signal in backtesting; "
            "the up/down split shows which tail leans but is rarer and noisier. The threshold k "
            "scales with horizon (5% at 20d, 10% at 30d, 15% at 60d) so it stays informative. "
            "A magnitude signal, not a directional call on the median."
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
            "interval_coverage": (
                {
                    "nominal_ci": round(r.conformal.ci_level, 2),
                    "empirical_coverage": round(r.conformal.empirical_coverage, 3),
                    "conformalized_coverage": (
                        round(r.conformal.conformalized_coverage, 3)
                        if r.conformal.conformalized_coverage is not None
                        else None
                    ),
                    "n_eval": r.conformal.n_eval,
                }
                if r.conformal is not None
                else None
            ),
            "interpretation_key": (
                "ECE is the average gap between predicted probability and realized frequency "
                "(0 = perfectly calibrated). In the reliability table, 'realized' below "
                "'predicted' means the model was overconfident in that probability band. "
                "interval_coverage.empirical_coverage is what the model's stated CI actually "
                "covered OOS — far from nominal_ci means the interval is mis-sized."
            ),
            "methodology": (
                f"Walk-forward out-of-sample, embargo = horizon, {r.n_folds} folds, "
                f"{c.n_bins}-bin reliability."
            ),
        }

    # ---- RAG: SEC-grounded filing QA + integrated summary (P8.5) --------------
    @staticmethod
    def _citations_payload(citations: list[Any]) -> list[dict[str, Any]]:
        """Flatten ``SourceCitation`` objects to the compact citation dicts tools return."""
        return [{"marker": c.marker, "label": c.label, "chunk_id": c.chunk_id} for c in citations]

    def _tool_search_filings(self, args: dict[str, Any]) -> dict[str, Any]:
        # Mirrors summarize_news: the tool makes its own guarded synthesis call, so the
        # citation guard + number grounding (P7) stay intact rather than handing raw chunks
        # to the agent. No LLM -> no synthesis possible.
        if self._llm is None:
            return {"error": "filing search unavailable (no LLM configured)"}
        ticker = str(args["ticker"]).upper()
        question = str(args["question"]).strip()
        if not question:
            return {"error": "search_filings needs a non-empty 'question'."}
        top_k = int(args.get("top_k", self._settings.rag_top_k))
        evidence = self._get_retriever().retrieve(
            question, top_k=top_k, where=ChunkFilter(ticker=ticker)
        )
        if evidence.is_empty:
            # Honest refusal (P7) + a hint that the corpus for this ticker isn't ingested.
            # The tool never downloads/ingests on the fly (too slow for a chat turn).
            return {
                "ticker": ticker,
                "answer": "Insufficient evidence found.",
                "insufficient_evidence": True,
                "n_sources": 0,
                "citations": [],
                "hint": (
                    f"No ingested SEC filings for {ticker}. Run "
                    f"`documents download-sec --ticker {ticker}` then "
                    f"`documents ingest --ticker {ticker}` to enable filing search."
                ),
            }
        try:
            answer = answer_question(question, evidence, llm=self._llm)
        except (ResearchGuardError, LLMError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "ticker": ticker,
            "answer": answer.answer,
            "insufficient_evidence": answer.insufficient_evidence,
            "n_sources": len(evidence.chunks),
            "citations": self._citations_payload(answer.citations),
        }

    def _tool_research_multistep(self, args: dict[str, Any]) -> dict[str, Any]:
        # A4: bounded ReAct multi-hop filing QA. Reuses the session retriever (retrieval between
        # steps is $0/local) and ends in the P7 guarded answer_question over the accumulated union,
        # so the citation + number guards stay intact; the agent just relays the cited result.
        # Wall-clock bounded like research_summary (agentic_max_steps decisions + 1 terminal call).
        if self._llm is None:
            return {"error": "multi-step filing research unavailable (no LLM configured)"}
        question = str(args.get("question", "")).strip()
        if not question:
            return {"error": "research_multistep needs a non-empty 'question'."}

        def _run() -> MultiStepAnswer:
            return answer_multistep(
                question,
                settings=self._settings,
                llm=self._llm,  # type: ignore[arg-type]  # guarded non-None above
                # A5.3: multi-hop path runs over the GraphRetriever (traversal-based bridging for
                # graph-built tickers; hybrid base + alias-bridge fallback otherwise).
                retriever=self._get_multistep_retriever(),
            )

        try:
            result = _run_with_timeout(_run, _RESEARCH_TIMEOUT_S)
        except (ResearchGuardError, LLMError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        except TimeoutError:
            secs = int(_RESEARCH_TIMEOUT_S)
            return {"error": f"multi-step research exceeded {secs}s; try again later."}
        ans = result.answer
        out: dict[str, Any] = {
            "question": question,
            "answer": ans.answer,
            "insufficient_evidence": ans.insufficient_evidence,
            "n_steps": result.n_steps,
            "n_evidence": result.n_evidence,
            "citations": self._citations_payload(ans.citations),
            # Compact transparency trace: the queries the loop issued + how much each returned.
            "trace": [
                {"query": st.query, "ticker": st.ticker, "n_retrieved": st.n_retrieved}
                for st in result.trace
            ],
        }
        if ans.insufficient_evidence and result.n_evidence == 0:
            # Nothing retrieved across all steps -> likely the relevant filings aren't ingested.
            out["hint"] = (
                "No SEC filing evidence was retrieved for this question. Ensure the relevant "
                "tickers' filings are ingested (`documents download-sec` then `documents ingest`)."
            )
        return out

    @staticmethod
    def _forecast_headline(forecasts: list[ScenarioForecast]) -> list[dict[str, Any]]:
        """Compact per-horizon forecast rows for the summary (numbers straight from the model)."""
        return [
            {
                "model": fc.model_name,
                "horizon_days": fc.horizon_days,
                "prob_up": fc.upside_prob,
                "expected_return": fc.expected_return,
                "var_95": fc.var_95,
            }
            for fc in forecasts
        ]

    @staticmethod
    def _forecast_buckets_payload(forecasts: list[ScenarioForecast]) -> list[dict[str, Any]]:
        """Per-horizon scenario-bucket distributions so the client can chart the brief.

        The bucket probabilities are the model's own numbers (never the LLM's); the chart
        builder (viz.charts._research_forecast_charts) renders one bar chart per horizon,
        since each horizon carries its own horizon-scaled return bands.
        """
        return [
            {
                "model_name": fc.model_name,
                "horizon_days": fc.horizon_days,
                "buckets": [
                    {"label": b.label, "probability": b.probability} for b in fc.buckets
                ],
            }
            for fc in forecasts
        ]

    def _tool_research_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        # Heaviest tool: P8's run_research (forecast + SEC retrieval + news summary + the memo
        # synthesis call). Numbers + filing citations are already guarded inside the memo, so
        # the agent just relays the compact result. Bound the wall-clock like run_backtest.
        if self._llm is None:
            return {"error": "research summary unavailable (no LLM configured)"}
        ticker = str(args["ticker"]).upper()
        days = int(args.get("days", DEFAULT_NEWS_LOOKBACK_DAYS))

        def _run() -> ResearchMemo:
            return run_research(
                ticker,
                settings=self._settings,
                registry=self._registry,
                llm=self._llm,
                days=days,
                retriever=self._get_retriever(),  # reuse the session embedder + store
            )

        try:
            memo = _run_with_timeout(_run, _RESEARCH_TIMEOUT_S)
        except ResearchPipelineError as exc:  # RuntimeError — outside dispatch's caught tuple
            return {"error": str(exc)}
        except TimeoutError:
            return {
                "error": f"research summary exceeded {int(_RESEARCH_TIMEOUT_S)}s; try again later."
            }
        # Compact dict (NOT the full Markdown — too long for a tool result), PLUS a
        # ready-to-present `brief_markdown`: the deterministic rich brief (tables from the numbers +
        # the memo's grounded prose). The runtime presents this verbatim (short-circuiting the
        # answer-turn re-narration) so the executive brief renders consistently and no figure is
        # transcribed by the LLM (see agent.runtime + research.brief).
        return {
            "ticker": memo.ticker,
            "as_of": memo.as_of.isoformat(),
            "brief_markdown": render_research_brief(memo),
            "executive_summary": memo.executive_summary,
            "business_drivers": memo.business_drivers,
            "risk_factors": memo.risk_factors,
            "bullish_evidence": memo.bullish_evidence,
            "bearish_evidence": memo.bearish_evidence,
            "uncertainty_notes": memo.uncertainty_notes,
            "recent_news": memo.recent_news,
            "news": memo.news.model_dump() if memo.news else None,
            "forecasts": self._forecast_headline(memo.forecasts),
            "forecast_buckets": self._forecast_buckets_payload(memo.forecasts),
            "technical_indicators": memo.technical_indicators,
            # Consolidated brief signals (were separate tools): recent price window, the
            # large-move tail split, and earnings context — so one call carries the full brief.
            "price_snapshot": memo.price_snapshot.model_dump(mode="json")
            if memo.price_snapshot
            else None,
            "large_move": memo.large_move.model_dump(mode="json") if memo.large_move else None,
            "earnings": memo.earnings.model_dump(mode="json") if memo.earnings else None,
            "citations": self._citations_payload(memo.citations),
        }
