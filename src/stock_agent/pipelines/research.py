"""Research-memo pipeline (RAG P8) — gather everything, then one grounded memo call.

Orchestration only: it reuses the existing building blocks (prices → indicators → forecast,
news summary) and adds RAG retrieval of SEC evidence, then hands the gathered inputs to
``research.memo.build_memo`` (the single synthesis call). The memo *is* the synthesis, so an
LLM is required (unlike ``analyze``, which degrades to a synthesis-free report).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from stock_agent.data.loader import PriceLoader
from stock_agent.features.news_features import build_news_features
from stock_agent.forecasting.ensemble import full_ensemble
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import AnthropicClient, LLMError, TextLLM
from stock_agent.llm.guards import NewsSummary
from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.pipelines.forecast import apply_conformal
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.rag.read_path import build_retrieval_system
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.research.memo import MemoGuardError, build_memo
from stock_agent.schemas.news import NewsBundle
from stock_agent.schemas.research import NewsAnalysis, ResearchMemo
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

log = get_logger(__name__)

# 20/30/60 are the horizons with trained pooled-ML artifacts AND exact return bands
# (see forecasting.buckets._HORIZON_BANDS), so the ensemble seats all 5 members at each.
DEFAULT_HORIZONS: tuple[int, ...] = (20, 30, 60)
# Recent-news lookback for the brief: 3 weeks captures the current news cycle without
# diluting it with stale items. Shared by the CLI `research` command and the agent's
# `research_summary` tool so the two never diverge.
DEFAULT_NEWS_LOOKBACK_DAYS = 21
_PRICE_LOOKBACK_DAYS = 420
# Targeted retrievals so the memo's filing-grounded sections each get coverage; the
# results are merged + deduped, then capped (more sources = more prompt tokens / cost).
_MEMO_QUERIES = (
    "material risk factors and risks to the business",
    "business drivers, demand, products, and growth strategy",
    "management discussion of results of operations, revenue, and outlook",
)
_MEMO_EVIDENCE_CAP = 10


class ResearchPipelineError(RuntimeError):
    """Raised when the memo cannot be produced (e.g. no LLM available)."""


def _round_robin_merge(
    result_lists: Sequence[Sequence[RetrievedChunk]], cap: int
) -> list[RetrievedChunk]:
    """Interleave per-query results so each query is fairly represented (section diversity).

    Pulls the next-best unseen chunk from each query's list in turn (round-robin), deduping by
    ``chunk_id``, until ``cap`` chunks are chosen or every list is exhausted. This stops one
    verbose section (e.g. Risk Factors, whose chunks score highest across *every* query) from
    crowding out the others — the failure mode a plain global score-cap had. Returned sorted by
    score for presentation; diversity comes from *which* chunks are picked, not their order.
    """
    selected: dict[str, RetrievedChunk] = {}
    cursors = [0] * len(result_lists)
    progressed = True
    while len(selected) < cap and progressed:
        progressed = False
        for qi, chunks in enumerate(result_lists):
            while cursors[qi] < len(chunks) and chunks[cursors[qi]].chunk.chunk_id in selected:
                cursors[qi] += 1  # skip a chunk an earlier query already claimed
            if cursors[qi] < len(chunks):
                rc = chunks[cursors[qi]]
                selected[rc.chunk.chunk_id] = rc
                cursors[qi] += 1
                progressed = True
                if len(selected) >= cap:
                    break
    return sorted(selected.values(), key=lambda rc: rc.score, reverse=True)


def _gather_sec_evidence(
    ticker: str,
    settings: Settings,
    per_query_k: int,
    *,
    retriever: RetrievalSystem | None = None,
) -> EvidenceSet:
    """Retrieve SEC chunks per memo-section query, then round-robin merge for balanced coverage."""
    # Default-OFF rerank: build_retrieval_system returns the plain dense Retriever unless
    # settings.rerank_provider is set, in which case it wraps it in a RerankingRetriever.
    retriever = retriever or build_retrieval_system(settings)
    where = ChunkFilter(ticker=ticker)
    per_query = [
        retriever.retrieve(q, top_k=per_query_k, where=where).chunks for q in _MEMO_QUERIES
    ]
    chunks = _round_robin_merge(per_query, _MEMO_EVIDENCE_CAP)
    return EvidenceSet(query=f"{ticker} research memo", chunks=chunks)


def _news_analysis(
    summary: NewsSummary, bundle: NewsBundle, *, lookback_days: int
) -> NewsAnalysis:
    """Flatten the LLM news summary + free provider-sentiment shares into the pure schema.

    Sentiment shares come from ``build_news_features`` (provider ``article.sentiment``
    scores, no LLM). They are ``None`` — not 0.0 — when no article carried a score, so a
    downstream chart no-ops instead of drawing a misleading "all neutral" bar.
    """
    feats = build_news_features(bundle)  # free AV scores; no LLM call
    has_scores = (feats.get("sentiment_coverage") or 0.0) > 0.0
    return NewsAnalysis(
        lookback_days=lookback_days,
        article_count=int(feats.get("article_count", 0.0)),
        overview=summary.overview,
        key_themes=list(summary.key_themes),
        bullish=[p.point for p in summary.bullish],
        bearish=[p.point for p in summary.bearish],
        risks=[p.point for p in summary.risks],
        catalysts=[p.point for p in summary.catalysts],
        pct_positive=feats["pct_positive"] if has_scores else None,
        pct_negative=feats["pct_negative"] if has_scores else None,
        sentiment_coverage=feats.get("sentiment_coverage"),
    )


def run_research(
    ticker: str,
    *,
    settings: Settings,
    registry: ProviderRegistry | None = None,
    llm: TextLLM | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    company_name: str | None = None,
    use_news: bool = True,
    retriever: RetrievalSystem | None = None,
) -> ResearchMemo:
    """Gather technicals + forecast + news + SEC evidence, then build the integrated memo.

    ``retriever`` lets a caller (e.g. the chat agent) inject a session-memoized retriever so
    the embedder + vector store aren't rebuilt per call; when ``None`` one is built from settings.
    """
    registry = registry or build_default_registry(settings)
    ticker = ticker.upper()

    client = llm or (AnthropicClient(settings) if settings.anthropic_api_key else None)
    if client is None:
        raise ResearchPipelineError(
            "The research memo requires an LLM. Set ANTHROPIC_API_KEY in your .env."
        )

    # Prices -> indicators -> ensemble forecasts (same model + conformal calibration the
    # standalone `run_forecast` serves, so the brief and a direct forecast query agree).
    # `full_ensemble` seats 3 stateless members + pooled logistic/lightgbm; an ML member
    # with no trained artifact for the horizon self-drops (see forecasting.ensemble).
    series = PriceLoader(registry).load_recent(ticker, _PRICE_LOOKBACK_DAYS, min_bars=30).series
    snapshot = compute_snapshot(series)
    as_of = series.bars[-1].date
    model = full_ensemble(registry)
    forecasts = []
    for h in horizons:
        try:
            fc = model.forecast(series, horizon_days=h, as_of=as_of)
            forecasts.append(apply_conformal(fc, settings))
        except ValueError:
            log.info("research.skip_horizon", ticker=ticker, horizon=h)

    # SEC evidence (RAG). Empty when nothing has been ingested for the ticker.
    evidence = _gather_sec_evidence(ticker, settings, settings.rag_top_k, retriever=retriever)
    if evidence.is_empty:
        log.warning("research.no_sec_evidence", ticker=ticker)

    # Recent-news pull (`days`-window) + qualitative analysis (optional; degrade gracefully).
    # The summary feeds the memo's grounding + narrative; `news_analysis` is the structured
    # block surfaced to the client (themes + insights + free provider-sentiment shares).
    news_summary = None
    news_analysis = None
    if use_news:
        try:
            bundle = NewsFetcher(registry).fetch(
                ticker, lookback_days=days, company_name=company_name, top_n=25
            )
            news_summary = summarize_news(bundle, client)
            news_analysis = _news_analysis(news_summary, bundle, lookback_days=days)
        except (LLMError, SummaryGuardError, ValueError, ValidationError) as exc:
            log.warning("research.news_failed", ticker=ticker, error=str(exc))

    # The memo IS the synthesis (no graceful no-LLM fallback) — surface a clean error rather
    # than a traceback if the call fails or its guards reject the output after the retry.
    try:
        return build_memo(
            ticker,
            as_of,
            snapshot=snapshot,
            forecasts=forecasts,
            evidence=evidence,
            llm=client,
            news_summary=news_summary,
            news_analysis=news_analysis,
        )
    except (MemoGuardError, LLMError, ValidationError) as exc:
        raise ResearchPipelineError(f"Memo synthesis failed: {exc}") from exc
