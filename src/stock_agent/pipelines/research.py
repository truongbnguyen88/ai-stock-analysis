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
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import AnthropicClient, LLMError, TextLLM
from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.rag.rerank import build_retrieval_system
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.research.memo import MemoGuardError, build_memo
from stock_agent.schemas.research import ResearchMemo
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

log = get_logger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 60)
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


def run_research(
    ticker: str,
    *,
    settings: Settings,
    registry: ProviderRegistry | None = None,
    llm: TextLLM | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    days: int = 30,
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

    # Prices -> indicators -> baseline forecasts.
    series = PriceLoader(registry).load_recent(ticker, _PRICE_LOOKBACK_DAYS, min_bars=30).series
    snapshot = compute_snapshot(series)
    as_of = series.bars[-1].date
    model = HistoricalSimulation()
    forecasts = []
    for h in horizons:
        try:
            forecasts.append(model.forecast(series, horizon_days=h, as_of=as_of))
        except ValueError:
            log.info("research.skip_horizon", ticker=ticker, horizon=h)

    # SEC evidence (RAG). Empty when nothing has been ingested for the ticker.
    evidence = _gather_sec_evidence(ticker, settings, settings.rag_top_k, retriever=retriever)
    if evidence.is_empty:
        log.warning("research.no_sec_evidence", ticker=ticker)

    # News summary (optional; degrade gracefully on failure).
    news_summary = None
    if use_news:
        try:
            bundle = NewsFetcher(registry).fetch(
                ticker, lookback_days=days, company_name=company_name, top_n=25
            )
            news_summary = summarize_news(bundle, client)
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
        )
    except (MemoGuardError, LLMError, ValidationError) as exc:
        raise ResearchPipelineError(f"Memo synthesis failed: {exc}") from exc
