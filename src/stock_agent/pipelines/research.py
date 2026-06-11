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
from stock_agent.rag.embeddings import build_embedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import build_vector_store
from stock_agent.research.memo import build_memo
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


def _gather_sec_evidence(ticker: str, settings: Settings, per_query_k: int) -> EvidenceSet:
    """Retrieve + merge SEC chunks across the memo's section queries (deduped, score-capped)."""
    retriever = Retriever(build_embedder(settings), build_vector_store(settings))
    where = ChunkFilter(ticker=ticker)
    best: dict[str, RetrievedChunk] = {}
    for query in _MEMO_QUERIES:
        for rc in retriever.retrieve(query, top_k=per_query_k, where=where).chunks:
            current = best.get(rc.chunk.chunk_id)
            if current is None or rc.score > current.score:
                best[rc.chunk.chunk_id] = rc
    ranked = sorted(best.values(), key=lambda rc: rc.score, reverse=True)[:_MEMO_EVIDENCE_CAP]
    return EvidenceSet(query=f"{ticker} research memo", chunks=ranked)


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
) -> ResearchMemo:
    """Gather technicals + forecast + news + SEC evidence, then build the integrated memo."""
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
    evidence = _gather_sec_evidence(ticker, settings, settings.rag_top_k)
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

    return build_memo(
        ticker,
        as_of,
        snapshot=snapshot,
        forecasts=forecasts,
        evidence=evidence,
        llm=client,
        news_summary=news_summary,
    )
