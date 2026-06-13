"""RAG pipeline (P6) — ingest downloaded filings into the vector store.

The write path: walk the filings already downloaded for a ticker (P1), parse (P2) + chunk (P3)
each, embed the chunks once (P4), and upsert them into the vector store (P5). Ingestion is
idempotent — the store upserts by ``chunk_id``, so re-running replaces rather than duplicates.

Thin orchestration only: it wires the existing pure stages together and does the disk walk.
Embedder + store are injected (the CLI builds them from settings; tests pass fakes).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from stock_agent.documents.parsers import load_filing
from stock_agent.rag.chunking import chunk_filing, estimate_tokens
from stock_agent.rag.embeddings import Embedder
from stock_agent.rag.sparse_store import SparseStore
from stock_agent.rag.vector_store import VectorStore
from stock_agent.schemas.documents import DocumentChunk

log = structlog.get_logger(__name__)

_DEFAULT_CHUNK_TOKENS = 900
_DEFAULT_CHUNK_OVERLAP = 0.15


class EmbedBudgetExceeded(RuntimeError):
    """Ingest refused because the estimated embedding tokens exceed the configured ceiling.

    A client-side hard stop (``settings.rag_max_embed_tokens``, RAG_TODO 9a) that aborts the
    run **before** any embedding call — independent of the provider's own dashboard limits.
    Protects the one-time paid voyage ingest (9c) from a surprise over-spend on a re-embed.
    """

    def __init__(self, ticker: str, estimated: int, ceiling: int) -> None:
        self.ticker = ticker
        self.estimated = estimated
        self.ceiling = ceiling
        super().__init__(
            f"{ticker}: estimated {estimated:,} embed tokens exceeds ceiling {ceiling:,} "
            f"(raise settings.rag_max_embed_tokens to proceed)"
        )


class IngestResult(BaseModel):
    """Summary of one ingestion run."""

    ticker: str
    filings: int
    chunks: int  # chunks embedded + stored this run (in incremental mode: only the NEW ones)
    embed_tokens: int = 0  # estimated embedding tokens (spend-guard proxy, RAG_TODO 9a)
    skipped_existing: int = 0  # chunks already in the store, skipped (incremental mode, 9e)


def iter_filing_dirs(documents_dir: Path, ticker: str) -> list[Path]:
    """Downloaded filing directories for ``ticker`` (each holds ``filing.html`` + metadata).

    Layout is ``{documents_dir}/sec/{TICKER}/{FORM}/{filing_date}/`` (see documents/download).
    Returns a sorted list (deterministic) — empty if nothing has been downloaded.
    """
    base = documents_dir / "sec" / ticker
    if not base.exists():
        return []
    return sorted(p.parent for p in base.rglob("filing.html"))


def build_chunks(
    ticker: str,
    *,
    documents_dir: Path,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Parse + chunk every downloaded filing for ``ticker`` (no embedding, no store).

    The shared write-path front half: reused by ``ingest_ticker`` and by the retrieval-eval
    harness (P9b), which embeds the *same* chunks with different embedders. Deterministic
    (``iter_filing_dirs`` is sorted).
    """
    chunks: list[DocumentChunk] = []
    for directory in iter_filing_dirs(documents_dir, ticker):
        parsed = load_filing(directory)
        chunks.extend(chunk_filing(parsed, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap))
    return chunks


def ingest_ticker(
    ticker: str,
    *,
    documents_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
    max_embed_tokens: int | None = None,
    incremental: bool = False,
    sparse_store: SparseStore | None = None,
) -> IngestResult:
    """Parse → chunk → embed → upsert every downloaded filing for ``ticker``.

    Embeds the ticker's chunks in one batch, then a single ``store.add`` (upsert).
    Re-ingesting the same corpus is a no-op on count (ids are stable).

    ``incremental`` (RAG_TODO 9e — quarterly refresh): when True, chunks whose ``chunk_id`` is
    already in the store are **skipped** (not re-embedded), so a refresh only pays to embed *new*
    filings' chunks. Without it a re-ingest re-embeds the whole ticker (full upsert) — fine for a
    one-off rebuild, far too costly for a recurring refresh against a paid embedder.

    ``max_embed_tokens`` (RAG_TODO 9a) is a client-side spend ceiling: when the estimated
    embedding tokens (of the chunks actually being embedded) exceed it, the run is **refused
    before any embedding call** (raising ``EmbedBudgetExceeded``). ``None`` disables the guard.

    ``sparse_store`` (advanced-RAG A3) keeps the BM25 index in lockstep with the vector store —
    indexing is $0 (no embedding). It uses the *sparse* store's own ``existing_ids`` for the
    incremental skip, so enabling hybrid on an existing corpus backfills the BM25 index on the next
    ingest/refresh without re-embedding anything.
    """
    dirs = iter_filing_dirs(documents_dir, ticker)
    all_chunks = build_chunks(
        ticker, documents_dir=documents_dir, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap
    )
    chunks = all_chunks
    skipped = 0
    if incremental and all_chunks:
        present = store.existing_ids([c.chunk_id for c in all_chunks])
        skipped = len(present)
        chunks = [c for c in all_chunks if c.chunk_id not in present]  # embed only the new ones
    embed_tokens = estimate_tokens([c.text for c in chunks]) if chunks else 0
    # Refuse BEFORE embedding so an over-budget run incurs no provider spend.
    if max_embed_tokens is not None and embed_tokens > max_embed_tokens:
        raise EmbedBudgetExceeded(ticker, embed_tokens, max_embed_tokens)
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
    sparse_added = _index_sparse(sparse_store, all_chunks, incremental=incremental)
    log.info(
        "rag.ingest", ticker=ticker, filings=len(dirs), chunks=len(chunks),
        embed_tokens=embed_tokens, skipped_existing=skipped, sparse_added=sparse_added,
    )
    return IngestResult(
        ticker=ticker, filings=len(dirs), chunks=len(chunks),
        embed_tokens=embed_tokens, skipped_existing=skipped,
    )


def _index_sparse(
    sparse_store: SparseStore | None, chunks: Sequence[DocumentChunk], *, incremental: bool
) -> int:
    """Add ``chunks`` to the sparse (BM25) index, skipping already-indexed ids when incremental.

    Uses the sparse store's OWN ``existing_ids`` (not the vector store's), so the BM25 index can
    backfill independently. Returns how many chunks were newly indexed. No-op when ``sparse_store``
    is None (hybrid off).
    """
    if sparse_store is None or not chunks:
        return 0
    to_add = list(chunks)
    if incremental:
        present = sparse_store.existing_ids([c.chunk_id for c in chunks])
        to_add = [c for c in chunks if c.chunk_id not in present]
    if to_add:
        sparse_store.add(to_add)
    return len(to_add)


class BulkIngestResult(BaseModel):
    """Aggregate summary of a multi-ticker ingest run (RAG_TODO 9c-run hardening)."""

    tickers: int
    chunks: int  # total chunks embedded + stored across all tickers (new ones in incremental mode)
    embed_tokens: int  # total proxy embed tokens
    skipped_existing: int = 0  # total chunks already present, skipped (incremental, RAG_TODO 9e)
    failed_tickers: list[str] = Field(default_factory=list)  # "TICKER: message"
    per_ticker: list[IngestResult] = Field(default_factory=list)


def bulk_ingest(
    tickers: Sequence[str],
    *,
    documents_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
    max_embed_tokens: int | None = None,
    incremental: bool = False,
    sparse_store: SparseStore | None = None,
    retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> BulkIngestResult:
    """Ingest many tickers, surviving transient per-ticker failures (RAG_TODO 9c-run hardening).

    A whole-corpus embed makes thousands of provider calls over ~hours, so a single transient blip
    (e.g. a dropped connection, as hit the first voyage embed) must **not** abort the run. Each
    ticker is retried up to ``retries`` times (linear backoff); a ticker that still fails is
    recorded in ``failed_tickers`` and the run **continues**. Idempotent (``ingest_ticker`` upserts
    by ``chunk_id``), so a re-run backfills exactly the failures.

    ``EmbedBudgetExceeded`` is deliberately **not** isolated — it's an intentional spend ceiling
    (9a), so it propagates and aborts the run (the caller decides whether to raise the ceiling).
    """
    syms = list(tickers)
    per_ticker: list[IngestResult] = []
    failed: list[str] = []
    for sym in syms:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                per_ticker.append(
                    ingest_ticker(
                        sym,
                        documents_dir=documents_dir,
                        embedder=embedder,
                        store=store,
                        chunk_tokens=chunk_tokens,
                        chunk_overlap=chunk_overlap,
                        max_embed_tokens=max_embed_tokens,
                        incremental=incremental,
                        sparse_store=sparse_store,
                    )
                )
                last_exc = None
                break
            except EmbedBudgetExceeded:
                raise  # deliberate budget stop — abort the whole run (not a transient failure)
            except Exception as exc:  # noqa: BLE001 — isolate transient/per-ticker embed failures
                last_exc = exc
                if attempt < retries:
                    log.warning(
                        "rag.ingest_retry", ticker=sym, attempt=attempt + 1, error=str(exc)
                    )
                    sleep(2.0 * (attempt + 1))  # linear backoff: 2s, 4s
        if last_exc is not None:
            log.warning("rag.bulk_ingest_ticker_failed", ticker=sym, error=str(last_exc))
            failed.append(f"{sym}: {last_exc}")
    return BulkIngestResult(
        tickers=len(syms),
        chunks=sum(r.chunks for r in per_ticker),
        embed_tokens=sum(r.embed_tokens for r in per_ticker),
        skipped_existing=sum(r.skipped_existing for r in per_ticker),
        failed_tickers=failed,
        per_ticker=per_ticker,
    )
