"""RAG pipeline (P6) — ingest downloaded filings into the vector store.

The write path: walk the filings already downloaded for a ticker (P1), parse (P2) + chunk (P3)
each, embed the chunks once (P4), and upsert them into the vector store (P5). Ingestion is
idempotent — the store upserts by ``chunk_id``, so re-running replaces rather than duplicates.

Thin orchestration only: it wires the existing pure stages together and does the disk walk.
Embedder + store are injected (the CLI builds them from settings; tests pass fakes).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel

from stock_agent.documents.parsers import load_filing
from stock_agent.rag.chunking import chunk_filing, estimate_tokens
from stock_agent.rag.embeddings import Embedder
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
    chunks: int
    embed_tokens: int = 0  # estimated embedding tokens (spend-guard proxy, RAG_TODO 9a)


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
) -> IngestResult:
    """Parse → chunk → embed → upsert every downloaded filing for ``ticker``.

    Embeds all of the ticker's chunks in one batch, then a single ``store.add`` (upsert).
    Re-ingesting the same corpus is a no-op on count (ids are stable).

    ``max_embed_tokens`` (RAG_TODO 9a) is a client-side spend ceiling: when the estimated
    embedding tokens exceed it, the run is **refused before any embedding call** (raising
    ``EmbedBudgetExceeded``) — nothing is embedded or stored. ``None`` disables the guard.
    """
    dirs = iter_filing_dirs(documents_dir, ticker)
    chunks = build_chunks(
        ticker, documents_dir=documents_dir, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap
    )
    embed_tokens = estimate_tokens([c.text for c in chunks]) if chunks else 0
    # Refuse BEFORE embedding so an over-budget run incurs no provider spend.
    if max_embed_tokens is not None and embed_tokens > max_embed_tokens:
        raise EmbedBudgetExceeded(ticker, embed_tokens, max_embed_tokens)
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
    log.info(
        "rag.ingest", ticker=ticker, filings=len(dirs), chunks=len(chunks),
        embed_tokens=embed_tokens,
    )
    return IngestResult(
        ticker=ticker, filings=len(dirs), chunks=len(chunks), embed_tokens=embed_tokens
    )
