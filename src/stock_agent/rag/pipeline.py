"""RAG pipeline (P6) — ingest downloaded filings into the vector store.

The write path: walk the filings already downloaded for a ticker (P1), parse (P2) + chunk (P3)
each, embed the chunks once (P4), and upsert them into the vector store (P5). Ingestion is
idempotent — the store upserts by ``chunk_id``, so re-running replaces rather than duplicates.

Thin orchestration only: it wires the existing pure stages together and does the disk walk.
Embedder + store are injected (the CLI builds them from settings; tests pass fakes).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from stock_agent.documents.parsers import load_filing
from stock_agent.rag.chunking import chunk_filing
from stock_agent.rag.embeddings import Embedder
from stock_agent.rag.vector_store import VectorStore

_DEFAULT_CHUNK_TOKENS = 900
_DEFAULT_CHUNK_OVERLAP = 0.15


class IngestResult(BaseModel):
    """Summary of one ingestion run."""

    ticker: str
    filings: int
    chunks: int


def iter_filing_dirs(documents_dir: Path, ticker: str) -> list[Path]:
    """Downloaded filing directories for ``ticker`` (each holds ``filing.html`` + metadata).

    Layout is ``{documents_dir}/sec/{TICKER}/{FORM}/{filing_date}/`` (see documents/download).
    Returns a sorted list (deterministic) — empty if nothing has been downloaded.
    """
    base = documents_dir / "sec" / ticker
    if not base.exists():
        return []
    return sorted(p.parent for p in base.rglob("filing.html"))


def ingest_ticker(
    ticker: str,
    *,
    documents_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
) -> IngestResult:
    """Parse → chunk → embed → upsert every downloaded filing for ``ticker``.

    Embeds all of the ticker's chunks in one batch, then a single ``store.add`` (upsert).
    Re-ingesting the same corpus is a no-op on count (ids are stable).
    """
    chunks = []
    dirs = iter_filing_dirs(documents_dir, ticker)
    for directory in dirs:
        parsed = load_filing(directory)
        chunks.extend(
            chunk_filing(parsed, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
        )
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
    return IngestResult(ticker=ticker, filings=len(dirs), chunks=len(chunks))
