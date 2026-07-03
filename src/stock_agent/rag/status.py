"""RAG corpus status (observability).

A read-only snapshot of **which embedder + vector collection the agent queries** and **how
fresh the SEC corpus is** — no network, no LLM, no embedding calls. One source of truth shared
by the `rag status` CLI, the CLI `chat` startup banner, and the Streamlit sidebar so all three
report identically. Lets a user confirm at a glance that the chat agent is using the intended
embeddings (e.g. voyage-4), not a stale local index.
"""

from __future__ import annotations

from pydantic import BaseModel

from stock_agent.documents.download import manifest_path
from stock_agent.documents.manifest import DownloadManifest
from stock_agent.logging_config import get_logger
from stock_agent.rag.embeddings import embedding_namespace
from stock_agent.rag.vector_store import build_vector_store, collection_name_for
from stock_agent.settings import Settings

log = get_logger(__name__)


class CorpusStatus(BaseModel):
    """Snapshot of the active RAG embedder/collection + corpus coverage/freshness."""

    provider: str  # settings.embedding_provider, e.g. "voyage"
    embedder: str  # embedding namespace, e.g. "voyage-voyage-4" (provider + model)
    collection: str  # vector-store collection, e.g. "filings-voyage-voyage-4"
    chunks: int  # vectors in the collection (-1 if the store can't be opened)
    filings: int  # downloaded filings (from the manifest)
    tickers: int  # distinct tickers downloaded
    earliest: str | None  # earliest filing_date (ISO) in the corpus
    latest: str | None  # latest filing_date (ISO) — i.e. how current the corpus is

    @property
    def one_line(self) -> str:
        """Compact one-line summary for a chat banner / sidebar caption."""
        n = "unavailable" if self.chunks < 0 else f"{self.chunks:,} chunks"
        fresh = f" · fresh to {self.latest}" if self.latest else ""
        return f"{self.embedder} · {n}{fresh}"


def corpus_status(settings: Settings) -> CorpusStatus:
    """Aggregate the active embedder/collection + corpus coverage/freshness. Never raises."""
    namespace = embedding_namespace(settings)
    try:
        chunks = build_vector_store(settings).count()
    except Exception as exc:  # noqa: BLE001 — a missing/unopened store must not break a status read
        # The store being unopenable is non-fatal (status must still report freshness), but the
        # cause must not be silent: without this a red "store unavailable" dot gives no signal to
        # diagnose whether it's a missing/locked store, a bad path, or an embedder-init failure.
        chunks = -1
        log.warning(
            "corpus.store_unavailable",
            collection=collection_name_for(namespace),
            provider=settings.embedding_provider,
            vector_store_dir=str(settings.vector_store_dir),
            error=repr(exc),
        )
    entries = DownloadManifest.load(manifest_path(settings.documents_dir)).entries
    tickers = {e.ticker for e in entries.values()}
    dates = sorted(e.filing_date for e in entries.values())
    return CorpusStatus(
        provider=settings.embedding_provider,
        embedder=namespace,
        collection=collection_name_for(namespace),
        chunks=chunks,
        filings=len(entries),
        tickers=len(tickers),
        earliest=dates[0] if dates else None,
        latest=dates[-1] if dates else None,
    )
