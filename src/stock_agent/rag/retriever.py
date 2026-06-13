"""Retrieval (RAG P6) — embed a query, search the store, dedup, return ranked evidence.

The ``Retriever`` joins an ``Embedder`` and a ``VectorStore`` into the read path: it embeds
the question once, runs a (filtered) cosine top-k against the store, removes verbatim duplicate
text, and returns an ``EvidenceSet`` — the *only* thing P7 hands to the synthesis LLM. NO LLM
calls here; this stage just decides *which* chunks become the grounding.
"""

from __future__ import annotations

from collections.abc import Iterable

from stock_agent.rag.embeddings import Embedder
from stock_agent.rag.vector_store import VectorStore
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk

# Fetch this many × top_k from the store before dedup, so removing duplicate text still
# leaves enough unique chunks to fill top_k.
_OVER_FETCH = 4


def _dedup_by_text(hits: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks with identical normalized text, keeping the highest-scored (first).

    SEC filings repeat boilerplate verbatim across years (and adjacent chunks overlap), so
    the same passage can be retrieved several times; without this, near-duplicate copies would
    crowd out the evidence budget. ``hits`` must already be sorted by score descending.
    """
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for rc in hits:
        key = " ".join(rc.chunk.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(rc)
    return out


class Retriever:
    """Embed → (filtered) cosine top-k → dedup → ``EvidenceSet``. No LLM; only the store.

    Satisfies the ``RetrievalSystem`` Protocol (a ``name`` + ``retrieve``), so the eval harness and
    any future composite retriever (rerank/hybrid/graph) can treat it interchangeably.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, *, name: str | None = None) -> None:
        self._embedder = embedder
        self._store = store
        # Label for eval reports; defaults to the embedder it wraps (e.g. "dense:voyage-4").
        self._name = name or f"dense:{embedder.name}"

    @property
    def name(self) -> str:
        """Identifier used in eval reports to distinguish this retrieval system from others."""
        return self._name

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: ChunkFilter | None = None,
        over_fetch: int = _OVER_FETCH,
    ) -> EvidenceSet:
        """Return up to ``top_k`` unique chunks most relevant to ``query`` (scoped by ``where``)."""
        query_vector = self._embedder.embed_query(query)
        raw = self._store.query(query_vector, top_k=max(top_k * over_fetch, top_k), where=where)
        deduped = _dedup_by_text(raw)[:top_k]
        return EvidenceSet(query=query, chunks=deduped)
