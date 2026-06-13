"""Read-path composition root (advanced-RAG) — assemble the configured retrieval system.

One factory builds the whole retrieval stack from config, composing the independently-shippable
stages of the advanced track. Each stage is a ``RetrievalSystem`` wrapping the previous one:

    dense Retriever                              (A1, always)
      → HybridRetriever(dense, sparse)           (A3, when retrieval_mode="hybrid")
        → RerankingRetriever(base, reranker)     (A2, when rerank_provider != "none")

So the production pipeline is ``rerank(hybrid(dense, sparse))`` with both stages on, or the plain
dense ``Retriever`` with the defaults — byte-identical to A1. This module is the *only* place that
knows the stack's shape; ``research/``, the agent tools, and the ``rag query`` CLI just call it.
"""

from __future__ import annotations

from stock_agent.rag.embeddings import build_embedder
from stock_agent.rag.hybrid import HybridRetriever
from stock_agent.rag.rerank import RerankingRetriever, build_reranker
from stock_agent.rag.retriever import RetrievalSystem, Retriever
from stock_agent.rag.sparse_store import SparseStore, build_sparse_store
from stock_agent.rag.vector_store import VectorStore, build_vector_store
from stock_agent.settings import Settings


def build_retrieval_system(
    settings: Settings,
    *,
    store: VectorStore | None = None,
    sparse_store: SparseStore | None = None,
) -> RetrievalSystem:
    """Build the configured read-path retriever — dense, optionally hybrid, optionally reranked.

    Defaults (``retrieval_mode="dense"``, ``rerank_provider="none"``) return the plain dense
    ``Retriever``, so the live system is unchanged until a stage is enabled. ``store`` and
    ``sparse_store`` are injectable (tests pass fakes; prod defaults to the built backends). All
    backends are lazy, so this stays cheap and offline.
    """
    dense = Retriever(build_embedder(settings), store or build_vector_store(settings))
    base: RetrievalSystem = dense
    if settings.retrieval_mode == "hybrid":
        base = HybridRetriever(
            base,
            sparse_store or build_sparse_store(settings),
            k_rrf=settings.hybrid_rrf_k,
            dense_k=settings.hybrid_dense_k,
            sparse_k=settings.hybrid_sparse_k,
        )
    if settings.rerank_provider != "none":
        base = RerankingRetriever(base, build_reranker(settings), fetch_k=settings.rerank_fetch_k)
    return base
