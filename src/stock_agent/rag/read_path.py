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

from typing import TYPE_CHECKING

from stock_agent.rag.embeddings import build_embedder
from stock_agent.rag.hybrid import HybridRetriever
from stock_agent.rag.rerank import RerankingRetriever, build_reranker
from stock_agent.rag.retriever import RetrievalSystem, Retriever
from stock_agent.rag.sparse_store import SparseStore, build_sparse_store
from stock_agent.rag.vector_store import VectorStore, build_vector_store
from stock_agent.settings import Settings

if TYPE_CHECKING:
    from stock_agent.graph.store import GraphStore


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


# The named, deployable retrieval configurations — each a (retrieval_mode, needs_rerank) recipe over
# `build_retrieval_system`. This is BOTH the eval-lattice's `--systems` menu AND the seed of the A6
# action space: a retrieval-selection policy (contextual bandit) chooses among exactly these per
# query. Promotion (the eval winner) only changes the *default* in settings — every config here
# stays reachable, so the A6 action set is preserved regardless of the eval outcome.
LATTICE_SYSTEMS: tuple[str, ...] = ("dense", "reranked", "hybrid", "hybrid+rerank")
_LATTICE: dict[str, tuple[str, bool]] = {
    "dense": ("dense", False),
    "reranked": ("dense", True),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}


def build_graph_system(
    settings: Settings,
    *,
    store: VectorStore | None = None,
    sparse_store: SparseStore | None = None,
    graph_store: GraphStore | None = None,
) -> RetrievalSystem:
    """Wrap the configured retriever in the A5 ``GraphRetriever`` (graph traversal ⊕ vector).

    The base is the normal ``build_retrieval_system`` stack (dense/hybrid/rerank by config); the
    GraphRetriever adds graph-traversed evidence and materializes provenance chunks via the sparse
    store's ``fetch`` (one shared instance feeds both the hybrid base and the provenance lookup).
    Graph imports are lazy so this stays a pure-wiring add-on (default-OFF: callers opt in by
    calling this factory instead of ``build_retrieval_system``). ``graph_store`` is for tests.
    """
    from stock_agent.graph.retriever import GraphRetriever
    from stock_agent.graph.store import build_graph_store

    sparse = sparse_store or build_sparse_store(settings)
    base = build_retrieval_system(settings, store=store, sparse_store=sparse)
    gstore = graph_store or build_graph_store(settings)
    return GraphRetriever(
        gstore,
        base,
        sparse.fetch,
        hops=settings.graph_hops,
        max_neighbors=settings.graph_max_neighbors,
        per_neighbor_k=settings.graph_per_neighbor_k,
    )


def build_named_system(
    name: str,
    settings: Settings,
    *,
    store: VectorStore | None = None,
    sparse_store: SparseStore | None = None,
    rerank_provider: str = "local",
) -> RetrievalSystem:
    """Build a named lattice config (``dense`` / ``reranked`` / ``hybrid`` / ``hybrid+rerank``).

    A thin recipe over ``build_retrieval_system``: it overrides ``retrieval_mode`` and
    ``rerank_provider`` per the named config, holding the embedder + chunking fixed, so the systems
    differ only in the retrieval *stages* being compared. ``rerank_provider`` selects which reranker
    the rerank-bearing configs use (``local`` onnx default, or ``voyage``). The same function is the
    intended A6 action-executor.
    """
    if name not in _LATTICE:
        raise ValueError(f"unknown retrieval system '{name}'; choose from {LATTICE_SYSTEMS}")
    mode, needs_rerank = _LATTICE[name]
    cfg = settings.model_copy(
        update={
            "retrieval_mode": mode,
            "rerank_provider": rerank_provider if needs_rerank else "none",
        }
    )
    return build_retrieval_system(cfg, store=store, sparse_store=sparse_store)
