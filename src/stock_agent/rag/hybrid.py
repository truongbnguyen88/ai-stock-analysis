"""Hybrid retrieval (advanced-RAG A3) — fuse dense (semantic) + sparse (BM25) by RRF.

Dense and sparse retrieval fail differently: embeddings match *meaning* but blur exact terms; BM25
matches exact *terms* but misses paraphrase. Running both and fusing their rankings recovers each
other's misses. We fuse with **Reciprocal Rank Fusion** (Cormack et al. 2009):

    score(d) = sum over lists L of  1 / (k_rrf + rank_L(d))

where ``rank_L(d)`` is d's 1-based position in list L (absent → that term is 0). RRF uses only the
**rank**, never the raw score — so we never have to reconcile cosine similarities with BM25 scores
(different scales, not comparable). ``k_rrf`` (≈60) damps the contribution of top ranks so a single
list can't dominate.

``HybridRetriever`` is a ``RetrievalSystem`` (like the dense ``Retriever`` and the A2
``RerankingRetriever``), so it slots into the same read path: the production pipeline becomes
``rerank(hybrid(dense, sparse))`` purely by composition — synthesis, memo, and the agent are
untouched. Fusion only reorders/selects chunks (never edits text), so citations + number-grounding
hold. With no sparse hits (empty index / no term match) it degrades to the dense ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.rag.sparse_store import SparseStore
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists into one, by Reciprocal Rank Fusion (rank-based).

    ``ranked_lists`` are best-first id orderings (e.g. dense ids, sparse ids). Returns
    ``(id, rrf_score)`` sorted best-first, where ``rrf_score = sum_L 1/(k + rank_L(id))`` over the
    lists that contain the id (rank is 1-based). Pure + deterministic (ties broken by id for
    stability). Larger ``k`` flattens the rank weighting.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    # Sort by score desc, then id asc for a deterministic order on ties.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """Dense ⊕ sparse retrieval fused by RRF; satisfies the ``RetrievalSystem`` contract.

    Over-fetches ``dense_k`` from the dense base and ``sparse_k`` from the BM25 index (both scoped
    by the same ``ChunkFilter``), fuses their rankings with RRF, and returns the top ``top_k`` as an
    ``EvidenceSet``. Chunks for sparse-only hits are materialized from the sparse store (it holds
    the text + metadata). Empty sparse result → the dense ranking is returned unchanged.
    """

    def __init__(
        self,
        dense: RetrievalSystem,
        sparse: SparseStore,
        *,
        k_rrf: int = 60,
        dense_k: int = 30,
        sparse_k: int = 30,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._k_rrf = k_rrf
        self._dense_k = dense_k
        self._sparse_k = sparse_k

    @property
    def name(self) -> str:
        return f"hybrid({self._dense.name}+{self._sparse.name})"

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        # Over-fetch each side independently (at least top_k), then fuse by rank.
        dense_hits = self._dense.retrieve(
            query, top_k=max(self._dense_k, top_k), where=where
        ).chunks
        sparse_hits = self._sparse.search(
            query, top_k=max(self._sparse_k, top_k), where=where
        )

        dense_ids = [rc.chunk.chunk_id for rc in dense_hits]
        sparse_ids = [cid for cid, _ in sparse_hits]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._k_rrf)[:top_k]

        # Resolve every fused id to its chunk: dense hits already carry chunk objects; materialize
        # any sparse-only ids from the sparse store. The RRF score replaces the per-list score.
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in dense_hits}
        missing = [cid for cid, _ in fused if cid not in by_id]
        by_id.update(self._sparse.fetch(missing))
        chunks = [
            RetrievedChunk(chunk=by_id[cid], score=score)
            for cid, score in fused
            if cid in by_id  # defensive: skip an id neither side can materialize
        ]
        return EvidenceSet(query=query, chunks=chunks)


class SparseRetriever:
    """BM25-only retrieval exposed as a ``RetrievalSystem`` — **diagnostic / eval use only**.

    Not a production read path: lexical-only retrieval loses all semantic recall (paraphrase,
    synonyms), which is exactly what the dense half exists for. This adapter lets the eval harness
    score sparse-on-its-own as a *baseline* (the ``rag eval --diagnostic`` column), to confirm BM25
    is contributing signal — it is deliberately **not** wired into ``build_retrieval_system``.
    """

    def __init__(self, sparse: SparseStore) -> None:
        self._sparse = sparse

    @property
    def name(self) -> str:
        return f"sparse({self._sparse.name})"

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        hits = self._sparse.search(query, top_k=top_k, where=where)
        by_id = self._sparse.fetch([cid for cid, _ in hits])
        chunks = [
            RetrievedChunk(chunk=by_id[cid], score=score) for cid, score in hits if cid in by_id
        ]
        return EvidenceSet(query=query, chunks=chunks)
