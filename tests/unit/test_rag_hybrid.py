"""Hybrid retrieval (advanced-RAG A3) — RRF fusion + HybridRetriever + read-path gating.

Offline: a canned dense system + an in-memory BM25 store; no embeddings, no network, no model.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from stock_agent.rag.read_path import build_retrieval_system
from stock_agent.rag.rerank import RerankingRetriever
from stock_agent.rag.retriever import RetrievalSystem, Retriever
from stock_agent.rag.sparse_store import InMemoryBM25Store
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings


def _chunk(idx: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"NVDA:10-K:2026-02-25:{idx}",
        document_id="NVDA:10-K:2026-02-25",
        chunk_index=idx,
        text=text,
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/Archives/edgar/data/x/nvda.htm",
        filing_date=date(2026, 2, 25),
        section="Item 1A. Risk Factors",
    )


class _FixedDense:
    """A RetrievalSystem returning a canned ranked EvidenceSet (no store needed)."""

    name = "dense:fake"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        return EvidenceSet(query=query, chunks=self._chunks[:top_k])


# ---- RRF (pure) --------------------------------------------------------------


def test_rrf_golden_fuses_two_rankings() -> None:
    # list A: [a, b, c]; list B: [b, d]. k=60.
    # a: 1/61; b: 1/62 + 1/61; c: 1/63; d: 1/62. So b > a > d > c.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert [i for i, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_is_deterministic_on_ties() -> None:
    # "x" and "y" each appear once at rank 1 → equal score → tie broken by id asc.
    fused = reciprocal_rank_fusion([["y"], ["x"]], k=60)
    assert [i for i, _ in fused] == ["x", "y"]


# ---- HybridRetriever ---------------------------------------------------------


def test_hybrid_recovers_a_sparse_only_exact_term_hit() -> None:
    # Dense returns two semantically-near chunks but MISSES the exact-term chunk (idx 2);
    # BM25 ranks idx 2 first on the rare term "hopper". RRF should surface idx 2 into the top-k,
    # and HybridRetriever must materialize it from the sparse store (dense never returned it).
    c0, c1, c2 = _chunk(0, "general compute discussion"), _chunk(1, "ai platform demand"), _chunk(
        2, "The Hopper architecture powers our data center GPUs."
    )
    dense = _FixedDense([RetrievedChunk(chunk=c0, score=0.9), RetrievedChunk(chunk=c1, score=0.8)])
    sparse = InMemoryBM25Store()
    sparse.add([c0, c1, c2])
    hybrid = HybridRetriever(dense, sparse, k_rrf=60, dense_k=10, sparse_k=10)

    out = hybrid.retrieve("hopper architecture", top_k=3)
    ids = [rc.chunk.chunk_id for rc in out.chunks]
    assert c2.chunk_id in ids  # the sparse-only exact-term hit was recovered + materialized
    # c2 (sparse rank 1, 1/61) outranks c1 (dense rank 2, 1/62); it ties c0 (dense rank 1) by RRF.
    assert ids.index(c2.chunk_id) < ids.index(c1.chunk_id)


def test_hybrid_empty_sparse_degrades_to_dense() -> None:
    c0, c1 = _chunk(0, "alpha"), _chunk(1, "beta")
    dense = _FixedDense([RetrievedChunk(chunk=c0, score=0.9), RetrievedChunk(chunk=c1, score=0.8)])
    hybrid = HybridRetriever(dense, InMemoryBM25Store(), dense_k=10, sparse_k=10)  # empty sparse
    out = hybrid.retrieve("anything", top_k=5)
    assert [rc.chunk.chunk_id for rc in out.chunks] == [c0.chunk_id, c1.chunk_id]  # dense order


def test_hybrid_applies_filter_to_both_sides() -> None:
    nvda, amd = _chunk(0, "export controls on china"), _chunk(1, "export controls on china")
    amd = amd.model_copy(update={"chunk_id": "AMD:10-K:2026-02-25:1", "ticker": "AMD"})
    dense = _FixedDense([RetrievedChunk(chunk=nvda, score=0.9)])  # dense already ticker-scoped
    sparse = InMemoryBM25Store()
    sparse.add([nvda, amd])
    hybrid = HybridRetriever(dense, sparse, dense_k=10, sparse_k=10)
    out = hybrid.retrieve("export controls", top_k=10, where=ChunkFilter(ticker="NVDA"))
    assert {rc.chunk.ticker for rc in out.chunks} == {"NVDA"}  # AMD excluded on the sparse side too


def test_hybrid_is_a_retrieval_system() -> None:
    hybrid = HybridRetriever(_FixedDense([]), InMemoryBM25Store())
    assert isinstance(hybrid, RetrievalSystem)
    assert hybrid.name == "hybrid(dense:fake+bm25-mem)"


# ---- read-path factory gating ------------------------------------------------


def test_build_retrieval_system_hybrid_and_hybrid_plus_rerank() -> None:
    store = InMemoryVectorStore()
    sparse = InMemoryBM25Store()
    # dense (default) → plain Retriever
    assert isinstance(build_retrieval_system(Settings(_env_file=None), store=store), Retriever)
    # hybrid → HybridRetriever
    hy = build_retrieval_system(
        Settings(_env_file=None, retrieval_mode="hybrid"), store=store, sparse_store=sparse
    )
    assert isinstance(hy, HybridRetriever)
    # hybrid + rerank → RerankingRetriever wrapping a HybridRetriever
    both = build_retrieval_system(
        Settings(_env_file=None, retrieval_mode="hybrid", rerank_provider="local"),
        store=store,
        sparse_store=sparse,
    )
    assert isinstance(both, RerankingRetriever)
    assert both.name.startswith("rerank(") and "hybrid(" in both.name


# ---- A1 ↔ A3 composition -----------------------------------------------------


def test_eval_harness_scores_a_hybrid_system() -> None:
    from stock_agent.rag.eval import LabeledQuery, evaluate_system

    corpus = [
        _chunk(0, "Export controls and licensing requirements affect China sales."),
        _chunk(1, "The Hopper architecture powers data center GPUs."),
    ]
    sparse = InMemoryBM25Store()
    sparse.add(corpus)
    dense = _FixedDense([RetrievedChunk(chunk=corpus[0], score=0.9)])
    system = HybridRetriever(dense, sparse, dense_k=10, sparse_k=10)

    q = LabeledQuery(query="hopper architecture", ticker="NVDA", relevant_spans=["Hopper"])
    report = evaluate_system(system, [q], top_k=2, corpus_chunks=corpus)
    assert report.system == "hybrid(dense:fake+bm25-mem)"
    assert report.per_query[0].hit == 1.0  # the exact-term chunk is recovered + scored relevant
