"""Eval-lattice (advanced-RAG, A1/A3 follow-up) — named retrieval configs + sparse-only baseline.

`build_named_system` is the menu the `rag eval --systems` lattice scores AND the seed of the A6
action space, so it gets its own tests. Offline: FakeEmbedder, in-memory stores, a FakeReranker
injected via `build_reranker` so rerank-bearing configs score without the onnx model.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.eval import LabeledQuery, evaluate_system
from stock_agent.rag.hybrid import HybridRetriever, SparseRetriever
from stock_agent.rag.read_path import LATTICE_SYSTEMS, build_named_system
from stock_agent.rag.rerank import RerankingRetriever
from stock_agent.rag.retriever import RetrievalSystem, Retriever
from stock_agent.rag.sparse_store import InMemoryBM25Store
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

_EMB = FakeEmbedder(dim=32)


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


_CORPUS = [
    _chunk(0, "Export controls and licensing requirements affect China sales."),
    _chunk(1, "The Hopper architecture powers our data center GPUs."),
    _chunk(2, "Competition in accelerated computing is intense."),
]


class _FakeReranker:
    """Deterministic stand-in for the onnx cross-encoder (score = query-term overlap)."""

    name = "fake-rerank"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        terms = set(query.lower().split())
        scored = [
            RetrievedChunk(
                chunk=rc.chunk, score=float(sum(t in rc.chunk.text.lower() for t in terms))
            )
            for rc in chunks
        ]
        return sorted(scored, key=lambda rc: rc.score, reverse=True)


def _stores() -> tuple[InMemoryVectorStore, InMemoryBM25Store]:
    store = InMemoryVectorStore()
    store.add(_CORPUS, _EMB.embed_documents([c.text for c in _CORPUS]))
    sparse = InMemoryBM25Store()
    sparse.add(_CORPUS)
    return store, sparse


# ---- build_named_system: the action registry --------------------------------


def test_lattice_systems_are_the_four_deployable_configs() -> None:
    assert LATTICE_SYSTEMS == ("dense", "reranked", "hybrid", "hybrid+rerank")


def test_build_named_system_composition_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_agent.rag.read_path.build_embedder", lambda s: _EMB)
    monkeypatch.setattr("stock_agent.rag.read_path.build_reranker", lambda s: _FakeReranker())
    store, sparse = _stores()
    s = Settings(_env_file=None)

    def _build(name: str) -> RetrievalSystem:
        return build_named_system(name, s, store=store, sparse_store=sparse)

    assert isinstance(_build("dense"), Retriever)
    assert isinstance(_build("reranked"), RerankingRetriever)
    assert isinstance(_build("hybrid"), HybridRetriever)
    both = _build("hybrid+rerank")
    assert isinstance(both, RerankingRetriever) and "hybrid(" in both.name


def test_build_named_system_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown retrieval system"):
        build_named_system("magic", Settings(_env_file=None))


# ---- every config scores through the A1 harness (offline) --------------------


@pytest.mark.parametrize("name", list(LATTICE_SYSTEMS))
def test_every_lattice_config_scores_through_evaluate_system(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stock_agent.rag.read_path.build_embedder", lambda s: _EMB)
    monkeypatch.setattr("stock_agent.rag.read_path.build_reranker", lambda s: _FakeReranker())
    store, sparse = _stores()
    system = build_named_system(
        name, Settings(_env_file=None), store=store, sparse_store=sparse, rerank_provider="local"
    )
    q = LabeledQuery(query="hopper architecture", ticker="NVDA", relevant_spans=["Hopper"])
    report = evaluate_system(system, [q], top_k=3, corpus_chunks=_CORPUS)
    assert report.n_queries == 1
    assert report.per_query[0].hit == 1.0  # the Hopper chunk is retrievable in every config


# ---- SparseRetriever (the --diagnostic baseline) -----------------------------


def test_sparse_retriever_is_a_retrieval_system_and_ranks_by_bm25() -> None:
    _, sparse = _stores()
    sr = SparseRetriever(sparse)
    assert isinstance(sr, RetrievalSystem)
    assert sr.name == "sparse(bm25-mem)"
    out = sr.retrieve("hopper architecture", top_k=2, where=ChunkFilter(ticker="NVDA"))
    assert out.chunks[0].chunk.chunk_index == 1  # the Hopper chunk ranks first lexically


def test_sparse_retriever_empty_on_no_match() -> None:
    _, sparse = _stores()
    out = SparseRetriever(sparse).retrieve("nonexistentterm", top_k=5)
    assert isinstance(out, EvidenceSet) and out.is_empty
