"""Reranking (advanced-RAG A2) — Protocol conformance, two-stage wrapper, gated factory.

All offline: a deterministic ``FakeReranker`` (lexical overlap) stands in for the onnx/Voyage
cross-encoders, so CI never downloads a model. The real ``FastEmbedReranker`` is exercised only
behind the ``RUN_RERANK_TESTS`` env gate (see ``test_real_fastembed_reranker_smoke``).
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.rerank import (
    NoOpReranker,
    Reranker,
    RerankingRetriever,
    build_reranker,
    build_retrieval_system,
)
from stock_agent.rag.retriever import RetrievalSystem, Retriever
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


class FakeReranker:
    """Deterministic stand-in: score = count of query words present in the chunk (lexical)."""

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


class _FixedSystem:
    """A RetrievalSystem returning a canned EvidenceSet (decouples the wrapper from a store)."""

    name = "fixed"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        return EvidenceSet(query=query, chunks=self._chunks[:top_k])


# ---- Protocol conformance ----------------------------------------------------


def test_rerankers_satisfy_protocol() -> None:
    assert isinstance(NoOpReranker(), Reranker)
    assert isinstance(FakeReranker(), Reranker)


def test_noop_reranker_is_passthrough() -> None:
    chunks = [RetrievedChunk(chunk=_chunk(i, f"text {i}"), score=1.0 / (i + 1)) for i in range(3)]
    out = NoOpReranker().rerank("anything", chunks)
    assert [rc.chunk.chunk_id for rc in out] == [rc.chunk.chunk_id for rc in chunks]  # order kept


# ---- RerankingRetriever ------------------------------------------------------


def test_reranking_retriever_reorders_and_truncates_to_top_k() -> None:
    # Dense order puts the most relevant chunk LAST; the reranker (lexical overlap) lifts it to top.
    base = _FixedSystem(
        [
            RetrievedChunk(chunk=_chunk(0, "unrelated boilerplate about offices"), score=0.9),
            RetrievedChunk(chunk=_chunk(1, "some competition discussion"), score=0.8),
            RetrievedChunk(chunk=_chunk(2, "export controls and licensing on China"), score=0.7),
        ]
    )
    rr = RerankingRetriever(base, FakeReranker(), fetch_k=10)
    out = rr.retrieve("export controls licensing China", top_k=2)

    assert len(out.chunks) == 2  # truncated to top_k
    assert out.chunks[0].chunk.chunk_index == 2  # the lexically-matching chunk reranked to the top
    assert out.chunks[0].score >= out.chunks[1].score  # descending rerank scores


def test_reranking_retriever_over_fetches_at_least_top_k() -> None:
    # fetch_k < top_k must not under-fetch: the wrapper asks the base for max(fetch_k, top_k).
    seen_top_k: list[int] = []

    class _Recorder:
        name = "rec"

        def retrieve(
            self, query: str, *, top_k: int, where: ChunkFilter | None = None
        ) -> EvidenceSet:
            seen_top_k.append(top_k)
            return EvidenceSet(query=query, chunks=[])

    RerankingRetriever(_Recorder(), FakeReranker(), fetch_k=3).retrieve("q", top_k=8)
    assert seen_top_k == [8]  # clamped up to top_k, never the smaller fetch_k


def test_reranking_retriever_empty_passthrough() -> None:
    rr = RerankingRetriever(_FixedSystem([]), FakeReranker(), fetch_k=10)
    out = rr.retrieve("q", top_k=5)
    assert out.is_empty  # nothing to rerank → the empty-retrieval refusal path is preserved


def test_reranking_retriever_is_a_retrieval_system() -> None:
    rr = RerankingRetriever(_FixedSystem([]), NoOpReranker(), fetch_k=10)
    assert isinstance(rr, RetrievalSystem)
    assert rr.name == "rerank(noop)+fixed"


# ---- build_reranker + build_retrieval_system (gating) ------------------------


def test_build_reranker_defaults_to_noop_when_disabled() -> None:
    assert isinstance(
        build_reranker(Settings(_env_file=None, rerank_provider="none")), NoOpReranker
    )


def test_build_retrieval_system_is_default_off() -> None:
    store = InMemoryVectorStore()
    # Default (rerank_provider="none") → the plain dense Retriever, byte-identical to A1.
    plain = build_retrieval_system(Settings(_env_file=None), store=store)
    assert isinstance(plain, Retriever)
    # rerank_provider="local" → wrapped (no model load — fastembed is lazy).
    wrapped = build_retrieval_system(Settings(_env_file=None, rerank_provider="local"), store=store)
    assert isinstance(wrapped, RerankingRetriever)
    assert isinstance(wrapped, RetrievalSystem)


# ---- A1 ↔ A2 composition: the eval harness scores a reranked system ----------


def test_eval_harness_scores_a_reranked_system() -> None:
    from stock_agent.rag.eval import LabeledQuery, evaluate_system

    corpus = [
        _chunk(0, "Export controls and licensing requirements affect China sales."),
        _chunk(1, "Data center revenue grew on strong demand."),
    ]
    emb = FakeEmbedder(dim=32)
    store = InMemoryVectorStore()
    store.add(corpus, emb.embed_documents([c.text for c in corpus]))
    system = RerankingRetriever(Retriever(emb, store), FakeReranker(), fetch_k=5)

    q = LabeledQuery(query=corpus[0].text, ticker="NVDA", relevant_spans=["export controls"])
    report = evaluate_system(system, [q], top_k=2, corpus_chunks=corpus)
    assert report.system == "rerank(fake-rerank)+dense:fake"
    assert report.n_queries == 1


# ---- real onnx cross-encoder (gated; downloads a model) ----------------------


@pytest.mark.skipif(
    os.environ.get("RUN_RERANK_TESTS") != "1",
    reason="downloads the onnx cross-encoder; set RUN_RERANK_TESTS=1 to run",
)
def test_real_fastembed_reranker_smoke() -> None:
    from stock_agent.rag.rerank import FastEmbedReranker

    chunks = [
        RetrievedChunk(chunk=_chunk(0, "The cafeteria menu changed this quarter."), score=0.5),
        RetrievedChunk(
            chunk=_chunk(1, "U.S. export controls restrict China GPU sales."), score=0.5
        ),
    ]
    out = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2").rerank("export controls China", chunks)
    assert out[0].chunk.chunk_index == 1  # the on-topic chunk ranks first
