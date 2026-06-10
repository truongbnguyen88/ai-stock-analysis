"""Retrieval (RAG P6) — Retriever: top-k, dedup, filter, empty path (offline)."""

from __future__ import annotations

from datetime import date

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk, DocumentType
from stock_agent.schemas.retrieval import ChunkFilter

_EMB = FakeEmbedder(dim=32)


def _chunk(
    idx: int, text: str, *, ticker: str = "NVDA", dtype: DocumentType = "10-K"
) -> DocumentChunk:
    doc_id = f"{ticker}:{dtype}:2026-02-25"
    return DocumentChunk(
        chunk_id=f"{doc_id}:{idx}",
        document_id=doc_id,
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type=dtype,
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=date(2026, 2, 25),
        section="Item 1A. Risk Factors",
    )


def _retriever(chunks: list[DocumentChunk]) -> Retriever:
    store = InMemoryVectorStore()
    store.add(chunks, _EMB.embed_documents([c.text for c in chunks]))
    return Retriever(_EMB, store)


def test_retrieve_returns_evidence_set_ranked() -> None:
    texts = ["export control and supply chain risk", "data center revenue", "share repurchases"]
    r = _retriever([_chunk(i, t) for i, t in enumerate(texts)])
    ev = r.retrieve(texts[0], top_k=3)
    assert ev.query == texts[0]
    assert ev.chunks[0].chunk.text == texts[0]  # self-match ranks first
    scores = [c.score for c in ev.chunks]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_top_k() -> None:
    r = _retriever([_chunk(i, f"text {i}") for i in range(6)])
    assert len(r.retrieve("text", top_k=2).chunks) == 2


def test_retrieve_dedups_identical_text() -> None:
    # Same passage stored under three different ids (e.g. repeated across years) -> one survives.
    dup = "the same verbatim risk factor paragraph about competition"
    chunks = [_chunk(0, dup), _chunk(1, dup, ticker="AVGO"), _chunk(2, "a unique passage")]
    ev = _retriever(chunks).retrieve(dup, top_k=5)
    texts = [c.chunk.text for c in ev.chunks]
    assert texts.count(dup) == 1  # deduped
    assert "a unique passage" in texts


def test_retrieve_empty_store_is_empty_evidence() -> None:
    ev = Retriever(_EMB, InMemoryVectorStore()).retrieve("anything", top_k=5)
    assert ev.is_empty and len(ev) == 0


def test_retrieve_passes_filter_through() -> None:
    chunks = [_chunk(0, "nvda risk", ticker="NVDA"), _chunk(1, "avgo risk", ticker="AVGO")]
    ev = _retriever(chunks).retrieve("risk", top_k=5, where=ChunkFilter(ticker="AVGO"))
    assert {c.chunk.ticker for c in ev.chunks} == {"AVGO"}


def test_evidence_citation_and_allowed_ids() -> None:
    ev = _retriever([_chunk(0, "risk factors text")]).retrieve("risk factors text", top_k=1)
    rc = ev.chunks[0]
    assert rc.citation_label() == "NVDA 10-K 2026-02-25 — Item 1A. Risk Factors"
    assert ev.allowed_chunk_ids() == {rc.chunk.chunk_id}
