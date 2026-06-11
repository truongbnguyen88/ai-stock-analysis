"""Vector store (RAG P5) — InMemoryVectorStore + ChunkFilter (offline, deterministic).

CI exercises the pure-Python backend and the filter/ranking contract via FakeEmbedder.
The real ChromaVectorStore is covered by a gated test (RUN_VECTORSTORE_TESTS) so CI never
needs chromadb or touches disk beyond a temp dir.
"""

from __future__ import annotations

import math
import os
import re
from datetime import date

import pytest

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.vector_store import (
    InMemoryVectorStore,
    VectorStore,
    build_vector_store,
    collection_name_for,
)
from stock_agent.schemas.documents import DocumentChunk, DocumentType
from stock_agent.schemas.retrieval import ChunkFilter
from stock_agent.settings import Settings

_EMB = FakeEmbedder(dim=32)


def _chunk(
    idx: int,
    text: str,
    *,
    ticker: str = "NVDA",
    dtype: DocumentType = "10-K",
    filing: date = date(2026, 2, 25),
    section: str | None = "Item 1A. Risk Factors",
) -> DocumentChunk:
    doc_id = f"{ticker}:{dtype}:{filing.isoformat()}"
    return DocumentChunk(
        chunk_id=f"{doc_id}:{idx}",
        document_id=doc_id,
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type=dtype,
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=filing,
        section=section,
    )


def _store(chunks: list[DocumentChunk]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(chunks, _EMB.embed_documents([c.text for c in chunks]))
    return store


# ---- Protocol + basic storage ------------------------------------------------
def test_in_memory_satisfies_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_add_and_count() -> None:
    store = _store([_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")])
    assert store.count() == 3


def test_add_is_upsert_by_chunk_id() -> None:
    store = InMemoryVectorStore()
    c = _chunk(0, "first")
    store.add([c], _EMB.embed_documents(["first"]))
    store.add([c], _EMB.embed_documents(["first"]))  # same id -> overwrite, not duplicate
    assert store.count() == 1


# ---- ranking -----------------------------------------------------------------
def test_query_ranks_self_match_first_and_scores_descend() -> None:
    texts = ["supply chain and export controls risk", "data center revenue", "dividend policy"]
    chunks = [_chunk(i, t) for i, t in enumerate(texts)]
    store = _store(chunks)
    hits = store.query(_EMB.embed_query(texts[0]), top_k=3)
    assert hits[0].chunk.text == texts[0]  # exact text embeds closest to its own query
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # descending similarity


def test_query_respects_top_k() -> None:
    store = _store([_chunk(i, f"text {i}") for i in range(5)])
    assert len(store.query(_EMB.embed_query("text"), top_k=2)) == 2


def test_score_is_cosine_self_match_is_one() -> None:
    store = _store([_chunk(0, "risk factors")])
    [hit] = store.query(_EMB.embed_query("risk factors"), top_k=1)
    # FakeEmbedder is deterministic but embed_query and embed_documents use the same map,
    # so a chunk's vector equals its own query vector -> cosine 1.0.
    assert math.isclose(hit.score, 1.0, abs_tol=1e-9)


def test_empty_store_returns_empty() -> None:
    assert InMemoryVectorStore().query(_EMB.embed_query("x"), top_k=5) == []


# ---- metadata filtering ------------------------------------------------------
def _mixed() -> InMemoryVectorStore:
    return _store(
        [
            _chunk(0, "nvda risk", ticker="NVDA", dtype="10-K", filing=date(2026, 2, 25)),
            _chunk(1, "nvda mdna", ticker="NVDA", dtype="10-Q", filing=date(2025, 5, 20)),
            _chunk(2, "avgo risk", ticker="AVGO", dtype="10-K", filing=date(2025, 12, 18)),
            _chunk(3, "nvda 8k", ticker="NVDA", dtype="8-K", filing=date(2026, 5, 20),
                   section="Item 8.01. Other Events"),
        ]
    )


def test_filter_by_ticker() -> None:
    hits = _mixed().query(_EMB.embed_query("x"), top_k=10, where=ChunkFilter(ticker="AVGO"))
    assert {h.chunk.ticker for h in hits} == {"AVGO"}


def test_filter_by_document_type() -> None:
    f = ChunkFilter(document_types=["10-K", "10-Q"])
    hits = _mixed().query(_EMB.embed_query("x"), top_k=10, where=f)
    assert {h.chunk.document_type for h in hits} == {"10-K", "10-Q"}  # no 8-K


def test_filter_by_section() -> None:
    f = ChunkFilter(sections=["Item 8.01. Other Events"])
    hits = _mixed().query(_EMB.embed_query("x"), top_k=10, where=f)
    assert [h.chunk.chunk_index for h in hits] == [3]


def test_filter_by_date_range_inclusive() -> None:
    f = ChunkFilter(date_from=date(2026, 1, 1), date_to=date(2026, 3, 1))
    hits = _mixed().query(_EMB.embed_query("x"), top_k=10, where=f)
    assert {h.chunk.filing_date for h in hits} == {date(2026, 2, 25)}  # only the in-window 10-K


def test_filter_combined_ticker_and_type() -> None:
    f = ChunkFilter(ticker="NVDA", document_types=["10-K"])
    hits = _mixed().query(_EMB.embed_query("x"), top_k=10, where=f)
    assert len(hits) == 1
    assert hits[0].chunk.ticker == "NVDA" and hits[0].chunk.document_type == "10-K"


# ---- ChunkFilter unit --------------------------------------------------------
def test_chunk_filter_is_empty() -> None:
    assert ChunkFilter().is_empty
    assert not ChunkFilter(ticker="NVDA").is_empty


def test_chunk_filter_matches_none_section() -> None:
    c = _chunk(0, "x", section=None)
    assert ChunkFilter().matches(c)
    assert not ChunkFilter(sections=["Item 1A. Risk Factors"]).matches(c)


# ---- real Chroma backend (gated; never runs in CI) ---------------------------
@pytest.mark.skipif(
    not os.environ.get("RUN_VECTORSTORE_TESTS"),
    reason="needs the chromadb [rag] extra; run locally with RUN_VECTORSTORE_TESTS=1",
)
def test_chroma_persists_and_matches_in_memory(tmp_path: object) -> None:
    pytest.importorskip("chromadb")
    from pathlib import Path

    from stock_agent.rag.vector_store import ChromaVectorStore

    path = Path(str(tmp_path)) / "vs"
    chunks = _mixed_list()
    vectors = _EMB.embed_documents([c.text for c in chunks])

    store = ChromaVectorStore(path)
    store.add(chunks, vectors)
    assert store.count() == len(chunks)

    # persistence: a fresh client over the same dir sees the data
    assert ChromaVectorStore(path).count() == len(chunks)

    # parity with the in-memory backend: same ranking + (cosine) scores
    mem = InMemoryVectorStore()
    mem.add(chunks, vectors)
    qv = _EMB.embed_query("nvda risk")
    chroma_hits = ChromaVectorStore(path).query(qv, top_k=4)
    mem_hits = mem.query(qv, top_k=4)
    assert [h.chunk.chunk_id for h in chroma_hits] == [h.chunk.chunk_id for h in mem_hits]
    for a, b in zip(chroma_hits, mem_hits, strict=True):
        assert math.isclose(a.score, b.score, abs_tol=1e-4)  # distance->similarity matches cosine

    # metadata filter survives the round-trip
    f = ChunkFilter(ticker="NVDA", document_types=["10-K"])
    filtered = ChromaVectorStore(path).query(qv, top_k=10, where=f)
    assert all(h.chunk.ticker == "NVDA" and h.chunk.document_type == "10-K" for h in filtered)


def _mixed_list() -> list[DocumentChunk]:
    return [
        _chunk(0, "nvda risk", ticker="NVDA", dtype="10-K", filing=date(2026, 2, 25)),
        _chunk(1, "nvda mdna", ticker="NVDA", dtype="10-Q", filing=date(2025, 5, 20)),
        _chunk(2, "avgo risk", ticker="AVGO", dtype="10-K", filing=date(2025, 12, 18)),
    ]


# ---- 9c: per-embedder collection namespacing (safe provider switch) ----------


def test_collection_name_for_is_chroma_safe() -> None:
    name = collection_name_for("local-BAAI/bge-small-en-v1.5")
    assert name == "filings-local-baai-bge-small-en-v1-5"
    # Chroma rules: 3–512 chars, only [a-zA-Z0-9._-], start + end alphanumeric.
    assert 3 <= len(name) <= 512
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name)
    assert collection_name_for("") == "filings"  # degenerate -> base name


def test_build_vector_store_isolates_collections_by_provider() -> None:
    # Switching local -> voyage (384-d -> 1024-d) must target a DIFFERENT collection, so the
    # two corpora never share one (a dimension mismatch would corrupt search). Construction is
    # lazy (no chromadb import), so this stays offline.
    local = build_vector_store(Settings(_env_file=None, embedding_provider="local"))
    voyage = build_vector_store(Settings(_env_file=None, embedding_provider="voyage"))
    assert local._collection_name != voyage._collection_name  # type: ignore[attr-defined]
    assert "voyage" in voyage._collection_name  # type: ignore[attr-defined]
