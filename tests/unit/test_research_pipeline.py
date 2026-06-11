"""Research-memo pipeline (P8) — evidence round-robin merge + gather wiring (offline)."""

from __future__ import annotations

from datetime import date

from stock_agent.pipelines.research import _gather_sec_evidence, _round_robin_merge
from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import RetrievedChunk
from stock_agent.settings import Settings


def _doc(cid: str, text: str, *, section: str = "Item 1A. Risk Factors") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id="NVDA:10-K:2026-02-25",
        chunk_index=0,
        text=text,
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=date(2026, 2, 25),
        section=section,
    )


def _rc(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=_doc(cid, cid), score=score)


# ---- _round_robin_merge (the section-diversity fix) --------------------------
def test_round_robin_gives_each_query_representation() -> None:
    # Query 0 out-scores everything; a plain global score-cap would pick ONLY its chunks.
    q0 = [_rc(f"a{i}", 0.90 - i * 0.01) for i in range(5)]
    q1 = [_rc(f"b{i}", 0.70 - i * 0.01) for i in range(5)]
    q2 = [_rc(f"c{i}", 0.60 - i * 0.01) for i in range(5)]
    merged = _round_robin_merge([q0, q1, q2], cap=6)
    ids = [rc.chunk.chunk_id for rc in merged]
    assert len(merged) == 6
    # each query contributes 2 (cap 6 / 3 queries) — none crowded out
    assert sum(i.startswith("a") for i in ids) == 2
    assert sum(i.startswith("b") for i in ids) == 2
    assert sum(i.startswith("c") for i in ids) == 2
    # presentation order is by score descending
    scores = [rc.score for rc in merged]
    assert scores == sorted(scores, reverse=True)


def test_round_robin_dedups_chunk_claimed_by_an_earlier_query() -> None:
    shared = _rc("shared", 0.9)
    merged = _round_robin_merge([[shared, _rc("a1", 0.8)], [shared, _rc("b1", 0.7)]], cap=10)
    ids = [rc.chunk.chunk_id for rc in merged]
    assert ids.count("shared") == 1 and set(ids) == {"shared", "a1", "b1"}


def test_round_robin_respects_cap() -> None:
    merged = _round_robin_merge([[_rc(f"x{i}", 0.5 + i * 0.1) for i in range(4)]], cap=2)
    assert len(merged) == 2


def test_round_robin_exhausts_without_infinite_loop() -> None:
    merged = _round_robin_merge([[_rc("a", 0.9)], [_rc("b", 0.8)]], cap=100)  # cap > available
    assert {rc.chunk.chunk_id for rc in merged} == {"a", "b"}


def test_round_robin_empty() -> None:
    assert _round_robin_merge([], cap=5) == []
    assert _round_robin_merge([[], []], cap=5) == []


# ---- _gather_sec_evidence wiring (injected Retriever; FakeEmbedder + InMemory) ----
def _retriever_with(n: int) -> Retriever:
    emb = FakeEmbedder(dim=16)
    store = InMemoryVectorStore()
    chunks = [_doc(f"NVDA:10-K:2026-02-25:{i}", f"sec chunk {i} on risk, revenue, and demand")
              for i in range(n)]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return Retriever(emb, store)


def test_gather_sec_evidence_caps_dedups_and_filters() -> None:
    ev = _gather_sec_evidence(
        "NVDA", Settings(_env_file=None), per_query_k=8, retriever=_retriever_with(15)
    )
    ids = [rc.chunk.chunk_id for rc in ev.chunks]
    assert 0 < len(ev.chunks) <= 10  # _MEMO_EVIDENCE_CAP
    assert len(ids) == len(set(ids))  # deduped across the 3 section queries
    assert all(rc.chunk.ticker == "NVDA" for rc in ev.chunks)


def test_gather_sec_evidence_empty_store_is_empty() -> None:
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore())
    ev = _gather_sec_evidence("NVDA", Settings(_env_file=None), per_query_k=8, retriever=retriever)
    assert ev.is_empty
