"""SEC-grounded QA (RAG P7) — synthesis call + citation/number guards (canned LLM)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.research.synthesis import ResearchGuardError, answer_question
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import EvidenceSet, RetrievedChunk


class _FakeLLM:
    """Returns canned JSON strings in order (last repeats); records call count."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out


def _resp(answer: str, citations: list[int], insufficient: bool = False) -> str:
    return json.dumps(
        {"answer": answer, "citations": citations, "insufficient_evidence": insufficient}
    )


def _chunk(idx: int, text: str) -> DocumentChunk:
    doc_id = "NVDA:10-K:2026-02-25"
    return DocumentChunk(
        chunk_id=f"{doc_id}:{idx}",
        document_id=doc_id,
        chunk_index=idx,
        text=text,
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=date(2026, 2, 25),
        section="Item 7. Management's Discussion and Analysis",
    )


def _evidence(*texts: str) -> EvidenceSet:
    chunks = [
        RetrievedChunk(chunk=_chunk(i, t), score=0.9 - 0.01 * i) for i, t in enumerate(texts)
    ]
    return EvidenceSet(query="q", chunks=chunks)


_EV = _evidence(
    "Data Center revenue was up 41% from a year ago on strong demand.",
    "Gross margin was 75.0% for the fiscal year.",
)


# ---- empty evidence: refuse, no LLM call -------------------------------------
def test_empty_evidence_refuses_without_llm_call() -> None:
    llm = _FakeLLM(_resp("should not be used", [1]))
    ga = answer_question("anything", EvidenceSet(query="q", chunks=[]), llm=llm)
    assert ga.insufficient_evidence is True
    assert ga.answer == "Insufficient evidence found."
    assert llm.calls == 0  # short-circuited — no paid call


# ---- valid answer: schema + citation resolution ------------------------------
def test_valid_answer_resolves_citations() -> None:
    llm = _FakeLLM(_resp("Data Center revenue rose 41% [1] and margin was 75.0% [2].", [1, 2]))
    ga = answer_question("How did data center do?", _EV, llm=llm)
    assert ga.insufficient_evidence is False
    assert {c.marker for c in ga.citations} == {1, 2}
    assert ga.citations[0].chunk_id == _EV.chunks[0].chunk.chunk_id
    assert "Item 7" in ga.citations[0].label
    assert llm.calls == 1  # no retry needed


def test_inline_markers_count_even_if_list_omits_them() -> None:
    # The model cites [1] inline but forgets the citations list -> still resolved + valid.
    llm = _FakeLLM(_resp("Revenue rose 41% [1].", []))
    ga = answer_question("q", _EV, llm=llm)
    assert {c.marker for c in ga.citations} == {1}


# ---- LLM-signaled insufficiency ----------------------------------------------
def test_llm_marks_insufficient() -> None:
    llm = _FakeLLM(_resp("Insufficient evidence found.", [], insufficient=True))
    ga = answer_question("unrelated question", _EV, llm=llm)
    assert ga.insufficient_evidence is True
    assert ga.answer == "Insufficient evidence found."
    assert ga.citations == []


# ---- citation guard: fabricated source ---------------------------------------
def test_fabricated_citation_retried_then_raises() -> None:
    # Cites [5] when only 2 sources exist; the corrected retry repeats the error -> raise.
    bad = _resp("Revenue grew per [5].", [5])
    llm = _FakeLLM(bad, bad)
    with pytest.raises(ResearchGuardError):
        answer_question("q", _EV, llm=llm)
    assert llm.calls == 2  # one retry


def test_fabricated_citation_recovers_on_retry() -> None:
    bad = _resp("Revenue grew per [9].", [9])
    good = _resp("Revenue rose 41% [1].", [1])
    llm = _FakeLLM(bad, good)
    ga = answer_question("q", _EV, llm=llm)
    assert {c.marker for c in ga.citations} == {1} and llm.calls == 2


# ---- number grounding: invented figure ---------------------------------------
def test_invented_number_retried_then_raises() -> None:
    # "62.5%" appears in no source -> flagged; retry repeats -> raise.
    bad = _resp("Margin expanded to 62.5% [2].", [2])
    llm = _FakeLLM(bad, bad)
    with pytest.raises(ResearchGuardError):
        answer_question("q", _EV, llm=llm)


def test_grounded_number_passes() -> None:
    # "41%" and "75.0%" are both in the sources -> clean, no retry.
    llm = _FakeLLM(_resp("Revenue +41% [1]; margin 75.0% [2].", [1, 2]))
    ga = answer_question("q", _EV, llm=llm)
    assert llm.calls == 1 and not ga.insufficient_evidence
