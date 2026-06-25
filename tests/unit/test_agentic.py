"""Agentic RAG (advanced-RAG A4) — bounded ReAct controller (canned LLM + fake retriever).

Offline: a scripted ``_FakeLLM`` returns ReAct decisions for the loop and the terminal answer JSON
for the reused P7 synthesis; a ``_FakeRetriever`` maps queries → canned chunks. No live calls, no
model. Asserts: scripted search→stop retrieves + dedups the union; never-stop caps at ``max_steps``;
empty/duplicate query terminates (anti-loop); the terminal reuses ``answer_question`` (citation
guard over the union; empty union → refusal with no LLM call); the LLM-call budget.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.research.agentic import answer_multistep
from stock_agent.research.prompts import REACT_SYSTEM
from stock_agent.research.synthesis import ResearchGuardError
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings


# ---- canned doubles ----------------------------------------------------------
class _FakeLLM:
    """Routes decision calls (REACT_SYSTEM) to scripted decisions; all else → the answer JSON.

    Tracks total calls and decision calls separately so the budget assertion (decisions + 1
    terminal) is exact. Scripted decisions clamp to the last entry (a never-stopping policy).
    """

    def __init__(self, decisions: list[str], answer: str) -> None:
        self._decisions = decisions
        self._answer = answer
        self.calls = 0
        self.decision_calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        if system == REACT_SYSTEM:
            out = self._decisions[min(self.decision_calls, len(self._decisions) - 1)]
            self.decision_calls += 1
            return out
        return self._answer  # terminal P7 synthesis (and its retry, if any)


class _FakeRetriever:
    """Maps query → canned chunks; records the queries it was asked, in order."""

    name = "fake"

    def __init__(self, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self._mapping = mapping
        self.queries: list[str] = []

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        self.queries.append(query)
        return EvidenceSet(query=query, chunks=self._mapping.get(query, []))


# ---- builders ----------------------------------------------------------------
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
        section="Item 1A. Risk Factors",
    )


def _rc(idx: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(idx, text), score=0.9 - 0.01 * idx)


def _decision(action: str, query: str | None = None, ticker: str | None = None) -> str:
    d: dict[str, object] = {"thought": "t", "action": action}
    if query is not None:
        d["query"] = query
    if ticker is not None:
        d["ticker"] = ticker
    return json.dumps(d)


def _answer(text: str, citations: list[int], insufficient: bool = False) -> str:
    return json.dumps(
        {"answer": text, "citations": citations, "insufficient_evidence": insufficient}
    )


_SETTINGS = Settings()


# ---- scripted search → search → stop: retrieves per step, dedups the union ----
def test_search_search_stop_dedups_union() -> None:
    # query "a" -> {1,2}; query "b" -> {2,3}: chunk 2 overlaps -> deduped union {1,2,3}.
    rc1, rc2, rc3 = _rc(1, "risk one"), _rc(2, "risk two"), _rc(3, "risk three")
    retriever = _FakeRetriever({"a": [rc1, rc2], "b": [rc2, rc3]})
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("search", "b", "NVDA"), _decision("stop")],
        _answer("Risks summarized [1].", [1]),
    )
    res = answer_multistep("compare risks", settings=_SETTINGS, llm=llm, retriever=retriever)

    assert retriever.queries == ["a", "b"]  # one retrieval per search step
    assert res.n_steps == 2 and len(res.trace) == 2
    assert res.n_evidence == 3  # union deduped chunk 2
    assert res.answer.insufficient_evidence is False
    assert {c.marker for c in res.answer.citations} == {1}


# ---- never stops: caps at max_steps ------------------------------------------
def test_never_stops_caps_at_max_steps() -> None:
    mapping = {f"q{i}": [_rc(i, f"risk {i}")] for i in range(5)}
    retriever = _FakeRetriever(mapping)
    decisions = [_decision("search", f"q{i}", "NVDA") for i in range(5)]  # never a "stop"
    llm = _FakeLLM(decisions, _answer("Summary [1].", [1]))
    res = answer_multistep(
        "q", settings=_SETTINGS, llm=llm, retriever=retriever, max_steps=3
    )
    assert res.n_steps == 3  # hard cap honored
    assert llm.decision_calls == 3
    assert retriever.queries == ["q0", "q1", "q2"]


# ---- anti-loop: duplicate query terminates -----------------------------------
def test_duplicate_query_terminates() -> None:
    retriever = _FakeRetriever({"a": [_rc(1, "risk one")]})
    # Second decision repeats "a" -> anti-loop break before a second retrieval.
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("search", "a", "NVDA")],
        _answer("Summary [1].", [1]),
    )
    res = answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever, max_steps=4)
    assert retriever.queries == ["a"]  # the duplicate never retrieved
    assert res.n_steps == 1


def test_same_query_different_ticker_not_deduped() -> None:
    # The comparative use case A4 exists for: one query string, two tickers. The anti-loop must key
    # on (query, scope), so the AMD hop is NOT mis-flagged as a duplicate of the NVDA hop.
    rc_nvda, rc_amd = _rc(1, "NVDA risk"), _rc(2, "AMD risk")

    class _ByTickerRetriever:
        name = "by-ticker"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def retrieve(
            self, query: str, *, top_k: int, where: ChunkFilter | None = None
        ) -> EvidenceSet:
            tic = where.ticker if where else None
            self.calls.append((query, tic))
            chunk = rc_nvda if tic == "NVDA" else rc_amd
            return EvidenceSet(query=query, chunks=[chunk])

    retriever = _ByTickerRetriever()
    q = "AI supply-chain risk"
    llm = _FakeLLM(
        [_decision("search", q, "NVDA"), _decision("search", q, "AMD"), _decision("stop")],
        _answer("NVDA [1] and AMD [2] both flag it.", [1, 2]),
    )
    res = answer_multistep("compare risks", settings=_SETTINGS, llm=llm, retriever=retriever)
    assert retriever.calls == [(q, "NVDA"), (q, "AMD")]  # both hops executed
    assert res.n_steps == 2 and res.n_evidence == 2


def test_malformed_decision_stops_gracefully() -> None:
    # An unparseable decision must not crash the run or discard prior evidence: it degrades to stop.
    retriever = _FakeRetriever({"a": [_rc(1, "risk one")]})
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), "not json at all — just prose"],
        _answer("Summary [1].", [1]),
    )
    res = answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever, max_steps=4)
    assert res.n_steps == 1 and res.n_evidence == 1  # first hop kept; bad decision ended the loop
    assert res.answer.insufficient_evidence is False


def test_empty_query_terminates() -> None:
    retriever = _FakeRetriever({"a": [_rc(1, "risk one")]})
    llm = _FakeLLM([_decision("search", query=None)], _answer("Insufficient", [], True))
    res = answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever)
    assert retriever.queries == []  # no query issued -> nothing retrieved
    assert res.n_steps == 0


# ---- empty union: refuse with NO terminal LLM call ---------------------------
def test_empty_union_refuses_without_llm_call() -> None:
    retriever = _FakeRetriever({})  # every query returns nothing
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("stop")],
        _answer("should not be used", [1]),
    )
    res = answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever)
    assert res.n_evidence == 0
    assert res.answer.insufficient_evidence is True
    assert res.answer.answer == "Insufficient evidence found."
    # Budget: only decision calls happened; the terminal synthesis was short-circuited.
    assert llm.calls == llm.decision_calls == 2


# ---- terminal reuses the P7 citation guard over the union --------------------
def test_terminal_citation_guard_over_union_raises() -> None:
    rc1, rc2 = _rc(1, "risk one"), _rc(2, "risk two")
    retriever = _FakeRetriever({"a": [rc1, rc2]})
    # Answer cites [5] when the union has only 2 sources; the guard retries then raises.
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("stop")],
        _answer("Per [5].", [5]),
    )
    with pytest.raises(ResearchGuardError):
        answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever)


# ---- budget: LLM calls == decision calls + 1 terminal ------------------------
def test_budget_decisions_plus_one_terminal() -> None:
    retriever = _FakeRetriever({"a": [_rc(1, "risk one")]})
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("stop")],
        _answer("Summary [1].", [1]),
    )
    answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever)
    assert llm.decision_calls == 2  # search + stop
    assert llm.calls == llm.decision_calls + 1  # + the single terminal synthesis


# ---- filter scoping: ticker folds into a ChunkFilter -------------------------
def test_ticker_scopes_retrieval_filter() -> None:
    captured: dict[str, ChunkFilter | None] = {}

    class _CapturingRetriever(_FakeRetriever):
        def retrieve(
            self, query: str, *, top_k: int, where: ChunkFilter | None = None
        ) -> EvidenceSet:
            captured["where"] = where
            return super().retrieve(query, top_k=top_k, where=where)

    retriever = _CapturingRetriever({"a": [_rc(1, "risk one")]})
    llm = _FakeLLM(
        [_decision("search", "a", "NVDA"), _decision("stop")], _answer("Summary [1].", [1])
    )
    answer_multistep("q", settings=_SETTINGS, llm=llm, retriever=retriever)
    assert captured["where"] == ChunkFilter(ticker="NVDA")
