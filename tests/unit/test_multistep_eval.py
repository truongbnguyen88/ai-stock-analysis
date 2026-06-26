"""Multi-step (A4) eval — union aspect coverage vs. single-shot (canned LLM + fake retriever).

Headline: a comparative question's loop union covers BOTH aspects (1.0) while one retrieval covers
only one (0.5) → a positive coverage gain — the empirical value of the extra hops. Plus the pure
coverage / citation-accuracy metrics, the empty-corpus refusal, and the aggregate summary. Offline.
"""

from __future__ import annotations

import json
import re
from datetime import date

from stock_agent.research.multistep_eval import (
    Aspect,
    MultiHopQuery,
    coverage,
    evaluate_multihop,
    evaluate_multihop_set,
    format_multihop_markdown,
    multihop_citation_accuracy,
)
from stock_agent.research.prompts import REACT_SYSTEM
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.research import GroundedAnswer, SourceCitation
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings


# ---- doubles -----------------------------------------------------------------
class _FakeLLM:
    """A STATELESS canned model (like the real one): each decision is derived from the evidence
    count in the prompt, so one instance can drive many independent queries. Decisions: 0 chunks →
    search hop A; 1 chunk → search hop B; ≥2 → stop. Non-decision calls return the answer JSON.
    """

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        if system != REACT_SYSTEM:
            return self._answer  # terminal synthesis
        match = re.search(r"EVIDENCE GATHERED \((\d+) chunks\)", user)
        n = int(match.group(1)) if match else 0
        if n == 0:
            return _decision("search", "A risk")
        if n == 1:
            return _decision("search", "B risk")
        return _decision("stop")


class _FakeRetriever:
    """Maps a query string → canned chunks (the single-shot baseline keys off the question)."""

    name = "fake"

    def __init__(self, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self._mapping = mapping

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        return EvidenceSet(query=query, chunks=self._mapping.get(query, []))


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


def _decision(action: str, query: str | None = None) -> str:
    d: dict[str, object] = {"thought": "t", "action": action}
    if query is not None:
        d["query"] = query
    return json.dumps(d)


def _answer(text: str, citations: list[int]) -> str:
    return json.dumps({"answer": text, "citations": citations, "insufficient_evidence": False})


_SETTINGS = Settings(_env_file=None)
# chunk 1 carries aspect-A's span ("alpha"), chunk 2 carries aspect-B's span ("beta").
_RC_A = _rc(1, "Risk alpha: export controls on NVIDIA accelerators.")
_RC_B = _rc(2, "Risk beta: AMD wafer capacity constraints.")
_ASPECTS = [Aspect(name="A", spans=["alpha"]), Aspect(name="B", spans=["beta"])]


# ---- pure coverage -----------------------------------------------------------
def test_coverage_full_partial_empty() -> None:
    frac, covered, missed = coverage([_RC_A, _RC_B], _ASPECTS)
    assert frac == 1.0 and covered == ["A", "B"] and missed == []
    frac, covered, missed = coverage([_RC_A], _ASPECTS)  # only aspect A present
    assert frac == 0.5 and covered == ["A"] and missed == ["B"]
    frac, covered, missed = coverage([], _ASPECTS)
    assert frac == 0.0 and covered == [] and missed == ["A", "B"]


def test_coverage_is_case_insensitive() -> None:
    rc = _rc(3, "RISK ALPHA in upper case")
    frac, _, _ = coverage([rc], [Aspect(name="A", spans=["alpha"])])
    assert frac == 1.0


# ---- citation accuracy -------------------------------------------------------
def test_citation_accuracy_none_when_no_citations() -> None:
    ans = GroundedAnswer(question="q", answer="no claim")
    assert multihop_citation_accuracy(ans, [_RC_A, _RC_B], _ASPECTS) is None


def test_citation_accuracy_scores_relevant_cites() -> None:
    # [1] -> chunk with "alpha" (relevant); a cite to a chunk-id not in the union counts wrong.
    ans = GroundedAnswer(
        question="q",
        answer="See [1] and [2].",
        citations=[
            SourceCitation(marker=1, chunk_id=_RC_A.chunk.chunk_id, label="A"),
            SourceCitation(marker=2, chunk_id="not-in-union", label="X"),
        ],
    )
    assert multihop_citation_accuracy(ans, [_RC_A, _RC_B], _ASPECTS) == 0.5


# ---- the headline: union covers both aspects, single-shot covers one ---------
def _headline_query() -> MultiHopQuery:
    return MultiHopQuery(question="compare A and B", aspects=_ASPECTS)


def _headline_retriever() -> _FakeRetriever:
    # Single-shot of the question returns only chunk A; the two hops fetch A then B.
    return _FakeRetriever(
        {"compare A and B": [_RC_A], "A risk": [_RC_A], "B risk": [_RC_B]}
    )


def _headline_llm() -> _FakeLLM:
    return _FakeLLM(_answer("NVDA flags alpha [1]; AMD flags beta [2].", [1, 2]))


def test_multistep_union_beats_single_shot() -> None:
    rep = evaluate_multihop(
        _headline_query(), settings=_SETTINGS, llm=_headline_llm(), retriever=_headline_retriever()
    )
    assert rep.multistep_coverage == 1.0  # union has both aspects
    assert rep.single_shot_coverage == 0.5  # one retrieval misses the AMD hop
    assert rep.coverage_gain == 0.5  # the value of the extra hop
    assert rep.covered_aspects == ["A", "B"] and rep.missed_aspects == []
    assert rep.n_steps == 2 and rep.n_evidence == 2
    assert rep.citation_accuracy == 1.0
    assert rep.insufficient_evidence is False


# ---- empty corpus: refusal, zero coverage, no citation accuracy --------------
def test_empty_corpus_zero_coverage_and_insufficient() -> None:
    retriever = _FakeRetriever({})  # every query returns nothing
    rep = evaluate_multihop(
        _headline_query(), settings=_SETTINGS, llm=_headline_llm(), retriever=retriever
    )
    assert rep.multistep_coverage == 0.0 and rep.single_shot_coverage == 0.0
    assert rep.coverage_gain == 0.0
    assert rep.insufficient_evidence is True
    assert rep.citation_accuracy is None  # honest refusal makes no citation


# ---- aggregate + markdown ----------------------------------------------------
def test_set_summary_and_markdown() -> None:
    queries = [_headline_query(), _headline_query()]
    reports, summary = evaluate_multihop_set(
        queries, settings=_SETTINGS, llm=_headline_llm(), retriever=_headline_retriever()
    )
    assert summary.n_queries == 2
    assert summary.mean_multistep_coverage == 1.0
    assert summary.mean_single_shot_coverage == 0.5
    assert summary.mean_coverage_gain == 0.5
    assert summary.mean_citation_accuracy == 1.0
    assert summary.mean_n_steps == 2.0

    md = format_multihop_markdown(reports, summary)
    assert "| gain |" in md  # table header column
    assert "over 2" in md and "gain +0.50" in md  # aggregate line
