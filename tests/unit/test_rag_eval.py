"""Retrieval-quality eval harness (RAG P9b + advanced-RAG A1) — pure metrics + offline fakes."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.eval import (
    EmbedderReport,
    LabeledQuery,
    RetrievalSystem,
    SystemReport,
    citation_accuracy,
    evaluate_query,
    evaluate_system,
    format_reports_markdown,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_ab,
)
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.research import GroundedAnswer, SourceCitation
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk

_EMB = FakeEmbedder(dim=32)


def _chunk(
    idx: int,
    text: str,
    *,
    ticker: str = "NVDA",
    section: str | None = "Item 1A. Risk Factors",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{ticker}:10-K:2026-02-25:{idx}",
        document_id=f"{ticker}:10-K:2026-02-25",
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type="10-K",
        source="SEC",
        source_url=f"https://www.sec.gov/Archives/edgar/data/x/{ticker.lower()}.htm",
        filing_date=date(2026, 2, 25),
        section=section,
    )


# ---- pure ranking metrics ----------------------------------------------------


def test_hit_at_k_any_relevant_in_topk() -> None:
    assert hit_at_k([False, False, True], 3) == 1.0
    assert hit_at_k([False, False, True], 2) == 0.0  # the hit is below the cutoff
    assert hit_at_k([], 5) == 0.0


def test_reciprocal_rank_is_one_over_first_hit() -> None:
    assert reciprocal_rank([False, True, True], 5) == pytest.approx(0.5)  # first hit at rank 2
    assert reciprocal_rank([True, False], 5) == 1.0
    assert reciprocal_rank([False, False], 5) == 0.0
    assert reciprocal_rank([False, True], 1) == 0.0  # hit outside top-1


def test_precision_at_k_over_returned() -> None:
    assert precision_at_k([True, False, True, False], 4) == pytest.approx(0.5)
    # Fewer than k returned -> denominator is what was actually returned (no penalty).
    assert precision_at_k([True, True], 5) == 1.0
    assert precision_at_k([], 5) == 0.0


def test_recall_at_k_against_corpus_relevant_count() -> None:
    assert recall_at_k([True, False, True], n_relevant=4, k=3) == pytest.approx(0.5)  # 2 of 4
    assert recall_at_k([True], n_relevant=1, k=3) == 1.0
    assert recall_at_k([True], n_relevant=0, k=3) == 0.0  # mislabeled -> 0 (excluded from mean)


def test_ndcg_at_k_self_ideal_golden() -> None:
    # gains [1,0,1]: DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5;
    # ideal [1,1,0]: IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093; nDCG = 0.91973.
    assert ndcg_at_k([1, 0, 1], 3) == pytest.approx(0.91973, abs=1e-4)
    # Perfectly ordered -> 1.0; reverse-of-ideal still <1 (discount bites later ranks).
    assert ndcg_at_k([1, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([], 5) == 0.0  # nothing retrieved / no relevance


def test_ndcg_at_k_ideal_pool_caps_unretrieved_relevant() -> None:
    # Retrieved one relevant chunk, but the corpus holds three relevant ones: the ideal DCG is
    # computed over all three, so the score is capped well below 1 even though every *retrieved*
    # item was relevant. IDCG = 1 + 1/log2(3) + 1/log2(4) = 2.13093; nDCG = 1/2.13093 = 0.46928.
    assert ndcg_at_k([1], 3, ideal_gains=[1, 1, 1]) == pytest.approx(0.46928, abs=1e-4)


# ---- relevance predicate + graded relevance ----------------------------------


def test_is_relevant_is_case_and_whitespace_insensitive_any_span() -> None:
    q = LabeledQuery(query="x", relevant_spans=["Export Control", "missing"])
    assert q.is_relevant(_chunk(0, "Our   EXPORT\ncontrol exposure is material."))  # normalized
    assert not q.is_relevant(_chunk(1, "Unrelated text about widgets."))


def test_labeled_query_requires_a_span() -> None:
    with pytest.raises(ValueError):
        LabeledQuery(query="x", relevant_spans=[])


def test_relevance_grade_counts_distinct_spans() -> None:
    q = LabeledQuery(query="x", relevant_spans=["export controls", "licensing", "missing"])
    # Chunk contains two of the three spans -> grade 2 (the nDCG gain), is_relevant True.
    chunk = _chunk(0, "Export controls and a licensing requirement constrain China sales.")
    assert q.relevance_grade(chunk) == 2
    assert q.is_relevant(chunk)


def test_metadata_constraints_gate_relevance() -> None:
    q = LabeledQuery(
        query="x",
        relevant_spans=["export controls"],
        expected_sections=["Item 1A. Risk Factors"],
        expected_document_types=["10-K"],
    )
    hit = _chunk(0, "Export controls apply.", section="Item 1A. Risk Factors")
    wrong_section = _chunk(1, "Export controls apply.", section="Item 7. MD&A")
    assert q.relevance_grade(hit) == 1 and q.is_relevant(hit)
    # Span present but section disallowed -> grade 0 (chunking-invariant metadata gate).
    assert q.relevance_grade(wrong_section) == 0 and not q.is_relevant(wrong_section)


# ---- citation accuracy (deterministic, canned answer; no LLM) ----------------


def test_citation_accuracy_is_precision_of_citations() -> None:
    relevant = _chunk(0, "Export controls and licensing requirements affect China sales.")
    irrelevant = _chunk(1, "Our campus operations expanded this year.")
    evidence = EvidenceSet(
        query="export controls?",
        chunks=[
            RetrievedChunk(chunk=relevant, score=0.9),
            RetrievedChunk(chunk=irrelevant, score=0.8),
        ],
    )
    labeled = LabeledQuery(query="export controls?", relevant_spans=["export controls"])
    answer = GroundedAnswer(
        question="export controls?",
        answer="Yes [1][2].",
        citations=[
            SourceCitation(marker=1, chunk_id=relevant.chunk_id, label="rel"),
            SourceCitation(marker=2, chunk_id=irrelevant.chunk_id, label="irrel"),
        ],
    )
    # One of two citations points to a relevant chunk -> 0.5.
    assert citation_accuracy(answer, evidence, labeled) == pytest.approx(0.5)


def test_citation_accuracy_none_when_no_citations() -> None:
    evidence = EvidenceSet(query="q", chunks=[])
    labeled = LabeledQuery(query="q", relevant_spans=["x"])
    refusal = GroundedAnswer(question="q", answer="Insufficient evidence found.", citations=[])
    # No citations -> None so the aggregate excludes it (an honest refusal isn't a 0).
    assert citation_accuracy(refusal, evidence, labeled) is None


# ---- evaluate_query + run_ab (offline, FakeEmbedder) -------------------------

# FakeEmbedder is hash-based: a query whose text equals a chunk's text self-matches (cosine 1)
# and ranks first, so retrieval is deterministic for these fixtures.
_CORPUS = [
    _chunk(0, "Export controls and supply chain disruption could harm our results."),
    _chunk(1, "Data center revenue grew on strong demand for our platforms."),
    _chunk(2, "Competition in accelerated computing is intense."),
]


class _FixedSystem:
    """A RetrievalSystem that returns a canned EvidenceSet (decouples eval from a real store)."""

    name = "fixed"

    def __init__(self, evidence: EvidenceSet) -> None:
        self._evidence = evidence

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        return self._evidence


def test_retriever_and_fake_satisfy_retrieval_system_protocol() -> None:
    store = InMemoryVectorStore()
    store.add(_CORPUS, _EMB.embed_documents([c.text for c in _CORPUS]))
    retriever = Retriever(_EMB, store)
    assert isinstance(retriever, RetrievalSystem)
    assert retriever.name == "dense:fake"  # defaults to the wrapped embedder's name
    assert isinstance(_FixedSystem(EvidenceSet(query="q")), RetrievalSystem)


def test_evaluate_system_uses_system_name_and_scores_ndcg() -> None:
    # Top result relevant (grade 1), second irrelevant; corpus has exactly one relevant chunk.
    evidence = EvidenceSet(
        query=_CORPUS[0].text,
        chunks=[
            RetrievedChunk(chunk=_CORPUS[0], score=0.9),
            RetrievedChunk(chunk=_CORPUS[1], score=0.5),
        ],
    )
    q = LabeledQuery(query=_CORPUS[0].text, ticker="NVDA", relevant_spans=["export controls"])
    report = evaluate_system(_FixedSystem(evidence), [q], top_k=2, corpus_chunks=_CORPUS)
    assert isinstance(report, SystemReport) and report.system == "fixed"
    assert report.hit_at_k == 1.0 and report.mrr == 1.0
    assert report.ndcg_at_k == pytest.approx(1.0)  # the single relevant chunk is ranked first
    assert report.per_query[0].ndcg == pytest.approx(1.0)


def test_evaluate_query_scores_a_self_match_first() -> None:
    store = InMemoryVectorStore()
    store.add(_CORPUS, _EMB.embed_documents([c.text for c in _CORPUS]))
    retriever = Retriever(_EMB, store)
    q = LabeledQuery(query=_CORPUS[0].text, ticker="NVDA", relevant_spans=["export controls"])

    rep = evaluate_query(retriever, q, top_k=3, corpus_chunks=_CORPUS)

    assert rep.hit == 1.0 and rep.reciprocal_rank == 1.0  # exact chunk ranks first
    assert rep.n_relevant_corpus == 1  # only chunk 0 contains the span
    assert rep.recall == 1.0


def test_run_ab_aggregates_per_embedder() -> None:
    queries = [
        LabeledQuery(query=_CORPUS[0].text, ticker="NVDA", relevant_spans=["supply chain"]),
        LabeledQuery(query=_CORPUS[1].text, ticker="NVDA", relevant_spans=["data center"]),
    ]
    reports = run_ab(_CORPUS, queries, {"fake": _EMB}, top_k=3)

    assert len(reports) == 1
    r = reports[0]
    assert isinstance(r, EmbedderReport) and r.embedder == "fake" and r.n_queries == 2
    assert r.hit_at_k == 1.0 and r.mrr == 1.0  # both self-match at rank 1
    assert r.recall_at_k == pytest.approx(1.0)


def test_run_ab_preserves_embedder_order_and_isolates_stores() -> None:
    # Two embedders -> two independent stores; report order follows the mapping order.
    reports = run_ab(
        _CORPUS,
        [LabeledQuery(query=_CORPUS[2].text, ticker="NVDA", relevant_spans=["competition"])],
        {"a": FakeEmbedder(dim=16), "b": FakeEmbedder(dim=32)},
        top_k=2,
    )
    assert [r.embedder for r in reports] == ["a", "b"]


def test_format_reports_markdown_table() -> None:
    reports = run_ab(
        _CORPUS,
        [LabeledQuery(query=_CORPUS[0].text, ticker="NVDA", relevant_spans=["export"])],
        {"fake": _EMB},
        top_k=3,
    )
    table = format_reports_markdown(reports)
    assert "| system | hit@3 | MRR | nDCG@3 |" in table
    assert "| fake |" in table
    assert format_reports_markdown([]) == "_no embedders evaluated_"
