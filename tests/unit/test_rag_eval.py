"""Retrieval-quality A/B harness (RAG P9b) — pure metrics + offline run with fakes."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.eval import (
    EmbedderReport,
    LabeledQuery,
    evaluate_query,
    format_reports_markdown,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_ab,
)
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk

_EMB = FakeEmbedder(dim=32)


def _chunk(idx: int, text: str, *, ticker: str = "NVDA") -> DocumentChunk:
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
        section="Item 1A. Risk Factors",
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


# ---- relevance predicate -----------------------------------------------------


def test_is_relevant_is_case_and_whitespace_insensitive_any_span() -> None:
    q = LabeledQuery(query="x", relevant_spans=["Export Control", "missing"])
    assert q.is_relevant(_chunk(0, "Our   EXPORT\ncontrol exposure is material."))  # normalized
    assert not q.is_relevant(_chunk(1, "Unrelated text about widgets."))


def test_labeled_query_requires_a_span() -> None:
    with pytest.raises(ValueError):
        LabeledQuery(query="x", relevant_spans=[])


# ---- evaluate_query + run_ab (offline, FakeEmbedder) -------------------------

# FakeEmbedder is hash-based: a query whose text equals a chunk's text self-matches (cosine 1)
# and ranks first, so retrieval is deterministic for these fixtures.
_CORPUS = [
    _chunk(0, "Export controls and supply chain disruption could harm our results."),
    _chunk(1, "Data center revenue grew on strong demand for our platforms."),
    _chunk(2, "Competition in accelerated computing is intense."),
]


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
    assert "| embedder | hit@3 | MRR |" in table
    assert "| fake |" in table
    assert format_reports_markdown([]) == "_no embedders evaluated_"
