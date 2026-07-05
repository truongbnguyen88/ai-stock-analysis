"""A6.1c — reward oracle adapter: composite math, label dispatch, reward-hacking sentinel (CI).

Deterministic fakes only (no embedder/model). Quality is reused from A1 (nDCG) and A6.0 (coverage),
so these pin the *composite* r = quality − λ_c·cost and the dispatch, not the metric internals.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.rag.eval import LabeledQuery
from stock_agent.rag.reward import DEFAULT_ARM_COSTS, reward, reward_matrix
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

SETTINGS = Settings(_env_file=None)


class FakeSystem:
    """RetrievalSystem returning canned chunks keyed by query string (ignores the where filter)."""

    def __init__(self, name: str, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self.name = name
        self._m = mapping

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        return EvidenceSet(query=query, chunks=self._m.get(query, [])[:top_k])


def _chunk(cid: str, text: str, ticker: str = "NVDA") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id=cid.rsplit(":", 1)[0],
        chunk_index=int(cid.rsplit(":", 1)[1]),
        text=text,
        ticker=ticker,
        document_type="10-K",
        source="SEC",
        source_url="https://sec.gov/x",
        filing_date=date(2026, 2, 25),
        section="Item 1A. Risk Factors",
    )


def _rc(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(cid, text), score=0.9)


# --- multi-hop coverage reward -------------------------------------------------
_MH = MultiHopQuery(
    question="Q",
    aspects=[Aspect(name="a1", spans=["hopper"]), Aspect(name="a2", spans=["blackwell"])],
)


def test_multihop_reward_is_coverage_minus_cost() -> None:
    # One chunk covers both aspects -> coverage 1.0; reward = 1.0 - 0.1*0.1.
    good = FakeSystem("good", {"Q": [_rc("D:1", "the hopper and blackwell GPUs")]})
    r = reward(
        good, _MH, arm="hybrid", settings=SETTINGS, lambda_cost=0.1, arm_costs={"hybrid": 0.1}
    )
    assert r == pytest.approx(1.0 - 0.1 * 0.1)


def test_reward_hacking_sentinel_scores_low() -> None:
    # An arm dumping many irrelevant chunks earns 0 quality; cost then pushes it below a good arm.
    sentinel = FakeSystem("sentinel", {"Q": [_rc(f"D:{i}", f"irrelevant {i}") for i in range(20)]})
    good = FakeSystem("good", {"Q": [_rc("D:1", "hopper and blackwell")]})
    r_sent = reward(
        sentinel, _MH, arm="graph", settings=SETTINGS, lambda_cost=0.05, arm_costs=DEFAULT_ARM_COSTS
    )
    r_good = reward(
        good, _MH, arm="hybrid", settings=SETTINGS, lambda_cost=0.05, arm_costs=DEFAULT_ARM_COSTS
    )
    assert r_sent < r_good
    assert r_sent == pytest.approx(0.0 - 0.05 * DEFAULT_ARM_COSTS["graph"])  # 0 quality, only cost


def test_partial_coverage() -> None:
    # Only aspect a1 covered -> coverage 0.5; dense arm is free -> reward exactly 0.5.
    half = FakeSystem("half", {"Q": [_rc("D:1", "only hopper here")]})
    r = reward(
        half, _MH, arm="dense", settings=SETTINGS, lambda_cost=0.05, arm_costs=DEFAULT_ARM_COSTS
    )
    assert r == pytest.approx(0.5)


# --- single-shot nDCG reward ---------------------------------------------------
def test_labeled_query_reward_is_ndcg_minus_cost() -> None:
    lq = LabeledQuery(query="LQ", relevant_spans=["hopper"], ticker="NVDA")
    corpus = [_chunk("NVDA:10-K:1", "about the hopper gpu"), _chunk("NVDA:10-K:2", "unrelated")]
    sys = FakeSystem("dense", {"LQ": [_rc("NVDA:10-K:1", "about the hopper gpu")]})
    # Retrieved the sole relevant chunk at rank 1 -> nDCG 1.0; dense cost 0 -> reward 1.0.
    r = reward(
        sys,
        lq,
        arm="dense",
        settings=SETTINGS,
        corpus_chunks=corpus,
        lambda_cost=0.05,
        arm_costs=DEFAULT_ARM_COSTS,
    )
    assert r == pytest.approx(1.0)


def test_labeled_query_without_corpus_raises() -> None:
    lq = LabeledQuery(query="LQ", relevant_spans=["hopper"], ticker="NVDA")
    sys = FakeSystem("dense", {"LQ": []})
    with pytest.raises(ValueError, match="corpus_chunks"):
        reward(sys, lq, arm="dense", settings=SETTINGS)


def test_unsupported_label_type_raises() -> None:
    sys = FakeSystem("dense", {})
    with pytest.raises(TypeError):
        reward(sys, object(), arm="dense", settings=SETTINGS)


# --- full-information matrix ---------------------------------------------------
def test_reward_matrix_shape_and_values() -> None:
    good = FakeSystem("good", {"Q": [_rc("D:1", "hopper and blackwell")]})
    sentinel = FakeSystem("sentinel", {"Q": [_rc("D:2", "nothing relevant")]})
    systems = {"dense": good, "graph": sentinel}  # column order = insertion order
    r_matrix = reward_matrix(
        systems, [_MH], settings=SETTINGS, lambda_cost=0.1, arm_costs={"dense": 0.0, "graph": 0.3}
    )
    assert r_matrix.shape == (1, 2)
    assert r_matrix[0, 0] == pytest.approx(1.0)  # good/dense: coverage 1, cost 0
    assert r_matrix[0, 1] == pytest.approx(0.0 - 0.1 * 0.3)  # sentinel/graph: coverage 0, cost 0.3
