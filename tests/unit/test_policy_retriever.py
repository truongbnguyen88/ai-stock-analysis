"""A6.1f — PolicyRetriever: arm selection, delegation, per-instance arm cache, gated logging (CI).

Pure fakes (no embedder/store/graph): a ``FakePolicy`` chooses the arm, a fake ``build_arm`` returns
a canned ``RetrievalSystem``. Featurization runs for real (a cheap pure function) with explicit
empty/known context sources so the test does no I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np

from stock_agent.rag.policy_features import FEATURE_NAMES
from stock_agent.rag.policy_retriever import PolicyRetriever
from stock_agent.rag.retrieval_log import read_log
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

ARM_NAMES = ("dense", "reranked", "hybrid", "hybrid+rerank", "graph")


class FakePolicy:
    """Deterministic policy: always returns ``action``; records the last context it saw."""

    def __init__(self, action: int, *, n_arms: int = 5, propensity: float = 1.0) -> None:
        self.action = action
        self.n_arms = n_arms
        self.name = "fake"
        self._propensity = propensity
        self.last_x: np.ndarray | None = None

    def act(self, x: np.ndarray) -> tuple[int, float]:
        self.last_x = x
        return self.action, self._propensity

    def prob(self, action: int, x: np.ndarray) -> float:
        return 1.0 if action == self.action else 0.0


class FakeArm:
    """A RetrievalSystem returning one canned chunk tagged with its own arm name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        self.calls += 1
        chunk = DocumentChunk(
            chunk_id=f"{self.name}:0",
            document_id=self.name,
            chunk_index=0,
            text=f"evidence from {self.name}",
            ticker="NVDA",
            document_type="10-K",
            source="SEC",
            source_url="https://sec.gov/x",
            filing_date=date(2026, 2, 25),
            section="Item 1A. Risk Factors",
        )
        return EvidenceSet(query=query, chunks=[RetrievedChunk(chunk=chunk, score=0.5)])


def _factory() -> tuple[dict[str, FakeArm], Callable[[str, Settings], FakeArm]]:
    """A build_arm callable that returns a shared FakeArm per name (so we can count builds)."""
    built: dict[str, FakeArm] = {}

    def build_arm(name: str, settings: Settings) -> FakeArm:
        if name not in built:
            built[name] = FakeArm(name)
        return built[name]

    return built, build_arm


def _retriever(policy: FakePolicy, settings: Settings) -> PolicyRetriever:
    _, build_arm = _factory()
    return PolicyRetriever(
        policy,
        settings,
        build_arm=build_arm,
        alias_map={},  # explicit empty ⇒ no mention detection, no I/O
        graph_universe={"NVDA"},
    )


def test_selects_the_arm_the_policy_returns() -> None:
    settings = Settings(_env_file=None)
    pol = FakePolicy(action=4)  # -> "graph"
    r = _retriever(pol, settings)
    ev = r.retrieve("What are the risks?", top_k=3)
    assert ev.chunks[0].chunk.text == "evidence from graph"
    assert r.name == "policy(fake)"


def test_arm_is_built_once_and_cached() -> None:
    settings = Settings(_env_file=None)
    pol = FakePolicy(action=2)  # -> "hybrid"
    built, build_arm = _factory()
    r = PolicyRetriever(pol, settings, build_arm=build_arm, alias_map={}, graph_universe=set())
    r.retrieve("q1", top_k=2)
    r.retrieve("q2", top_k=2)
    assert built["hybrid"].calls == 2  # same arm served twice
    assert set(built) == {"hybrid"}  # only the chosen arm was ever built


def test_ticker_from_where_reaches_featurizer() -> None:
    # where.ticker="NVDA" (in the graph universe) ⇒ has_ticker=1 AND in_graph_universe=1 in x.
    settings = Settings(_env_file=None)
    pol = FakePolicy(action=0)
    r = _retriever(pol, settings)
    r.retrieve("overview please", top_k=2, where=ChunkFilter(ticker="NVDA"))
    x = pol.last_x
    assert x is not None
    assert x[FEATURE_NAMES.index("has_ticker")] == 1.0
    assert x[FEATURE_NAMES.index("in_graph_universe")] == 1.0


def test_logging_off_writes_nothing(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, retrieval_logging=False, retrieval_log_dir=tmp_path)
    r = _retriever(FakePolicy(action=1), settings)
    r.retrieve("q", top_k=2)
    assert read_log(settings.retrieval_log_dir / "retrieval_log.jsonl") == []


def test_logging_on_records_decision(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, retrieval_logging=True, retrieval_log_dir=tmp_path)
    r = _retriever(FakePolicy(action=4, propensity=0.2), settings)
    r.retrieve("supplier risk?", top_k=2, where=ChunkFilter(ticker="NVDA"))
    entries = read_log(settings.retrieval_log_dir / "retrieval_log.jsonl")
    assert len(entries) == 1
    e = entries[0]
    assert e.action == "graph"  # arm_names[4]
    assert e.propensity == 0.2
    assert e.ticker == "NVDA"
    assert e.context is not None and e.context["in_graph_universe"] == 1.0
    assert e.retrieved and e.retrieved[0].chunk_id == "graph:0"
