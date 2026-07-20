"""A6.2f — training driver + checkpointing: standardizer, wrappers, train, freeze/load round-trip.

CI-floor (torch-free): exercises the numpy ``reinforce``/``bc`` paths only (the ``ppo`` checkpoint
round-trip lives in the torch-gated ``test_rl_ppo.py``). Uses the same deterministic NVDA→MU fake
bridge env as ``test_rl_env`` (a ticker-scoped ``FakeSystem`` + one labeled episode — no embedder,
no corpus, no LLM). The load-bearing test is the **round-trip equivalence**: the frozen ``(μ, σ)``
applied loader-side on the *raw* env reproduces the train-time (std-env + raw policy) behavior
exactly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from stock_agent.rag.rl.action import named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv, RetrieverFactory, StepInfo
from stock_agent.rag.rl.reinforce import greedy_rollout
from stock_agent.rag.rl.state import N_STATE_FEATURES, STATE_FEATURE_NAMES
from stock_agent.rag.rl.train import (
    CHECKPOINT_VERSION,
    StandardizedPolicy,
    Standardizer,
    StandardizingEnv,
    TrainConfig,
    fit_state_standardizer,
    load_checkpoint,
    save_checkpoint,
    train_policy,
)
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

S = Settings(_env_file=None)
ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"]}
UNIVERSE = {"NVDA", "MU"}

# pruned indices (pinned A6.2b): 0 STOP, 1 hybrid@self, 2 hybrid@disc0, 3 hybrid@disc1,
# 4 graph@self, 5 graph@disc0, 6 graph@disc1.
STOP_A, HY_SELF, HY_DISC0 = 0, 1, 2


def _rc(cid: str, text: str, *, ticker: str) -> RetrievedChunk:
    chunk = DocumentChunk(
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
    return RetrievedChunk(chunk=chunk, score=0.9)


class FakeSystem:
    """RetrievalSystem returning canned chunks scoped by ``where.ticker`` (the env scoping path)."""

    name = "fake"  # satisfies the RetrievalSystem protocol's ``name`` member

    def __init__(self, by_ticker: dict[str | None, list[RetrievedChunk]]) -> None:
        self._by_ticker = by_ticker

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        ticker = where.ticker if where is not None else None
        return EvidenceSet(query=query, chunks=list(self._by_ticker.get(ticker, []))[:top_k])


NVDA_CHUNK = _rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", ticker="NVDA")
MU_CHUNK = _rc("MU:0", "Micron cites critical information infrastructure rules.", ticker="MU")
EP = MultiHopQuery(
    question="Which memory supplier NVIDIA depends on discloses Chinese cybersecurity rules?",
    aspects=[
        Aspect(name="A1 names supplier", spans=["Micron"]),
        Aspect(name="A2 supplier CAC", spans=["critical information infrastructure"]),
    ],
    seed="NVDA",
    stratum="HARD",
    relation="depends_on",
    group_id="NVDA:MU",
)
# hybrid solves the bridge; graph returns junk (a dominated arm) so the optimum is clearly hybrid.
HYBRID = FakeSystem({"NVDA": [NVDA_CHUNK], "MU": [MU_CHUNK]})
GRAPH_JUNK = FakeSystem(
    {t: [_rc(f"J:{t}:0", "weather sunny; markets open", ticker=t)] for t in ("NVDA", "MU")}
)


def _factory() -> RetrieverFactory:
    return lambda arm: {"hybrid": HYBRID, "graph": GRAPH_JUNK}[arm]


def _env() -> RagRetrievalEnv:
    return RagRetrievalEnv(
        [EP],
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=_factory(),
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.05,
    )


# ---- Standardizer -------------------------------------------------------------------------------
def test_standardizer_transform_and_constant_feature_guard() -> None:
    std = Standardizer(mean=np.array([1.0, 5.0, 2.0]), std=np.array([2.0, 1.0, 4.0]))
    assert np.allclose(std.transform(np.array([3.0, 5.0, 10.0])), [1.0, 0.0, 2.0])


def test_fit_standardizer_shapes_and_no_zero_std() -> None:
    std = fit_state_standardizer(_env(), [0], seed=0, n_rollouts=4)
    assert std.mean.shape == (N_STATE_FEATURES,) and std.std.shape == (N_STATE_FEATURES,)
    assert np.all(std.std > 0.0)  # constant features guarded to 1.0, never 0 (no div-by-zero)
    # evidence-block features (step_idx, budget, …) vary across a random rollout ⇒ genuine σ>~0.
    assert std.std[N_STATE_FEATURES - 7] > 1e-6  # step_idx


def test_fit_standardizer_guards() -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        fit_state_standardizer(_env(), [], seed=0, n_rollouts=4)
    with pytest.raises(ValueError, match="n_rollouts"):
        fit_state_standardizer(_env(), [0], seed=0, n_rollouts=0)


# ---- wrappers -----------------------------------------------------------------------------------
def test_standardizing_env_applies_transform_and_passes_through() -> None:
    env = _env()
    std = fit_state_standardizer(env, [0], seed=0, n_rollouts=4)
    wrapped = StandardizingEnv(env, std)
    raw0 = env.reset(0)
    w0 = wrapped.reset(0)
    assert np.allclose(w0, std.transform(raw0))
    assert wrapped.n_actions == env.n_actions
    assert list(wrapped.action_space) == list(env.action_space)
    ws, wr, wdone, winfo = wrapped.step(HY_SELF)
    env.reset(0)
    rs, rr, rdone, rinfo = env.step(HY_SELF)
    assert np.allclose(ws, std.transform(rs))  # state standardized …
    # state standardized above; reward/done/info pass through unchanged.
    assert isinstance(winfo, StepInfo) and winfo.coverage == rinfo.coverage
    assert wr == rr and wdone == rdone


def test_standardized_policy_matches_raw_on_transformed_input() -> None:
    env = _env()
    trained = train_policy(env, [0], TrainConfig(algo="reinforce", iterations=5, seed=0))
    wrapped = StandardizedPolicy(trained.policy, trained.standardizer)
    raw = env.reset(0)
    mask = env.legal_mask()
    assert np.allclose(
        wrapped.action_probs(raw, mask=mask),
        trained.policy.action_probs(trained.standardizer.transform(raw), mask=mask),
    )
    assert wrapped.greedy_action(raw, mask=mask) == trained.policy.greedy_action(
        trained.standardizer.transform(raw), mask=mask
    )


# ---- training -----------------------------------------------------------------------------------
def test_reinforce_learns_the_bridge() -> None:
    env = _env()
    cfg = TrainConfig(
        algo="reinforce", iterations=300, episodes_per_batch=16, lr=0.25, seed=0, n_fit_rollouts=3
    )
    trained = train_policy(env, [0], cfg)
    assert trained.config.n_train == 1
    # greedy should search self first (not STOP), bridge, and reach full coverage.
    std_env = StandardizingEnv(env, trained.standardizer)
    g = greedy_rollout(std_env, trained.policy, 0)
    assert int(g.actions[0]) == HY_SELF
    assert g.total_return > 0.9  # ≈ 1.0 coverage − two small hybrid costs, minus nothing wasted


def test_reinforce_is_deterministic() -> None:
    cfg = TrainConfig(algo="reinforce", iterations=40, seed=7)
    a = train_policy(_env(), [0], cfg)
    b = train_policy(_env(), [0], cfg)
    assert np.allclose(a.policy.W, b.policy.W)
    assert np.allclose(a.standardizer.mean, b.standardizer.mean)


def test_bc_clones_the_scripted_expert() -> None:
    env = _env()
    trained = train_policy(env, [0], TrainConfig(algo="bc", bc_epochs=60, bc_lr=0.5, seed=0))
    assert trained.history_metric == "cross_entropy"
    assert trained.history[0] > trained.history[-1]  # BC reduces cross-entropy
    std_env = StandardizingEnv(env, trained.standardizer)
    g = greedy_rollout(std_env, trained.policy, 0)
    assert list(g.actions) == [HY_SELF, HY_DISC0, STOP_A]  # self → bridge → reflective stop


# ---- checkpoint freeze / load round-trip (the load-bearing correctness test) --------------------
@pytest.mark.parametrize("algo", ["reinforce", "bc"])
def test_checkpoint_round_trip_reproduces_behavior(algo: str, tmp_path: Path) -> None:
    env = _env()
    cfg = TrainConfig(algo=algo, iterations=120, lr=0.25, bc_epochs=60, seed=1, n_fit_rollouts=3)
    trained = train_policy(env, [0], cfg)

    # Train-time behavior: std-env + raw policy.
    std_env = StandardizingEnv(env, trained.standardizer)
    g_train = greedy_rollout(std_env, trained.policy, 0)

    # Reload from the in-memory dict AND from a written file; both must reproduce it on the RAW env.
    sources: list[dict[str, object] | Path] = [
        trained.to_dict(),
        save_checkpoint(trained, tmp_path / "policy.json"),
    ]
    for source in sources:
        loaded = load_checkpoint(source)
        g_load = greedy_rollout(env, loaded.policy, 0)  # raw env + loader-standardized policy
        assert list(g_load.actions) == list(g_train.actions)
        raw0 = env.reset(0)
        mask = env.legal_mask()
        assert np.allclose(
            loaded.policy.action_probs(raw0, mask=mask),
            trained.policy.action_probs(trained.standardizer.transform(raw0), mask=mask),
        )
        assert loaded.algo == algo and loaded.action_space == "pruned"


def test_checkpoint_meta_and_version() -> None:
    trained = train_policy(_env(), [0], TrainConfig(algo="reinforce", iterations=3, seed=0))
    data = trained.to_dict()
    assert data["checkpoint_version"] == CHECKPOINT_VERSION
    assert data["state_feature_names"] == list(STATE_FEATURE_NAMES)
    assert data["action_labels"][HY_SELF] == "hybrid@self"
    assert data["weights"]["kind"] == "linear_softmax"


# ---- drift guards: a reorder silently remaps weights, so the loader must reject it ---------------
def test_load_rejects_version_drift() -> None:
    data = train_policy(_env(), [0], TrainConfig(algo="reinforce", iterations=2)).to_dict()
    with pytest.raises(ValueError, match="version"):
        load_checkpoint({**data, "checkpoint_version": CHECKPOINT_VERSION + 1})


def test_load_rejects_feature_drift() -> None:
    data = train_policy(_env(), [0], TrainConfig(algo="reinforce", iterations=2)).to_dict()
    bad = {**data, "state_feature_names": ["renamed", *data["state_feature_names"][1:]]}
    with pytest.raises(ValueError, match="state_feature_names"):
        load_checkpoint(bad)


def test_load_rejects_action_label_drift() -> None:
    data = train_policy(_env(), [0], TrainConfig(algo="reinforce", iterations=2)).to_dict()
    bad = {**data, "action_labels": ["STOP", "graph@self", *data["action_labels"][2:]]}
    with pytest.raises(ValueError, match="action_labels"):
        load_checkpoint(bad)


def test_load_rejects_weight_shape_mismatch() -> None:
    data = train_policy(_env(), [0], TrainConfig(algo="reinforce", iterations=2)).to_dict()
    bad = {**data, "weights": {**data["weights"], "W": [row[:-1] for row in data["weights"]["W"]]}}
    with pytest.raises(ValueError, match="shape"):
        load_checkpoint(bad)
