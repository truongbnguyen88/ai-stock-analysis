"""A6.2d — numpy REINFORCE + BC: gradient correctness, convergence, masking, scripted expert.

Deterministic + torch-free (the CI-floor learner). Pins the closed-form ``grad_log_prob`` against
finite differences, REINFORCE convergence on a 2-step toy with a known step-dependent optimum,
behavior cloning reducing cross-entropy toward an expert, masked-softmax correctness, and the A4/A5
scripted-expert / replay helpers against the real ``RagRetrievalEnv`` (fed a fake retriever).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stock_agent.rag.rl.action import named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv, RetrieverFactory
from stock_agent.rag.rl.policy import LinearSoftmaxPolicy, LinearValueBaseline
from stock_agent.rag.rl.reinforce import (
    behavior_clone,
    bridge_expert_rollout,
    discounted_returns_to_go,
    reinforce_update,
    replay,
    rollout,
)
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings


# ---- a tiny toy MDP for learner tests (no corpus, no env dependency) -----------------------------
class ToyEnv:
    """2-step, 3-action MDP with a step-dependent optimum (STOP=0, arm A=1, arm B=2).

    step 0 rewards arm A; step 1 rewards arm B; STOP ends the episode with 0. The optimum (A then B,
    return 2) requires *using the state* (one-hot step), so it exercises the weight matrix rather
    than just the per-action bias. Satisfies the ``EpisodeMDP`` structural interface.
    """

    n_actions = 3

    def __init__(self, max_steps: int = 2) -> None:
        self.max_steps = max_steps
        self.action_space = list(range(self.n_actions))
        self._t = 0
        self._done = True

    def reset(self, episode_idx: int) -> np.ndarray:
        self._t = 0
        self._done = False
        return self._state()

    def _state(self) -> np.ndarray:
        v = np.zeros(2)
        if self._t < 2:
            v[self._t] = 1.0  # one-hot of the current step (all-zero at the terminal state)
        return v

    def legal_mask(self) -> np.ndarray:
        return np.ones(self.n_actions, dtype=bool)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, object]:
        if self._done:
            raise RuntimeError("step on a done ToyEnv")
        if action_idx == 0:  # STOP terminates with 0
            self._done = True
            return self._state(), 0.0, True, None
        good = (self._t == 0 and action_idx == 1) or (self._t == 1 and action_idx == 2)
        self._t += 1
        self._done = self._t >= self.max_steps
        return self._state(), (1.0 if good else 0.0), self._done, None


# ---- gradient correctness ------------------------------------------------------------------------
def test_grad_log_prob_matches_finite_difference() -> None:
    policy = LinearSoftmaxPolicy(d=4, n_actions=5, seed=3, init_scale=0.4)
    x = np.random.default_rng(0).standard_normal(4)
    a = 2
    analytic = policy.grad_log_prob(a, x)
    eps = 1e-6
    numeric = np.zeros_like(analytic)
    for k in range(policy.n_actions):
        for j in range(policy.W.shape[1]):
            orig = policy.W[k, j]
            policy.W[k, j] = orig + eps
            plus = policy.log_prob(a, x)
            policy.W[k, j] = orig - eps
            minus = policy.log_prob(a, x)
            policy.W[k, j] = orig
            numeric[k, j] = (plus - minus) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_grad_log_prob_matches_finite_difference_masked() -> None:
    policy = LinearSoftmaxPolicy(d=3, n_actions=5, seed=7, init_scale=0.5)
    x = np.random.default_rng(1).standard_normal(3)
    mask = np.array([True, True, True, False, True])  # action 3 illegal
    a = 2  # a legal action
    analytic = policy.grad_log_prob(a, x, mask=mask)
    eps = 1e-6
    numeric = np.zeros_like(analytic)
    for k in range(policy.n_actions):
        for j in range(policy.W.shape[1]):
            orig = policy.W[k, j]
            policy.W[k, j] = orig + eps
            plus = policy.log_prob(a, x, mask=mask)
            policy.W[k, j] = orig - eps
            minus = policy.log_prob(a, x, mask=mask)
            policy.W[k, j] = orig
            numeric[k, j] = (plus - minus) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6)
    assert np.allclose(analytic[3], 0.0)  # masked row has zero gradient


# ---- masking -------------------------------------------------------------------------------------
def test_masking_zeroes_illegal_actions() -> None:
    policy = LinearSoftmaxPolicy(d=3, n_actions=4, seed=1, init_scale=0.5)
    x = np.ones(3)
    mask = np.array([True, False, True, False])
    p = policy.action_probs(x, mask=mask)
    assert p[1] == 0.0 and p[3] == 0.0
    assert p.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    picks = {policy.act(x, rng=rng, mask=mask)[0] for _ in range(200)}
    assert picks <= {0, 2}  # never samples a masked action


# ---- determinism ---------------------------------------------------------------------------------
def test_policy_init_is_seeded_deterministic() -> None:
    a = LinearSoftmaxPolicy(d=5, n_actions=4, seed=42, init_scale=0.3)
    b = LinearSoftmaxPolicy(d=5, n_actions=4, seed=42, init_scale=0.3)
    c = LinearSoftmaxPolicy(d=5, n_actions=4, seed=43, init_scale=0.3)
    assert np.array_equal(a.W, b.W)
    assert not np.array_equal(a.W, c.W)


def test_rollout_is_seeded_deterministic() -> None:
    env = ToyEnv()
    policy = LinearSoftmaxPolicy(d=2, n_actions=3, seed=0)  # uniform (zero init)
    t1 = rollout(env, policy, 0, rng=np.random.default_rng(123))
    t2 = rollout(env, policy, 0, rng=np.random.default_rng(123))
    assert np.array_equal(t1.actions, t2.actions)
    assert np.array_equal(t1.states, t2.states)
    assert np.array_equal(t1.rewards, t2.rewards)


# ---- REINFORCE convergence -----------------------------------------------------------------------
def test_reinforce_converges_on_toy() -> None:
    env = ToyEnv()
    policy = LinearSoftmaxPolicy(d=2, n_actions=3, seed=0)  # start uniform
    baseline = LinearValueBaseline(2)
    rng = np.random.default_rng(0)
    stats = None
    for _ in range(300):
        trajs = [rollout(env, policy, 0, rng=rng) for _ in range(64)]
        stats = reinforce_update(policy, trajs, gamma=1.0, lr=0.5, baseline=baseline)
    assert policy.greedy_action(np.array([1.0, 0.0])) == 1  # step 0 → arm A
    assert policy.greedy_action(np.array([0.0, 1.0])) == 2  # step 1 → arm B
    assert stats is not None and stats.mean_return > 1.5  # near the optimum (2.0)


def test_discounted_returns_to_go() -> None:
    g = discounted_returns_to_go(np.array([1.0, 2.0, 3.0]), gamma=0.5)
    # G_2=3; G_1=2+0.5·3=3.5; G_0=1+0.5·3.5=2.75
    assert np.allclose(g, [2.75, 3.5, 3.0])


# ---- behavior cloning ----------------------------------------------------------------------------
def test_behavior_clone_reduces_cross_entropy() -> None:
    env = ToyEnv()
    expert = replay(env, 0, [1, 2])  # the optimal demonstration: A@step0, B@step1
    policy = LinearSoftmaxPolicy(d=2, n_actions=3, seed=5, init_scale=0.1)
    history = behavior_clone(policy, [expert], lr=0.3, epochs=200)
    assert history[-1] < history[0]  # cross-entropy fell
    pairs = zip(history, history[1:], strict=False)  # offset pairs ⇒ intentionally unequal length
    assert all(later <= earlier + 1e-9 for earlier, later in pairs)  # monotone non-increasing
    assert policy.greedy_action(np.array([1.0, 0.0])) == 1
    assert policy.greedy_action(np.array([0.0, 1.0])) == 2


# ---- linear value baseline -----------------------------------------------------------------------
def test_baseline_recovers_linear_function() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))
    true = np.array([1.0, -2.0, 0.5])
    y = x @ true + 3.0
    baseline = LinearValueBaseline(3, ridge=1e-8).fit(x, y)
    assert np.allclose(baseline.predict(x), y, atol=1e-4)


# ---- scripted A4/A5 expert + replay against the real env -----------------------------------------
S = Settings(_env_file=None)
ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"]}
UNIVERSE = {"NVDA", "MU"}


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


class _FakeSystem:
    name = "hybrid"

    def __init__(self, by_ticker: dict[str | None, list[RetrievedChunk]]) -> None:
        self._by_ticker = by_ticker

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        ticker = where.ticker if where is not None else None
        return EvidenceSet(query=query, chunks=list(self._by_ticker.get(ticker, []))[:top_k])


_NVDA = _rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", ticker="NVDA")
_MU = _rc("MU:0", "Micron cites critical information infrastructure review rules.", ticker="MU")
_EP = MultiHopQuery(
    question="Which memory supplier NVIDIA depends on discloses Chinese cybersecurity rules?",
    aspects=[
        Aspect(name="A1", spans=["Micron"]),
        Aspect(name="A2", spans=["critical information infrastructure"]),
    ],
    seed="NVDA",
    stratum="HARD",
    relation="depends_on",
    group_id="NVDA:MU",
)


def _bridge_env() -> RagRetrievalEnv:
    factory: RetrieverFactory = lambda arm: _FakeSystem({"NVDA": [_NVDA], "MU": [_MU]})  # noqa: E731
    return RagRetrievalEnv(
        [_EP],
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=factory,
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.0,  # isolate coverage (the scripted expert's return == terminal coverage)
    )


def test_bridge_expert_rollout_scripts_a4_a5() -> None:
    traj = bridge_expert_rollout(_bridge_env(), 0, prod_arm="hybrid")
    # pruned indices: 1 = hybrid@self, 2 = hybrid@disc0, 0 = STOP.
    assert list(traj.actions) == [1, 2, 0]  # self-search → bridge MU → STOP
    assert traj.total_return == pytest.approx(1.0)  # both aspects covered, no cost


def test_replay_matches_action_sequence_and_stops_early() -> None:
    env = _bridge_env()
    full = replay(env, 0, [1, 2, 0])
    assert list(full.actions) == [1, 2, 0]
    assert full.total_return == pytest.approx(1.0)
    early = replay(env, 0, [0, 1, 2])  # STOP first ⇒ env terminates, rest ignored
    assert list(early.actions) == [0]
