"""A6.2e — torch PPO actor-critic: convergence, clip surrogate, GAE, masking, backend-agnostic.

Skipped entirely when the optional ``[rl]`` extra (torch) is absent, and **collect-ignored by
default** even when torch IS installed (conftest ``RUN_RL_TORCH_TESTS``) — torch and lightgbm's
bundled OpenMP runtimes abort if loaded in one interpreter, so this file runs in its own session::

    RUN_RL_TORCH_TESTS=1 pytest tests/unit/test_rl_ppo.py

The final ``test_torch_lightgbm_openmp_smoke`` verifies the ``KMP_DUPLICATE_LIB_OK`` isolation
directly, in a fresh subprocess so it can never take this session down.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from stock_agent.rag.rl.ppo import (  # noqa: E402
    PPOPolicy,
    clipped_surrogate,
    compute_gae,
    train_ppo,
)
from stock_agent.rag.rl.reinforce import greedy_rollout, rollout  # noqa: E402


# ---- a tiny toy MDP (mirrors test_rl_policy.ToyEnv; local so this torch file is self-contained) --
class ToyEnv:
    """2-step, 3-action MDP with a step-dependent optimum (STOP=0, arm A=1, arm B=2).

    step 0 rewards arm A, step 1 rewards arm B; STOP ends with 0. The optimum (A then B, return 2)
    needs the state (one-hot step), so it exercises the shared torso, not just the action bias.
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
            v[self._t] = 1.0
        return v

    def legal_mask(self) -> np.ndarray:
        return np.ones(self.n_actions, dtype=bool)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, object]:
        if self._done:
            raise RuntimeError("step on a done ToyEnv")
        if action_idx == 0:
            self._done = True
            return self._state(), 0.0, True, None
        good = (self._t == 0 and action_idx == 1) or (self._t == 1 and action_idx == 2)
        self._t += 1
        self._done = self._t >= self.max_steps
        return self._state(), (1.0 if good else 0.0), self._done, None


# ---- clipped surrogate: the four (sign of Â) × (ratio in/out of band) quadrants ------------------
def test_clipped_surrogate_four_quadrants() -> None:
    eps = 0.2  # trust band [0.8, 1.2]
    ratio = torch.tensor([1.5, 0.5, 1.5, 0.5])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    out = clipped_surrogate(ratio, adv, eps).numpy()
    # Â>0, ρ=1.5 → clip to 1.2·1 = 1.2 (gain capped); Â>0, ρ=0.5 → min(0.5, 0.8) = 0.5 (unclipped);
    # Â<0, ρ=1.5 → min(-1.5,-1.2) = -1.5 (unclipped, penalized); Â<0, ρ=0.5 → min(-0.5,-0.8) = -0.8.
    assert np.allclose(out, [1.2, 0.5, -1.5, -0.8])


def test_clipped_surrogate_ratio_one_is_identity() -> None:
    # ρ=1 is inside every band ⇒ surrogate == Â (no clipping at the start of a PPO epoch).
    adv = torch.tensor([0.3, -0.7, 2.0])
    out = clipped_surrogate(torch.ones(3), adv, 0.2).numpy()
    assert np.allclose(out, adv.numpy())


# ---- GAE golden (hand-computed) ------------------------------------------------------------------
def test_compute_gae_lambda_one_is_monte_carlo() -> None:
    # rewards [1,0], values [0.5,0.2], γ=1, λ=1, terminal bootstrap 0.
    # λ=1 ⇒ Â_t = (reward-to-go) − V(s_t): Â_0 = 1 − 0.5 = 0.5; Â_1 = 0 − 0.2 = -0.2.
    adv, ret = compute_gae(np.array([1.0, 0.0]), np.array([0.5, 0.2]), gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv, [0.5, -0.2])
    assert np.allclose(ret, [1.0, 0.0])  # value target = Â + V = reward-to-go


def test_compute_gae_lambda_zero_is_one_step_td() -> None:
    # λ=0 ⇒ Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t). δ_0 = 1 + 0.2 − 0.5 = 0.7; δ_1 = −0.2.
    adv, _ = compute_gae(np.array([1.0, 0.0]), np.array([0.5, 0.2]), gamma=1.0, gae_lambda=0.0)
    assert np.allclose(adv, [0.7, -0.2])


# ---- masking (backend-agnostic inference surface) ------------------------------------------------
def test_masking_zeroes_illegal_actions() -> None:
    policy = PPOPolicy(d=3, n_actions=4, hidden=8, seed=1)
    x = np.ones(3)
    mask = np.array([True, False, True, False])
    p = policy.action_probs(x, mask=mask)
    assert p[1] == 0.0 and p[3] == 0.0
    assert p.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    picks = {policy.act(x, rng=rng, mask=mask)[0] for _ in range(200)}
    assert picks <= {0, 2}  # never samples a masked action


def test_ppo_policy_is_seeded_deterministic() -> None:
    a = PPOPolicy(d=4, n_actions=3, hidden=8, seed=7)
    b = PPOPolicy(d=4, n_actions=3, hidden=8, seed=7)
    x = np.array([0.2, -0.5, 1.0, 0.1])
    assert np.allclose(a.action_probs(x), b.action_probs(x))  # identical init ⇒ identical policy
    assert a.value(x) == pytest.approx(b.value(x))


# ---- backend-agnostic: the numpy rollout helpers drive the torch policy unchanged ----------------
def test_rollout_helpers_accept_ppo_policy() -> None:
    env = ToyEnv()
    policy = PPOPolicy(d=2, n_actions=3, hidden=8, seed=0)
    traj = rollout(env, policy, 0, rng=np.random.default_rng(0))  # RolloutPolicy protocol
    assert traj.length >= 1 and traj.masks.shape[1] == 3
    greedy = greedy_rollout(env, policy, 0)
    assert greedy.length >= 1


# ---- convergence on the toy (the escalation rung solves what REINFORCE solves) -------------------
def test_ppo_converges_on_toy() -> None:
    env = ToyEnv()
    policy, history = train_ppo(
        env,
        [0],
        iterations=80,
        episodes_per_batch=32,
        hidden=32,
        seed=0,
        gamma=1.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr=0.02,
        epochs=4,
        entropy_coef=0.0,
    )
    assert policy.greedy_action(np.array([1.0, 0.0])) == 1  # step 0 → arm A
    assert policy.greedy_action(np.array([0.0, 1.0])) == 2  # step 1 → arm B
    assert history[-1].mean_return > 1.5  # near the optimum (2.0)


# ---- A6.2f: PPO checkpoint freeze/load round-trip (torch weights → JSON → identical inference) ---
def _fake_bridge_env() -> object:
    """A one-episode NVDA→MU RagRetrievalEnv over a ticker-scoped FakeSystem (no corpus/LLM)."""
    from datetime import date

    from stock_agent.rag.rl.action import named_action_space
    from stock_agent.rag.rl.env import RagRetrievalEnv
    from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
    from stock_agent.schemas.documents import DocumentChunk
    from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
    from stock_agent.settings import Settings

    def rc(cid: str, text: str, ticker: str) -> RetrievedChunk:
        chunk = DocumentChunk(
            chunk_id=cid, document_id=cid.rsplit(":", 1)[0], chunk_index=0, text=text,
            ticker=ticker, document_type="10-K", source="SEC", source_url="https://sec.gov/x",
            filing_date=date(2026, 2, 25), section="Item 1A. Risk Factors",
        )
        return RetrievedChunk(chunk=chunk, score=0.9)

    by_ticker: dict[str | None, list[RetrievedChunk]] = {
        "NVDA": [rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", "NVDA")],
        "MU": [rc("MU:0", "Micron cites critical information infrastructure rules.", "MU")],
    }

    class FakeSystem:
        name = "fake"  # satisfies the RetrievalSystem protocol's ``name`` member

        def retrieve(
            self, query: str, *, top_k: int, where: ChunkFilter | None = None
        ) -> EvidenceSet:
            t = where.ticker if where is not None else None
            return EvidenceSet(query=query, chunks=list(by_ticker.get(t, []))[:top_k])

        # graph arm reuses the same canned corpus (irrelevant to the serialization round-trip).

    ep = MultiHopQuery(
        question="Which memory supplier NVIDIA depends on discloses Chinese cybersecurity rules?",
        aspects=[
            Aspect(name="A1", spans=["Micron"]),
            Aspect(name="A2", spans=["critical information infrastructure"]),
        ],
        seed="NVDA", stratum="HARD", relation="depends_on", group_id="NVDA:MU",
    )
    return RagRetrievalEnv(
        [ep], settings=Settings(_env_file=None), action_space=named_action_space("pruned"),
        retriever_factory=lambda arm: FakeSystem(),
        alias_map={"NVDA": ["NVIDIA"], "MU": ["Micron"]},
        graph_universe={"NVDA", "MU"}, gamma=1.0, lambda_cost=0.05,
    )


def test_ppo_checkpoint_round_trip() -> None:
    from stock_agent.rag.rl.train import (
        StandardizingEnv,
        TrainConfig,
        load_checkpoint,
        train_policy,
    )

    env = _fake_bridge_env()
    cfg = TrainConfig(algo="ppo", iterations=5, episodes_per_batch=8, hidden=16, seed=0)
    trained = train_policy(env, [0], cfg)  # type: ignore[arg-type]
    assert trained.weights["kind"] == "ppo_mlp" and trained.weights["hidden"] == 16

    raw0 = env.reset(0)  # type: ignore[attr-defined]
    mask = env.legal_mask()  # type: ignore[attr-defined]
    loaded = load_checkpoint(trained.to_dict())
    # loader-standardized policy on the RAW state == trained raw policy on the standardized state.
    p_ref = trained.policy.action_probs(trained.standardizer.transform(raw0), mask=mask)
    p_load = loaded.policy.action_probs(raw0, mask=mask)
    assert np.allclose(p_load, p_ref, atol=1e-6)
    assert loaded.algo == "ppo"
    # greedy trajectory identical between (std-env + raw policy) and (raw env + loaded policy).
    from stock_agent.rag.rl.reinforce import greedy_rollout

    std_env = StandardizingEnv(env, trained.standardizer)  # type: ignore[arg-type]
    g_ref = greedy_rollout(std_env, trained.policy, 0)
    g_load = greedy_rollout(env, loaded.policy, 0)  # type: ignore[arg-type]
    assert list(g_ref.actions) == list(g_load.actions)


# ---- day-1 OpenMP isolation smoke test (subprocess: cannot crash this session) -------------------
def test_torch_lightgbm_openmp_smoke() -> None:
    """torch + lightgbm import together (KMP_DUPLICATE_LIB_OK=TRUE) without the macOS OpenMP abort.

    Runs in a fresh subprocess so a genuine segfault surfaces as a non-zero return code here rather
    than taking down the pytest process. Skipped if lightgbm is not installed.
    """
    pytest.importorskip("lightgbm")
    code = (
        "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE');"
        "import torch; import lightgbm;"
        "print('openmp-ok', torch.__version__, lightgbm.__version__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"torch+lightgbm co-import failed: {proc.stderr}"
    assert "openmp-ok" in proc.stdout
