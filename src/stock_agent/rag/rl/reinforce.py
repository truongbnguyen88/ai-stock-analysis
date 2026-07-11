"""Rollouts, REINFORCE, and behavior cloning for the numpy policy (advanced-RAG A6.2d).

The CI-floor training loop (no torch): collect on-policy rollouts through the deterministic A6.2c
env, then improve the A6.2d ``LinearSoftmaxPolicy`` by either (i) **REINFORCE** with reward-to-go
and an optional linear value baseline, or (ii) **behavior cloning** on scripted expert demos —
the A4 self-search + A5 entity-bridge heuristic replayed into the action space — as a warm start.

**REINFORCE (policy-gradient theorem, reward-to-go form).** For a stochastic policy ``π_θ`` and
trajectory ``τ = (s_0,a_0,r_0,…)``, the gradient of the expected return
``J(θ)=E[Σ_t γ^t r_t]`` is estimated by::

    ∇_θ J ≈ (1/N) Σ_τ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)),   G_t = Σ_{k≥t} γ^{k−t} r_k

``G_t`` is the discounted **reward-to-go** (only rewards after step ``t`` are causally affected by
``a_t`` — the lower-variance form of the estimator); ``b(s_t)`` is a state-only baseline (does not
bias the estimator, only cuts variance). One gradient-ascent step: ``θ ← θ + lr·∇_θ J``.

**Behavior cloning.** Supervised max-likelihood of the expert action under the policy — minimize
cross-entropy ``CE = −(1/M) Σ log π_θ(a*|s)``. Since ``∂CE/∂θ = −∇ log π``, a BC step is the *same*
score-function update with the "advantage" pinned to ``+1`` for the expert action (ascent on
log-likelihood). It is only a warm start: the A4/A5 expert never varies the config, so it is a weak
teacher (plan §8.4), not a substitute for the learned policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stock_agent.rag.rl.action import Action, ScopeKind
from stock_agent.rag.rl.policy import LinearSoftmaxPolicy, LinearValueBaseline

# STOP is index 0 in every A6.2b action space (asserted at env construction).
STOP_INDEX = 0


class EpisodeMDP(Protocol):
    """Structural interface the rollout helpers need — satisfied by ``RagRetrievalEnv`` (and fakes).

    Deliberately narrow (no ``action_space`` requirement) so a tiny toy env in tests conforms; the
    scripted-expert helper below additionally needs ``action_space`` and takes the concrete env.
    """

    @property
    def n_actions(self) -> int: ...
    def reset(self, episode_idx: int) -> np.ndarray: ...
    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, object]: ...
    def legal_mask(self) -> np.ndarray: ...


class BridgeMDP(EpisodeMDP, Protocol):
    """``EpisodeMDP`` that also exposes its ordered ``Action`` space (for the scripted expert)."""

    @property
    def action_space(self) -> Sequence[Action]: ...


@dataclass(frozen=True)
class Trajectory:
    """One rolled-out episode: the state at each decision, the action taken, reward, and legal mask.

    ``states`` are the states at which each action was chosen (pre-transition), so ``states[t]`` is
    aligned with ``actions[t]``/``rewards[t]``/``masks[t]``. ``masks[t]`` is the legal mask *at that
    state* — needed so REINFORCE/BC recompute ``π`` under exactly the mask used when acting.
    """

    states: np.ndarray  # (T, d)
    actions: np.ndarray  # (T,) int64
    rewards: np.ndarray  # (T,) float64
    masks: np.ndarray  # (T, n_actions) bool

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def total_return(self) -> float:
        """Undiscounted sum of rewards (episode return)."""
        return float(self.rewards.sum())


def _build_trajectory(
    states: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    masks: list[np.ndarray],
) -> Trajectory:
    n = len(actions)
    return Trajectory(
        states=np.asarray(states, dtype=np.float64).reshape(n, -1),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        masks=np.asarray(masks, dtype=bool).reshape(n, -1),
    )


# ---- rollout collection -----------------------------------------------------------
def rollout(
    env: EpisodeMDP, policy: LinearSoftmaxPolicy, episode_idx: int, *, rng: np.random.Generator
) -> Trajectory:
    """Sample one on-policy episode: act by ``π`` under the env's legal mask until ``done``."""
    state = env.reset(episode_idx)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    masks: list[np.ndarray] = []
    done = False
    while not done:
        mask = env.legal_mask()
        action, _ = policy.act(state, rng=rng, mask=mask)
        states.append(np.asarray(state, dtype=np.float64))
        actions.append(action)
        masks.append(mask)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return _build_trajectory(states, actions, rewards, masks)


def greedy_rollout(
    env: EpisodeMDP, policy: LinearSoftmaxPolicy, episode_idx: int
) -> Trajectory:
    """Deterministic (argmax) rollout — the policy evaluated without exploration noise."""
    state = env.reset(episode_idx)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    masks: list[np.ndarray] = []
    done = False
    while not done:
        mask = env.legal_mask()
        action = policy.greedy_action(state, mask=mask)
        states.append(np.asarray(state, dtype=np.float64))
        actions.append(action)
        masks.append(mask)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return _build_trajectory(states, actions, rewards, masks)


def replay(env: EpisodeMDP, episode_idx: int, action_indices: Sequence[int]) -> Trajectory:
    """Replay a fixed action-index sequence through the env; stop early if the env terminates.

    Used to turn any scripted policy into a ``Trajectory`` (e.g. building expert demonstrations for
    behavior cloning). Records the legal mask at each visited state.
    """
    state = env.reset(episode_idx)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    masks: list[np.ndarray] = []
    done = False
    for action_idx in action_indices:
        if done:
            break
        mask = env.legal_mask()
        states.append(np.asarray(state, dtype=np.float64))
        actions.append(int(action_idx))
        masks.append(mask)
        state, reward, done, _ = env.step(int(action_idx))
        rewards.append(reward)
    return _build_trajectory(states, actions, rewards, masks)


# ---- REINFORCE + baseline ---------------------------------------------------------
def discounted_returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Reward-to-go ``G_t = Σ_{k≥t} γ^{k−t} r_k`` (backward accumulation), aligned to rewards."""
    g = np.zeros_like(rewards, dtype=np.float64)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        g[t] = acc
    return g


@dataclass(frozen=True)
class UpdateStats:
    """Diagnostics from one REINFORCE step (logging only; not consumed by the math)."""

    mean_return: float
    grad_norm: float
    n_trajectories: int
    n_steps: int


def reinforce_update(
    policy: LinearSoftmaxPolicy,
    trajectories: Sequence[Trajectory],
    *,
    gamma: float,
    lr: float,
    baseline: LinearValueBaseline | None = None,
) -> UpdateStats:
    """One gradient-ascent REINFORCE step on ``policy`` from a batch of trajectories.

    Estimator: ``(1/N) Σ_τ Σ_t (G_t − b(s_t)) ∇log π(a_t|s_t)`` averaged over the ``N`` episodes.
    When a ``baseline`` is given, advantages use its *current* fit (so this batch's gradient stays
    unbiased) and it is refit on the batch returns *after* — ready for the next update.
    """
    if not trajectories:
        raise ValueError("reinforce_update needs at least one trajectory")
    states = np.vstack([t.states for t in trajectories])
    actions = np.concatenate([t.actions for t in trajectories])
    masks = np.vstack([t.masks for t in trajectories])
    returns = np.concatenate([discounted_returns_to_go(t.rewards, gamma) for t in trajectories])

    if baseline is not None:
        advantages = returns - baseline.predict(states)
        baseline.fit(states, returns)  # refit AFTER advantage ⇒ this batch's gradient stays fair
    else:
        advantages = returns

    grad = np.zeros_like(policy.W)
    for i in range(states.shape[0]):
        grad += advantages[i] * policy.grad_log_prob(int(actions[i]), states[i], mask=masks[i])
    grad /= len(trajectories)  # Monte-Carlo mean over episodes (per-episode gradient)
    policy.W += lr * grad

    return UpdateStats(
        mean_return=float(np.mean([t.total_return for t in trajectories])),
        grad_norm=float(np.linalg.norm(grad)),
        n_trajectories=len(trajectories),
        n_steps=int(states.shape[0]),
    )


# ---- behavior cloning -------------------------------------------------------------
def behavior_clone(
    policy: LinearSoftmaxPolicy,
    demos: Sequence[Trajectory],
    *,
    lr: float,
    epochs: int,
) -> list[float]:
    """Full-batch behavior cloning: minimize cross-entropy of the expert action under ``π``.

    Each epoch takes one gradient-ascent step on mean log-likelihood (= descent on cross-entropy),
    reusing the exact per-state legal masks. Returns the per-epoch cross-entropy history (monotone
    non-increasing for a small enough ``lr``) so a test can assert BC actually reduces the loss.
    """
    if not demos:
        raise ValueError("behavior_clone needs at least one demonstration")
    states = np.vstack([d.states for d in demos])
    actions = np.concatenate([d.actions for d in demos])
    masks = np.vstack([d.masks for d in demos])
    n = states.shape[0]

    history: list[float] = []
    for _ in range(epochs):
        grad = np.zeros_like(policy.W)
        cross_entropy = 0.0
        for i in range(n):
            action = int(actions[i])
            cross_entropy += -policy.log_prob(action, states[i], mask=masks[i])
            grad += policy.grad_log_prob(action, states[i], mask=masks[i])
        history.append(cross_entropy / n)
        policy.W += lr * (grad / n)  # ascent on log-likelihood ⇒ descent on cross-entropy
    return history


# ---- scripted A4/A5 expert (for the BC warm start) --------------------------------
def _action_index(
    action_space: Sequence[Action],
    config: str,
    scope_kind: ScopeKind,
    scope_slot: int | None = None,
) -> int | None:
    """Index of the ``(config, scope)`` action in the space, or ``None`` if it is not present."""
    target = Action(config=config, scope_kind=scope_kind, scope_slot=scope_slot)
    for i, action in enumerate(action_space):
        if action == target:
            return i
    return None


def bridge_expert_rollout(
    env: BridgeMDP, episode_idx: int, *, prod_arm: str = "hybrid"
) -> Trajectory:
    """The A4/A5 heuristic replayed as an action sequence: self-search, then bridge, then STOP.

    Mirrors production control flow — one self-scoped search with the production arm, then pull the
    first discovered-but-unretrieved entity (``disc0``) as long as one exists and budget remains
    (each bridge consumes that entity, so ``disc0`` re-points to the next discovery), then STOP.
    This is the (weak) demonstrator behavior cloning warm-starts from. Raises if ``prod_arm`` has no
    self-scope action in the space.
    """
    self_idx = _action_index(env.action_space, prod_arm, ScopeKind.SELF)
    if self_idx is None:
        raise ValueError(f"prod arm {prod_arm!r} has no self-scope action in the action space")
    disc0_idx = _action_index(env.action_space, prod_arm, ScopeKind.DISCOVERED, 0)

    state = env.reset(episode_idx)
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    masks: list[np.ndarray] = []
    took_self = False
    done = False
    while not done:
        mask = env.legal_mask()
        if not took_self:
            action = self_idx  # step 1: self-scoped search (A4)
            took_self = True
        elif disc0_idx is not None and bool(mask[disc0_idx]):
            action = disc0_idx  # A5 bridge: pull the first discovered-but-unretrieved entity
        else:
            action = STOP_INDEX  # nothing left to bridge ⇒ reflective STOP
        states.append(np.asarray(state, dtype=np.float64))
        actions.append(action)
        masks.append(mask)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return _build_trajectory(states, actions, rewards, masks)
