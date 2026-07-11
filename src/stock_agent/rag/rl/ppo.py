"""Torch PPO actor-critic for retrieval RL — the escalation rung (advanced-RAG A6.2e).

The numpy REINFORCE of A6.2d is the CI floor; PPO is the sample-efficient, lower-variance learner
for the same deterministic ``$0`` MDP (A6.2c). It adds three things over REINFORCE:

- a **learned critic** ``V(s)`` and **GAE(λ)** advantages (bias/variance-tuned credit assignment),
- a **clipped surrogate** ``L^CLIP`` that lets each batch take several gradient epochs without the
  update walking off the trust region (the sample-reuse REINFORCE cannot do), and
- an **entropy bonus** to keep exploration alive early.

**torch isolation (plan §3c).** torch lives in the optional ``[rl]`` extra and is **lazy-imported**
inside this module only — importing ``stock_agent.rag.rl.ppo`` never pulls torch, so the CI gate
(torch-free) still imports the package. On macOS torch and lightgbm each bundle their own OpenMP
runtime and loading both in one process aborts; we set ``KMP_DUPLICATE_LIB_OK=TRUE`` *before* the
first ``import torch`` so they coexist (a subprocess smoke test guards this on this hardware).

**Backend-agnostic surface.** ``PPOPolicy`` exposes the *same* numpy inference methods as the A6.2d
``LinearSoftmaxPolicy`` (``act`` / ``greedy_action`` / ``action_probs`` / ``log_prob``), so the
rollout helpers (`reinforce.rollout`) and the A6.2g eval harness drive either backend unchanged.

**Clipped surrogate (Schulman et al. 2017).** With importance ratio ``ρ_t = π_θ(a_t|s_t) /
π_{θ_old}(a_t|s_t)`` and advantage ``Â_t``::

    L^CLIP(θ) = E_t[ min( ρ_t Â_t,  clip(ρ_t, 1−ε, 1+ε) Â_t ) ]

The ``min`` + ``clip`` removes the incentive to push ``ρ_t`` far from 1: when ``Â_t > 0`` the
gain is capped at ``1+ε``; when ``Â_t < 0`` it is capped at ``1−ε``. Maximizing ``L^CLIP``
(minimizing its negation) is the policy loss; a value loss and an entropy bonus complete it.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from stock_agent.rag.rl.reinforce import EpisodeMDP

# Must be set before torch (hence its bundled OpenMP) first loads, so torch and lightgbm's OpenMP
# runtimes coexist instead of aborting on macOS. Setting it at module import is early enough because
# torch is only imported lazily (below), never at module load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _torch() -> Any:
    """Lazy ``import torch`` (keeps ``[rl]`` optional; the numpy CI path never imports it)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    return torch


def _seed_torch(seed: int) -> None:
    """Seed every RNG torch touches + force deterministic single-threaded CPU (bit-stable runs)."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _build_actor_critic(d: int, n_actions: int, hidden: int, layers: int) -> Any:
    """A shared-torso MLP: ``layers`` Tanh blocks, then a logit head + a scalar value head."""
    from torch import nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torso: list[Any] = []
            in_dim = d
            for _ in range(layers):
                torso += [nn.Linear(in_dim, hidden), nn.Tanh()]
                in_dim = hidden
            self.torso = nn.Sequential(*torso)
            self.actor = nn.Linear(hidden, n_actions)  # action logits
            self.critic = nn.Linear(hidden, 1)  # state value V(s)

        def forward(self, x: Any) -> Any:
            h = self.torso(x)
            return self.actor(h), self.critic(h).squeeze(-1)

    return ActorCritic()


# ---- masked-softmax numerics (numpy inference, mirrors LinearSoftmaxPolicy exactly) --------------
def _masked_softmax(logits: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Stable softmax over legal actions (illegal → 0); STOP is always legal so a max exists."""
    z = logits if mask is None else np.where(np.asarray(mask, dtype=bool), logits, -np.inf)
    finite = np.isfinite(z)
    m = float(np.max(z[finite]))
    ez = np.exp(z - m)
    ez[~finite] = 0.0
    probs: np.ndarray = ez / float(ez.sum())
    return probs


def _masked_log_prob(logits: np.ndarray, a: int, mask: np.ndarray | None) -> float:
    """``log π(a|s)`` via a stable log-sum-exp over the legal logits (matches _masked_softmax)."""
    z = logits if mask is None else np.where(np.asarray(mask, dtype=bool), logits, -np.inf)
    finite = np.isfinite(z)
    m = float(np.max(z[finite]))
    lse = m + float(np.log(np.sum(np.exp(z[finite] - m))))
    return float(z[a] - lse)


class PPOPolicy:
    """Torch MLP actor-critic with the A6.2d numpy inference surface (backend-agnostic).

    Inference (``act`` / ``greedy_action`` / ``action_probs`` / ``log_prob``) runs the net under
    ``no_grad`` and reproduces the numpy masked-softmax numerics of ``LinearSoftmaxPolicy`` — so a
    rollout / eval harness cannot tell the two policies apart. ``value`` + ``_forward_batch`` expose
    the critic and a grad-enabled forward for the PPO update. Seeded at construction (deterministic
    weights); like the numpy policy it holds no RNG (``act`` takes an explicit ``Generator``).
    """

    def __init__(
        self, d: int, n_actions: int, *, hidden: int = 64, layers: int = 2, seed: int = 0
    ) -> None:
        if d <= 0 or n_actions <= 0:
            raise ValueError("d and n_actions must be positive")
        _seed_torch(seed)
        self.d = int(d)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)  # kept so a checkpoint can rebuild the exact architecture (A6.2f)
        self.layers = int(layers)
        self.net: Any = _build_actor_critic(self.d, self.n_actions, hidden, layers)

    def parameters(self) -> Any:
        """The trainable torch parameters (fed to the optimizer)."""
        return self.net.parameters()

    def _forward_batch(self, states: Any) -> Any:
        """Grad-enabled forward on a ``(B, d)`` float tensor → ``(logits[B, K], values[B])``."""
        return self.net(states)

    def _logits(self, x: np.ndarray) -> np.ndarray:
        """Raw (unmasked) action logits for one state, computed under ``no_grad`` → numpy."""
        torch = _torch()
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
            logits_t, _ = self.net(xt)
        return np.asarray(logits_t.squeeze(0).numpy(), dtype=np.float64)

    def action_probs(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
        """Masked softmax probabilities over actions (0 on illegal actions; sums to 1)."""
        return _masked_softmax(self._logits(x), mask)

    def log_prob(self, a: int, x: np.ndarray, *, mask: np.ndarray | None = None) -> float:
        """``log π(a|s)`` under the masked softmax."""
        return _masked_log_prob(self._logits(x), a, mask)

    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        """Sample ``a ~ π(·|s)``; return ``(a, log π(a|s))`` (masked-safe; caller supplies RNG)."""
        logits = self._logits(x)
        p = _masked_softmax(logits, mask)
        a = int(rng.choice(self.n_actions, p=p))
        return a, _masked_log_prob(logits, a, mask)

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        """Deterministic ``argmax_a π(a|s)`` over legal actions (evaluation)."""
        z = self._logits(x)
        if mask is not None:
            z = np.where(np.asarray(mask, dtype=bool), z, -np.inf)
        return int(np.argmax(z))

    def value(self, x: np.ndarray) -> float:
        """Critic estimate ``V(s)`` for one state (no_grad)."""
        torch = _torch()
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
            _, v = self.net(xt)
        return float(v.squeeze(0).item())

    # ---- checkpoint weight I/O (JSON-friendly; the A6.2f freeze path) --------------------------
    def state_arrays(self) -> dict[str, Any]:
        """Serialize the net weights to JSON-friendly nested lists (checkpoint freeze, A6.2f).

        Keyed by the torch ``state_dict`` parameter names (e.g. ``"torso.0.weight"``); reload with
        ``load_arrays`` into a policy rebuilt with the *same* ``(d, n_actions, hidden, layers)`` so
        the shapes and key set line up exactly.
        """
        return {k: v.detach().cpu().numpy().tolist() for k, v in self.net.state_dict().items()}

    def load_arrays(self, arrays: Mapping[str, Any]) -> None:
        """Load net weights previously produced by ``state_arrays`` (architecture must match)."""
        torch = _torch()
        state = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in arrays.items()}
        self.net.load_state_dict(state)


# ---- GAE + rollout collection -----------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over ONE terminated episode (Schulman et al. 2016).

    ``δ_t = r_t + γ V(s_{t+1}) − V(s_t)``; ``Â_t = δ_t + γλ Â_{t+1}`` (backward). The episode always
    terminates in this MDP (STOP or the horizon), so the bootstrap ``V(s_T) = last_value = 0``. The
    value target is ``Â_t + V(s_t)`` (the TD(λ) return). ``λ=1`` recovers the Monte-Carlo advantage
    ``G_t − V(s_t)``; ``λ=0`` the one-step TD advantage.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    t_max = len(r)
    adv = np.zeros(t_max, dtype=np.float64)
    gae = 0.0
    for t in range(t_max - 1, -1, -1):
        next_v = v[t + 1] if t + 1 < t_max else last_value
        delta = r[t] + gamma * next_v - v[t]
        gae = delta + gamma * gae_lambda * gae
        adv[t] = gae
    returns = adv + v
    return adv, returns


@dataclass(frozen=True)
class PPOBatch:
    """One flattened on-policy batch: per-transition arrays + per-episode returns (for logging)."""

    states: np.ndarray  # (M, d)
    actions: np.ndarray  # (M,) int64
    masks: np.ndarray  # (M, K) bool
    logp_old: np.ndarray  # (M,) log π_old(a|s) at collection time (the PPO ratio denominator)
    advantages: np.ndarray  # (M,) GAE(λ)
    returns: np.ndarray  # (M,) value targets (Â + V)
    episode_returns: list[float]  # undiscounted episode returns (batch mean = mean_return diag)


def collect_ppo_batch(
    env: EpisodeMDP,
    policy: PPOPolicy,
    episode_indices: Sequence[int],
    *,
    gamma: float,
    gae_lambda: float,
    rng: np.random.Generator,
) -> PPOBatch:
    """Roll out each episode under the current ``π`` and assemble a GAE-annotated PPO batch.

    Records, at each visited state, the value ``V(s)`` and the acting log-prob ``log π_old(a|s)`` —
    both needed by the update (GAE and the importance ratio) and impossible to recover once the
    policy has moved. Advantages/returns are computed per-episode (each terminates), then pooled.
    """
    states: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    logp_old: list[float] = []
    advantages: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    episode_returns: list[float] = []

    for episode_idx in episode_indices:
        state = env.reset(int(episode_idx))
        ep_rewards: list[float] = []
        ep_values: list[float] = []
        done = False
        while not done:
            mask = env.legal_mask()
            action, logp = policy.act(state, rng=rng, mask=mask)
            states.append(np.asarray(state, dtype=np.float64))
            actions.append(action)
            masks.append(mask)
            logp_old.append(logp)
            ep_values.append(policy.value(state))
            state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
        adv, ret = compute_gae(
            np.asarray(ep_rewards), np.asarray(ep_values), gamma=gamma, gae_lambda=gae_lambda
        )
        advantages.append(adv)
        returns.append(ret)
        episode_returns.append(float(np.sum(ep_rewards)))

    n = len(actions)
    return PPOBatch(
        states=np.asarray(states, dtype=np.float64).reshape(n, -1),
        actions=np.asarray(actions, dtype=np.int64),
        masks=np.asarray(masks, dtype=bool).reshape(n, -1),
        logp_old=np.asarray(logp_old, dtype=np.float64),
        advantages=np.concatenate(advantages) if advantages else np.zeros(0),
        returns=np.concatenate(returns) if returns else np.zeros(0),
        episode_returns=episode_returns,
    )


# ---- clipped surrogate + PPO update ------------------------------------------------------------
def clipped_surrogate(ratio: Any, advantage: Any, clip_eps: float) -> Any:
    """Per-sample ``min(ρ Â, clip(ρ, 1−ε, 1+ε) Â)`` (torch tensors in, tensor out).

    The core of ``L^CLIP``: the ``clip`` bounds the ratio to ``[1−ε, 1+ε]`` and the ``min`` makes
    the surrogate a *pessimistic* (lower) bound of the unclipped objective, so an update is never
    rewarded for leaving the trust region (positive Â capped at ``1+ε``, negative Â capped at
    ``1−ε``). The policy loss is ``−mean`` of this.
    """
    torch = _torch()
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return torch.minimum(surr1, surr2)


@dataclass(frozen=True)
class PPOUpdateStats:
    """Diagnostics from one PPO iteration (logging only; never consumed by the math)."""

    mean_return: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    n_steps: int


def ppo_update(
    policy: PPOPolicy,
    batch: PPOBatch,
    optimizer: Any,
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    rng: np.random.Generator,
) -> PPOUpdateStats:
    """Several minibatch epochs of ``L^CLIP`` + value regression − entropy bonus on one batch.

    Advantages are batch-normalized (zero mean, unit std) for scale stability. Each epoch reshuffles
    the transitions (via the supplied ``rng``, for reproducibility) and steps Adam on
    ``−L^CLIP + c_v · MSE(V, returns) − c_e · H[π]``. The persistent ``optimizer`` carries Adam's
    moments across iterations (owned by ``train_ppo``).
    """
    torch = _torch()
    if batch.states.shape[0] == 0:
        raise ValueError("ppo_update needs a non-empty batch")
    states = torch.as_tensor(batch.states, dtype=torch.float32)
    actions = torch.as_tensor(batch.actions, dtype=torch.long)
    masks = torch.as_tensor(batch.masks, dtype=torch.bool)
    logp_old = torch.as_tensor(batch.logp_old, dtype=torch.float32)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32)
    adv_np = batch.advantages
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)  # normalize for a stable step size
    advantages = torch.as_tensor(adv_np, dtype=torch.float32)

    m = batch.states.shape[0]
    neg_inf = torch.tensor(float("-inf"))
    pol_loss = val_loss = ent = kl = 0.0
    for _ in range(epochs):
        perm = rng.permutation(m)
        for start in range(0, m, minibatch_size):
            idx = torch.as_tensor(perm[start : start + minibatch_size], dtype=torch.long)
            logits, values = policy._forward_batch(states[idx])
            mb_mask = masks[idx]
            masked_logits = torch.where(mb_mask, logits, neg_inf)
            logp_all = torch.log_softmax(masked_logits, dim=-1)
            logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - logp_old[idx])
            policy_loss = -clipped_surrogate(ratio, advantages[idx], clip_eps).mean()
            value_loss = torch.mean((values - returns[idx]) ** 2)
            probs = torch.softmax(masked_logits, dim=-1)
            # Zero −inf log-probs BEFORE multiplying so masked entries add 0 (avoids 0·−inf = nan).
            safe_logp = torch.where(mb_mask, logp_all, torch.zeros_like(logp_all))
            entropy = -(probs * safe_logp).sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pol_loss, val_loss, ent = (
                float(policy_loss.item()),
                float(value_loss.item()),
                float(entropy.item()),
            )
            kl = float((logp_old[idx] - logp).mean().item())  # approx KL(π_old‖π_new) diagnostic

    return PPOUpdateStats(
        mean_return=float(np.mean(batch.episode_returns)) if batch.episode_returns else 0.0,
        policy_loss=pol_loss,
        value_loss=val_loss,
        entropy=ent,
        approx_kl=kl,
        n_steps=m,
    )


def train_ppo(
    env: EpisodeMDP,
    episode_indices: Sequence[int],
    *,
    iterations: int,
    episodes_per_batch: int = 32,
    hidden: int = 64,
    layers: int = 2,
    seed: int = 0,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-3,
    epochs: int = 4,
    minibatch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    policy: PPOPolicy | None = None,
) -> tuple[PPOPolicy, list[PPOUpdateStats]]:
    """Train a ``PPOPolicy`` on the retrieval MDP: collect → GAE → clipped-surrogate update, repeat.

    ``episode_indices`` are the training episodes (e.g. the group-split train fold); each iteration
    samples ``episodes_per_batch`` of them (with replacement, via the seeded ``rng``) for one
    on-policy batch. A single Adam optimizer spans all iterations, deterministic given ``seed``.
    Returns the trained policy and the per-iteration diagnostics.
    """
    torch = _torch()
    idxs = [int(i) for i in episode_indices]
    if not idxs:
        raise ValueError("train_ppo needs at least one episode index")
    if policy is None:
        d = int(np.asarray(env.reset(idxs[0])).shape[0])
        policy = PPOPolicy(d, env.n_actions, hidden=hidden, layers=layers, seed=seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history: list[PPOUpdateStats] = []
    for _ in range(iterations):
        chosen = [idxs[i] for i in rng.integers(0, len(idxs), size=episodes_per_batch)]
        batch = collect_ppo_batch(
            env, policy, chosen, gamma=gamma, gae_lambda=gae_lambda, rng=rng
        )
        stats = ppo_update(
            policy,
            batch,
            optimizer,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            rng=rng,
        )
        history.append(stats)
    return policy, history
