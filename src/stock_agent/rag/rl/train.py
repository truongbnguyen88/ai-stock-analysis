"""Training driver + checkpointing for retrieval RL (advanced-RAG A6.2f).

The orchestration layer over the A6.2d/e learners: fit a feature standardizer, run one of
``{bc, reinforce, ppo}`` on the group-split train fold, and **freeze** the result into a single
``policy.json`` a loader can reconstruct — so the held-out eval (A6.2g) and any real deploy run the
exact policy that was trained. Mirrors the ``backtesting/`` experiment discipline: every run logs
its full config + seed, and the frozen artifact pins the load-bearing orderings.

**Scale-agnostic policies ⇒ two standardizer placements (the one correctness subtlety).** The
policies (``LinearSoftmaxPolicy`` / ``PPOPolicy``) hold no internal standardizer (A6.2d note), so
feature standardization is the trainer's job. Placement differs by phase, deliberately:

- **Train-time — env-side** (``StandardizingEnv``): the wrapper standardizes every state the env
  emits, so the states the policy *samples on* are the states *recorded in the trajectory* and the
  states the gradient *recomputes on* — all standardized, all consistent. (Standardizing policy-side
  during training would record raw states but act on standardized ones, desyncing the gradient.)
- **Inference-time — policy-side** (``StandardizedPolicy``): the frozen ``(μ, σ)`` is applied by the
  loader *before* ``act``, so the policy runs on a **raw** env (the A6.2g harness, a real deploy)
  with no env wrapper needed. Same ``Standardizer`` object both sides ⇒ identical behavior, which
  the checkpoint round-trip test asserts end-to-end.

**Standardizer fit.** Seeded uniform-random rollouts over the train fold sample the reachable state
manifold broadly and reproducibly (a random policy's state coverage is a superset of any single
learned policy's). Constant features (σ ≈ 0, e.g. a static block field that never varies on the
fold) get σ := 1 — centered but unscaled — so the transform never divides by zero.

**torch isolation.** ``train.py`` imports torch **only** inside the ``ppo`` dispatch/load branches
(lazy), so importing this module — and training/loading numpy checkpoints — stays torch-free (the CI
floor). The ppo path requires the ``[rl]`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from stock_agent.rag.rl.action import (
    DEFAULT_DISCOVERED_SLOTS,
    Action,
    action_label,
    named_action_space,
)
from stock_agent.rag.rl.policy import LinearSoftmaxPolicy
from stock_agent.rag.rl.reinforce import BridgeMDP, rollout, train_bc, train_reinforce
from stock_agent.rag.rl.state import STATE_FEATURE_NAMES

# Bump when the on-disk schema changes incompatibly; the loader rejects mismatches (fail loud, not
# silently load stale weights against a new feature/action order).
CHECKPOINT_VERSION = 1
_ALGOS = ("bc", "reinforce", "ppo")
_SPACES = ("pruned", "full")


# ---- feature standardizer --------------------------------------------------------------------
@dataclass(frozen=True)
class Standardizer:
    """Per-feature affine transform ``(x − μ) / σ`` with constant features left unscaled (σ := 1).

    ``mean`` / ``std`` are length-``d`` arrays aligned to ``STATE_FEATURE_NAMES``. Frozen at train
    time and applied identically at inference (loader-side), so a stored policy's inputs match the
    distribution it was fit on. ``std`` never contains a 0 (constant features are set to 1 at fit).
    """

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize one state (or a batch): ``(x − μ) / σ`` (broadcasts over the last axis)."""
        out: np.ndarray = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        return out

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-friendly ``{"mean": [...], "std": [...]}`` for the checkpoint."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Standardizer:
        """Rebuild from ``to_dict`` output."""
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
        )


def fit_state_standardizer(
    env: BridgeMDP,
    episode_indices: Sequence[int],
    *,
    seed: int,
    n_rollouts: int,
    eps: float = 1e-8,
) -> Standardizer:
    """Fit ``(μ, σ)`` from seeded uniform-random rollouts over the train episodes (label-free).

    A zero-weight ``LinearSoftmaxPolicy`` is uniform over the legal actions, so it explores the
    reachable state manifold broadly; ``n_rollouts`` per episode (seeded) makes the sample
    reproducible. Features with σ < ``eps`` (constant on the fold) get σ := 1 so ``transform`` only
    centers them. Returns a ``Standardizer`` over the raw state dimension.
    """
    idxs = [int(i) for i in episode_indices]
    if not idxs:
        raise ValueError("fit_state_standardizer needs at least one episode index")
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be >= 1")
    d = int(np.asarray(env.reset(idxs[0])).shape[0])
    explorer = LinearSoftmaxPolicy(d, env.n_actions, seed=seed, init_scale=0.0)  # uniform / legal
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for episode_idx in idxs:
        for _ in range(n_rollouts):
            rows.extend(rollout(env, explorer, episode_idx, rng=rng).states)
    states = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # constant feature ⇒ center only, never divide by ~0
    return Standardizer(mean=mean, std=std)


# ---- train-time env wrapper / inference-time policy wrapper ----------------------------------
class StandardizingEnv:
    """Wrap a retrieval env so every state it emits is standardized (train-time placement).

    Delegates ``reset`` / ``step`` and standardizes only the returned state; reward, done, info,
    ``legal_mask``, ``n_actions`` and ``action_space`` pass through unchanged. Satisfies the
    ``BridgeMDP`` protocol, so the rollout/REINFORCE/BC/PPO drivers consume it exactly like the raw
    env — but now the recorded and acted-on states are the standardized ones (gradient-consistent).
    """

    def __init__(self, env: BridgeMDP, standardizer: Standardizer) -> None:
        self._env = env
        self._std = standardizer

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def action_space(self) -> Sequence[Action]:
        return self._env.action_space

    def reset(self, episode_idx: int) -> np.ndarray:
        return self._std.transform(self._env.reset(episode_idx))

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, object]:
        state, reward, done, info = self._env.step(action_idx)
        return self._std.transform(state), reward, done, info

    def legal_mask(self) -> np.ndarray:
        return self._env.legal_mask()


class StandardizedPolicy:
    """Wrap a trained policy so it standardizes raw states before acting (inference-time placement).

    Exposes the full ``RolloutPolicy`` surface (``act`` / ``greedy_action`` / ``action_probs`` /
    ``log_prob``), applying the frozen ``(μ, σ)`` to the incoming *raw* state and delegating to the
    underlying backend policy. This is what the loader returns, so the A6.2g eval harness (and any
    deploy) runs it directly on the raw ``RagRetrievalEnv`` with no env wrapper.
    """

    def __init__(self, policy: Any, standardizer: Standardizer) -> None:
        self.policy = policy
        self.standardizer = standardizer

    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        a, logp = self.policy.act(self.standardizer.transform(x), rng=rng, mask=mask)
        return int(a), float(logp)

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        return int(self.policy.greedy_action(self.standardizer.transform(x), mask=mask))

    def action_probs(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
        probs: np.ndarray = np.asarray(
            self.policy.action_probs(self.standardizer.transform(x), mask=mask), dtype=np.float64
        )
        return probs

    def log_prob(self, a: int, x: np.ndarray, *, mask: np.ndarray | None = None) -> float:
        return float(self.policy.log_prob(a, self.standardizer.transform(x), mask=mask))


# ---- train config + trained-policy bundle ----------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """Full, serializable training config (logged verbatim into the checkpoint — reproducibility).

    ``lr`` is REINFORCE's step size; ``bc_lr`` / ``ppo_lr`` are the BC / PPO learning rates (PPO's
    Adam wants a smaller step than plain REINFORCE ascent), so a single knob never has to serve two
    optimizers. ``iterations`` / ``episodes_per_batch`` drive reinforce+ppo; bc uses ``bc_epochs``.
    """

    algo: str = "reinforce"
    action_space: str = "pruned"
    n_discovered_slots: int = DEFAULT_DISCOVERED_SLOTS
    seed: int = 0
    gamma: float = 1.0
    lambda_cost: float | None = None
    n_fit_rollouts: int = 4
    # reinforce / ppo shared
    iterations: int = 200
    episodes_per_batch: int = 16
    lr: float = 0.1
    use_baseline: bool = True
    # behavior cloning
    bc_epochs: int = 50
    bc_lr: float = 0.5
    bc_prod_arm: str = "hybrid"
    # ppo
    ppo_lr: float = 3e-3
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    hidden: int = 64
    layers: int = 2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    # provenance (filled by the CLI / driver)
    queries_path: str | None = None
    test_frac: float = 0.0
    n_train: int | None = None

    def __post_init__(self) -> None:
        if self.algo not in _ALGOS:
            raise ValueError(f"algo must be one of {_ALGOS}; got {self.algo!r}")
        if self.action_space not in _SPACES:
            raise ValueError(f"action_space must be one of {_SPACES}; got {self.action_space!r}")


@dataclass
class TrainedPolicy:
    """A trained policy + everything needed to freeze and later reconstruct it (A6.2f artifact)."""

    policy: Any  # LinearSoftmaxPolicy | PPOPolicy (RolloutPolicy inference surface)
    standardizer: Standardizer
    config: TrainConfig
    d: int
    n_actions: int
    action_labels: tuple[str, ...]
    state_feature_names: tuple[str, ...]
    weights: dict[str, Any]
    history_metric: str  # "mean_return" (reinforce/ppo) | "cross_entropy" (bc)
    history: list[float]
    created_at: str

    def standardized(self) -> StandardizedPolicy:
        """The inference-ready policy (raw state in, standardized internally)."""
        return StandardizedPolicy(self.policy, self.standardizer)

    def to_dict(self) -> dict[str, Any]:
        """The full JSON checkpoint (weights + all load-bearing orderings + config + provenance)."""
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "created_at": self.created_at,
            "algo": self.config.algo,
            "action_space": self.config.action_space,
            "n_discovered_slots": self.config.n_discovered_slots,
            "d": self.d,
            "n_actions": self.n_actions,
            "seed": self.config.seed,
            # Load-bearing orderings — weights index into these; the loader rejects any drift.
            "state_feature_names": list(self.state_feature_names),
            "action_labels": list(self.action_labels),
            "standardizer": self.standardizer.to_dict(),
            "config": asdict(self.config),
            "history": {"metric": self.history_metric, "values": self.history},
            "weights": self.weights,
        }


def _linear_weights(policy: LinearSoftmaxPolicy) -> dict[str, Any]:
    """Freeze a numpy linear-softmax policy: the augmented weight matrix + bias flag."""
    return {"kind": "linear_softmax", "bias": bool(policy.bias), "W": policy.W.tolist()}


def _ppo_weights(policy: Any) -> dict[str, Any]:
    """Freeze a torch PPO policy: architecture + JSON-friendly state-dict arrays."""
    return {
        "kind": "ppo_mlp",
        "hidden": int(policy.hidden),
        "layers": int(policy.layers),
        "arrays": policy.state_arrays(),
    }


# ---- the training driver (dispatch on algo) --------------------------------------------------
def train_policy(
    env: BridgeMDP, train_indices: Sequence[int], config: TrainConfig
) -> TrainedPolicy:
    """Fit the standardizer, train ``config.algo`` on the standardized fold, bundle it to freeze.

    ``env`` is the **raw** retrieval MDP over the training episodes; ``train_indices`` selects the
    fold (the whole env, or a group-split subset). Returns a ``TrainedPolicy`` ready to serialize.
    The PPO branch lazy-imports torch (``[rl]`` extra) — the numpy branches never touch it.
    """
    idxs = [int(i) for i in train_indices]
    if not idxs:
        raise ValueError("train_policy needs at least one training episode index")

    standardizer = fit_state_standardizer(
        env, idxs, seed=config.seed, n_rollouts=config.n_fit_rollouts
    )
    std_env = StandardizingEnv(env, standardizer)
    d = int(standardizer.mean.shape[0])
    cfg = replace(config, n_train=len(idxs))

    policy: Any
    if cfg.algo == "reinforce":
        policy, hist = train_reinforce(
            std_env,
            idxs,
            iterations=cfg.iterations,
            episodes_per_batch=cfg.episodes_per_batch,
            lr=cfg.lr,
            gamma=cfg.gamma,
            seed=cfg.seed,
            use_baseline=cfg.use_baseline,
        )
        history_metric, history = "mean_return", [h.mean_return for h in hist]
        weights = _linear_weights(policy)
    elif cfg.algo == "bc":
        policy, ce = train_bc(
            std_env,
            idxs,
            epochs=cfg.bc_epochs,
            lr=cfg.bc_lr,
            seed=cfg.seed,
            prod_arm=cfg.bc_prod_arm,
        )
        history_metric, history = "cross_entropy", list(ce)
        weights = _linear_weights(policy)
    else:  # "ppo" — lazy torch import keeps this module CI-floor torch-free
        from stock_agent.rag.rl.ppo import train_ppo

        policy, phist = train_ppo(
            std_env,
            idxs,
            iterations=cfg.iterations,
            episodes_per_batch=cfg.episodes_per_batch,
            hidden=cfg.hidden,
            layers=cfg.layers,
            seed=cfg.seed,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_eps=cfg.clip_eps,
            lr=cfg.ppo_lr,
            epochs=cfg.ppo_epochs,
            minibatch_size=cfg.minibatch_size,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
        )
        history_metric, history = "mean_return", [h.mean_return for h in phist]
        weights = _ppo_weights(policy)

    return TrainedPolicy(
        policy=policy,
        standardizer=standardizer,
        config=cfg,
        d=d,
        n_actions=env.n_actions,
        action_labels=tuple(action_label(a) for a in env.action_space),
        state_feature_names=STATE_FEATURE_NAMES,
        weights=weights,
        history_metric=history_metric,
        history=history,
        created_at=datetime.now(UTC).isoformat(),
    )


# ---- checkpoint I/O --------------------------------------------------------------------------
def save_checkpoint(trained: TrainedPolicy, path: Path) -> Path:
    """Write the frozen ``policy.json`` (creating parent dirs); returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trained.to_dict(), indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class LoadedPolicy:
    """A reconstructed checkpoint: the inference-ready policy + the metadata A6.2g needs to eval."""

    policy: StandardizedPolicy
    algo: str
    action_space: str
    n_discovered_slots: int
    action_labels: tuple[str, ...]
    config: dict[str, Any]


def load_checkpoint(source: Path | str | Mapping[str, Any]) -> LoadedPolicy:
    """Reconstruct a checkpoint into a raw-state-ready ``StandardizedPolicy`` (+ eval metadata).

    Enforces the drift guards before touching weights: the on-disk ``checkpoint_version``,
    ``state_feature_names`` and ``action_labels`` must match the current code (a reorder would
    silently remap every weight). The PPO branch lazy-imports torch (``[rl]`` extra).
    """
    data: dict[str, Any] = (
        dict(source)
        if isinstance(source, Mapping)
        else json.loads(Path(source).read_text(encoding="utf-8"))
    )
    if data.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint version {data.get('checkpoint_version')} != {CHECKPOINT_VERSION}"
        )
    if tuple(data["state_feature_names"]) != STATE_FEATURE_NAMES:
        raise ValueError("checkpoint state_feature_names drifted from current STATE_FEATURE_NAMES")

    space = named_action_space(
        data["action_space"], n_discovered_slots=int(data["n_discovered_slots"])
    )
    labels = [action_label(a) for a in space]
    if labels != list(data["action_labels"]):
        raise ValueError("checkpoint action_labels drifted from the current action space")

    d, n = int(data["d"]), int(data["n_actions"])
    if n != len(space):
        raise ValueError(f"checkpoint n_actions {n} != rebuilt action space {len(space)}")

    standardizer = Standardizer.from_dict(data["standardizer"])
    weights = data["weights"]
    kind = weights.get("kind")
    if kind == "linear_softmax":
        policy: Any = LinearSoftmaxPolicy(d, n, bias=bool(weights["bias"]))
        w = np.asarray(weights["W"], dtype=np.float64)
        if w.shape != policy.W.shape:
            raise ValueError(f"weight shape {w.shape} != expected {policy.W.shape}")
        policy.W = w
    elif kind == "ppo_mlp":
        from stock_agent.rag.rl.ppo import PPOPolicy

        policy = PPOPolicy(
            d,
            n,
            hidden=int(weights["hidden"]),
            layers=int(weights["layers"]),
            seed=int(data["seed"]),
        )
        policy.load_arrays(weights["arrays"])
    else:
        raise ValueError(f"unknown weights kind {kind!r}")

    return LoadedPolicy(
        policy=StandardizedPolicy(policy, standardizer),
        algo=str(data["algo"]),
        action_space=str(data["action_space"]),
        n_discovered_slots=int(data["n_discovered_slots"]),
        action_labels=tuple(data["action_labels"]),
        config=dict(data.get("config", {})),
    )
