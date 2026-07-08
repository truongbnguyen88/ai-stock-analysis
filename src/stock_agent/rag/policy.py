"""Retrieval-selection policies (advanced-RAG A6.1e) — the contextual bandit ``pi(a|x)``.

A policy maps a featurized query (context ``x`` from A6.1b) to one of the retrieval **arms** and,
crucially for off-policy evaluation, exposes the full action distribution ``pi(a|x)``. All policies
here are **pure numpy** (leakage rule 5 — torch is A6.2-only) and, where they learn, are fit
**offline** on the group-wise train split (leakage rule 2); nothing calls the retriever or the LLM.

Arms are integer indices ``0..K-1`` in the fixed ``ARMS`` order below — the SAME ordering used for
``reward_matrix`` columns (A6.1c) and OPE ``target_probs`` columns (A6.1d), so a policy's action
index, the reward matrix, and the importance weights all line up without a name lookup. Policies are
name-agnostic in their math; ``FixedPolicy`` resolves an arm name against ``ARMS`` for ergonomics.

Five policies:

- **UniformPolicy** — the logging policy ``mu`` (``mu(a|x)=1/K``, full support ⇒ OPE is exact).
  This is what synthesizes the logged dataset in A6.1f.
- **FixedPolicy** — always the same arm (the baseline to beat = the promoted default; multi-hop
  baseline ``graph``, single-shot ``hybrid``). Deterministic ⇒ ``pi`` is a one-hot.
- **EpsilonGreedy** — with prob ``eps`` explore uniformly, else exploit ``argmax_a q_hat(x,a)``,
  the A6.1d ridge reward model (weights ``[K, d]``). ``pi(a|x) = eps/K + (1-eps)*1[a=greedy]``.
- **LinUCB** — disjoint linear UCB (Li et al. 2010). Per arm ``A_a = lambda I + sum x xT`` and
  ``b_a = sum r x`` give ``theta_a = A_a^{-1} b_a``; pick the arm maximizing
  ``theta_a . x + alpha*sqrt(x . A_a^{-1} x)`` (mean + optimism). Fit offline in one pass over the
  train log; deterministic argmax ⇒ ``pi`` is a one-hot.
- **GatedPolicy** — a two-branch router (the A6.1 verdict follow-up). A label-free gate sends easy
  queries to a cheap *easy* branch and the rest to a *hard* branch (a fixed arm or the bandit), so
  the bandit never touches the easy queries it regressed on. ``pi(a|x)`` delegates to the active
  branch, so it composes with any of the above.

``target_prob_matrix(policy, contexts)`` turns any policy into the ``[N, K]`` ``pi(a|x_i)`` matrix
the OPE layer consumes — the single bridge from this module to ``rag.ope``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

# Canonical A6.1 action space + ordering: the four lattice configs (read_path.LATTICE_SYSTEMS) plus
# the A5 graph arm. This tuple defines the integer arm index used EVERYWHERE in A6.1 — reward_matrix
# columns, OPE target_probs columns, and policy actions must all use this order. Kept as a literal
# (not imported from read_path) so this stays a pure-numpy module with no retrieval-backend imports;
# a unit test pins ARMS == (*LATTICE_SYSTEMS, "graph") to catch drift.
ARMS: tuple[str, ...] = ("dense", "reranked", "hybrid", "hybrid+rerank", "graph")
N_ARMS: int = len(ARMS)

# The fixed baselines A6.1 must beat (the promoted defaults for each benchmark kind). Multi-hop
# bridges favour the graph arm (A5.3); single-shot LabeledQuery favours tuned hybrid.
DEFAULT_MULTIHOP_ARM = "graph"
DEFAULT_SINGLESHOT_ARM = "hybrid"


@runtime_checkable
class Policy(Protocol):
    """A contextual retrieval policy: choose an arm and expose ``pi(a|x)`` for off-policy eval.

    ``act`` is the serving path (returns the chosen arm + propensity ``pi(action|x)``, logged for
    OPE); ``prob`` gives ``pi(a|x)`` for an arbitrary arm so a target policy can be turned into the
    OPE ``target_probs`` matrix. ``n_arms`` is the action-space size (== ``N_ARMS`` for real arms).
    """

    name: str
    n_arms: int

    def act(self, x: np.ndarray) -> tuple[int, float]:
        """Select an arm for context ``x``; return ``(action, propensity=pi(action|x))``."""
        ...

    def prob(self, action: int, x: np.ndarray) -> float:
        """``pi(action | x)`` — the probability this policy assigns to ``action`` at ``x``."""
        ...


def target_prob_matrix(policy: Policy, contexts: np.ndarray) -> np.ndarray:
    """``[N, K]`` matrix ``pi(a | x_i)`` for OPE — row ``i``, col ``a`` = ``policy.prob(a, x_i)``.

    The one bridge from a ``Policy`` to ``rag.ope``: the OPE estimators take these target
    probabilities as a plain array, keeping OPE decoupled from the policy classes.
    """
    n = len(contexts)
    matrix = np.zeros((n, policy.n_arms), dtype=np.float64)
    for i in range(n):
        xi = contexts[i]
        for a in range(policy.n_arms):
            matrix[i, a] = policy.prob(a, xi)
    return matrix


class UniformPolicy:
    """Logging policy ``mu``: pick an arm uniformly at random ⇒ ``mu(a|x)=1/K`` for every arm.

    Full support over all arms is what makes IPS/SNIPS/DR exact (no unobserved arm), so this is the
    dataset-synthesis policy in A6.1f. Seeded for reproducible logs; ``act`` advances the RNG.
    """

    def __init__(self, n_arms: int = N_ARMS, *, seed: int = 42) -> None:
        self.n_arms = n_arms
        self.name = "uniform"
        self._rng = np.random.default_rng(seed)

    def act(self, x: np.ndarray) -> tuple[int, float]:
        action = int(self._rng.integers(self.n_arms))
        return action, 1.0 / self.n_arms

    def prob(self, action: int, x: np.ndarray) -> float:
        return 1.0 / self.n_arms


class FixedPolicy:
    """Always choose one fixed arm — the baseline the learned policy must beat.

    Construct from an arm name (resolved against ``arm_names``) or an integer index. Deterministic,
    so ``pi(a|x)`` is a one-hot on the fixed arm and ``act`` reports propensity ``1.0``.
    """

    def __init__(self, arm: int | str, *, arm_names: Sequence[str] = ARMS) -> None:
        if isinstance(arm, str):
            action = list(arm_names).index(arm)  # raises ValueError on an unknown arm name
            label = arm
        else:
            action = int(arm)
            label = arm_names[action] if 0 <= action < len(arm_names) else str(action)
        self.action = action
        self.n_arms = len(arm_names)
        self.name = f"fixed({label})"

    def act(self, x: np.ndarray) -> tuple[int, float]:
        return self.action, 1.0

    def prob(self, action: int, x: np.ndarray) -> float:
        return 1.0 if action == self.action else 0.0


class EpsilonGreedy:
    """Epsilon-greedy over a ridge reward model ``q_hat`` (the A6.1d ``[K, d]`` weight matrix).

    Greedy arm ``= argmax_a x . theta_a`` (identical to ``ope.predict_q`` for one row); with prob
    ``epsilon`` explore uniformly instead. So ``pi(a|x) = epsilon/K + (1-epsilon)*1[a = greedy(x)]``
    (sums to 1). Seeded ⇒ ``act`` is reproducible. ``epsilon=1`` degenerates to ``UniformPolicy``.
    """

    def __init__(self, q_weights: np.ndarray, *, epsilon: float, seed: int = 42) -> None:
        self.q = np.asarray(q_weights, dtype=np.float64)  # [n_arms, d]
        self.n_arms = int(self.q.shape[0])
        self.epsilon = float(epsilon)
        self.name = f"epsilon_greedy(eps={self.epsilon:g})"
        self._rng = np.random.default_rng(seed)

    def _greedy(self, x: np.ndarray) -> int:
        # x . theta_a for every arm == ope.predict_q(x[None], q)[0]; argmax ties -> lowest index.
        scores = self.q @ x
        return int(np.argmax(scores))

    def act(self, x: np.ndarray) -> tuple[int, float]:
        if self._rng.random() < self.epsilon:
            action = int(self._rng.integers(self.n_arms))
        else:
            action = self._greedy(x)
        # Propensity is the MARGINAL pi(action|x) (not the branch prob) — what IPS needs.
        return action, self.prob(action, x)

    def prob(self, action: int, x: np.ndarray) -> float:
        base = self.epsilon / self.n_arms
        return base + (1.0 - self.epsilon) * (1.0 if action == self._greedy(x) else 0.0)


class LinUCB:
    """Disjoint linear UCB (Li, Chu, Langford & Schapire 2010) — optimism under uncertainty.

    Per arm ``a`` maintain ``A_a = lambda I + sum_t x_t x_t^T`` and ``b_a = sum_t r_t x_t`` over the
    rows where ``a`` was logged; the ridge estimate is ``theta_a = A_a^{-1} b_a``. For a query ``x``
    the score is ``theta_a . x + alpha * sqrt(x . A_a^{-1} x)`` = predicted reward + an exploration
    bonus that grows with posterior uncertainty in direction ``x`` (an unseen arm has ``theta=0``
    but a large bonus). ``act`` picks the argmax (deterministic ⇒ one-hot ``pi``).

    Fit **offline** in one batch pass (``fit``) over the train log — no online updates during eval,
    so a given train set yields a fixed, reproducible policy. ``alpha`` sets the exploration weight,
    ``ridge_lambda`` the prior precision (keeps ``A_a`` invertible on tiny/collinear data).
    """

    def __init__(
        self,
        d: int,
        n_arms: int = N_ARMS,
        *,
        alpha: float = 1.0,
        ridge_lambda: float = 1.0,
    ) -> None:
        self.d = d
        self.n_arms = n_arms
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.name = f"linucb(alpha={self.alpha:g})"
        # A_a starts at lambda*I (ridge prior), b_a at 0. Recompute theta / A^{-1} so they are
        # always defined (an unfit LinUCB is pure exploration, theta=0), avoiding Optional handling.
        self.A = np.array([self.ridge_lambda * np.eye(d) for _ in range(n_arms)])  # [K, d, d]
        self.b = np.zeros((n_arms, d), dtype=np.float64)
        self._recompute()

    def _recompute(self) -> None:
        self.A_inv = np.array([np.linalg.inv(self.A[a]) for a in range(self.n_arms)])
        self.theta = np.array([self.A_inv[a] @ self.b[a] for a in range(self.n_arms)])

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> LinUCB:
        """Accumulate ``A_a``/``b_a`` over the logged ``(x, a, r)`` rows, then refresh ``theta``.

        One batch pass (offline training on the group-wise train split). Returns ``self``.
        """
        for x, a, r in zip(contexts, actions.astype(int), rewards, strict=True):
            self.A[a] += np.outer(x, x)
            self.b[a] += r * x
        self._recompute()
        return self

    def ucb_scores(self, x: np.ndarray) -> np.ndarray:
        """Per-arm UCB score ``theta_a . x + alpha*sqrt(x . A_a^{-1} x)`` (shape ``[K]``)."""
        mean = self.theta @ x  # predicted reward per arm
        bonus = np.array(
            [self.alpha * float(np.sqrt(x @ self.A_inv[a] @ x)) for a in range(self.n_arms)]
        )
        scores: np.ndarray = mean + bonus
        return scores

    def act(self, x: np.ndarray) -> tuple[int, float]:
        action = int(np.argmax(self.ucb_scores(x)))
        return action, 1.0  # deterministic argmax

    def prob(self, action: int, x: np.ndarray) -> float:
        return 1.0 if action == int(np.argmax(self.ucb_scores(x))) else 0.0


def baseline_fixed_policy(*, multihop: bool, arm_names: Sequence[str] = ARMS) -> FixedPolicy:
    """Pre-registered ``FixedPolicy`` baseline (multi-hop ⇒ graph, single-shot ⇒ hybrid)."""
    arm = DEFAULT_MULTIHOP_ARM if multihop else DEFAULT_SINGLESHOT_ARM
    return FixedPolicy(arm, arm_names=arm_names)


class GatedPolicy:
    """Two-branch router: a cheap **easy** branch and a **hard** branch, split by a label-free gate.

    Motivation (A6.1 verdict): the contextual bandit *regressed* on easy (CTRL) queries because it
    misrouted them to heavy retrievers where cheap ``dense`` already suffices, and the ``lambda_c``
    sweep showed cost-tuning cannot fix it (~97% of the loss was retrieval-quality, not cost). A
    gate removes that failure **by construction**: a deploy-time predicate routes easy queries to
    the cheap arm and sends only hard queries to the ``hard`` branch (a fixed arm, or the bandit
    that must *earn* the branch).

    Gate: ``hard`` iff ``x[gate_index] > gate_threshold``, else ``easy``. With the default
    ``gate_threshold=0.0`` and ``gate_index`` = the ``is_bridging`` feature, the gate recovers
    ``is_bridging == 1`` under **both** raw (0/1) and train-fit **z-standardized** contexts: a 0/1
    feature's standardized images straddle 0 — value ``1`` maps to ``(1-mean)/std > 0`` and value
    ``0`` to ``-mean/std < 0`` for any ``0 < mean < 1``; a constant column (all-0 or all-1 in the
    fit fold) passes through unchanged, so ``> 0`` still selects the ``1`` rows. Hence the *same*
    policy object works whether or not the caller standardizes — no separate raw-context plumbing.

    ``pi(a|x)`` delegates to whichever branch the gate selects, so ``GatedPolicy`` composes with any
    ``Policy`` on either branch. Because the gate is a **deterministic** function of ``x``,
    conditioning on ``x`` fixes the branch and ``pi_gated(.|x) == pi_activebranch(.|x)`` exactly —
    so the propensity the active branch reports is already the correct gated propensity for OPE
    (a deterministic branch ⇒ one-hot ``pi``; an ``EpsilonGreedy`` branch ⇒ its eps-mixed ``pi``).
    """

    def __init__(
        self,
        easy: Policy,
        hard: Policy,
        *,
        gate_index: int,
        gate_threshold: float = 0.0,
        name: str | None = None,
    ) -> None:
        if easy.n_arms != hard.n_arms:
            raise ValueError(
                f"branch arm-space mismatch: easy has {easy.n_arms} arms, hard has {hard.n_arms}"
            )
        self.easy = easy
        self.hard = hard
        self.gate_index = int(gate_index)
        self.gate_threshold = float(gate_threshold)
        self.n_arms = easy.n_arms
        self.name = name or f"gated({easy.name}|{hard.name})"

    def is_hard(self, x: np.ndarray) -> bool:
        """Gate predicate: True ⇒ route to the hard branch (``x[gate_index] > gate_threshold``)."""
        return bool(x[self.gate_index] > self.gate_threshold)

    def _branch(self, x: np.ndarray) -> Policy:
        return self.hard if self.is_hard(x) else self.easy

    def act(self, x: np.ndarray) -> tuple[int, float]:
        # The active branch's propensity IS the gated propensity (gate deterministic given x).
        return self._branch(x).act(x)

    def prob(self, action: int, x: np.ndarray) -> float:
        return self._branch(x).prob(action, x)


def build_gated_policy(
    hard: Policy,
    *,
    easy_arm: str = "dense",
    gate_feature: str = "is_bridging",
    arm_names: Sequence[str] = ARMS,
) -> GatedPolicy:
    """Compose the pre-registered gated router: easy queries ⇒ ``easy_arm``, hard ⇒ ``hard`` policy.

    ``easy_arm`` defaults to the cheapest arm (``dense``, cost 0) — the A6.1 finding is that CTRL
    queries are already well served by it. The gate feature index is resolved from ``FEATURE_NAMES``
    (lazy import so this module stays free of featurizer imports at load time); the gate uses only
    that label-free feature, never the gold stratum (leakage rule 1).
    """
    from stock_agent.rag.policy_features import FEATURE_NAMES

    gate_index = FEATURE_NAMES.index(gate_feature)  # ValueError on an unknown feature name
    easy = FixedPolicy(easy_arm, arm_names=arm_names)
    return GatedPolicy(easy, hard, gate_index=gate_index, name=f"gated({easy_arm}|{hard.name})")
