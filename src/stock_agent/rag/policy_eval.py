"""Offline policy training + verdict harness (advanced-RAG A6.1f) — the ``rag policy-eval`` engine.

This is the batch "training procedure" for the contextual bandit. It never trains online; given the
$0, deterministic reward oracle (A6.1c) it runs one offline pass:

    1. build the full-information reward matrix ``R[N,K]`` + label-free contexts ``X[N,d]``;
    2. **group-wise** split (``split_multihop``) into train / test folds (leakage rule 2);
    3. synthesize a logged dataset on each fold under the uniform logging policy ``mu`` (full
       support ⇒ OPE is exact) — each row sees only the *sampled* arm's reward (partial feedback);
    4. **fit** the learned policies on the TRAIN log — ``LinUCB`` and an ``EpsilonGreedy`` over a
       ridge reward model ``q_hat`` (both numpy normal-equations; no gradient descent, rule 5);
    5. **evaluate off-policy** on the TEST log — IPS / SNIPS / DR (+ group bootstrap CI) for every
       candidate (the 5 fixed arms + the learned policies), plus the *true* value from ``R_test``
       (a full-information ground-truth check the deterministic oracle uniquely affords);
    6. apply the **pre-registered verdict**: promote iff ``DR(bandit) − DR(best fixed) > 0`` clears
       the group bootstrap 95% CI on the paired difference AND the bandit does not regress CTRL.

The heavy, corpus-touching part (``build_dataset``: it calls ``reward_matrix`` → ``retrieve``) is
separated from the pure-numpy scoring (``evaluate_offline``) so the scoring/verdict logic is fully
CI-testable on a synthetic reward matrix with a *known* contextual optimum, while the CLI feeds the
real (local, $0) matrix. Nothing here calls an LLM or emits a forecast — it only scores retrieval.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from stock_agent.rag.ope import (
    OPEEstimate,
    bootstrap_delta_stats,
    dr_value,
    fit_q_ridge,
    importance_weights,
    off_policy_evaluate,
    predict_q,
)
from stock_agent.rag.policy import (
    EpsilonGreedy,
    FixedPolicy,
    LinUCB,
    Policy,
    UniformPolicy,
    build_gated_policy,
    target_prob_matrix,
)
from stock_agent.rag.policy_features import featurize
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.rag.reward import DEFAULT_ARM_COSTS, reward_matrix
from stock_agent.settings import Settings

if TYPE_CHECKING:
    from stock_agent.research.multistep_eval import MultiHopQuery

# The learned (non-fixed) policies the verdict can promote. "fixed" is a baseline, never the bandit.
LEARNED_POLICIES = ("linucb", "epsilon_greedy")
_STRATA = ("HARD", "MED", "CTRL")


# ---- dataset -----------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """The offline bandit dataset for a benchmark: contexts, the full-info reward matrix, and keys.

    ``contexts`` ``[N,d]`` (label-free features); ``reward_matrix`` ``[N,K]`` = reward of every arm
    on every query ($0 full information); ``groups`` ``[N]`` int-encoded split keys (bridge pairs)
    for the group bootstrap; ``strata`` ``[N]`` HARD/MED/CTRL for per-stratum reporting;
    ``arm_names`` the K arm labels in column order (== the policy action order).
    """

    contexts: np.ndarray
    reward_matrix: np.ndarray
    groups: np.ndarray
    strata: np.ndarray
    arm_names: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def d(self) -> int:
        return int(self.contexts.shape[1])

    @property
    def n_arms(self) -> int:
        return len(self.arm_names)

    def select(self, idx: np.ndarray) -> Dataset:
        """Row-subset (a fold) — slices every aligned array by ``idx`` (arms/keys unchanged)."""
        return Dataset(
            contexts=self.contexts[idx],
            reward_matrix=self.reward_matrix[idx],
            groups=self.groups[idx],
            strata=self.strata[idx],
            arm_names=self.arm_names,
        )


def _group_key(query: MultiHopQuery) -> str:
    """Split key: explicit ``group_id`` (bridge pair) else the question (mirrors multistep_gen)."""
    return query.group_id or query.question


def build_dataset(
    queries: Sequence[MultiHopQuery],
    systems: Mapping[str, RetrievalSystem],
    *,
    settings: Settings,
    alias_map: Mapping[str, Sequence[str]] | None = None,
    graph_universe: Collection[str] | None = None,
    lambda_cost: float | None = None,
    arm_costs: Mapping[str, float] = DEFAULT_ARM_COSTS,
    top_k: int | None = None,
) -> Dataset:
    """Materialize the offline dataset: reward matrix (via ``reward_matrix``) + label-free contexts.

    ``systems`` maps arm name → retrieval system; its **insertion order fixes the arm/column order**
    (pass it in ``ARMS`` order so policy action indices line up). Contexts use ``q.seed`` as the
    scoped ticker (the filing the bridge starts from) so ``has_ticker`` / ``in_graph_universe``
    reflect the real query scope. This is the only step that runs retrieval ($0, needs the corpus).
    """
    arm_names = tuple(systems.keys())
    r_matrix = reward_matrix(
        systems,
        queries,
        settings=settings,
        lambda_cost=lambda_cost,
        arm_costs=arm_costs,
        top_k=top_k,
    )
    contexts = np.vstack(
        [
            featurize(
                q.question, ticker=q.seed, alias_map=alias_map, graph_universe=graph_universe
            ).values
            for q in queries
        ]
    )
    # Int-encode group keys so the bootstrap can resample groups; strata stay strings for reporting.
    keys = np.array([_group_key(q) for q in queries])
    _, groups = np.unique(keys, return_inverse=True)
    strata = np.array([q.stratum or "NA" for q in queries])
    return Dataset(contexts, r_matrix, groups.astype(int), strata, arm_names)


def split_dataset(
    dataset: Dataset, queries: Sequence[MultiHopQuery], *, test_frac: float, seed: int
) -> tuple[Dataset, Dataset]:
    """Group-wise train/test split reusing the canonical ``split_multihop`` (no split-logic drift).

    Recovers row indices from the returned query objects (identity), so a bridge pair's variants
    never straddle the fold — the same guarantee ``split_multihop`` gives, on the aligned arrays.
    """
    from stock_agent.research.multistep_gen import split_multihop

    train_q, _ = split_multihop(list(queries), test_frac=test_frac, seed=seed)
    train_ids = {id(q) for q in train_q}
    train_idx = np.array([i for i, q in enumerate(queries) if id(q) in train_ids], dtype=int)
    test_idx = np.array([i for i, q in enumerate(queries) if id(q) not in train_ids], dtype=int)
    return dataset.select(train_idx), dataset.select(test_idx)


# ---- standardization (fit on train only) -------------------------------------
def _fit_standardizer(contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) from TRAIN rows; constant columns (std 0, e.g. bias) pass through.

    Zero-variance columns get mean 0 / std 1 so ``(x-0)/1`` leaves them unchanged — this keeps the
    constant ``bias`` feature at 1.0 instead of annihilating it (a plain z-score would zero it).
    """
    mean = contexts.mean(axis=0)
    std = contexts.std(axis=0)
    const = std == 0.0
    mean = np.where(const, 0.0, mean)
    std = np.where(const, 1.0, std)
    return mean, std


def _standardize(contexts: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z: np.ndarray = (contexts - mean) / std
    return z


# ---- synthetic logged dataset under mu ---------------------------------------
def synthesize_log(
    contexts: np.ndarray, r_matrix: np.ndarray, mu: Policy
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one logged ``(action, propensity, reward)`` per row under logging policy ``mu``.

    Partial feedback: row ``i`` observes only ``R[i, a_i]`` for the arm ``mu`` sampled (the other
    arms' rewards are unseen by OPE — the full matrix is only the reward *lookup* here, and the
    ground-truth check elsewhere). ``mu`` is seeded ⇒ the log is reproducible.
    """
    n = len(contexts)
    actions = np.empty(n, dtype=int)
    propensities = np.empty(n, dtype=np.float64)
    rewards = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, p = mu.act(contexts[i])
        actions[i] = a
        propensities[i] = p
        rewards[i] = r_matrix[i, a]
    return actions, propensities, rewards


def _true_value(target_probs: np.ndarray, r_matrix: np.ndarray) -> float:
    """Full-information value ``V(pi) = mean_i sum_a pi(a|x_i) R[i,a]`` (ground truth for DR)."""
    return float(np.mean((target_probs * r_matrix).sum(axis=1)))


# ---- report types ------------------------------------------------------------
@dataclass(frozen=True)
class PolicyValue:
    """One candidate policy's off-policy value (DR headline) + diagnostics + the true value."""

    name: str
    dr: float
    dr_ci: tuple[float, float]
    ips: float
    snips: float
    true_value: float
    ess: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dr": self.dr,
            "dr_ci": list(self.dr_ci),
            "ips": self.ips,
            "snips": self.snips,
            "true_value": self.true_value,
            "ess": self.ess,
        }


@dataclass(frozen=True)
class StratumDelta:
    """Per-stratum DR gap of the bandit vs the (globally) best fixed arm — the regression guard."""

    stratum: str
    n: int
    dr_bandit: float
    dr_fixed: float
    delta: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "n": self.n,
            "dr_bandit": self.dr_bandit,
            "dr_fixed": self.dr_fixed,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class PolicyEvalReport:
    """The verdict: every candidate's value, the bandit-vs-best-fixed delta, and promote/reject."""

    arms: tuple[str, ...]
    n_train: int
    n_test: int
    seed: int
    lambda_cost: float
    values: list[PolicyValue]
    best_fixed: str
    bandit: str
    delta_dr: float
    delta_ci: tuple[float, float]
    delta_p_positive: float  # exact one-sided bootstrap P(Δ>0) — reported, not gated
    per_stratum: list[StratumDelta]
    promote: bool
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "seed": self.seed,
            "lambda_cost": self.lambda_cost,
            "values": [v.to_json_dict() for v in self.values],
            "best_fixed": self.best_fixed,
            "bandit": self.bandit,
            "delta_dr": self.delta_dr,
            "delta_ci": list(self.delta_ci),
            "delta_p_positive": self.delta_p_positive,
            "per_stratum": [s.to_json_dict() for s in self.per_stratum],
            "promote": self.promote,
            "rationale": self.rationale,
        }


# ---- the offline evaluation --------------------------------------------------
def fit_learned_policies(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    n_arms: int,
    alpha: float,
    epsilon: float,
    ridge_lambda: float,
    seed: int,
) -> dict[str, Policy]:
    """Fit the learned policies on the TRAIN log (offline, one pass, numpy normal-equations).

    ``linucb`` = per-arm ridge + UCB; ``epsilon_greedy`` = ridge ``q_hat`` + eps exploration.
    """
    d = contexts.shape[1]
    q_weights = fit_q_ridge(contexts, actions, rewards, n_arms=n_arms, ridge_lambda=ridge_lambda)
    linucb = LinUCB(d, n_arms, alpha=alpha, ridge_lambda=ridge_lambda).fit(
        contexts, actions, rewards
    )
    return {
        "linucb": linucb,
        "epsilon_greedy": EpsilonGreedy(q_weights, epsilon=epsilon, seed=seed),
    }


def fit_bandit_baseline(
    train: Dataset,
    *,
    alpha: float = 1.0,
    ridge_lambda: float = 1.0,
    seed: int = 42,
) -> tuple[LinUCB, np.ndarray, np.ndarray]:
    """Fit the A6.1 LinUCB on a train fold and return it with its context standardizer (A6.2g).

    Reproduces ``evaluate_offline``'s training path exactly — train-fit z-score, a uniform
    ``mu``-log of the train fold (seed ``seed``), one batch ridge/UCB pass — so the RL eval's bandit
    baseline is *the* A6.1 policy, not a re-implementation. Returns ``(linucb, mean, std)``; the
    caller must standardize contexts with the same ``(mean, std)`` before scoring (the scale LinUCB
    was fit on).

    Typed to the concrete ``LinUCB`` (not ``Policy``) because the A6.2g baseline needs
    ``ucb_scores`` to re-argmax over the MDP's restricted arm menu.
    """
    mean, std = _fit_standardizer(train.contexts)  # fit on TRAIN only (leakage-safe)
    train_x = _standardize(train.contexts, mean, std)
    actions, _, rewards = synthesize_log(
        train_x, train.reward_matrix, UniformPolicy(train.n_arms, seed=seed)
    )
    linucb = LinUCB(train.d, train.n_arms, alpha=alpha, ridge_lambda=ridge_lambda).fit(
        train_x, actions, rewards
    )
    return linucb, mean, std


def _evaluate_policy(
    policy: Policy,
    test_x: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    r_test: np.ndarray,
    groups: np.ndarray,
    *,
    n_arms: int,
    ridge_lambda: float,
    n_boot: int,
    seed: int,
) -> tuple[PolicyValue, np.ndarray]:
    """OPE headline (IPS/SNIPS/DR + CI) plus the true value for one policy.

    Also returns the policy's target-prob matrix (reused by the paired-delta stage).
    """
    tp = target_prob_matrix(policy, test_x)
    est: dict[str, OPEEstimate] = off_policy_evaluate(
        test_x,
        actions,
        propensities,
        rewards,
        tp,
        groups=groups,
        n_arms=n_arms,
        ridge_lambda=ridge_lambda,
        n_boot=n_boot,
        seed=seed,
    )
    value = PolicyValue(
        name=policy.name,
        dr=est["dr"].value,
        dr_ci=(est["dr"].ci_low, est["dr"].ci_high),
        ips=est["ips"].value,
        snips=est["snips"].value,
        true_value=_true_value(tp, r_test),
        ess=est["dr"].ess,
    )
    return value, tp


def _paired_delta(
    bandit_tp: np.ndarray,
    fixed_tp: np.ndarray,
    test_x: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    *,
    n_arms: int,
    ridge_lambda: float,
    n_boot: int,
    seed: int,
) -> tuple[float, tuple[float, float], float, list[StratumDelta]]:
    """Paired DR delta ``DR(bandit) − DR(fixed)`` with a group bootstrap CI + per-stratum breakdown.

    The reward model ``q_hat`` (policy-independent) is fit once on the test log; only the importance
    weights and target-prob matrices differ between the two policies. The bootstrap resamples the
    SAME groups for both DR values each draw (paired) so the CI is on their difference — the
    pre-registered promotion statistic. Per-stratum deltas reuse those weights/``q_hat`` on masks.

    Returns ``(point, ci, p_positive, per_stratum)`` where ``p_positive = P(Δ > 0)`` is the exact
    one-sided bootstrap probability the paired delta is positive (companion to the CI, reported not
    gated — see ``bootstrap_delta_stats``).
    """
    q_weights = fit_q_ridge(test_x, actions, rewards, n_arms=n_arms, ridge_lambda=ridge_lambda)
    q_hat_all = predict_q(test_x, q_weights)
    w_b = importance_weights(actions, propensities, bandit_tp)
    w_f = importance_weights(actions, propensities, fixed_tp)

    def _delta(idx: np.ndarray) -> float:
        dr_b = dr_value(actions[idx], rewards[idx], w_b[idx], bandit_tp[idx], q_hat_all[idx])
        dr_f = dr_value(actions[idx], rewards[idx], w_f[idx], fixed_tp[idx], q_hat_all[idx])
        return dr_b - dr_f

    full = np.arange(len(rewards))
    point = _delta(full)
    lo, hi, p_positive = bootstrap_delta_stats(
        _delta, len(rewards), groups=groups, n_boot=n_boot, seed=seed
    )
    ci = (lo, hi)

    per_stratum: list[StratumDelta] = []
    for s in _STRATA:
        mask = strata == s
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        dr_b = dr_value(actions[idx], rewards[idx], w_b[idx], bandit_tp[idx], q_hat_all[idx])
        dr_f = dr_value(actions[idx], rewards[idx], w_f[idx], fixed_tp[idx], q_hat_all[idx])
        per_stratum.append(StratumDelta(s, int(mask.sum()), dr_b, dr_f, dr_b - dr_f))
    return point, ci, p_positive, per_stratum


def evaluate_offline(
    train: Dataset,
    test: Dataset,
    *,
    bandit: str = "linucb",
    alpha: float = 1.0,
    epsilon: float = 0.1,
    ridge_lambda: float = 1.0,
    lambda_cost: float = 0.05,  # reported only; the cost penalty is already baked into R by build
    standardize: bool = True,
    n_boot: int = 1000,
    seed: int = 42,
) -> PolicyEvalReport:
    """Train the bandit offline on ``train`` and off-policy-evaluate it vs the fixed arms on test.

    Fits ``LinUCB``/``EpsilonGreedy`` on a ``mu``-log of the train fold, evaluates all candidates (5
    fixed + 2 learned) on a separate ``mu``-log of the test fold, picks the best fixed arm by DR,
    and applies the pre-registered verdict for the chosen ``bandit`` policy. ``standardize``
    z-scores the contexts using **train** statistics only (leakage-safe; constant columns pass
    through). Deterministic given ``seed`` (mu-train = seed, mu-test = seed+1, bootstrap = seed).
    """
    if bandit not in LEARNED_POLICIES:
        raise ValueError(f"bandit must be one of {LEARNED_POLICIES}; got {bandit!r}")
    n_arms = train.n_arms
    train_x, test_x = train.contexts, test.contexts
    if standardize:
        mean, std = _fit_standardizer(train.contexts)  # fit on TRAIN only
        train_x = _standardize(train.contexts, mean, std)
        test_x = _standardize(test.contexts, mean, std)

    # 1) synthesize the mu-logs (uniform logging policy, full support ⇒ OPE exact).
    mu_train = UniformPolicy(n_arms, seed=seed)
    a_tr, _, r_tr = synthesize_log(train_x, train.reward_matrix, mu_train)
    mu_test = UniformPolicy(n_arms, seed=seed + 1)
    a_te, p_te, r_te = synthesize_log(test_x, test.reward_matrix, mu_test)

    # 2) fit learned policies on the train log; the fixed baselines need no fitting.
    learned = fit_learned_policies(
        train_x,
        a_tr,
        r_tr,
        n_arms=n_arms,
        alpha=alpha,
        epsilon=epsilon,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )
    candidates: dict[str, Policy] = {
        f"fixed({name})": FixedPolicy(name, arm_names=train.arm_names) for name in train.arm_names
    }
    candidates.update({p.name: p for p in learned.values()})

    # 3) evaluate every candidate on the test log (DR headline + true-value ground truth).
    values: list[PolicyValue] = []
    target_probs: dict[str, np.ndarray] = {}
    for name, policy in candidates.items():
        v, tp = _evaluate_policy(
            policy,
            test_x,
            a_te,
            p_te,
            r_te,
            test.reward_matrix,
            test.groups,
            n_arms=n_arms,
            ridge_lambda=ridge_lambda,
            n_boot=n_boot,
            seed=seed,
        )
        values.append(v)
        target_probs[name] = tp

    # 4) pick the best FIXED arm by DR; the bandit is the chosen learned policy.
    fixed_values = [v for v in values if v.name.startswith("fixed(")]
    best_fixed = max(fixed_values, key=lambda v: v.dr).name
    bandit_name = learned[bandit].name

    # 5) pre-registered verdict: paired DR delta clears the group bootstrap 95% CI AND no CTRL loss.
    delta, delta_ci, delta_p_pos, per_stratum = _paired_delta(
        target_probs[bandit_name],
        target_probs[best_fixed],
        test_x,
        a_te,
        p_te,
        r_te,
        test.groups,
        test.strata,
        n_arms=n_arms,
        ridge_lambda=ridge_lambda,
        n_boot=n_boot,
        seed=seed,
    )
    ctrl_regression = any(s.stratum == "CTRL" and s.delta < -1e-9 for s in per_stratum)
    promote = delta > 0 and delta_ci[0] > 0 and not ctrl_regression
    rationale = _verdict_rationale(delta, delta_ci, ctrl_regression, bandit_name, best_fixed)

    return PolicyEvalReport(
        arms=train.arm_names,
        n_train=train.n,
        n_test=test.n,
        seed=seed,
        lambda_cost=lambda_cost,
        values=values,
        best_fixed=best_fixed,
        bandit=bandit_name,
        delta_dr=delta,
        delta_ci=delta_ci,
        delta_p_positive=delta_p_pos,
        per_stratum=per_stratum,
        promote=promote,
        rationale=rationale,
    )


def _verdict_rationale(
    delta: float, ci: tuple[float, float], ctrl_regression: bool, bandit: str, best_fixed: str
) -> str:
    """Human-readable one-liner explaining the promote/reject decision."""
    lo, hi = ci
    if delta <= 0:
        return f"REJECT: {bandit} DR ≤ {best_fixed} (Δ={delta:+.4f})."
    if lo <= 0:
        return f"REJECT: Δ={delta:+.4f} does not clear the 95% CI [{lo:+.4f}, {hi:+.4f}]."
    if ctrl_regression:
        return f"REJECT: Δ={delta:+.4f} clears CI but regresses the CTRL stratum."
    return (
        f"PROMOTE: {bandit} beats {best_fixed} by Δ={delta:+.4f}, "
        f"CI [{lo:+.4f}, {hi:+.4f}] > 0, no CTRL regression."
    )


# ---- gated router (A6.1 verdict follow-up) -----------------------------------
@dataclass(frozen=True)
class GatedEvalReport:
    """Gated-router verdict: overall promotion **and** whether the bandit earns the hard branch.

    Two pre-registered questions (see ``evaluate_gated``): (1) does ``gated(easy_arm | linucb)``
    beat the best fixed arm on the held-out fold (paired group-bootstrap CI, no CTRL regression)?
    (2) does ``linucb`` beat ``fixed(graph)`` on the HARD+MED rows only — i.e. does the bandit
    *earn* the hard branch, or stay the A5.3 fixed default? The nested ``hard_branch`` answers (2).
    """

    arms: tuple[str, ...]
    n_train: int
    n_test: int
    seed: int
    lambda_cost: float
    gate_feature: str
    easy_arm: str
    values: list[PolicyValue]
    best_fixed: str
    # Verdict 1 — promote the gated router (headline = gated(easy_arm | linucb)) vs best fixed arm.
    gated_policy: str
    delta_dr: float
    delta_ci: tuple[float, float]
    delta_p_positive: float  # exact one-sided bootstrap P(Δ>0) for verdict 1 — reported, not gated
    per_stratum: list[StratumDelta]
    promote: bool
    rationale: str
    # Verdict 2 — does the bandit EARN the hard branch? linucb vs fixed(graph) on HARD+MED only.
    hard_bandit: str
    hard_fixed: str
    hard_n: int
    hard_delta_dr: float
    hard_delta_ci: tuple[float, float]
    hard_delta_p_positive: float  # exact one-sided bootstrap P(Δ>0) for verdict 2
    hard_per_stratum: list[StratumDelta]
    bandit_earns_hard: bool
    hard_rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "seed": self.seed,
            "lambda_cost": self.lambda_cost,
            "gate_feature": self.gate_feature,
            "easy_arm": self.easy_arm,
            "values": [v.to_json_dict() for v in self.values],
            "best_fixed": self.best_fixed,
            "gated_policy": self.gated_policy,
            "delta_dr": self.delta_dr,
            "delta_ci": list(self.delta_ci),
            "delta_p_positive": self.delta_p_positive,
            "per_stratum": [s.to_json_dict() for s in self.per_stratum],
            "promote": self.promote,
            "rationale": self.rationale,
            "hard_branch": {
                "bandit": self.hard_bandit,
                "fixed": self.hard_fixed,
                "n": self.hard_n,
                "delta_dr": self.hard_delta_dr,
                "delta_ci": list(self.hard_delta_ci),
                "delta_p_positive": self.hard_delta_p_positive,
                "per_stratum": [s.to_json_dict() for s in self.hard_per_stratum],
                "bandit_earns_hard": self.bandit_earns_hard,
                "rationale": self.hard_rationale,
            },
        }


def _hard_branch_rationale(delta: float, ci: tuple[float, float], bandit: str, fixed: str) -> str:
    """One-liner: does the bandit earn the hard branch, or does it stay the fixed default?"""
    lo, hi = ci
    if delta <= 0:
        return f"HARD BRANCH = {fixed}: {bandit} DR ≤ {fixed} on HARD+MED (Δ={delta:+.4f})."
    if lo <= 0:
        return (
            f"HARD BRANCH = {fixed}: {bandit} leads by Δ={delta:+.4f} on HARD+MED but the 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] includes 0 (not certified — keep the fixed default)."
        )
    return (
        f"HARD BRANCH = {bandit}: beats {fixed} on HARD+MED by Δ={delta:+.4f}, "
        f"CI [{lo:+.4f}, {hi:+.4f}] > 0 — the bandit earns the branch."
    )


def evaluate_gated(
    train: Dataset,
    test: Dataset,
    *,
    easy_arm: str = "dense",
    gate_feature: str = "is_bridging",
    hard_fixed_arm: str = "graph",
    alpha: float = 1.0,
    epsilon: float = 0.1,
    ridge_lambda: float = 1.0,
    lambda_cost: float = 0.05,
    standardize: bool = True,
    n_boot: int = 1000,
    seed: int = 42,
) -> GatedEvalReport:
    """Evaluate the gated router and answer the two pre-registered questions.

    **Verdict 1 (promotion).** Does ``gated(easy_arm | linucb)`` beat the best fixed arm on the
    group-wise held-out fold — clearing the paired group-bootstrap 95% CI with no CTRL regression?
    The gate (``gate_feature`` = ``is_bridging``) routes easy queries to ``easy_arm`` (cheap
    ``dense``) so the bandit never touches the CTRL queries it regressed on in the A6.1 verdict.

    **Verdict 2 (does the bandit earn the hard branch?).** Restricted to HARD+MED rows only, does
    ``linucb`` beat ``fixed(hard_fixed_arm)`` (=graph, the A5.3 tiered default) clearing the CI? If
    not, the hard branch should stay the fixed arm — the bandit is not *assumed* onto it (refinement
    2). ``linucb`` is fit on the full train log (it only serves hard queries once gated).

    The logged folds + standardizer are built exactly as ``evaluate_offline`` (train-fit z-score;
    mu-train seed, mu-test seed+1), so the two harnesses agree; only the candidate set and the two
    verdicts differ. The gate reads the standardized context: ``x[gate_index] > 0`` recovers
    ``is_bridging == 1`` because a 0/1 feature's standardized images straddle 0 (see GatedPolicy).
    """
    arms = train.arm_names
    n_arms = train.n_arms
    if standardize:
        mean, std = _fit_standardizer(train.contexts)  # fit on TRAIN only (leakage-safe)
        train_x = _standardize(train.contexts, mean, std)
        test_x = _standardize(test.contexts, mean, std)
    else:
        train_x, test_x = train.contexts, test.contexts

    # Logged folds under uniform mu (context-independent log; identical to evaluate_offline).
    mu_train = UniformPolicy(n_arms, seed=seed)
    a_tr, _, r_tr = synthesize_log(train_x, train.reward_matrix, mu_train)
    mu_test = UniformPolicy(n_arms, seed=seed + 1)
    a_te, p_te, r_te = synthesize_log(test_x, test.reward_matrix, mu_test)

    learned = fit_learned_policies(
        train_x,
        a_tr,
        r_tr,
        n_arms=n_arms,
        alpha=alpha,
        epsilon=epsilon,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )
    linucb = learned["linucb"]
    graph_fixed = FixedPolicy(hard_fixed_arm, arm_names=arms)

    # Candidates: 5 fixed + 2 learned standalone + 2 gated (cheap gate + fixed-graph / bandit hard).
    gated_bandit = build_gated_policy(
        linucb, easy_arm=easy_arm, gate_feature=gate_feature, arm_names=arms
    )
    gated_graph = build_gated_policy(
        graph_fixed, easy_arm=easy_arm, gate_feature=gate_feature, arm_names=arms
    )
    candidates: dict[str, Policy] = {
        f"fixed({name})": FixedPolicy(name, arm_names=arms) for name in arms
    }
    candidates.update({p.name: p for p in learned.values()})
    candidates[gated_graph.name] = gated_graph
    candidates[gated_bandit.name] = gated_bandit

    values: list[PolicyValue] = []
    target_probs: dict[str, np.ndarray] = {}
    for name, policy in candidates.items():
        v, tp = _evaluate_policy(
            policy,
            test_x,
            a_te,
            p_te,
            r_te,
            test.reward_matrix,
            test.groups,
            n_arms=n_arms,
            ridge_lambda=ridge_lambda,
            n_boot=n_boot,
            seed=seed,
        )
        values.append(v)
        target_probs[name] = tp

    fixed_values = [v for v in values if v.name.startswith("fixed(")]
    best_fixed = max(fixed_values, key=lambda v: v.dr).name

    # Verdict 1: gated(easy|linucb) vs best fixed — the promotion decision.
    delta, delta_ci, delta_p_pos, per_stratum = _paired_delta(
        target_probs[gated_bandit.name],
        target_probs[best_fixed],
        test_x,
        a_te,
        p_te,
        r_te,
        test.groups,
        test.strata,
        n_arms=n_arms,
        ridge_lambda=ridge_lambda,
        n_boot=n_boot,
        seed=seed,
    )
    ctrl_regression = any(s.stratum == "CTRL" and s.delta < -1e-9 for s in per_stratum)
    promote = delta > 0 and delta_ci[0] > 0 and not ctrl_regression
    rationale = _verdict_rationale(delta, delta_ci, ctrl_regression, gated_bandit.name, best_fixed)

    # Verdict 2: does the bandit earn the hard branch? linucb vs fixed(graph) on HARD+MED only.
    hard_idx = np.flatnonzero(np.isin(test.strata, ("HARD", "MED")))
    h_per: list[StratumDelta]
    if hard_idx.size == 0:  # degenerate fold with no hard rows — nothing to earn.
        h_delta, h_ci, h_p_pos, h_per, earns = 0.0, (0.0, 0.0), 0.0, [], False
    else:
        h_delta, h_ci, h_p_pos, h_per = _paired_delta(
            target_probs[linucb.name][hard_idx],
            target_probs[graph_fixed.name][hard_idx],
            test_x[hard_idx],
            a_te[hard_idx],
            p_te[hard_idx],
            r_te[hard_idx],
            test.groups[hard_idx],
            test.strata[hard_idx],
            n_arms=n_arms,
            ridge_lambda=ridge_lambda,
            n_boot=n_boot,
            seed=seed,
        )
        earns = h_delta > 0 and h_ci[0] > 0
    hard_rationale = _hard_branch_rationale(h_delta, h_ci, linucb.name, graph_fixed.name)

    return GatedEvalReport(
        arms=arms,
        n_train=train.n,
        n_test=test.n,
        seed=seed,
        lambda_cost=lambda_cost,
        gate_feature=gate_feature,
        easy_arm=easy_arm,
        values=values,
        best_fixed=best_fixed,
        gated_policy=gated_bandit.name,
        delta_dr=delta,
        delta_ci=delta_ci,
        delta_p_positive=delta_p_pos,
        per_stratum=per_stratum,
        promote=promote,
        rationale=rationale,
        hard_bandit=linucb.name,
        hard_fixed=graph_fixed.name,
        hard_n=int(hard_idx.size),
        hard_delta_dr=h_delta,
        hard_delta_ci=h_ci,
        hard_delta_p_positive=h_p_pos,
        hard_per_stratum=h_per,
        bandit_earns_hard=earns,
        hard_rationale=hard_rationale,
    )
