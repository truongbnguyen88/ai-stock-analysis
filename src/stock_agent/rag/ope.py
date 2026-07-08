"""Off-policy evaluation for retrieval policies (advanced-RAG A6.1d) — score a policy from logs.

We want the *value* ``V(pi) = E_x E_{a~pi(.|x)}[r(x,a)]`` of a candidate retrieval policy ``pi``
WITHOUT deploying it — using only data logged under a different (logging) policy ``mu``. Three
estimators, in increasing sophistication (all pure numpy, no torch):

- **IPS** (inverse propensity scoring) — reweight each logged reward by the ratio
  ``w_i = pi(a_i|x_i)/mu(a_i|x_i)`` so the ``mu``-drawn sample estimates the ``pi`` expectation.
  *Unbiased* when ``mu(a|x)>0`` wherever ``pi(a|x)>0`` (full support) and propensities
  are known — but *high variance* when the weights are large (``pi`` and ``mu`` disagree).

    V_IPS = (1/N) Σ_i w_i r_i

- **SNIPS** (self-normalized IPS) — divide by the sum of weights instead of ``N``. This removes the
  systematic scale bias of raw IPS (E[Σw]≈N but the realized Σw fluctuates), trading a little bias
  for much lower variance; a weighted average of rewards, so it is *bounded* in ``[min r, max r]``.

    V_SNIPS = (Σ_i w_i r_i) / (Σ_i w_i)

- **DR** (doubly robust) — combine a **direct method** reward model ``q̂(x,a)`` with an IPS
  *correction* on its residual. Consistent if *either* ``q̂`` is accurate *or* the propensities are
  correct (hence "doubly" robust), and lower variance than IPS because ``q̂`` absorbs most signal:

    V_DR = (1/N) Σ_i [ Σ_a pi(a|x_i) q̂(x_i,a) + w_i ( r_i − q̂(x_i,a_i) ) ]

``q̂`` is a per-arm **ridge** regression ``r ≈ x·θ_a`` (numpy normal equations; the same linear form
as LinUCB's estimate), fit in-sample on the log. When ``q̂≡0``, DR collapses exactly to IPS (tested).

**Confidence intervals** are a **group-level bootstrap**: resample *groups* (e.g. bridge pairs) with
replacement and recompute the estimate, so the CI honours the A6.0 group structure (rows sharing a
group are near-duplicates — a row-level bootstrap would understate variance / pseudo-replicate).

All functions take plain arrays (contexts ``X[N,d]``, integer ``actions[N]``, ``propensities[N]``,
``rewards[N]``, ``target_probs[N,K]`` = ``pi(a|x_i)`` for every arm) so OPE is decoupled from the
Policy class (A6.1e): the caller turns a policy into ``target_probs``. Reference: Dudík, Langford &
Li (2011), "Doubly Robust Policy Evaluation and Learning."
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OPEEstimate:
    """One estimator's value + bootstrap CI + importance-weight diagnostics."""

    estimator: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    ess: float  # effective sample size (Σw)²/Σw² — low ESS ⇒ a few rows dominate ⇒ untrustworthy


# ---- importance weights & diagnostics ----------------------------------------
def importance_weights(
    actions: np.ndarray, propensities: np.ndarray, target_probs: np.ndarray
) -> np.ndarray:
    """``w_i = pi(a_i|x_i) / mu(a_i|x_i)`` — the per-row importance ratio.

    ``target_probs[i, a]`` is ``pi(a|x_i)``; the logged action's target prob is gathered by
    ``actions``. ``propensities`` are ``mu(a_i|x_i)`` (must be > 0 — logging must have support).
    """
    idx = np.arange(len(actions))
    pi_logged = target_probs[idx, actions]
    weights: np.ndarray = pi_logged / propensities
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS ``(Σw)² / Σw²`` — the effective # independent samples after reweighting.

    Equals ``N`` when all weights are equal; collapses toward 1 when one weight dominates (the IPS
    high-variance failure mode). A cheap trust diagnostic for an OPE estimate.
    """
    s1 = float(weights.sum())
    s2 = float((weights**2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


# ---- point estimators (operate on precomputed weights) -----------------------
def ips_value(rewards: np.ndarray, weights: np.ndarray) -> float:
    """``V_IPS = mean(w_i r_i)`` — unbiased under full support, high variance."""
    return float(np.mean(weights * rewards))


def snips_value(rewards: np.ndarray, weights: np.ndarray) -> float:
    """``V_SNIPS = Σ w_i r_i / Σ w_i`` — self-normalized; bounded in ``[min r, max r]``."""
    denom = float(weights.sum())
    return float((weights * rewards).sum() / denom) if denom > 0 else 0.0


def dr_value(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    target_probs: np.ndarray,
    q_hat_all: np.ndarray,
) -> float:
    """``V_DR = mean( Σ_a pi(a|x_i) q̂(x_i,a) + w_i (r_i − q̂(x_i,a_i)) )``.

    ``q_hat_all[i, a]`` is the reward model's prediction for every arm; the *direct* term is its
    expectation under ``pi`` and the second term is the IPS-corrected residual on the logged action.
    With ``q_hat_all ≡ 0`` this is exactly ``ips_value`` (the direct term vanishes).
    """
    direct = (target_probs * q_hat_all).sum(axis=1)  # V_pi^model(x_i)
    idx = np.arange(len(actions))
    q_logged = q_hat_all[idx, actions]
    return float(np.mean(direct + weights * (rewards - q_logged)))


# ---- direct-method reward model q̂(x,a) --------------------------------------
def fit_q_ridge(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    n_arms: int,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """Per-arm ridge ``θ_a = (Xᵀ_a X_a + λI)⁻¹ Xᵀ_a r_a`` → weight matrix ``Q[n_arms, d]``.

    Fits an independent linear reward model per arm on the rows where that arm was chosen. Arms with
    no logged rows keep ``θ_a = 0`` (no evidence ⇒ predicts the neutral 0 — DR then leans entirely
    on the IPS correction for those, which is correct). ``predict_q`` turns ``Q`` into the
    ``q_hat_all`` matrix. ``ridge_lambda`` > 0 keeps ``XᵀX + λI`` invertible on tiny/collinear data.
    """
    d = contexts.shape[1]
    weight = np.zeros((n_arms, d), dtype=np.float64)
    eye = ridge_lambda * np.eye(d)
    for a in range(n_arms):
        mask = actions == a
        if not mask.any():
            continue
        x_a = contexts[mask]
        r_a = rewards[mask]
        gram = x_a.T @ x_a + eye
        weight[a] = np.linalg.solve(gram, x_a.T @ r_a)
    return weight


def predict_q(contexts: np.ndarray, q_weights: np.ndarray) -> np.ndarray:
    """``q_hat_all[i, a] = x_i · θ_a`` — reward-model prediction per (row, arm), shape ``[N,K]``."""
    q_hat_all: np.ndarray = contexts @ q_weights.T
    return q_hat_all


# ---- group-level bootstrap ---------------------------------------------------
def bootstrap_ci(
    estimate_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for an estimator, resampling **groups** (not rows) when given.

    ``estimate_fn(idx)`` recomputes the estimate on the row indices ``idx`` (so it works for ratio
    estimators like SNIPS too). With ``groups``, each bootstrap draws groups with replacement and
    concatenates their member rows — honouring the A6.0 group structure (anti-pseudo-replication).
    Deterministic given ``seed``. Returns the ``(alpha/2, 1-alpha/2)`` percentiles.
    """
    boot = _bootstrap_samples(estimate_fn, n, groups=groups, n_boot=n_boot, seed=seed)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def _bootstrap_samples(
    estimate_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """The raw ``[n_boot]`` bootstrap distribution of ``estimate_fn`` (groups resampled if given).

    Shared core for ``bootstrap_ci`` and ``bootstrap_delta_stats`` so both read the SAME resamples
    for a given seed — the CI and the one-sided P(estimate>0) are then mutually consistent.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        unique_groups = np.arange(n)
        members = {g: np.array([g]) for g in unique_groups}
    else:
        unique_groups = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in unique_groups}
    g_count = len(unique_groups)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        drawn = rng.choice(unique_groups, size=g_count, replace=True)
        idx = np.concatenate([members[g] for g in drawn])
        boot[b] = estimate_fn(idx)
    return boot


def bootstrap_delta_stats(
    estimate_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile CI **and** the one-sided bootstrap ``P(estimate > 0)`` from ONE resampling pass.

    ``p_positive`` = fraction of bootstrap replicates that are strictly positive — the bootstrap
    achieved-significance for the one-sided alternative ``H1: Δ > 0`` (ties at exactly 0 are
    measure-zero for continuous DR values, so ``> 0`` vs ``>= 0`` is immaterial). This is the exact
    empirical companion to the CI: it does not gate the pre-registered promote rule (that gates on
    ``CI_low > 0``) but quantifies *how close* a within-CI delta came. Same ``seed``/``groups`` as
    ``bootstrap_ci`` ⇒ the returned ``(lo, hi)`` are identical to ``bootstrap_ci``'s.
    """
    boot = _bootstrap_samples(estimate_fn, n, groups=groups, n_boot=n_boot, seed=seed)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    p_positive = float(np.mean(boot > 0.0))
    return lo, hi, p_positive


# ---- high-level convenience --------------------------------------------------
def off_policy_evaluate(
    contexts: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    target_probs: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_arms: int | None = None,
    ridge_lambda: float = 1.0,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, OPEEstimate]:
    """Compute IPS, SNIPS, and DR (each with a group bootstrap CI) for one target policy.

    ``target_probs[i, a] = pi(a|x_i)``; the DR reward model ``q̂`` is fit in-sample by per-arm ridge.
    Returns ``{"ips":…, "snips":…, "dr":…}``. Column count of ``target_probs`` defines the arm count
    unless ``n_arms`` is given.
    """
    actions = actions.astype(int)
    k = n_arms if n_arms is not None else target_probs.shape[1]
    weights = importance_weights(actions, propensities, target_probs)
    q_weights = fit_q_ridge(contexts, actions, rewards, n_arms=k, ridge_lambda=ridge_lambda)
    q_hat_all = predict_q(contexts, q_weights)
    ess = effective_sample_size(weights)
    n = len(rewards)

    def _ips(idx: np.ndarray) -> float:
        return ips_value(rewards[idx], weights[idx])

    def _snips(idx: np.ndarray) -> float:
        return snips_value(rewards[idx], weights[idx])

    def _dr(idx: np.ndarray) -> float:
        return dr_value(actions[idx], rewards[idx], weights[idx], target_probs[idx], q_hat_all[idx])

    out: dict[str, OPEEstimate] = {}
    for name, fn, point in (
        ("ips", _ips, ips_value(rewards, weights)),
        ("snips", _snips, snips_value(rewards, weights)),
        ("dr", _dr, dr_value(actions, rewards, weights, target_probs, q_hat_all)),
    ):
        lo, hi = bootstrap_ci(fn, n, groups=groups, n_boot=n_boot, seed=seed, alpha=alpha)
        out[name] = OPEEstimate(estimator=name, value=point, ci_low=lo, ci_high=hi, n=n, ess=ess)
    return out
