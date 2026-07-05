"""A6.1f — offline verdict harness: log synthesis, contextual promote/reject, leakage-safe split.

The scoring logic (``evaluate_offline``) is exercised on a **synthetic** reward matrix with a
*known* contextual optimum (no corpus, no model), so the promote/reject verdict is checked against
ground truth. ``build_dataset`` (which calls ``retrieve``) is not run here — the reward-adapter
test (A6.1c) covers ``reward_matrix``; this pins the numpy scoring + the pre-registered rule.
"""

from __future__ import annotations

import numpy as np

from stock_agent.rag.policy import UniformPolicy
from stock_agent.rag.policy_eval import (
    Dataset,
    _fit_standardizer,
    _standardize,
    evaluate_offline,
    split_dataset,
    synthesize_log,
)
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery

_ARMS = ("arm0", "arm1")


def _dataset(cvals: list[int], r_rows: list[list[float]], strata: list[str]) -> Dataset:
    """Build a 2-feature ([bias, c]) 2-arm Dataset; each row is its own bootstrap group."""
    contexts = np.array([[1.0, float(c)] for c in cvals])
    return Dataset(
        contexts=contexts,
        reward_matrix=np.array(r_rows),
        groups=np.arange(len(cvals)),
        strata=np.array(strata),
        arm_names=_ARMS,
    )


def _contextual(n_per: int) -> Dataset:
    # c=0 (CTRL): arm0 best (1,0); c=1 (HARD): arm1 best (0,1). No fixed arm can win both.
    cvals = [0] * n_per + [1] * n_per
    r_rows = [[1.0, 0.0]] * n_per + [[0.0, 1.0]] * n_per
    strata = ["CTRL"] * n_per + ["HARD"] * n_per
    return _dataset(cvals, r_rows, strata)


def _no_signal(n_per: int) -> Dataset:
    # arm0 dominates everywhere ⇒ the best fixed arm is already optimal; the bandit cannot beat it.
    cvals = [0] * n_per + [1] * n_per
    r_rows = [[1.0, 0.0]] * (2 * n_per)
    strata = ["CTRL"] * n_per + ["HARD"] * n_per
    return _dataset(cvals, r_rows, strata)


# --- log synthesis ------------------------------------------------------------
def test_synthesize_log_is_uniform_and_looks_up_the_sampled_reward() -> None:
    ds = _contextual(10)
    a, p, r = synthesize_log(ds.contexts, ds.reward_matrix, UniformPolicy(2, seed=0))
    assert a.shape == (20,) and set(np.unique(a)).issubset({0, 1})
    assert np.allclose(p, 0.5)  # mu(a|x) = 1/2 on every row
    # each logged reward equals the sampled arm's entry in R
    assert np.allclose(r, ds.reward_matrix[np.arange(20), a])


# --- standardizer preserves the constant bias column --------------------------
def test_standardizer_leaves_bias_and_centers_others() -> None:
    ds = _contextual(10)
    mean, std = _fit_standardizer(ds.contexts)
    z = _standardize(ds.contexts, mean, std)
    assert np.allclose(z[:, 0], 1.0)  # constant bias column passes through unchanged
    assert abs(z[:, 1].mean()) < 1e-9  # varying column is centered


# --- the verdict --------------------------------------------------------------
def test_contextual_optimum_promotes_the_bandit() -> None:
    report = evaluate_offline(_contextual(20), _contextual(20), bandit="linucb", n_boot=200, seed=1)
    bandit = next(v for v in report.values if v.name == report.bandit)
    best_fixed = next(v for v in report.values if v.name == report.best_fixed)
    # the bandit realizes the per-context optimum; a single fixed arm is stuck near 0.5.
    assert bandit.true_value > best_fixed.true_value
    assert best_fixed.true_value < 0.6
    assert report.delta_dr > 0 and report.delta_ci[0] > 0
    assert report.promote
    assert "PROMOTE" in report.rationale
    # no CTRL regression is part of the promote decision
    ctrl = next(s for s in report.per_stratum if s.stratum == "CTRL")
    assert ctrl.delta >= -1e-9


def test_no_contextual_signal_rejects() -> None:
    report = evaluate_offline(_no_signal(20), _no_signal(20), bandit="linucb", n_boot=200, seed=1)
    assert not report.promote
    assert "REJECT" in report.rationale


def test_epsilon_greedy_also_learns_the_optimum() -> None:
    report = evaluate_offline(
        _contextual(20), _contextual(20), bandit="epsilon_greedy", epsilon=0.1, n_boot=100, seed=1
    )
    bandit = next(v for v in report.values if v.name == report.bandit)
    best_fixed = next(v for v in report.values if v.name == report.best_fixed)
    assert bandit.true_value > best_fixed.true_value


def test_dr_true_value_agree_under_full_support() -> None:
    # deterministic reward + uniform full-support mu ⇒ DR should track the ground-truth value well.
    report = evaluate_offline(_contextual(25), _contextual(25), bandit="linucb", n_boot=50, seed=3)
    for v in report.values:
        assert abs(v.dr - v.true_value) < 0.15


# --- leakage-safe split (the slice-e "split disjoint" obligation, exercised in f) ---
def test_split_dataset_keeps_groups_disjoint() -> None:
    # three bridge pairs, two variants each; a pair must never straddle the train/test fold.
    def q(gid: str, name: str) -> MultiHopQuery:
        return MultiHopQuery(
            question=f"{name}?", aspects=[Aspect(name="a", spans=["x"])], group_id=gid, seed="NVDA"
        )

    queries = [
        q("NVDA|AMD", "a1"),
        q("NVDA|AMD", "a2"),
        q("NVDA|MU", "b1"),
        q("NVDA|MU", "b2"),
        q("AMD|MU", "c1"),
        q("AMD|MU", "c2"),
    ]
    ds = Dataset(
        contexts=np.ones((6, 2)),
        reward_matrix=np.zeros((6, 2)),
        groups=np.arange(6),
        strata=np.array(["HARD"] * 6),
        arm_names=_ARMS,
    )
    train, test = split_dataset(ds, queries, test_frac=0.34, seed=0)
    assert train.n + test.n == 6
    assert set(train.groups.tolist()).isdisjoint(
        test.groups.tolist()
    )  # no group straddles the fold
