"""A6.1d — off-policy evaluation: hand-computed IPS/SNIPS/DR goldens + invariants (offline, CI).

Golden 3-sample set (2 arms; target policy always picks arm 0):
    actions=[0,1,0]  mu=[0.5,0.5,0.25]  r=[1.0,0.0,0.4]  target_probs=[[1,0]]*3
    weights w = pi(a_i|x_i)/mu = [1/0.5, 0/0.5, 1/0.25] = [2, 0, 4]
    IPS   = mean(2*1, 0*0, 4*0.4) = mean(2,0,1.6)   = 1.2
    SNIPS = (2*1+0+4*0.4)/(2+0+4) = 3.6/6           = 0.6
    DR(q̂≡0) == IPS                                  = 1.2
"""

from __future__ import annotations

import numpy as np

from stock_agent.rag.ope import (
    dr_value,
    effective_sample_size,
    fit_q_ridge,
    importance_weights,
    ips_value,
    off_policy_evaluate,
    predict_q,
    snips_value,
)

ACTIONS = np.array([0, 1, 0])
MU = np.array([0.5, 0.5, 0.25])
REWARDS = np.array([1.0, 0.0, 0.4])
TARGET = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # always arm 0
CONTEXTS = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # bias + one feature


def _weights() -> np.ndarray:
    return importance_weights(ACTIONS, MU, TARGET)


def test_importance_weights_golden() -> None:
    assert _weights().tolist() == [2.0, 0.0, 4.0]


def test_ips_golden() -> None:
    assert ips_value(REWARDS, _weights()) == 1.2


def test_snips_golden() -> None:
    assert snips_value(REWARDS, _weights()) == 0.6


def test_dr_equals_ips_when_qhat_zero() -> None:
    q_hat_all = np.zeros((3, 2))
    assert dr_value(ACTIONS, REWARDS, _weights(), TARGET, q_hat_all) == ips_value(
        REWARDS, _weights()
    )


def test_snips_is_within_reward_hull() -> None:
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 1, size=50)
    w = rng.uniform(0.1, 5.0, size=50)
    v = snips_value(r, w)
    assert r.min() <= v <= r.max()


def test_ips_exact_on_full_information_matrix() -> None:
    # Enumerate EVERY (context, arm) once with mu=0.5 -> IPS is exact = mean_i max_a R[i,a].
    reward_matrix = np.array([[0.2, 0.9], [0.7, 0.1], [0.5, 0.5]])
    actions = np.array([0, 1, 0, 1, 0, 1])
    mu = np.full(6, 0.5)
    rewards = np.array([0.2, 0.9, 0.7, 0.1, 0.5, 0.5])  # R flattened row-major
    # Target = argmax per context (ties -> arm 0): ctx0->arm1, ctx1->arm0, ctx2->arm0.
    target = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
    w = importance_weights(actions, mu, target)
    assert w.tolist() == [0.0, 2.0, 2.0, 0.0, 2.0, 0.0]
    got = ips_value(rewards, w)
    assert got == np.mean(np.max(reward_matrix, axis=1))  # 0.7 exactly


def test_fit_q_ridge_recovers_linear_reward() -> None:
    # Per-arm true reward r = x·θ_a; ridge with tiny λ recovers θ_a.
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 2))
    theta = np.array([[1.0, 0.0], [0.0, 2.0]])  # arm0 -> feature0, arm1 -> feature1
    actions = np.array([0, 1] * 10)
    rewards = np.array([x[i] @ theta[actions[i]] for i in range(20)])
    q_weights = fit_q_ridge(x, actions, rewards, n_arms=2, ridge_lambda=1e-8)
    assert np.allclose(q_weights, theta, atol=1e-3)
    # predict_q reproduces the training rewards for the chosen arm.
    q_hat_all = predict_q(x, q_weights)
    chosen = q_hat_all[np.arange(20), actions]
    assert np.allclose(chosen, rewards, atol=1e-3)


def test_fit_q_ridge_zero_for_unseen_arm() -> None:
    x = np.array([[1.0, 1.0], [1.0, 2.0]])
    actions = np.array([0, 0])  # arm 1 never chosen
    rewards = np.array([0.5, 0.5])
    q_weights = fit_q_ridge(x, actions, rewards, n_arms=2, ridge_lambda=1.0)
    assert q_weights[1].tolist() == [0.0, 0.0]  # no evidence -> neutral 0


def test_effective_sample_size() -> None:
    assert effective_sample_size(np.ones(5)) == 5.0
    assert effective_sample_size(np.array([10.0, 0.0, 0.0, 0.0, 0.0])) == 1.0


def test_off_policy_evaluate_matches_point_estimators_and_is_deterministic() -> None:
    out1 = off_policy_evaluate(CONTEXTS, ACTIONS, MU, REWARDS, TARGET, n_boot=200, seed=7)
    out2 = off_policy_evaluate(CONTEXTS, ACTIONS, MU, REWARDS, TARGET, n_boot=200, seed=7)
    assert out1["ips"].value == 1.2
    assert out1["snips"].value == 0.6
    # DR with an in-sample ridge q̂ need not equal IPS, but the CI must bracket its own point value.
    for name in ("ips", "snips", "dr"):
        est = out1[name]
        assert est.ci_low <= est.value <= est.ci_high
        assert (est.ci_low, est.ci_high) == (out2[name].ci_low, out2[name].ci_high)  # seeded
        assert est.n == 3


def test_group_bootstrap_is_deterministic_and_uses_groups() -> None:
    groups = np.array([0, 0, 1])  # rows 0,1 share a group
    a = off_policy_evaluate(
        CONTEXTS, ACTIONS, MU, REWARDS, TARGET, groups=groups, n_boot=200, seed=3
    )
    b = off_policy_evaluate(
        CONTEXTS, ACTIONS, MU, REWARDS, TARGET, groups=groups, n_boot=200, seed=3
    )
    assert (a["ips"].ci_low, a["ips"].ci_high) == (b["ips"].ci_low, b["ips"].ci_high)
