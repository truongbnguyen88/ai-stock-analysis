"""A6.1e — retrieval policies: seeded determinism, prob-sums-to-1, hand-checked LinUCB toy (CI).

Pure numpy, no retriever/model. The LinUCB test pins A, theta, and the UCB scores by hand so the
optimism-under-uncertainty arithmetic is verified, not just its shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from stock_agent.rag.policy import (
    ARMS,
    N_ARMS,
    EpsilonGreedy,
    FixedPolicy,
    GatedPolicy,
    LinUCB,
    Policy,
    UniformPolicy,
    baseline_fixed_policy,
    build_gated_policy,
    target_prob_matrix,
)
from stock_agent.rag.policy_features import FEATURE_NAMES, N_FEATURES
from stock_agent.rag.read_path import LATTICE_SYSTEMS


# --- action-space contract ----------------------------------------------------
def test_arms_order_matches_lattice_plus_graph() -> None:
    # Anti-drift: policy action indices must equal reward_matrix / OPE column order.
    expected = (*LATTICE_SYSTEMS, "graph")
    assert tuple(ARMS) == expected
    assert N_ARMS == 5


# --- FixedPolicy --------------------------------------------------------------
def test_fixed_policy_by_name_and_index_agree() -> None:
    by_name = FixedPolicy("graph")
    by_index = FixedPolicy(ARMS.index("graph"))
    assert by_name.action == by_index.action == 4
    assert by_name.act(np.zeros(3)) == (4, 1.0)  # deterministic, propensity 1
    assert by_name.name == "fixed(graph)"


def test_fixed_policy_prob_is_one_hot() -> None:
    p = FixedPolicy("hybrid")  # index 2
    x = np.zeros(3)
    assert p.prob(2, x) == 1.0
    assert p.prob(0, x) == 0.0
    assert sum(p.prob(a, x) for a in range(p.n_arms)) == 1.0


def test_fixed_policy_unknown_arm_raises() -> None:
    try:
        FixedPolicy("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown arm name")


def test_baseline_fixed_policy_picks_registered_defaults() -> None:
    assert baseline_fixed_policy(multihop=True).name == "fixed(graph)"
    assert baseline_fixed_policy(multihop=False).name == "fixed(hybrid)"


# --- UniformPolicy (logging mu) -----------------------------------------------
def test_uniform_policy_is_full_support_and_seeded() -> None:
    mu = UniformPolicy(N_ARMS, seed=0)
    x = np.zeros(3)
    assert mu.prob(0, x) == 1.0 / N_ARMS
    assert sum(mu.prob(a, x) for a in range(N_ARMS)) == pytest.approx(1.0)
    # Two identically-seeded policies produce the same *stream* (act advances the RNG each call).
    other = UniformPolicy(N_ARMS, seed=0)
    seq_a = [mu.act(x) for _ in range(10)]
    seq_b = [other.act(x) for _ in range(10)]
    assert seq_a == seq_b  # same seed -> identical draw stream
    assert len({a for a, _ in seq_a}) > 1  # and the stream actually varies (not a constant arm)
    assert all(prop == 1.0 / N_ARMS for _, prop in seq_a)


# --- EpsilonGreedy ------------------------------------------------------------
# q_hat weights [K=3, d=2]: score = q @ x. For x=[1,1] -> [1, 2, 0.5] -> greedy arm 1.
_Q = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 0.5]])
_X = np.array([1.0, 1.0])


def test_epsilon_greedy_distribution_sums_to_one_and_matches_formula() -> None:
    eps = 0.3
    pol = EpsilonGreedy(_Q, epsilon=eps, seed=0)
    base = eps / 3
    assert pol.prob(1, _X) == base + (1 - eps)  # greedy arm
    assert pol.prob(0, _X) == base
    assert pol.prob(2, _X) == base
    assert sum(pol.prob(a, _X) for a in range(3)) == pytest.approx(1.0)


def test_epsilon_one_degenerates_to_uniform() -> None:
    pol = EpsilonGreedy(_Q, epsilon=1.0, seed=0)
    assert all(pol.prob(a, _X) == 1.0 / 3 for a in range(3))


def test_epsilon_greedy_act_is_seeded_deterministic() -> None:
    a = [EpsilonGreedy(_Q, epsilon=0.5, seed=7).act(_X) for _ in range(20)]
    b = [EpsilonGreedy(_Q, epsilon=0.5, seed=7).act(_X) for _ in range(20)]
    # Each fresh policy at the same seed makes the same first draw.
    assert a == b
    # And a single policy's stream is reproducible across two runs.
    p1, p2 = EpsilonGreedy(_Q, epsilon=0.5, seed=1), EpsilonGreedy(_Q, epsilon=0.5, seed=1)
    assert [p1.act(_X) for _ in range(20)] == [p2.act(_X) for _ in range(20)]


# --- LinUCB: hand-checked 2-arm toy -------------------------------------------
def test_linucb_toy_A_theta_and_ucb_are_hand_verified() -> None:
    # d=1, lambda=1, alpha=1. arm0 pulled twice (x=1, r=1); arm1 once (x=1, r=0).
    #   A0 = 1 + 1 + 1 = 3,  b0 = 1 + 1 = 2,  theta0 = 2/3,  A0^{-1} = 1/3
    #   A1 = 1 + 1 = 2,      b1 = 0,          theta1 = 0,    A1^{-1} = 1/2
    lin = LinUCB(d=1, n_arms=2, alpha=1.0, ridge_lambda=1.0)
    lin.fit(np.array([[1.0], [1.0], [1.0]]), np.array([0, 0, 1]), np.array([1.0, 1.0, 0.0]))
    assert np.allclose(lin.A[0], [[3.0]])
    assert np.allclose(lin.A[1], [[2.0]])
    assert np.allclose(lin.theta[0], [2.0 / 3.0])
    assert np.allclose(lin.theta[1], [0.0])
    assert np.allclose(lin.A_inv[0], [[1.0 / 3.0]])
    assert np.allclose(lin.A_inv[1], [[1.0 / 2.0]])
    x = np.array([1.0])
    ucb = lin.ucb_scores(x)
    assert np.allclose(ucb, [2.0 / 3.0 + np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 2.0)])
    assert lin.act(x) == (0, 1.0)  # arm0 wins (higher mean + bonus)
    assert lin.prob(0, x) == 1.0
    assert lin.prob(1, x) == 0.0


def test_linucb_unseen_arm_is_pure_exploration_bonus() -> None:
    # alpha=2, lambda=4: an arm with NO logged rows keeps theta=0 and A=lambda*I, so its UCB is
    # only the bonus alpha*sqrt(x . (1/lambda) I . x) = alpha*sqrt(||x||^2/lambda).
    lin = LinUCB(d=2, n_arms=2, alpha=2.0, ridge_lambda=4.0)
    lin.fit(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 0]), np.array([1.0, 1.0]))
    x = np.array([3.0, 4.0])  # ||x||^2 = 25
    assert np.allclose(lin.theta[1], [0.0, 0.0])
    bonus = 2.0 * np.sqrt(25.0 / 4.0)  # = 2 * 2.5 = 5.0
    assert lin.ucb_scores(x)[1] == bonus


# --- target_prob_matrix (the OPE bridge) --------------------------------------
def test_target_prob_matrix_shape_and_rows_sum_to_one() -> None:
    contexts = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    for pol in (
        FixedPolicy("dense"),
        UniformPolicy(N_ARMS),
        EpsilonGreedy(np.zeros((N_ARMS, 2)), epsilon=0.2, seed=0),
    ):
        m = target_prob_matrix(pol, contexts)
        assert m.shape == (3, pol.n_arms)
        assert np.allclose(m.sum(axis=1), 1.0)


def test_policies_satisfy_the_protocol() -> None:
    assert isinstance(FixedPolicy("dense"), Policy)
    assert isinstance(UniformPolicy(N_ARMS), Policy)
    assert isinstance(LinUCB(d=2), Policy)
    assert isinstance(build_gated_policy(FixedPolicy("graph")), Policy)


# --- GatedPolicy (A6.1 verdict follow-up: the two-branch router) ---------------
_GATE_IDX = FEATURE_NAMES.index("is_bridging")


def _ctx(is_bridging: float) -> np.ndarray:
    """An 11-dim context with the bias set and the is_bridging feature = ``is_bridging``."""
    x = np.zeros(N_FEATURES)
    x[0] = 1.0
    x[_GATE_IDX] = is_bridging
    return x


def test_gated_policy_routes_by_gate_feature() -> None:
    # easy=dense (idx 0), hard=graph (idx 4); is_bridging gates the branch.
    g = GatedPolicy(FixedPolicy("dense"), FixedPolicy("graph"), gate_index=_GATE_IDX)
    x_easy, x_hard = _ctx(0.0), _ctx(1.0)
    assert g.is_hard(x_hard) is True and g.is_hard(x_easy) is False
    assert g.act(x_easy) == (0, 1.0)  # dense on an easy query
    assert g.act(x_hard) == (4, 1.0)  # graph on a hard query
    assert g.prob(0, x_easy) == 1.0 and g.prob(4, x_easy) == 0.0
    assert g.prob(4, x_hard) == 1.0 and g.prob(0, x_hard) == 0.0


def test_gated_policy_prob_delegates_to_active_branch_and_sums_to_one() -> None:
    # Hard branch = eps-greedy whose greedy arm is graph when is_bridging fires.
    q = np.zeros((N_ARMS, N_FEATURES))
    q[ARMS.index("graph"), _GATE_IDX] = 1.0  # graph scores highest on a bridging row
    hard = EpsilonGreedy(q, epsilon=0.2, seed=0)
    g = GatedPolicy(FixedPolicy("dense"), hard, gate_index=_GATE_IDX)

    x_hard = _ctx(1.0)
    assert abs(sum(g.prob(a, x_hard) for a in range(N_ARMS)) - 1.0) < 1e-12
    for a in range(N_ARMS):  # gated prob == the hard branch's eps-mixed prob, exactly
        assert g.prob(a, x_hard) == hard.prob(a, x_hard)

    x_easy = _ctx(0.0)  # easy row: one-hot on dense, sums to 1
    assert g.prob(0, x_easy) == 1.0
    assert sum(g.prob(a, x_easy) for a in range(N_ARMS)) == 1.0


def test_gate_threshold_zero_is_standardization_invariant() -> None:
    """`> 0` recovers is_bridging==1 under BOTH raw (0/1) and z-standardized contexts."""
    g = GatedPolicy(FixedPolicy("dense"), FixedPolicy("graph"), gate_index=_GATE_IDX)
    # (a) raw 0/1 — this is also the constant-column standardizer passthrough case.
    assert g.is_hard(_ctx(1.0)) and not g.is_hard(_ctx(0.0))
    # (b) z-standardized: a fold's is_bridging column with both 0s and 1s (mean in (0,1)).
    col = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    mean, std = col.mean(), col.std()
    assert 0.0 < mean < 1.0 and std > 0.0
    z1, z0 = (1.0 - mean) / std, (0.0 - mean) / std
    assert z1 > 0.0 > z0  # the images of 1 and 0 straddle the threshold
    assert g.is_hard(_ctx(z1)) and not g.is_hard(_ctx(z0))


def test_build_gated_policy_resolves_gate_and_easy_arm() -> None:
    hard = FixedPolicy("graph")
    g = build_gated_policy(hard, easy_arm="dense")
    assert g.gate_index == FEATURE_NAMES.index("is_bridging")
    assert isinstance(g.easy, FixedPolicy) and g.easy.action == ARMS.index("dense")
    assert g.hard is hard
    assert g.name == "gated(dense|fixed(graph))"


def test_gated_policy_branch_arm_mismatch_raises() -> None:
    easy = FixedPolicy("dense", arm_names=("dense", "graph"))  # n_arms=2
    hard = FixedPolicy("graph")  # n_arms=5
    with pytest.raises(ValueError):
        GatedPolicy(easy, hard, gate_index=0)
