"""A6.1f — offline verdict harness: log synthesis, contextual promote/reject, leakage-safe split.

The scoring logic (``evaluate_offline``) is exercised on a **synthetic** reward matrix with a
*known* contextual optimum (no corpus, no model), so the promote/reject verdict is checked against
ground truth. ``build_dataset`` (which calls ``retrieve``) is not run here — the reward-adapter
test (A6.1c) covers ``reward_matrix``; this pins the numpy scoring + the pre-registered rule.
"""

from __future__ import annotations

import numpy as np

from stock_agent.rag.policy import ARMS, FixedPolicy, UniformPolicy, build_gated_policy
from stock_agent.rag.policy_eval import (
    Dataset,
    _fit_standardizer,
    _standardize,
    evaluate_gated,
    evaluate_offline,
    split_dataset,
    synthesize_log,
)
from stock_agent.rag.policy_features import FEATURE_NAMES, N_FEATURES
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery

_ARMS = ("arm0", "arm1")

# --- gated-router fixtures (G4) ------------------------------------------------
# Column indices in the real 11-d feature layout (append-only FEATURE_NAMES).
_DENSE = ARMS.index("dense")
_RERANKED = ARMS.index("reranked")
_GRAPH = ARMS.index("graph")
_GATE_IDX = FEATURE_NAMES.index("is_bridging")  # the deploy-time gate signal
_SUB_IDX = FEATURE_NAMES.index("qtype_risk")  # a within-hard discriminator (NOT the gate)


def _ctx_row(is_bridging: int, sub: int = 0) -> np.ndarray:
    """A full 11-d context: bias=1, is_bridging at the gate index, an extra binary at `sub`."""
    x = np.zeros(N_FEATURES)
    x[0] = 1.0  # bias intercept the linear policies expect
    x[_GATE_IDX] = float(is_bridging)
    x[_SUB_IDX] = float(sub)
    return x


def _reward(best_arm: int, hi: float = 1.0, lo: float = 0.2) -> list[float]:
    """5-arm reward: `hi` for the context-optimal arm, `lo` elsewhere (linearly separable)."""
    r = [lo] * len(ARMS)
    r[best_arm] = hi
    return r


def _full_dataset(rows: list[tuple[np.ndarray, list[float], str]]) -> Dataset:
    """Assemble a Dataset over the real 5-arm space; each row is its own bootstrap group."""
    return Dataset(
        contexts=np.array([r[0] for r in rows]),
        reward_matrix=np.array([r[1] for r in rows]),
        groups=np.arange(len(rows)),
        strata=np.array([r[2] for r in rows]),
        arm_names=ARMS,
    )


def _split_optimum(n_ctrl: int, n_hard: int) -> Dataset:
    """CTRL(is_bridging=0)→dense optimal; HARD(is_bridging=1)→graph optimal (uniformly on hard).

    No single fixed arm wins both strata, so the gate (CTRL→dense, HARD→bandit) should beat any
    fixed arm. Because graph is *uniformly* best on the hard rows, linucb ≈ fixed(graph) there — so
    the bandit does NOT earn the hard branch (refinement 2's "keep the fixed default" path).
    """
    rows: list[tuple[np.ndarray, list[float], str]] = []
    rows += [(_ctx_row(0), _reward(_DENSE), "CTRL")] * n_ctrl
    rows += [(_ctx_row(1), _reward(_GRAPH), "HARD")] * n_hard
    return _full_dataset(rows)


def _hard_substructure(n_ctrl: int, n_hard_each: int) -> Dataset:
    """CTRL→dense; HARD&sub=0→graph; MED&sub=1→reranked.

    On the hard rows no single fixed arm is optimal (graph wins the sub=0 half, reranked the sub=1
    half), so linucb (which can read `sub`) beats fixed(graph) → the bandit EARNS the hard branch.
    Using MED for the sub=1 rows also checks that MED is routed to the hard branch alongside HARD.
    """
    rows: list[tuple[np.ndarray, list[float], str]] = []
    rows += [(_ctx_row(0, 0), _reward(_DENSE), "CTRL")] * n_ctrl
    rows += [(_ctx_row(1, 0), _reward(_GRAPH), "HARD")] * n_hard_each
    rows += [(_ctx_row(1, 1), _reward(_RERANKED), "MED")] * n_hard_each
    return _full_dataset(rows)


def _dense_dominates(n_ctrl: int, n_hard: int) -> Dataset:
    """dense optimal on EVERY row (no signal) → the gated router cannot beat fixed(dense)."""
    rows: list[tuple[np.ndarray, list[float], str]] = []
    rows += [(_ctx_row(0), _reward(_DENSE), "CTRL")] * n_ctrl
    rows += [(_ctx_row(1), _reward(_DENSE), "HARD")] * n_hard
    return _full_dataset(rows)


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


# --- gated router: verdict 1 (promotion) --------------------------------------
def test_gated_promotes_on_context_split_optimum() -> None:
    # dense best on CTRL, graph best on HARD: the gate routes each stratum to its optimum, so the
    # gated router beats any single fixed arm — CTRL rows are in the test fold, with no regression.
    train, test = _split_optimum(40, 40), _split_optimum(40, 40)
    report = evaluate_gated(train, test, alpha=0.5, n_boot=200, seed=1)

    assert report.gated_policy.startswith("gated(dense|linucb")
    gated = next(v for v in report.values if v.name == report.gated_policy)
    best_fixed = next(v for v in report.values if v.name == report.best_fixed)
    assert gated.true_value > best_fixed.true_value  # realizes the per-stratum optimum
    assert best_fixed.true_value < 0.7  # a single fixed arm is stuck near the 0.6 blend

    assert report.delta_dr > 0 and report.delta_ci[0] > 0
    assert report.promote
    assert "PROMOTE" in report.rationale

    # CTRL rows ARE in the test fold and the gate protects them (no regression vs best fixed).
    ctrl = next(s for s in report.per_stratum if s.stratum == "CTRL")
    assert ctrl.n > 0 and ctrl.delta >= -1e-9


def test_gated_does_not_promote_without_contextual_signal() -> None:
    # dense dominates every row: the gate + bandit cannot beat fixed(dense) → reject.
    ds = _dense_dominates(40, 40)
    report = evaluate_gated(ds, ds, alpha=0.5, n_boot=200, seed=1)
    assert not report.promote
    assert "REJECT" in report.rationale


# --- gated router: verdict 2 (does the bandit EARN the hard branch?) -----------
def test_bandit_does_not_earn_hard_branch_when_graph_uniformly_optimal() -> None:
    # graph is uniformly best on the hard rows ⇒ linucb ≈ fixed(graph) there ⇒ bandit does not earn.
    train, test = _split_optimum(40, 40), _split_optimum(40, 40)
    report = evaluate_gated(train, test, alpha=0.5, n_boot=200, seed=1)

    assert report.hard_n == 40  # only the HARD rows are the hard branch here (no MED)
    assert not report.bandit_earns_hard
    assert (
        report.hard_delta_ci[0] <= 0
    )  # CI includes 0 (or below) — not certified over fixed(graph)
    assert "HARD BRANCH = fixed(graph)" in report.hard_rationale


def test_bandit_earns_hard_branch_on_within_hard_structure() -> None:
    # within the hard rows, graph wins the sub=0 half and reranked the sub=1 half: no single fixed
    # arm is optimal, so linucb beats fixed(graph) → the bandit earns the branch (MED counts hard).
    train, test = _hard_substructure(30, 30), _hard_substructure(30, 30)
    report = evaluate_gated(train, test, alpha=0.5, n_boot=200, seed=1)

    assert report.hard_n == 60  # HARD (sub=0) + MED (sub=1) both routed to the hard branch
    assert report.bandit_earns_hard
    assert report.hard_delta_dr > 0 and report.hard_delta_ci[0] > 0
    assert "earns the branch" in report.hard_rationale


# --- gated router: JSON schema + gate/standardization integration -------------
def test_gated_report_json_has_nested_hard_branch_block() -> None:
    report = evaluate_gated(_split_optimum(30, 30), _split_optimum(30, 30), n_boot=50, seed=1)
    d = report.to_json_dict()
    for key in (
        "gate_feature",
        "easy_arm",
        "gated_policy",
        "best_fixed",
        "promote",
        "per_stratum",
        "delta_p_positive",  # exact one-sided bootstrap P(Δ>0), verdict 1 (A6.1-b)
    ):
        assert key in d
    hard = d["hard_branch"]
    for key in (
        "bandit",
        "fixed",
        "n",
        "delta_dr",
        "delta_ci",
        "delta_p_positive",  # verdict 2
        "bandit_earns_hard",
        "rationale",
    ):
        assert key in hard
    assert d["gate_feature"] == "is_bridging" and d["easy_arm"] == "dense"
    assert 0.0 <= d["delta_p_positive"] <= 1.0 and 0.0 <= hard["delta_p_positive"] <= 1.0


def test_gate_recovers_is_bridging_after_train_fit_standardization() -> None:
    # End-to-end standardization invariance: fit the standardizer on a mixed train fold, z-score a
    # mixed test fold, and confirm the gate still recovers is_bridging==1 on every standardized
    # row — AND routes easy→dense / hard→graph. This is the property evaluate_gated relies on.
    train = _split_optimum(30, 30)
    test = _split_optimum(
        10, 20
    )  # different is_bridging mix → standardized threshold must still hold
    mean, std = _fit_standardizer(train.contexts)
    z_test = _standardize(test.contexts, mean, std)

    gate = build_gated_policy(FixedPolicy("graph"))
    for row, z in zip(test.contexts, z_test, strict=True):
        expected_hard = row[_GATE_IDX] > 0.5
        assert gate.is_hard(z) == bool(expected_hard)
        action, _ = gate.act(z)
        assert action == (_GRAPH if expected_hard else _DENSE)
