"""Tests for split-conformal interval calibration."""

from __future__ import annotations

import math

import numpy as np

from stock_agent.forecasting.conformal import (
    conformal_correction,
    conformal_metrics,
    conformalize_interval,
    empirical_coverage,
    nonconformity,
)


def test_nonconformity_sign_inside_vs_outside() -> None:
    # Inside → negative (distance to nearest edge); outside → positive (how far).
    assert abs(nonconformity(-0.1, 0.1, 0.0) - (-0.1)) < 1e-9  # centered, 0.1 from each edge
    assert abs(nonconformity(-0.1, 0.1, 0.15) - 0.05) < 1e-9  # 0.05 above upper
    assert abs(nonconformity(-0.1, 0.1, -0.18) - 0.08) < 1e-9  # 0.08 below lower


def test_correction_widens_undercovering_interval() -> None:
    # Model interval [-0.05, 0.05] but realized moves are larger → scores positive → q>0.
    rng = np.random.default_rng(0)
    ys = rng.normal(0, 0.10, size=200)
    scores = [nonconformity(-0.05, 0.05, float(y)) for y in ys]
    q = conformal_correction(scores, alpha=0.10)
    assert q > 0  # interval must widen to reach 90% coverage
    lo, hi = conformalize_interval(-0.05, 0.05, q)
    # The conformalized interval should now cover ~>= 90% of the SAME sample.
    cov = empirical_coverage([lo] * len(ys), [hi] * len(ys), list(ys))
    assert cov >= 0.90


def test_correction_tightens_overcovering_interval() -> None:
    # Interval far too wide → realized always well inside → scores very negative → q<0.
    ys = [0.0, 0.01, -0.01, 0.02, -0.02] * 20
    scores = [nonconformity(-0.5, 0.5, y) for y in ys]
    q = conformal_correction(scores, alpha=0.10)
    assert q < 0  # can tighten
    lo, hi = conformalize_interval(-0.5, 0.5, q)
    assert hi - lo < 1.0  # narrower than the original width of 1.0


def test_too_few_points_returns_inf() -> None:
    # For 90% coverage you need ceil((n+1)*0.9) <= n  → n >= 9.
    assert conformal_correction([0.0] * 5, alpha=0.10) == math.inf
    assert math.isfinite(conformal_correction([0.0] * 20, alpha=0.10))


def test_coverage_guarantee_on_holdout() -> None:
    # Honest split: fit q on calibration half, verify coverage on a disjoint test half.
    rng = np.random.default_rng(1)
    cal = rng.normal(0, 0.08, size=500)
    test = rng.normal(0, 0.08, size=500)
    lo0, hi0 = -0.04, 0.04  # a deliberately too-narrow nominal interval
    q = conformal_correction([nonconformity(lo0, hi0, float(y)) for y in cal], alpha=0.10)
    lo, hi = conformalize_interval(lo0, hi0, q)
    cov = empirical_coverage([lo] * len(test), [hi] * len(test), list(test))
    assert cov >= 0.86  # ~90% target, allowing finite-sample slack on the holdout


def test_interval_never_inverts() -> None:
    lo, hi = conformalize_interval(-0.05, 0.05, q=10.0)  # absurd tightening
    assert lo <= hi


def test_conformal_metrics_reports_undercoverage_and_fixes_it() -> None:
    # A too-narrow 90% CI on wider realized moves → stated coverage well below 0.90,
    # conformalized coverage near/above 0.90 on the held-out half.
    rng = np.random.default_rng(2)
    ys = rng.normal(0, 0.10, size=400)
    lowers = [-0.04] * 400
    uppers = [0.04] * 400
    m = conformal_metrics(lowers, uppers, list(ys), alpha=0.10)
    assert m is not None
    assert m.empirical_coverage < 0.80  # stated 90% CI under-covers badly
    assert m.correction > 0  # needs widening
    assert m.conformalized_coverage is not None and m.conformalized_coverage >= 0.85
    assert m.mean_width_conformal is not None and m.mean_width_conformal > m.mean_width


def test_conformal_metrics_too_few_points_no_split() -> None:
    m = conformal_metrics([-0.05] * 8, [0.05] * 8, [0.0] * 8, alpha=0.10)
    assert m is not None
    assert m.conformalized_coverage is None  # not enough to split honestly
    assert m.n_eval == 8


def test_conformal_metrics_empty_is_none() -> None:
    assert conformal_metrics([], [], [], alpha=0.10) is None
