"""Calibration: reliability bins, ECE/MCE golden values, post-hoc improvement."""

from __future__ import annotations

import pytest

from stock_agent.backtesting.calibration import (
    calibration_report,
    expected_calibration_error,
    fit_isotonic,
    reliability_bins,
)

# Two bins [0,0.5) and [0.5,1.0]:
#   bin0: p={0.2,0.3} conf=0.25, y={0,1} acc=0.50
#   bin1: p={0.8,0.9} conf=0.85, y={1,1} acc=1.00
_P = [0.2, 0.3, 0.8, 0.9]
_Y = [0.0, 1.0, 1.0, 1.0]


def test_reliability_bins_golden() -> None:
    bins = reliability_bins(_P, _Y, n_bins=2)
    assert len(bins) == 2
    assert bins[0].count == 2
    assert bins[0].mean_pred == pytest.approx(0.25)
    assert bins[0].frequency == pytest.approx(0.5)
    assert bins[1].mean_pred == pytest.approx(0.85)
    assert bins[1].frequency == pytest.approx(1.0)


def test_ece_mce_golden() -> None:
    # ECE = 0.5*|0.25-0.5| + 0.5*|0.85-1.0| = 0.125 + 0.075 = 0.20
    ece, mce = expected_calibration_error(_P, _Y, n_bins=2)
    assert ece == pytest.approx(0.20)
    assert mce == pytest.approx(0.25)


def test_perfectly_calibrated_has_zero_ece() -> None:
    # Each bin's confidence equals its realized frequency.
    p = [0.0, 0.0, 1.0, 1.0]
    y = [0.0, 0.0, 1.0, 1.0]
    ece, mce = expected_calibration_error(p, y, n_bins=2)
    assert ece == pytest.approx(0.0)
    assert mce == pytest.approx(0.0)


def test_isotonic_recalibration_reduces_error() -> None:
    # Overconfident model: always predicts 0.6, true rate ~0.3 (deterministic).
    pattern = [1.0, 0.0, 0.0]  # 1/3 positive
    y = (pattern * 140)[:400]
    p = [0.6] * len(y)
    cal = fit_isotonic(p[:200], y[:200])
    raw_ece, _ = expected_calibration_error(p[200:], y[200:], n_bins=10)
    cal_ece, _ = expected_calibration_error(cal(p[200:]), y[200:], n_bins=10)  # type: ignore[arg-type]
    assert raw_ece > 0.2  # badly miscalibrated raw
    assert cal_ece < raw_ece  # isotonic fixes most of it


def test_calibration_report_post_hoc_split() -> None:
    pattern = [1.0, 0.0, 0.0]
    y = (pattern * 140)[:400]
    p = [0.6] * len(y)
    report = calibration_report(p, y, n_bins=10, post_method="isotonic", min_post_samples=100)
    assert report.n == 400
    assert report.method_post == "isotonic"
    assert report.ece_post is not None and report.ece_pre_holdout is not None
    assert report.ece_post < report.ece_pre_holdout  # held-out improvement


def test_calibration_report_skips_post_when_too_few() -> None:
    report = calibration_report(_P, _Y, n_bins=2, min_post_samples=100)
    assert report.ece_post is None  # only 4 points
    assert report.ece == pytest.approx(0.20)
