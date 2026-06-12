"""Offline tests for feature-group ablation aggregation + promote gate."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.backtesting.ablation import (
    ablation_table,
    aggregate,
    extract_metrics,
    improvement,
    is_promotion_candidate,
)
from stock_agent.schemas.backtest import (
    BacktestResult,
    CalibrationReport,
    ConformalReport,
    ThresholdMetrics,
)


def _result(
    *, brier: float, ece: float, bm_brier: float = 0.2, bm_auc: float = 0.6,
    emp_cov: float = 0.88, ci: float = 0.90,
) -> BacktestResult:
    return BacktestResult(
        ticker="TST", horizon_days=20, model_name="logistic", n_folds=3, n_predictions=30,
        as_of_start=date(2023, 1, 1), as_of_end=date(2023, 12, 1), thresholds=[],
        calibration=CalibrationReport(n=30, n_bins=10, bins=[], ece=ece, mce=ece + 0.05),
        big_move=ThresholdMetrics(
            threshold=0.10, n=30, n_positive=6, base_rate=0.2, brier=bm_brier,
            log_loss=0.4, accuracy=0.7, precision=0.5, recall=0.5, roc_auc=bm_auc,
        ),
        conformal=ConformalReport(
            ci_level=ci, n_eval=30, empirical_coverage=emp_cov, mean_width=0.2
        ),
        mean_brier=brier, mean_log_loss=0.5, folds=[], seed=0,
    )


def test_extract_metrics_picks_headline_fields() -> None:
    m = extract_metrics(_result(brier=0.18, ece=0.04, bm_auc=0.65, emp_cov=0.85, ci=0.90))
    assert m["mean_brier"] == pytest.approx(0.18)
    assert m["ece"] == pytest.approx(0.04)
    assert m["big_move_auc"] == pytest.approx(0.65)
    assert m["coverage_abs_gap"] == pytest.approx(0.05)


def test_extract_metrics_omits_absent_optionals() -> None:
    r = _result(brier=0.2, ece=0.03)
    r.big_move = None
    r.conformal = None
    m = extract_metrics(r)
    assert set(m) == {"mean_brier", "ece"}


def test_aggregate_means_across_tickers_and_skips_missing() -> None:
    agg = aggregate([_result(brier=0.10, ece=0.02), _result(brier=0.20, ece=0.06)])
    assert agg["mean_brier"] == pytest.approx(0.15)
    assert agg["ece"] == pytest.approx(0.04)


def test_improvement_direction() -> None:
    # lower-is-better: baseline 0.20, candidate 0.18 → +0.02 (better)
    assert improvement("mean_brier", 0.20, 0.18) == pytest.approx(0.02)
    # higher-is-better: baseline 0.60, candidate 0.65 → +0.05 (better)
    assert improvement("big_move_auc", 0.60, 0.65) == pytest.approx(0.05)


def test_promote_gate_requires_brier_gain_and_no_ece_regression() -> None:
    by_label = {
        "baseline": [_result(brier=0.200, ece=0.040)],
        "winner": [_result(brier=0.180, ece=0.035)],   # brier↓, ece↓ → promote
        "noisy": [_result(brier=0.19995, ece=0.040)],  # brier gain below 1e-4 threshold
        "miscal": [_result(brier=0.180, ece=0.080)],   # brier↓ but ece↑ → reject
    }
    rows = {r["label"]: r for r in ablation_table(by_label)}
    assert is_promotion_candidate(rows["winner"]) is True
    assert is_promotion_candidate(rows["noisy"]) is False
    assert is_promotion_candidate(rows["miscal"]) is False
    assert is_promotion_candidate(rows["baseline"]) is False  # no deltas on baseline


def test_ablation_table_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        ablation_table({"volume": [_result(brier=0.2, ece=0.04)]})
