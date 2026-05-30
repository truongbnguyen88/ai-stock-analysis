"""Backtest metrics: hand-verified golden values + edge cases."""

from __future__ import annotations

import math

import pytest

from stock_agent.backtesting.metrics import (
    accuracy,
    base_rate,
    brier_score,
    log_loss,
    precision,
    recall,
    roc_auc,
    threshold_metrics,
)

# Mixed case with errors so precision/recall/AUC are < 1.
_Y = [1.0, 0.0, 1.0, 0.0]
_P = [0.9, 0.2, 0.3, 0.6]


def test_brier_golden() -> None:
    # mean((0.9-1)^2, (0.2-0)^2, (0.3-1)^2, (0.6-0)^2) = (0.01+0.04+0.49+0.36)/4
    assert brier_score(_Y, _P) == pytest.approx(0.225)


def test_log_loss_golden() -> None:
    expected = -(math.log(0.9) + math.log(0.8) + math.log(0.3) + math.log(0.4)) / 4
    assert log_loss(_Y, _P) == pytest.approx(expected, abs=1e-9)


def test_accuracy_precision_recall_golden() -> None:
    # preds at 0.5: [1,0,0,1]; y=[1,0,1,0] → 2/4 correct.
    assert accuracy(_Y, _P) == pytest.approx(0.5)
    # predicted positive at idx 0,3 → tp=1 (idx0), fp=1 (idx3).
    assert precision(_Y, _P) == pytest.approx(0.5)
    # actual positive at idx 0,2 → tp=1 (idx0), fn=1 (idx2).
    assert recall(_Y, _P) == pytest.approx(0.5)


def test_roc_auc_golden() -> None:
    # positives p={0.9,0.3}, negatives p={0.2,0.6}; 3 of 4 pairs correctly ordered.
    assert roc_auc(_Y, _P) == pytest.approx(0.75)


def test_roc_auc_single_class_is_none() -> None:
    assert roc_auc([1.0, 1.0, 1.0], [0.2, 0.8, 0.5]) is None


def test_precision_none_when_no_predicted_positives() -> None:
    assert precision([1.0, 0.0], [0.1, 0.2]) is None


def test_perfect_predictions() -> None:
    y = [1.0, 0.0, 1.0, 0.0]
    p = [1.0, 0.0, 1.0, 0.0]
    assert brier_score(y, p) == pytest.approx(0.0)
    assert roc_auc(y, p) == pytest.approx(1.0)
    assert log_loss(y, p) == pytest.approx(0.0, abs=1e-10)  # clipped, ~0


def test_threshold_metrics_bundle() -> None:
    m = threshold_metrics(_Y, _P, threshold=0.05)
    assert m.threshold == 0.05
    assert m.n == 4
    assert m.n_positive == 2
    assert m.base_rate == pytest.approx(0.5)
    assert m.brier == pytest.approx(0.225)
    assert m.roc_auc == pytest.approx(0.75)


def test_base_rate() -> None:
    assert base_rate([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)
