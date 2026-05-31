"""Backtesting & calibration domain models.

Typed contracts for out-of-sample evaluation results, so the runner, the CLI,
the experiment log, and (Phase 6.5) the agent's ``get_calibration`` tool all
share one shape. As with ``ScenarioForecast``, every number here is produced by
deterministic evaluation code — never by the LLM.

Probability calibration is the headline deliverable: ``CalibrationReport.ece``
is the trustworthiness measure surfaced to the user.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field


class ThresholdMetrics(BaseModel):
    """Binary-classification metrics for one return threshold ``P(r > θ)``.

    Each forecast yields an exceedance probability at θ; the realized label is
    ``1[r > θ]``. Metrics aggregate over all out-of-sample (prob, label) pairs.
    """

    threshold: float  # fractional return boundary, e.g. 0.05 = +5%
    n: int
    n_positive: int
    base_rate: float  # P(label = 1) in the OOS sample
    brier: float  # mean squared error of probability; lower is better
    log_loss: float  # cross-entropy; lower is better
    accuracy: float  # at a 0.5 decision threshold
    precision: float | None  # None when the model predicts no positives
    recall: float | None  # None when there are no actual positives
    roc_auc: float | None  # rank discrimination; None when a single class is present


class ReliabilityBin(BaseModel):
    """One bin of a reliability diagram (predicted vs. realized frequency)."""

    lower: float  # bin's predicted-probability lower edge (inclusive)
    upper: float  # upper edge (exclusive, except the last bin)
    count: int
    mean_pred: float  # mean predicted probability in the bin (confidence)
    frequency: float  # empirical positive rate in the bin (accuracy)


class CalibrationReport(BaseModel):
    """Probability calibration over the pooled OOS predictions.

    ``ece`` (Expected Calibration Error) is the |confidence − accuracy| gap,
    bin-count weighted: 0 is perfectly calibrated. ``ece_post`` is an HONEST OOS
    estimate after a post-hoc fit (isotonic/Platt fit on an earlier half,
    evaluated on a later held-out half) — it answers "would recalibration help?"
    """

    n: int
    n_bins: int
    bins: list[ReliabilityBin]
    ece: float
    mce: float  # max calibration error across non-empty bins
    method_post: str | None = None  # "isotonic" / "platt" when ece_post is set
    ece_pre_holdout: float | None = None  # raw ECE on the held-out half (baseline for ece_post)
    ece_post: float | None = None  # ECE on the held-out half after post-hoc calibration


class FoldSummary(BaseModel):
    """Per-fold bookkeeping for dispersion + reproducibility."""

    index: int
    train_start: Date
    train_end: Date
    test_start: Date
    test_end: Date
    n_test: int
    mean_brier: float  # averaged across thresholds within the fold


class BacktestResult(BaseModel):
    """Full out-of-sample evaluation of one forecaster on one ticker/horizon."""

    ticker: str
    horizon_days: int = Field(gt=0)
    model_name: str
    n_folds: int
    n_predictions: int  # number of OOS as-of forecast points
    as_of_start: Date
    as_of_end: Date
    thresholds: list[ThresholdMetrics]
    calibration: CalibrationReport
    # Big-move ("large move regardless of direction") prediction quality:
    # P(|return| > big_move_k) = P(r < -k) + P(r > +k), scored vs 1[|r| > k]. This is
    # where ML's volatility/tail skill is NOT redundant with the directional baselines.
    big_move_k: float = 0.10
    big_move: ThresholdMetrics | None = None
    mean_brier: float  # mean Brier across thresholds (headline scalar; lower better)
    mean_log_loss: float
    folds: list[FoldSummary]
    seed: int
    notes: list[str] = Field(default_factory=list)
