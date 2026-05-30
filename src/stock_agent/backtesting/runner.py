"""Walk-forward backtest runner.

Drives any ``ForecastModel`` over leakage-safe walk-forward folds and scores its
out-of-sample probabilities. The bridge that makes every forecaster comparable:
a ``ScenarioForecast`` is reduced to per-threshold exceedance probabilities
``P(r > θ_k)`` (θ_k = the bucket boundaries = the ML ``THRESHOLDS``), and the
realized label is ``1[r > θ_k]`` — the exact target the ML models train on.

Point-in-time discipline: at each test as-of ``t`` the model is given only
``bars[:t+1]``, so historical-sim and Monte-Carlo are leakage-safe by
construction. Refittable models (pooled ML) are rebuilt per fold via
``build_model(train_end_date)`` so they never see data at/after the test fold.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as Date

import numpy as np

from stock_agent.backtesting.calibration import calibration_report
from stock_agent.backtesting.metrics import threshold_metrics
from stock_agent.backtesting.splitter import assert_no_leakage, walk_forward_splits
from stock_agent.features.assembler import THRESHOLDS
from stock_agent.forecasting.base import ForecastModel
from stock_agent.logging_config import get_logger
from stock_agent.schemas.backtest import BacktestResult, FoldSummary
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

# build_model(train_end_date) -> a model trained only on data <= train_end_date.
ModelBuilder = Callable[[Date], ForecastModel]

_TOL = 1e-9


def stateless_builder(model: ForecastModel) -> ModelBuilder:
    """Adapt a stateless forecaster (historical-sim, MC) to the builder API.

    These models carry no fitted state — they read everything from the sliced
    series at forecast time — so the same instance is reused for every fold.
    """
    return lambda _train_end: model


def exceedance_probabilities(
    forecast: ScenarioForecast, thresholds: list[float] = THRESHOLDS
) -> list[float]:
    """Reduce a bucketed forecast to ``P(r > θ)`` at each threshold.

    Because the bucket boundaries equal the thresholds, ``P(r > θ)`` is the sum
    of probabilities of all buckets whose lower edge is ``>= θ`` (the survival
    function at the boundary). Works identically for every forecaster.
    """
    return [
        float(
            sum(
                b.probability
                for b in forecast.buckets
                if b.lower is not None and b.lower >= theta - _TOL
            )
        )
        for theta in thresholds
    ]


def _slice(series: PriceSeries, end_idx: int) -> PriceSeries:
    """Point-in-time view: bars up to and including ``end_idx``."""
    return PriceSeries(ticker=series.ticker, bars=series.bars[: end_idx + 1])


def run_backtest(
    series: PriceSeries,
    build_model: ModelBuilder,
    *,
    model_name: str,
    horizon_days: int,
    thresholds: list[float] = THRESHOLDS,
    min_train: int = 252,
    test_size: int = 6,
    stride: int | None = None,
    embargo: int | None = None,
    expanding: bool = True,
    rolling_window: int | None = None,
    n_bins: int = 10,
    seed: int = 42,
) -> BacktestResult:
    """Run a leakage-safe walk-forward backtest and score the OOS probabilities.

    Returns per-threshold metrics, a pooled calibration report, and per-fold
    dispersion. Raises ``ValueError`` if the series is too short for any fold.
    """
    closes = np.asarray(series.closes, dtype=float)
    dates = series.dates
    n = len(series)

    folds = walk_forward_splits(
        n_bars=n,
        horizon=horizon_days,
        min_train=min_train,
        test_size=test_size,
        stride=stride,
        embargo=embargo,
        expanding=expanding,
        rolling_window=rolling_window,
    )
    assert_no_leakage(folds, horizon=horizon_days)  # cheap invariant guard

    # Pooled OOS (prob, label) per threshold, plus chronological pools for calibration.
    n_thresh = len(thresholds)
    probs_by_thresh: list[list[float]] = [[] for _ in range(n_thresh)]
    labels_by_thresh: list[list[float]] = [[] for _ in range(n_thresh)]
    cal_p: list[float] = []  # all predictions, time-ordered, for calibration
    cal_y: list[float] = []
    fold_summaries: list[FoldSummary] = []
    notes: list[str] = []

    for fold in folds:
        model = build_model(dates[fold.train_end])
        fold_sq_err: list[float] = []
        for t in fold.test_as_of:
            sub = _slice(series, t)
            try:
                fc = model.forecast(sub, horizon_days=horizon_days, as_of=dates[t])
            except (ValueError, RuntimeError) as exc:  # degrade: skip unforecastable as-of
                log.warning("backtest.forecast_failed", ticker=series.ticker, idx=t, error=str(exc))
                continue
            ex = exceedance_probabilities(fc, thresholds)
            realized = float(closes[t + horizon_days] / closes[t] - 1.0)
            for k, theta in enumerate(thresholds):
                label = 1.0 if realized > theta else 0.0
                probs_by_thresh[k].append(ex[k])
                labels_by_thresh[k].append(label)
                cal_p.append(ex[k])
                cal_y.append(label)
                fold_sq_err.append((ex[k] - label) ** 2)
        if fold_sq_err:
            fold_summaries.append(
                FoldSummary(
                    index=fold.index,
                    train_start=dates[fold.train_start],
                    train_end=dates[fold.train_end],
                    test_start=dates[fold.test_start],
                    test_end=dates[fold.test_end],
                    n_test=len(fold.test_as_of),
                    mean_brier=float(np.mean(fold_sq_err)),
                )
            )

    if not cal_p:
        raise ValueError("backtest produced no out-of-sample predictions")

    per_threshold = [
        threshold_metrics(labels_by_thresh[k], probs_by_thresh[k], threshold=thresholds[k])
        for k in range(n_thresh)
    ]
    calibration = calibration_report(cal_p, cal_y, n_bins=n_bins)

    n_predictions = len(probs_by_thresh[0])
    mean_brier = float(np.mean([m.brier for m in per_threshold]))
    mean_log_loss = float(np.mean([m.log_loss for m in per_threshold]))
    test_idxs = [t for f in folds for t in f.test_as_of]

    log.info(
        "backtest.done",
        ticker=series.ticker,
        model=model_name,
        horizon=horizon_days,
        folds=len(folds),
        predictions=n_predictions,
        mean_brier=round(mean_brier, 4),
        ece=round(calibration.ece, 4),
    )

    return BacktestResult(
        ticker=series.ticker,
        horizon_days=horizon_days,
        model_name=model_name,
        n_folds=len(fold_summaries),
        n_predictions=n_predictions,
        as_of_start=dates[min(test_idxs)],
        as_of_end=dates[max(test_idxs)],
        thresholds=per_threshold,
        calibration=calibration,
        mean_brier=mean_brier,
        mean_log_loss=mean_log_loss,
        folds=fold_summaries,
        seed=seed,
        notes=notes,
    )
