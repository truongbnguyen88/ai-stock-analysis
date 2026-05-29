"""ML-based probabilistic forecasters.

Each ``MLForecaster`` trains one binary classifier per return-threshold boundary
on the stock's own price history (stock-specific, no cross-sectional data), then
combines the per-threshold probabilities into the six-bucket scenario distribution.

Architecture:
  - One classifier per threshold in [-0.10, -0.05, 0.00, +0.05, +0.10].
  - Training: price features only (historical news unavailable from free APIs).
  - Inference: price features from the current series.
  - Lazy training: ``forecast()`` triggers ``_fit()`` on the first call.
  - Isotonicity: P(return > θ) is enforced to decrease as θ increases by
    clipping; separate classifiers have no such guarantee.
  - Fallback: if fewer than ``min_train_rows`` usable rows exist, falls back
    to the historical-simulation baseline with a warning note.

Supported model types: "logistic", "xgboost", "lightgbm", "random_forest".
"""

from __future__ import annotations

from datetime import date as Date
from typing import Any, Literal

import numpy as np
import pandas as pd

from stock_agent.features.assembler import (
    THRESHOLDS,
    _target_col,
    build_training_matrix,
    current_features_vector,
)
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.logging_config import get_logger
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

ModelType = Literal["logistic", "xgboost", "lightgbm", "random_forest"]
_MIN_TRAIN_ROWS = 60  # below this, fall back to historical_sim


def _make_classifier(model_type: ModelType) -> Any:
    """Instantiate a fresh, unfitted classifier for the given type."""
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=42)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    raise ValueError(f"unknown model_type '{model_type}'")


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column mean (for sklearn models; XGBoost/LGBM handle NaN natively).

    ``keep_empty_features=True`` preserves all-NaN columns (e.g. ma50_to_ma200 on a
    short series) as zeros rather than dropping them, keeping the column count stable
    across training and inference.
    """
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    return pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)


def _exceedance_to_buckets(probs_gt: list[float]) -> list[ProbBucket]:
    """Convert P(return > θ) for each threshold into the six bucket probabilities.

    The thresholds partition the return space, so bucket probs are differences
    of consecutive exceedance probs. Isotonicity is enforced first (P(return > θ)
    must decrease as θ increases).

    Bucket order (matches DEFAULT_BUCKETS):
      P(< -10%) = 1 - P(> -10%)
      P(-10% to -5%) = P(> -10%) - P(> -5%)
      ...
      P(> +10%) = P(> +10%)
    """
    from stock_agent.forecasting.buckets import DEFAULT_BUCKETS

    # Enforce monotone decreasing exceedance (higher threshold → lower prob).
    monotone = list(probs_gt)
    for i in range(1, len(monotone)):
        monotone[i] = min(monotone[i], monotone[i - 1])

    # Build bucket probs from the six regions.
    bucket_probs = [
        1.0 - monotone[0],  # P(< -10%)
        monotone[0] - monotone[1],  # P(-10% to -5%)
        monotone[1] - monotone[2],  # P(-5% to 0%)
        monotone[2] - monotone[3],  # P(0% to +5%)
        monotone[3] - monotone[4],  # P(+5% to +10%)
        monotone[4],  # P(> +10%)
    ]
    # Clip to [0,1] and renormalize (floating-point and isotonic clipping may
    # produce tiny negatives or a sum slightly off 1).
    bucket_probs = [max(0.0, p) for p in bucket_probs]
    total = sum(bucket_probs)
    if total > 0:
        bucket_probs = [p / total for p in bucket_probs]

    return [
        ProbBucket(label=label, lower=lo, upper=hi, probability=p)
        for (label, lo, hi), p in zip(DEFAULT_BUCKETS, bucket_probs, strict=True)
    ]


class MLForecaster:
    """``ForecastModel`` using binary classifiers on price features.

    Trains lazily on the first ``forecast()`` call. The model is stock-specific:
    each call with a different series re-trains from scratch.
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        horizon_days: int = 20,
        min_train_rows: int = _MIN_TRAIN_ROWS,
    ) -> None:
        self.name = f"ml_{model_type}"
        self._model_type = model_type
        self._horizon_days = horizon_days
        self._min_train_rows = min_train_rows
        self._classifiers: dict[float, Any] = {}
        self._needs_imputation = model_type in ("logistic", "random_forest")

    def _fit(self, series: PriceSeries) -> str | None:
        """Train one classifier per threshold. Returns a note if quality is low."""
        try:
            X, y = build_training_matrix(series, self._horizon_days, min_rows=self._min_train_rows)
        except ValueError as exc:
            return str(exc)

        X_fit = _impute(X) if self._needs_imputation else X

        self._classifiers = {}
        for thresh in THRESHOLDS:
            col = _target_col(thresh)
            labels = y[col]
            # Skip thresholds where all labels are the same (classifier can't learn).
            if labels.nunique() < 2:
                log.debug("ml.skip_threshold", threshold=thresh, reason="single class")
                continue
            clf = _make_classifier(self._model_type)
            clf.fit(X_fit, labels)
            self._classifiers[thresh] = clf

        n_trained = len(self._classifiers)
        if n_trained < len(THRESHOLDS):
            return (
                f"{len(THRESHOLDS) - n_trained} thresholds skipped (single class in training data)."
            )
        return None

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        # Horizon mismatch: retrain if needed (the model bakes in a fixed horizon).
        if horizon_days != self._horizon_days:
            self._horizon_days = horizon_days
            self._classifiers = {}

        note = None
        if not self._classifiers:
            note = self._fit(series)

        as_of = as_of or series.bars[-1].date

        # Fall back to historical_sim when training failed.
        if not self._classifiers:
            reason = note or "insufficient training data"
            log.warning("ml.fallback", model=self.name, reason=reason)
            fc = HistoricalSimulation().forecast(series, horizon_days=horizon_days, as_of=as_of)
            fallback_note = f"[{self.name}] Fell back to historical_sim: {reason}"
            return fc.model_copy(update={"model_name": self.name, "notes": fallback_note})

        # Build inference features for the most recent bar.
        x_current = current_features_vector(series)
        x_df = pd.DataFrame([x_current])
        if self._needs_imputation:
            x_df = _impute(x_df)

        # Predict P(return > threshold) for each trained classifier.
        probs_gt: list[float] = []
        for thresh in THRESHOLDS:
            clf = self._classifiers.get(thresh)
            if clf is None:
                # Fallback for skipped thresholds: use historical base rate.
                closes = np.array(series.closes)
                fwd = closes[horizon_days:] / closes[:-horizon_days] - 1
                probs_gt.append(float((fwd > thresh).mean()) if len(fwd) > 0 else 0.5)
            else:
                probs_gt.append(float(clf.predict_proba(x_df)[0, 1]))

        buckets = _exceedance_to_buckets(probs_gt)
        expected_return = sum(
            # Midpoint of each bucket * its probability.
            _bucket_midpoint(b) * b.probability
            for b in buckets
        )

        return ScenarioForecast(
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
            buckets=buckets,
            expected_return=expected_return,
            upside_prob=sum(b.probability for b in buckets if (b.lower or 0.0) >= 0.0),
            downside_prob=sum(b.probability for b in buckets if (b.upper or 0.0) <= 0.0),
            calibration_status="unknown",
            notes=note,
        )


def _bucket_midpoint(b: ProbBucket) -> float:
    """Return a representative return for a bucket (midpoint or edge estimate)."""
    lo = b.lower if b.lower is not None else -0.20  # open left tail floor
    hi = b.upper if b.upper is not None else 0.20  # open right tail cap
    return (lo + hi) / 2.0
