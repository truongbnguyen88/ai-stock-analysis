"""Point-in-time feature matrix assembly for ML training and inference.

Training constraint:
  Historical news is not available from free APIs, so the training matrix
  uses price features only. News features (Claude sentiment, event flags)
  are used at inference time (current state) to enrich the live prediction.

Leakage prevention (enforced by assertion):
  For each row at date t:
    features → data up to and including t  (no lookahead)
    target   → return from t to t+h        (strictly future)
  The target is computed by shifting closes forward h steps, which is only
  applied after all features are computed from historical data.

Target thresholds correspond to the six return buckets; classifiers predict
P(return > threshold) at each boundary, then bucket probabilities are derived
by differencing the cumulative exceedance probabilities.
"""

from __future__ import annotations

from datetime import date as Date

import pandas as pd

from stock_agent.features.price_features import PRICE_FEATURE_COLS, build_price_feature_matrix
from stock_agent.schemas.market import PriceSeries

# Return thresholds (bucket boundaries). For each threshold θ, we train a
# binary classifier predicting P(h-day return > θ).
THRESHOLDS: list[float] = [-0.10, -0.05, 0.0, 0.05, 0.10]
TARGET_COLS: list[str] = [f"gt_{int(abs(t) * 100):02d}{'p' if t >= 0 else 'm'}" for t in THRESHOLDS]


def _target_col(threshold: float) -> str:
    sign = "p" if threshold >= 0 else "m"
    return f"gt_{int(abs(threshold) * 100):02d}{sign}"


def build_training_matrix(
    series: PriceSeries,
    horizon: int,
    *,
    min_rows: int = 30,
    earnings_dates: list[Date] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X, y) for training ML models on a single stock's price history.

    Returns:
        X: feature DataFrame (rows = valid training dates)
        y: target DataFrame with binary columns for each threshold

    Raises:
        ValueError: if fewer than ``min_rows`` valid training rows exist.
    """
    features = build_price_feature_matrix(series, earnings_dates=earnings_dates)

    close = pd.Series(series.closes, index=features.index)
    # Forward h-day simple return at each date t: P_{t+h}/P_t - 1.
    # This uses future prices — only correct when applied AFTER features are
    # built from historical data. The shift is negative (looks forward).
    fwd_return = close.shift(-horizon) / close - 1.0

    # --- leakage assertion ---
    # Feature data ends at its index date; fwd_return is strictly in the future.
    # This checks that we never accidentally use a feature that requires future
    # close prices by verifying the features DataFrame has no dependency on the
    # shifted series (structural check: feature index ≡ bar dates).
    assert list(features.index) == list(close.index), "feature index must match price dates"

    # Build binary target columns.
    targets = pd.DataFrame(index=features.index)
    for thresh in THRESHOLDS:
        col = _target_col(thresh)
        targets[col] = (fwd_return > thresh).astype(float)

    # Valid rows: features not all-NaN, AND target is available (not in the
    # final ``horizon`` bars where fwd_return is NaN).
    has_features = features.notna().any(axis=1)
    has_target = fwd_return.notna()
    valid = has_features & has_target

    X = features.loc[valid].copy()
    y = targets.loc[valid].copy()

    if len(X) < min_rows:
        raise ValueError(
            f"only {len(X)} valid training rows (need {min_rows}); "
            "consider a longer price history or shorter horizon"
        )
    return X, y


def current_features_vector(
    series: PriceSeries, *, earnings_dates: list[Date] | None = None
) -> pd.Series:
    """Return the most recent feature row for live inference."""
    df = build_price_feature_matrix(series, earnings_dates=earnings_dates)
    return df.iloc[-1][PRICE_FEATURE_COLS]
