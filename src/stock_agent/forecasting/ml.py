"""ML-based probabilistic forecaster (pooled, price-only).

``MLForecaster`` loads a *pooled* model artifact — one binary classifier per
return-threshold, trained across a universe of tickers (see ``pooled.py`` and
``train_pooled.py``) — and applies it to the target ticker's current features.

Design:
  - PRICE-ONLY (Option A): news sentiment is display context, never a model input.
  - Pooled, not per-ticker: train once offline, persist, load at inference.
  - One classifier per threshold in [-0.10, -0.05, 0.00, +0.05, +0.10]; the
    per-threshold P(return > θ) are combined into the six-bucket distribution,
    with isotonicity enforced (P decreases as θ increases).
  - Fallback: if no trained artifact exists for the (model_type, horizon), falls
    back to the historical-simulation baseline with a note telling you to train.

Train an artifact with:  python -m stock_agent train --model xgboost --horizon 20
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from stock_agent.features.assembler import THRESHOLDS, current_features_vector
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.pooled import PooledModel, default_model_path
from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import get_settings

log = get_logger(__name__)

ModelType = Literal["logistic", "xgboost", "lightgbm", "random_forest"]


def _exceedance_to_buckets(probs_gt: list[float]) -> list[ProbBucket]:
    """Convert P(return > θ) for each threshold into the six bucket probabilities.

    The thresholds partition the return space, so bucket probs are differences of
    consecutive exceedance probs. Isotonicity is enforced first (P(return > θ)
    must decrease as θ increases).
    """
    from stock_agent.forecasting.buckets import DEFAULT_BUCKETS

    # Enforce monotone-decreasing exceedance (higher threshold → lower prob).
    monotone = list(probs_gt)
    for i in range(1, len(monotone)):
        monotone[i] = min(monotone[i], monotone[i - 1])

    bucket_probs = [
        1.0 - monotone[0],  # P(< -10%)
        monotone[0] - monotone[1],  # P(-10% to -5%)
        monotone[1] - monotone[2],  # P(-5% to 0%)
        monotone[2] - monotone[3],  # P(0% to +5%)
        monotone[3] - monotone[4],  # P(+5% to +10%)
        monotone[4],  # P(> +10%)
    ]
    # Clip and renormalize (isotonic clipping / float noise may drift off 1).
    bucket_probs = [max(0.0, p) for p in bucket_probs]
    total = sum(bucket_probs)
    if total > 0:
        bucket_probs = [p / total for p in bucket_probs]

    return [
        ProbBucket(label=label, lower=lo, upper=hi, probability=p)
        for (label, lo, hi), p in zip(DEFAULT_BUCKETS, bucket_probs, strict=True)
    ]


def _bucket_midpoint(b: ProbBucket) -> float:
    """A representative return for a bucket (midpoint, with tail floors/caps)."""
    lo = b.lower if b.lower is not None else -0.20
    hi = b.upper if b.upper is not None else 0.20
    return (lo + hi) / 2.0


class MLForecaster:
    """``ForecastModel`` that applies a pooled, price-only classifier ensemble."""

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        *,
        models_dir: Path | None = None,
        model: PooledModel | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self.name = f"ml_{model_type}"
        self._model_type = model_type
        self._models_dir = models_dir  # default resolved from settings at load time
        self._model = model  # optional preloaded artifact (tests / warm cache)
        self._registry = registry  # optional: fetch earnings dates for the feature

    def _earnings_dates(self, ticker: str) -> list[Date] | None:
        """Fetch the target ticker's earnings dates (or None if unavailable)."""
        if self._registry is None:
            return None
        try:
            return self._registry.get_earnings_dates(ticker)
        except ProviderError:
            return None

    def _resolve_model(self, horizon_days: int) -> PooledModel | None:
        """Return a pooled model for this horizon (cached, then disk), or None."""
        if self._model is not None and self._model.horizon_days == horizon_days:
            return self._model
        models_dir = self._models_dir or (Path(get_settings().output_dir) / "models")
        path = default_model_path(models_dir, self._model_type, horizon_days)
        if not path.exists():
            return None
        self._model = PooledModel.load(path)
        return self._model

    def _base_rate(self, series: PriceSeries, horizon_days: int, threshold: float) -> float:
        """Historical exceedance rate, used to fill a threshold with no classifier."""
        closes = np.array(series.closes)
        if len(closes) <= horizon_days:
            return 0.5
        fwd = closes[horizon_days:] / closes[:-horizon_days] - 1.0
        return float((fwd > threshold).mean())

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        as_of = as_of or series.bars[-1].date
        model = self._resolve_model(horizon_days)

        # No trained artifact → fall back to the historical baseline.
        if model is None:
            fc = HistoricalSimulation().forecast(series, horizon_days=horizon_days, as_of=as_of)
            note = (
                f"[{self.name}] no trained pooled model for horizon {horizon_days}; "
                f"run `python -m stock_agent train --model {self._model_type} "
                f"--horizon {horizon_days}`. Fell back to historical_sim."
            )
            log.warning("ml.no_artifact", model=self.name, horizon=horizon_days)
            return fc.model_copy(update={"model_name": self.name, "notes": note})

        # Inference: current features (incl. earnings) → per-threshold exceedance → buckets.
        earnings_dates = self._earnings_dates(series.ticker)
        x_df = pd.DataFrame([current_features_vector(series, earnings_dates=earnings_dates)])
        probs_raw = model.predict_exceedance(x_df)
        probs_gt = [
            p if p is not None else self._base_rate(series, horizon_days, thresh)
            for thresh, p in zip(THRESHOLDS, probs_raw, strict=True)
        ]

        buckets = _exceedance_to_buckets(probs_gt)
        expected_return = sum(_bucket_midpoint(b) * b.probability for b in buckets)
        note = f"Pooled {self._model_type}: {model.n_tickers} tickers, {model.n_train_rows} rows."

        return ScenarioForecast(
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
            buckets=buckets,
            expected_return=expected_return,
            # A bucket is upside if it lies entirely at/above 0 (lower >= 0) and
            # downside if entirely at/below 0 (upper <= 0). Guard against None so
            # the open tails are classified by their finite bound, not coalesced
            # to 0 (which would miscount < -10% as upside and > +10% as downside).
            upside_prob=sum(
                b.probability for b in buckets if b.lower is not None and b.lower >= 0.0
            ),
            downside_prob=sum(
                b.probability for b in buckets if b.upper is not None and b.upper <= 0.0
            ),
            calibration_status="unknown",  # calibrated in Phase 6
            notes=note,
        )
