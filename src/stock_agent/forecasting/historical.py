"""Historical-simulation baseline forecaster.

Estimates the forward-return distribution empirically from the stock's own
history: it samples every overlapping ``horizon``-day simple return and reads the
scenario probabilities, expected return, VaR, and a predictive interval directly
off that sample. No distributional assumptions — a strong, hard-to-beat baseline
and the reference against which Phase 5 models are judged.

Caveat: overlapping windows induce autocorrelation (effective sample size < count),
so this understates tail uncertainty somewhat; acceptable for a baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

import numpy as np

from stock_agent.forecasting.buckets import (
    bucket_probabilities,
    buckets_for_horizon,
    make_prob_buckets,
)
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

_MODEL_NAME = "historical_sim"
_MIN_SAMPLES = 30  # below this, flag low confidence


def _horizon_returns(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Overlapping ``horizon``-day simple returns P_{t+h}/P_t - 1."""
    if horizon < 1 or len(closes) <= horizon:
        return np.empty(0, dtype=float)
    return np.asarray(closes[horizon:] / closes[:-horizon] - 1.0, dtype=float)


def sample_to_forecast(
    sample: np.ndarray,
    *,
    ticker: str,
    as_of: Date,
    horizon_days: int,
    model_name: str,
    min_samples: int = _MIN_SAMPLES,
) -> ScenarioForecast:
    """Build a ``ScenarioForecast`` from any sample of forward simple returns.

    Shared by all forecasters (historical sim, Monte Carlo, ML) so the output
    shape is identical and models are directly comparable.
    """
    n = len(sample)
    if n == 0:
        raise ValueError(f"empty sample; cannot build forecast for horizon {horizon_days}")

    notes = None
    if n < min_samples:
        notes = f"Only {n} samples; estimates are low-confidence."

    return ScenarioForecast(
        ticker=ticker,
        as_of=as_of,
        horizon_days=horizon_days,
        model_name=model_name,
        buckets=make_prob_buckets(
            bucket_probabilities(sample, buckets_for_horizon(horizon_days)),
            buckets_for_horizon(horizon_days),
        ),
        expected_return=float(sample.mean()),
        upside_prob=float((sample > 0).mean()),
        downside_prob=float((sample < 0).mean()),
        # 5th / 1st percentile as (typically negative) VaR levels.
        var_95=float(np.quantile(sample, 0.05)),
        var_99=float(np.quantile(sample, 0.01)),
        ci_level=0.90,
        ci_low=float(np.quantile(sample, 0.05)),
        ci_high=float(np.quantile(sample, 0.95)),
        calibration_status="unknown",
        notes=notes,
    )


def historical_forecast(
    closes: Sequence[float],
    horizon_days: int,
    *,
    ticker: str,
    as_of: Date,
    min_samples: int = _MIN_SAMPLES,
) -> ScenarioForecast:
    """Build a ``ScenarioForecast`` from the empirical horizon-return distribution."""
    arr = np.asarray(closes, dtype=float)
    sample = _horizon_returns(arr, horizon_days)
    if len(sample) == 0:
        raise ValueError(f"insufficient price history ({len(arr)} bars) for horizon {horizon_days}")
    return sample_to_forecast(
        sample,
        ticker=ticker,
        as_of=as_of,
        horizon_days=horizon_days,
        model_name=_MODEL_NAME,
        min_samples=min_samples,
    )


class HistoricalSimulation:
    """``ForecastModel`` using empirical historical horizon returns."""

    name = _MODEL_NAME

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        return historical_forecast(
            series.closes,
            horizon_days,
            ticker=series.ticker,
            as_of=as_of or series.bars[-1].date,
        )
