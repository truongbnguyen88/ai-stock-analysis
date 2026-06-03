"""Regime-conditional forecaster (Task 8 spike): conditioning, determinism, fallback."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import historical_forecast
from stock_agent.forecasting.regime import RegimeForecaster
from stock_agent.schemas.market import PriceBar, PriceSeries


def _series_from_logrets(logrets: np.ndarray, *, ticker: str = "TST") -> PriceSeries:
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], logrets])))
    bars = [
        PriceBar(
            date=date(2018, 1, 1) + timedelta(days=i),
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
        )
        for i, c in enumerate(closes)
    ]
    return PriceSeries(ticker=ticker, bars=bars)


def _calm_then_volatile() -> PriceSeries:
    # 450 calm days then 150 high-vol days: two clean regimes, ending in high-vol.
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0, 0.004, 450)
    vol = rng.normal(0.0, 0.030, 150)
    return _series_from_logrets(np.concatenate([calm, vol]))


def test_regime_is_forecastmodel() -> None:
    assert isinstance(RegimeForecaster(), ForecastModel)


def test_regime_conditioning_engages_and_is_deterministic() -> None:
    series = _calm_then_volatile()
    fc1 = RegimeForecaster(n_states=2).forecast(series, horizon_days=20)
    fc2 = RegimeForecaster(n_states=2).forecast(series, horizon_days=20)

    assert fc1.model_name == "regime_hmm"
    assert sum(b.probability for b in fc1.buckets) == 1.0 or abs(
        sum(b.probability for b in fc1.buckets) - 1.0
    ) < 1e-6
    # The regime path engaged (did not fall back to unconditional history).
    assert fc1.notes is not None and "Conditioned on regime" in fc1.notes
    # Fixed random_state → bit-identical reruns.
    assert fc1 == fc2


def test_regime_in_high_vol_widens_tails_vs_unconditional() -> None:
    # Ending in the high-vol regime, the conditioned forward-return distribution
    # should be wider than the unconditional history (which is mostly calm days).
    series = _calm_then_volatile()
    regime_fc = RegimeForecaster(n_states=2).forecast(series, horizon_days=20)
    uncond = historical_forecast(
        series.closes, 20, ticker=series.ticker, as_of=series.bars[-1].date
    )
    regime_width = regime_fc.ci_high - regime_fc.ci_low  # type: ignore[operator]
    uncond_width = uncond.ci_high - uncond.ci_low  # type: ignore[operator]
    assert regime_width > uncond_width


def test_regime_falls_back_on_short_history() -> None:
    # Below min_fit_bars → graceful unconditional fallback (still a valid forecast).
    rng = np.random.default_rng(1)
    series = _series_from_logrets(rng.normal(0.0, 0.01, 120))
    fc = RegimeForecaster().forecast(series, horizon_days=20)
    assert fc.model_name == "regime_hmm"
    assert fc.notes is not None and "Regime fallback" in fc.notes
    assert abs(sum(b.probability for b in fc.buckets) - 1.0) < 1e-6
