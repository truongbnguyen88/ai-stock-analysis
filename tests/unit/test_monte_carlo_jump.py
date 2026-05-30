"""Earnings-jump Monte Carlo: jump calibration, fallback notes, fatter tails."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from stock_agent.forecasting.monte_carlo import MonteCarlo, historical_earnings_moves
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings

_START = date(2024, 1, 1)


def _bars(closes: list[float]) -> PriceSeries:
    bars = [
        PriceBar(date=_START + timedelta(days=i), open=c, high=c * 1.02, low=c * 0.98, close=c)
        for i, c in enumerate(closes)
    ]
    return PriceSeries(ticker="NVDA", bars=bars)


def test_historical_earnings_moves_golden() -> None:
    # earnings on day 1: close-before=110, close-after=121 → log(121/110).
    series = _bars([100.0, 110.0, 121.0, 120.0])
    moves = historical_earnings_moves(series, [_START + timedelta(days=1)])
    assert len(moves) == 1
    assert moves[0] == pytest.approx(math.log(121.0 / 110.0))


def test_historical_earnings_moves_skips_future_dates() -> None:
    series = _bars([100.0, 101.0, 102.0])
    # A date after the last bar has no realized move → skipped.
    assert len(historical_earnings_moves(series, [_START + timedelta(days=99)])) == 0


def _series_with_earnings() -> tuple[PriceSeries, list[date]]:
    """150 bars: big post-earnings moves early, calm recent window."""
    earnings_idx = list(range(8, 90, 9))  # ~10 historical earnings
    closes: list[float] = []
    price = 100.0
    for i in range(150):
        if i > 0 and (i - 1) in earnings_idx:
            price *= 1.12 if i % 2 == 0 else 0.90  # large post-earnings move
        else:
            price *= 1.006 if i % 2 == 0 else 0.997  # small oscillation
        closes.append(price)
    dates = [_START + timedelta(days=idx) for idx in earnings_idx]
    upcoming = _START + timedelta(days=149 + 5)  # within a 20-trading-day horizon
    return _bars(closes), dates + [upcoming]


def _registry(earnings: list[date], series: PriceSeries) -> ProviderRegistry:
    # Serve both prices (the jump's calibration fetch) and earnings from the fake,
    # so the test exercises the real extended-history calibration path.
    fake = FakeProvider("fake", prices=series, earnings_dates=earnings)
    settings = Settings(
        _env_file=None,
        provider_price_priority="fake",
        provider_earnings_priority="fake",
    )
    return ProviderRegistry([fake], settings)


def _tail(fc: ScenarioForecast) -> float:
    return fc.buckets[0].probability + fc.buckets[-1].probability  # P(<-10%) + P(>+10%)


def test_jump_applied_and_fattens_tails() -> None:
    series, earnings = _series_with_earnings()
    reg = _registry(earnings, series)
    jump = MonteCarlo("jump", vol_window=30, n_paths=5000, registry=reg).forecast(
        series, horizon_days=20
    )
    gbm = MonteCarlo("gbm", vol_window=30, n_paths=5000).forecast(series, horizon_days=20)

    assert jump.notes is not None and "Earnings jump applied" in jump.notes
    assert sum(b.probability for b in jump.buckets) == pytest.approx(1.0, abs=1e-6)
    # The earnings jump must widen the distribution → more tail mass than plain GBM.
    assert _tail(jump) > _tail(gbm)


def test_jump_without_registry_falls_back_to_gbm() -> None:
    series, _ = _series_with_earnings()
    fc = MonteCarlo("jump", vol_window=30, n_paths=2000).forecast(series, horizon_days=20)
    assert fc.notes is not None and "GBM only" in fc.notes


def test_jump_with_earnings_beyond_horizon_falls_back() -> None:
    series, earnings = _series_with_earnings()
    # Replace the upcoming date with one far in the future.
    far = [d for d in earnings if d <= series.dates[-1]] + [_START + timedelta(days=149 + 200)]
    reg = _registry(far, series)
    fc = MonteCarlo("jump", vol_window=30, n_paths=2000, registry=reg).forecast(
        series, horizon_days=20
    )
    assert fc.notes is not None and "beyond" in fc.notes
