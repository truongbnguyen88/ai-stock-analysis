"""Tests for the high-level indicator snapshot (PriceSeries -> latest values)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.schemas.market import PriceBar, PriceSeries


def _series(closes: list[float]) -> PriceSeries:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i), open=c, high=c + 0.5, low=c - 0.5, close=c
        )
        for i, c in enumerate(closes)
    ]
    return PriceSeries(ticker="T", bars=bars)


def test_uptrend_with_partial_history() -> None:
    closes = [100.0 + i for i in range(60)]  # strictly increasing, 60 bars
    snap = compute_snapshot(_series(closes))

    assert snap.ma20 is not None and snap.ma50 is not None
    assert snap.ma200 is None  # only 60 bars (< 200)
    assert snap.ma20 == pytest.approx(sum(closes[-20:]) / 20)
    assert snap.trend_label == "uptrend"
    assert snap.price_above_ma50 is True
    assert snap.ma50_above_ma200 is None
    assert snap.max_drawdown == pytest.approx(0.0)  # monotonic increase

    numeric = snap.numeric_indicators()
    assert "ma200" not in numeric  # None values excluded
    assert "ma50" in numeric and "last_close" in numeric


def test_insufficient_history() -> None:
    snap = compute_snapshot(_series([10.0, 11.0, 12.0, 11.0, 10.0]))
    assert snap.ma50 is None
    assert snap.rsi14 is None  # needs >= period (14)
    assert snap.trend_label == "insufficient_data"
    assert "last_close" in snap.numeric_indicators()
