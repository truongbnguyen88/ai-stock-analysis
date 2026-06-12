"""Golden-value tests for moving averages."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.trend import moving_averages, pct_from_high, sma


def test_sma_window_3() -> None:
    out = sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), window=3).tolist()
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_moving_averages_columns_and_nan_when_short() -> None:
    df = moving_averages(pd.Series([float(i) for i in range(1, 11)]))  # 10 points
    assert list(df.columns) == ["ma20", "ma50", "ma200"]
    assert bool(df["ma20"].isna().all())  # < 20 points -> all NaN


def test_pct_from_high_is_zero_at_new_high_and_negative_below() -> None:
    # Rising then pulling back: at each new high pct==0; on the pullback it's < 0.
    close = pd.Series([10.0, 11.0, 12.0, 9.0])
    out = pct_from_high(close, window=252, min_periods=1).tolist()
    assert out[:3] == pytest.approx([0.0, 0.0, 0.0])  # each bar is a fresh high
    assert out[3] == pytest.approx(9.0 / 12.0 - 1.0)  # -0.25 off the peak of 12


def test_pct_from_high_window_caps_lookback() -> None:
    # window=2 only "remembers" the last 2 bars: after a high rolls out of the
    # window, the anchor resets to the recent local max.
    close = pd.Series([20.0, 5.0, 6.0])
    out = pct_from_high(close, window=2, min_periods=1).tolist()
    # t2 window = {5,6} → high 6 (the 20 has rolled off) → 6/6 - 1 = 0
    assert out[2] == pytest.approx(0.0)


def test_pct_from_high_min_periods_warmup() -> None:
    out = pct_from_high(pd.Series([10.0, 11.0]), window=252, min_periods=3).tolist()
    assert all(math.isnan(x) for x in out)  # fewer than min_periods bars
