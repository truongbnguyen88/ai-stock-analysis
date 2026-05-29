"""Golden-value tests for moving averages."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.trend import moving_averages, sma


def test_sma_window_3() -> None:
    out = sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), window=3).tolist()
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_moving_averages_columns_and_nan_when_short() -> None:
    df = moving_averages(pd.Series([float(i) for i in range(1, 11)]))  # 10 points
    assert list(df.columns) == ["ma20", "ma50", "ma200"]
    assert bool(df["ma20"].isna().all())  # < 20 points -> all NaN
