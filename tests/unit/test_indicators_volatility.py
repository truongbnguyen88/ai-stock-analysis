"""Golden-value tests for volatility / risk indicators.

ATR (period=2) hand-derivation:
  high=[10.5,11.5,12.5] low=[9.5,10.5,11.5] close=[10,11,12]
  TR=[1.0, 1.5, 1.5]; Wilder seed at idx1 = mean(1.0,1.5)=1.25;
  idx2 = (1.25*1 + 1.5)/2 = 1.375
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.volatility import (
    atr,
    drawdown_series,
    historical_volatility,
    max_drawdown,
)


def test_drawdown() -> None:
    series = pd.Series([10.0, 12.0, 6.0, 9.0])
    assert drawdown_series(series).tolist() == pytest.approx([0.0, 0.0, -0.5, -0.25])
    assert max_drawdown(series) == pytest.approx(-0.5)


def test_atr_wilder() -> None:
    high = pd.Series([10.5, 11.5, 12.5])
    low = pd.Series([9.5, 10.5, 11.5])
    close = pd.Series([10.0, 11.0, 12.0])
    out = atr(high, low, close, period=2).tolist()
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(1.25)
    assert out[2] == pytest.approx(1.375)


def test_historical_volatility_annualized() -> None:
    # log returns = [nan, 0.1, 0.2]; window=2 std(ddof=1) of [0.1,0.2] = 0.0707106781
    close = pd.Series([1.0, math.exp(0.1), math.exp(0.3)])
    out = historical_volatility(close, window=2).tolist()
    assert math.isnan(out[0]) and math.isnan(out[1])
    expected = 0.07071067811865472 * math.sqrt(252)
    assert out[2] == pytest.approx(expected, rel=1e-9)
