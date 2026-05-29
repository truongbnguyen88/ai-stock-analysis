"""Golden-value tests for RSI and MACD.

RSI values are hand-derived with period=2 so the Wilder recursion is checkable:
  series [1,2,1,2]: gains=[-,1,0,1], losses=[-,0,1,0]
  seed at idx 2: avg_gain=mean(1,0)=0.5, avg_loss=mean(0,1)=0.5 -> RS=1 -> RSI=50
  idx 3: avg_gain=(0.5+1)/2=0.75, avg_loss=(0.5+0)/2=0.25 -> RS=3 -> RSI=75
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.indicators.momentum import ema, macd, rsi


def test_rsi_all_gains_is_100() -> None:
    out = rsi(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), period=2).tolist()
    assert all(v == pytest.approx(100.0) for v in out[2:])  # seeded from idx 2


def test_rsi_known_mixed_values() -> None:
    out = rsi(pd.Series([1.0, 2.0, 1.0, 2.0]), period=2).tolist()
    assert out[2] == pytest.approx(50.0)
    assert out[3] == pytest.approx(75.0)


def test_ema_adjust_false_seeds_with_first_value() -> None:
    out = ema(pd.Series([2.0, 2.0, 2.0]), span=2).tolist()
    assert out == pytest.approx([2.0, 2.0, 2.0])


def test_macd_constant_series_is_zero() -> None:
    df = macd(pd.Series([5.0] * 40))
    assert df["macd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert df["histogram"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_macd_histogram_identity() -> None:
    series = pd.Series([float(i) for i in range(1, 60)])
    df = macd(series)
    residual = (df["histogram"] - (df["macd"] - df["signal"])).abs().max()
    assert residual == pytest.approx(0.0, abs=1e-12)
