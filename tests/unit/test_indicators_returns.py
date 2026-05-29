"""Golden-value tests for return series."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.returns import cumulative_returns, daily_returns, log_returns


def test_daily_returns() -> None:
    out = daily_returns(pd.Series([10.0, 11.0, 12.0])).tolist()
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.1)
    assert out[2] == pytest.approx(1.0 / 11.0)


def test_log_returns() -> None:
    out = log_returns(pd.Series([10.0, 11.0, 12.0])).tolist()
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(math.log(1.1))
    assert out[2] == pytest.approx(math.log(12.0 / 11.0))


def test_cumulative_returns() -> None:
    out = cumulative_returns(pd.Series([10.0, 11.0, 12.0])).tolist()
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.2)  # 12/10 - 1
