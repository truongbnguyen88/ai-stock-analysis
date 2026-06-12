"""Golden-value tests for the overnight/intraday return decomposition."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.returns import intraday_return, overnight_return


def test_overnight_return_close_to_open() -> None:
    # open=[_,110,121], prev_close=[100,110]; overnight_t = O_t/C_{t-1}-1
    open_ = pd.Series([100.0, 110.0, 121.0])
    close = pd.Series([100.0, 110.0, 120.0])
    out = overnight_return(open_, close).tolist()
    assert math.isnan(out[0])  # no prior close
    assert out[1] == pytest.approx(0.10)  # 110/100 - 1
    assert out[2] == pytest.approx(0.10)  # 121/110 - 1


def test_intraday_return_open_to_close() -> None:
    open_ = pd.Series([100.0, 110.0])
    close = pd.Series([105.0, 99.0])
    out = intraday_return(open_, close).tolist()
    assert out[0] == pytest.approx(0.05)  # 105/100 - 1
    assert out[1] == pytest.approx(-0.10)  # 99/110 - 1


def test_session_split_composes_to_daily() -> None:
    # (1+overnight)*(1+intraday) == C_t / C_{t-1} == 1 + daily_return.
    open_ = pd.Series([100.0, 108.0, 103.0])
    close = pd.Series([100.0, 105.0, 110.0])
    on = overnight_return(open_, close)
    intra = intraday_return(open_, close)
    composed = (1 + on) * (1 + intra)
    daily = close / close.shift(1)
    assert composed.iloc[1] == pytest.approx(daily.iloc[1])
    assert composed.iloc[2] == pytest.approx(daily.iloc[2])
