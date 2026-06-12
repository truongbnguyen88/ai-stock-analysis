"""Return series: simple, log, and cumulative.

All functions are pure and vectorized over a close-price ``pd.Series`` and do not
look ahead (each value at t uses only data up to t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns r_t = P_t / P_{t-1} - 1 (first value NaN)."""
    # fill_method=None: do not forward-fill missing prices before differencing.
    return close.pct_change(fill_method=None)


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns ln(P_t / P_{t-1}) (first value NaN). Additive across time."""
    return np.log(close / close.shift(1))


def cumulative_returns(close: pd.Series) -> pd.Series:
    """Cumulative simple return from the first bar: (1 + r).cumprod() - 1."""
    return (1.0 + daily_returns(close)).cumprod() - 1.0


def overnight_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight (close-to-open) return: ``O_t / C_{t-1} - 1`` (first value NaN).

    Captures the gap accrued while the market is closed (reaction to after-hours
    news / overnight order flow). Documented to carry distinct drift vs. the
    intraday session. Scale-free; uses only data through t.
    """
    return open_ / close.shift(1) - 1.0


def intraday_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Intraday (open-to-close) return: ``C_t / O_t - 1``.

    The regular-session move. Together with ``overnight_return`` it decomposes
    the daily return into its two sessions, which often behave differently
    (overnight momentum vs. intraday mean-reversion). Scale-free.
    """
    return close / open_ - 1.0
