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
