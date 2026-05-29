"""Trend indicators: simple moving averages and trend-regime flags.

Trend flags are deterministic, documented heuristics for *description* only —
they are not trading signals and the report never derives buy/sell calls from
them (non-advisory invariant).
"""

from __future__ import annotations

import pandas as pd

# Default MA windows required by the MVP spec.
DEFAULT_WINDOWS: tuple[int, ...] = (20, 50, 200)


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until ``window`` observations are available."""
    return close.rolling(window=window, min_periods=window).mean()


def moving_averages(close: pd.Series, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Return a DataFrame with one ``ma{w}`` column per requested window."""
    return pd.DataFrame({f"ma{w}": sma(close, w) for w in windows})
