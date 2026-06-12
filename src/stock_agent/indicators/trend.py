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


def pct_from_high(close: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    """Signed distance from the trailing rolling high: ``C_t / max(C, window) - 1``.

    Always <= 0 (the running max includes the current bar). Proxies the
    52-week-high "nearness to high" anchoring momentum effect (George & Hwang
    2004): stocks near their high tend to continue. ``window`` caps the lookback
    at ~one trading year; ``min_periods`` lets it populate early using the
    max-so-far (the anchor is approximate by construction). Scale-free, trailing.
    """
    rolling_high = close.rolling(window=window, min_periods=min_periods).max()
    return close / rolling_high - 1.0
