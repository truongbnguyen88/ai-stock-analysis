"""Volatility and risk indicators: historical volatility, ATR, drawdowns.

Conventions:
- Historical volatility: rolling sample std (ddof=1) of LOG returns, annualized by
  sqrt(periods_per_year) with a 252 trading-day year.
- ATR: Wilder's smoothing of True Range.
- Drawdown: close / running-max - 1 (<= 0); max drawdown is its minimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.indicators._smoothing import wilder_smooth
from stock_agent.indicators.returns import log_returns

TRADING_DAYS_PER_YEAR = 252


def historical_volatility(
    close: pd.Series, window: int = 20, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized rolling volatility (std of log returns). NaN until ``window`` returns."""
    returns = log_returns(close)
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    First bar has no previous close, so TR reduces to high-low there (NaN terms
    are skipped by the row-wise max).
    """
    prev_close = close.shift(1)
    components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return components.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range via Wilder's smoothing of True Range."""
    return wilder_smooth(true_range(high, low, close), period)


def bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B: where the close sits inside its Bollinger Bands.

    Bands = SMA(window) ± num_std · rolling sample std (ddof=1).
    %B = (close - lower) / (upper - lower): ~0 at the lower band, ~1 at the upper
    band, and outside [0, 1] on breakouts. Scale-free (a position ratio), so it is
    safe for pooled cross-ticker models. NaN until ``window`` observations, and
    NaN when the band has zero width (flat prices).
    """
    mid = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=1)
    lower = mid - num_std * sd
    width = 2.0 * num_std * sd  # = upper - lower
    return (close - lower) / width


def drawdown_series(close: pd.Series) -> pd.Series:
    """Drawdown at each point relative to the running peak (values <= 0)."""
    running_max = close.cummax()
    return close / running_max - 1.0


def max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough decline over the series (<= 0)."""
    series = drawdown_series(close)
    return float(series.min()) if len(series) else 0.0
