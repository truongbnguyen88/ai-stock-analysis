"""Momentum indicators: RSI (Wilder) and MACD.

Conventions:
- RSI period 14, Wilder's smoothing (see ``_smoothing.wilder_smooth``).
- MACD = EMA(fast) - EMA(slow); signal = EMA(signal) of MACD; histogram = MACD - signal.
  EMAs use ``adjust=False`` (recursive form), matching standard charting tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.indicators._smoothing import wilder_smooth


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index in [0, 100] (NaN before the Wilder seed).

    Conventions at the boundaries:
    - no losses in the window  -> RSI = 100
    - no gains and no losses    -> RSI = 50 (flat)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)  # keeps NaN at index 0 (from diff)
    loss = (-delta).clip(lower=0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)

    # Boundary handling where rs is 0/0 or x/0.
    out = out.where(avg_loss != 0, 100.0)  # avg_loss==0 (and some gain) -> 100
    flat = (avg_gain == 0) & (avg_loss == 0)
    out = out.mask(flat, 50.0)
    # Preserve NaN before the seed (avg_gain is NaN there).
    out = out.mask(avg_gain.isna(), np.nan)
    return out


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive, ``adjust=False``)."""
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Return MACD components as columns: ``macd``, ``signal``, ``histogram``."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})
