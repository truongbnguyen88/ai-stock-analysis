"""Volume / liquidity indicators — scale-free, point-in-time.

Raw share volume is not comparable across tickers (it scales with shares
outstanding and price), so every feature here is a *normalization* of volume
against its own recent history — a ratio or z-score — making it valid for the
pooled cross-ticker model. All windows are trailing (no lookahead).
"""

from __future__ import annotations

import pandas as pd


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its own trailing average: ``v_t / mean(v, window)``.

    > 1 = today trades above its recent norm (participation surge), < 1 = quiet.
    Scale-free (a ratio against the ticker's own baseline). NaN until ``window``
    observations are available. The trailing mean excludes the current bar to
    keep the ratio a clean "vs. recent history" comparison (and avoid the bar
    diluting its own baseline).
    """
    baseline = volume.shift(1).rolling(window=window, min_periods=window).mean()
    return volume / baseline


def dollar_volume_zscore(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of dollar volume (``close * volume``) vs its trailing distribution.

    Dollar volume proxies traded *notional* (liquidity). The z-score against the
    trailing window makes it scale-free and captures liquidity surprises
    (abnormally heavy/light trading) rather than the level. Trailing stats exclude
    the current bar (``shift(1)``) so the bar is scored against its own past.
    NaN until ``window`` prior observations exist or when the trailing std is 0.
    """
    dollar_vol = close * volume
    prior = dollar_vol.shift(1)
    mean = prior.rolling(window=window, min_periods=window).mean()
    std = prior.rolling(window=window, min_periods=window).std(ddof=1)
    return (dollar_vol - mean) / std
