"""Wilder's smoothing (RMA) — shared by RSI and ATR.

Wilder's moving average seeds with the simple mean of the first ``period`` valid
observations, then applies the recursive update::

    a_i = (a_{i-1} * (period - 1) + x_i) / period

Values before the seed index are NaN. This is the canonical definition used by
most reference implementations (TA-Lib, Wilder's original), chosen here so our
RSI/ATR match standard tools and our golden tests are unambiguous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_smooth(x: pd.Series, period: int) -> pd.Series:
    """Return Wilder's smoothed series of ``x`` (NaN-padded before the seed).

    Handles leading NaNs (e.g. RSI gains start at index 1 because they derive
    from a diff): the seed window begins at the first valid observation.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    arr = x.to_numpy(dtype="float64")
    out = np.full(arr.shape, np.nan)

    valid = ~np.isnan(arr)
    if not valid.any():
        return pd.Series(out, index=x.index)

    first = int(np.argmax(valid))  # index of the first non-NaN value
    seed_idx = first + period - 1
    if seed_idx >= len(arr):
        return pd.Series(out, index=x.index)  # not enough data to seed

    # Seed = SMA of the first `period` valid observations.
    out[seed_idx] = float(np.nanmean(arr[first : first + period]))
    for i in range(seed_idx + 1, len(arr)):
        xi = arr[i]
        # Carry the previous average through any internal NaN (defensive; clean
        # series have none after the first valid index).
        out[i] = out[i - 1] if np.isnan(xi) else (out[i - 1] * (period - 1) + xi) / period

    return pd.Series(out, index=x.index)
