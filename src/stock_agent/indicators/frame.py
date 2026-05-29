"""Adapter: PriceSeries (domain model) -> pandas DataFrame for indicator math.

Keeps the schema layer pandas-free while giving the vectorized indicator
functions the DataFrame they expect. This is the single conversion boundary.
"""

from __future__ import annotations

import pandas as pd

from stock_agent.schemas.market import PriceSeries


def to_ohlcv_frame(series: PriceSeries) -> pd.DataFrame:
    """Build a date-indexed OHLCV DataFrame from a ``PriceSeries``.

    Columns: open, high, low, close, adj_close (may be NaN), volume.
    Index is a ``DatetimeIndex`` in ascending order (guaranteed by the schema).
    """
    index = pd.DatetimeIndex([b.date for b in series.bars])
    frame = pd.DataFrame(
        {
            "open": [b.open for b in series.bars],
            "high": [b.high for b in series.bars],
            "low": [b.low for b in series.bars],
            "close": [b.close for b in series.bars],
            "adj_close": [b.adj_close for b in series.bars],
            "volume": [b.volume for b in series.bars],
        },
        index=index,
    )
    # All-None adj_close would infer an object dtype (breaks numeric ufuncs like
    # np.log); coerce to float64 with None -> NaN.
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    return frame


def adjusted_close(frame: pd.DataFrame) -> pd.Series:
    """Analysis close: adjusted close where present, else raw close.

    Mirrors ``PriceSeries.closes`` so indicator outputs are consistent with the
    rest of the system.
    """
    return frame["adj_close"].fillna(frame["close"])
