"""yfinance DataFrame -> PriceSeries normalization (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_agent.providers.base import SymbolNotFound
from stock_agent.providers.yfinance_provider import _frame_to_series


def _frame(data: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def test_maps_rows_and_prefers_adj_close() -> None:
    frame = _frame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 12.5],
            "Low": [9.0, 10.5],
            "Close": [11.0, 12.0],
            "Adj Close": [10.8, 11.9],
            "Volume": [1000, 2000],
        },
        ["2025-01-02", "2025-01-03"],
    )
    series = _frame_to_series("ABC", frame)
    assert series.ticker == "ABC"
    assert len(series) == 2
    assert series.bars[0].date == date(2025, 1, 2)
    assert series.closes == [10.8, 11.9]  # adjusted preferred


def test_without_adj_close_falls_back_to_close() -> None:
    frame = _frame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [500]},
        ["2025-01-02"],
    )
    series = _frame_to_series("ABC", frame)
    assert series.bars[0].adj_close is None
    assert series.closes == [10.5]


def test_empty_frame_raises_symbol_not_found() -> None:
    with pytest.raises(SymbolNotFound):
        _frame_to_series("ABC", pd.DataFrame())


def test_nan_rows_are_skipped() -> None:
    frame = _frame(
        {
            "Open": [10.0, np.nan],
            "High": [11.0, 1.0],
            "Low": [9.0, 1.0],
            "Close": [10.5, np.nan],
            "Volume": [500, 0],
        },
        ["2025-01-02", "2025-01-03"],
    )
    series = _frame_to_series("ABC", frame)
    assert len(series) == 1
