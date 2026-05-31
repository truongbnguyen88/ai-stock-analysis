"""VIX market-context feature: population, alignment, and graceful absence."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stock_agent.features.price_features import PRICE_FEATURE_COLS, build_price_feature_matrix
from stock_agent.schemas.market import PriceBar, PriceSeries

_START = date(2024, 1, 1)


def _series(n: int = 60) -> PriceSeries:
    bars = [
        PriceBar(date=_START + timedelta(days=i), open=100.0, high=101.0, low=99.0, close=100.0)
        for i in range(n)
    ]
    return PriceSeries(ticker="TST", bars=bars)


def _vix(n: int = 60, level: float = 20.0) -> pd.Series:
    idx = pd.DatetimeIndex([_START + timedelta(days=i) for i in range(n)])
    return pd.Series([level] * n, index=idx, dtype=float)


def test_vix_columns_are_in_the_feature_set() -> None:
    assert "vix_level" in PRICE_FEATURE_COLS
    assert "vix_rel" in PRICE_FEATURE_COLS


def test_vix_populates_and_scales() -> None:
    df = build_price_feature_matrix(_series(60), vix=_vix(60, level=20.0))
    # vix_level = VIX / 100 (annualized vol fraction).
    assert df["vix_level"].iloc[-1] == pytest.approx(0.20)
    # Constant VIX → it equals its own 20-day average → vix_rel == 1 (after warmup).
    assert df["vix_rel"].iloc[-1] == pytest.approx(1.0)
    # vix_rel needs the 20-day window → NaN before it fills.
    assert pd.isna(df["vix_rel"].iloc[5])
    assert int(df["vix_level"].isna().sum()) == 0  # level needs no warmup


def test_vix_absent_is_nan_not_an_error() -> None:
    df = build_price_feature_matrix(_series(60))  # no vix supplied
    assert df["vix_level"].isna().all()
    assert df["vix_rel"].isna().all()
