"""Opt-in candidate feature groups: backward-compat + correct wiring.

The central guarantee is BC: with no ``feature_groups`` the matrix is byte-for-byte
the production baseline, so committed model artifacts are unaffected. Groups only
appear when explicitly requested.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.price_features import (
    FEATURE_GROUPS,
    PRICE_FEATURE_COLS,
    build_price_feature_matrix,
    resolve_feature_cols,
)
from stock_agent.schemas.market import PriceBar, PriceSeries

_START = date(2024, 1, 1)


def _series(n: int = 300, seed: int = 0) -> PriceSeries:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    bars = [
        PriceBar(
            date=_START + timedelta(days=i),
            open=float(close[i] * (1.0 + rng.normal(0.0, 0.002))),
            high=float(close[i] * 1.01),
            low=float(close[i] * 0.99),
            close=float(close[i]),
            volume=int(1_000_000 + rng.integers(0, 500_000)),
        )
        for i in range(n)
    ]
    return PriceSeries(ticker="TST", bars=bars)


def _market(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n)))
    idx = pd.DatetimeIndex([_START + timedelta(days=i) for i in range(n)])
    return pd.Series(close, index=idx, dtype=float)


def test_default_matrix_is_baseline_only_bc() -> None:
    df = build_price_feature_matrix(_series())
    assert list(df.columns) == PRICE_FEATURE_COLS  # no extra columns leak in


def test_resolve_feature_cols_appends_in_order() -> None:
    cols = resolve_feature_cols(["high52w", "relstr"])
    assert cols == [*PRICE_FEATURE_COLS, "pct_from_52w_high",
                    "rel_strength_20d", "rel_strength_60d"]


# The Phase 1.6 ablation promoted these into the always-computed baseline.
PROMOTED_COLS = [
    "rvol_20", "dollar_vol_z_20", "overnight_ret_20d",
    "intraday_ret_20d", "realized_skew_60", "semivol_ratio_60",
]


def test_promoted_features_are_in_baseline() -> None:
    # Promoted candidates now live in the default matrix (no feature_groups needed),
    # and are NOT opt-in groups anymore.
    df = build_price_feature_matrix(_series())
    for col in PROMOTED_COLS:
        assert col in df.columns and df[col].notna().any(), col
    assert not ({"volume", "session", "shape"} & set(FEATURE_GROUPS))


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="unknown feature group"):
        build_price_feature_matrix(_series(60), feature_groups=["bogus"])


def _insider_frame(series: PriceSeries, n_filings: int = 8) -> pd.DataFrame:
    # Sparse Form 4 activity on in-range trading dates; buy-heavy (6 buys, 2 sells)
    # with one senior buy, so all three re-engineered columns are nonzero.
    dates = [pd.Timestamp(b.date) for b in series.bars[40:40 + n_filings]]
    is_buy = [i < 6 for i in range(n_filings)]
    return pd.DataFrame(
        {
            "buy_conviction": [0.05 if b else 0.0 for b in is_buy],
            "senior_buy_n": [1.0 if i == 0 else 0.0 for i in range(n_filings)],
            "sell_pressure": [0.0 if b else 0.10 for b in is_buy],
        },
        index=pd.DatetimeIndex(dates),
    )


@pytest.mark.parametrize("group", sorted(FEATURE_GROUPS))
def test_group_adds_exactly_its_columns(group: str) -> None:
    series = _series()
    df = build_price_feature_matrix(
        series,
        feature_groups=[group],
        market=_market() if group == "relstr" else None,
        insider=_insider_frame(series) if group == "insider" else None,
    )
    assert list(df.columns) == [*PRICE_FEATURE_COLS, *FEATURE_GROUPS[group]]
    # The new columns must populate (not be entirely NaN) on a 300-bar series.
    for col in FEATURE_GROUPS[group]:
        assert df[col].notna().any(), f"{col} never populated"


def test_insider_is_nan_without_frame() -> None:
    df = build_price_feature_matrix(_series(), feature_groups=["insider"])  # no insider=
    for col in FEATURE_GROUPS["insider"]:
        assert df[col].isna().all()


def test_insider_features_populate_and_are_price_level_invariant() -> None:
    # The re-engineered insider features are Δ-ownership / role based — purely from the
    # Form 4 frame — so they're invariant to the stock's price level by construction.
    base = _series(seed=11)
    scaled = PriceSeries(
        ticker="TST",
        bars=[
            PriceBar(date=b.date, open=b.open * 10, high=b.high * 10, low=b.low * 10,
                     close=b.close * 10, volume=b.volume)
            for b in base.bars
        ],
    )
    ins = _insider_frame(base)  # filings on bars[40:48]
    # Evaluate at row 70: the trailing 63-day window still contains the filings.
    a = build_price_feature_matrix(base, feature_groups=["insider"], insider=ins).iloc[70]
    b = build_price_feature_matrix(scaled, feature_groups=["insider"], insider=ins).iloc[70]
    # Trailing sums of the in-window filings: 6 buys × 0.05, 1 senior buy, 2 sells × 0.10.
    assert a["insider_buy_conviction_63d"] == pytest.approx(6 * 0.05)
    assert a["insider_senior_buy_63d"] == pytest.approx(1.0)
    assert a["insider_sell_pressure_63d"] == pytest.approx(2 * 0.10)
    for col in FEATURE_GROUPS["insider"]:  # identical at 10x price level
        assert a[col] == pytest.approx(b[col], rel=1e-9)


def test_relstr_is_nan_without_market() -> None:
    df = build_price_feature_matrix(_series(), feature_groups=["relstr"])  # market omitted
    assert df["rel_strength_20d"].isna().all()
    assert df["rel_strength_60d"].isna().all()


def test_candidate_columns_are_scale_free_across_price_level() -> None:
    # Same return path, 10x price level: scale-free features must be ~identical.
    # Covers the promoted baseline features + the remaining high52w group.
    base = _series(seed=7)
    scaled_bars = [
        PriceBar(
            date=b.date, open=b.open * 10, high=b.high * 10, low=b.low * 10,
            close=b.close * 10, volume=b.volume,
        )
        for b in base.bars
    ]
    scaled = PriceSeries(ticker="TST", bars=scaled_bars)
    cols = [*PROMOTED_COLS, *FEATURE_GROUPS["high52w"]]
    a = build_price_feature_matrix(base, feature_groups=["high52w"]).iloc[-1]
    b = build_price_feature_matrix(scaled, feature_groups=["high52w"]).iloc[-1]
    for col in cols:
        assert a[col] == pytest.approx(b[col], rel=1e-9), f"{col} not scale-free"
