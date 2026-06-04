"""Golden tests for the source-agnostic GDELT daily aggregation (Task 10).

These pin the exact semantics the BigQuery server-side GROUP BY must reproduce:
neutral-band tone signs, per-(date,ticker) and market-stream rollups, ddof=1 std
with a 0.0 single-article convention, and multi-ticker fan-out.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from stock_agent.news.aggregate import (
    MARKET_COLS,
    PER_TICKER_COLS,
    GkgRecord,
    aggregate,
    aggregate_market,
    aggregate_per_ticker,
)


def _rec(day: str, tone: float, tickers: tuple[str, ...] = (), **flags: bool) -> GkgRecord:
    dt = datetime.fromisoformat(f"{day}T12:00:00")
    return GkgRecord(dt=dt, tone=tone, tickers=tickers, **flags)


def test_per_ticker_counts_and_tone_sign() -> None:
    records = [
        _rec("2024-01-02", 3.0, ("NVDA",)),  # positive (> +1)
        _rec("2024-01-02", -4.0, ("NVDA",)),  # negative (< -1)
        _rec("2024-01-02", 0.5, ("NVDA",)),  # neutral (|tone| <= 1)
    ]
    df = aggregate_per_ticker(records)
    row = df.loc[(records[0].dt.date(), "NVDA")]
    assert row["article_count"] == 3.0
    assert row["pos_count"] == 1.0
    assert row["neg_count"] == 1.0
    # mean of [3, -4, 0.5] = -0.1667
    assert row["tone_mean"] == pd.Series([3.0, -4.0, 0.5]).mean()
    # sample std (ddof=1)
    assert abs(row["tone_std"] - pd.Series([3.0, -4.0, 0.5]).std(ddof=1)) < 1e-12


def test_single_article_std_is_zero_not_nan() -> None:
    df = aggregate_per_ticker([_rec("2024-01-02", 5.0, ("AAPL",))])
    assert df.loc[(df.index[0]), "tone_std"] == 0.0


def test_multi_ticker_article_counts_for_each() -> None:
    # One article mentioning two tickers contributes to both.
    df = aggregate_per_ticker([_rec("2024-01-02", 2.0, ("NVDA", "AMD"))])
    assert set(t for _, t in df.index) == {"NVDA", "AMD"}
    assert df["article_count"].sum() == 2.0


def test_untagged_articles_excluded_from_per_ticker() -> None:
    df = aggregate_per_ticker([_rec("2024-01-02", 2.0, ()), _rec("2024-01-02", 2.0, ("KO",))])
    assert list(df.index.get_level_values("ticker")) == ["KO"]


def test_market_stream_political_epu_presidential() -> None:
    records = [
        _rec("2024-01-02", 5.0, political=True),
        _rec("2024-01-02", -3.0, political=True, epu=True),
        _rec("2024-01-02", 1.5, political=True, presidential=True),
        _rec("2024-01-02", 9.0, tickers=("NVDA",)),  # non-political → ignored by market
    ]
    df = aggregate_market(records)
    row = df.loc[records[0].dt.date()]
    assert row["pol_article_count"] == 3.0
    assert row["pol_tone_mean"] == pd.Series([5.0, -3.0, 1.5]).mean()
    assert row["epu_count"] == 1.0
    assert row["pres_article_count"] == 1.0
    assert row["pres_tone_mean"] == 1.5


def test_market_day_without_political_coverage_is_omitted() -> None:
    df = aggregate_market([_rec("2024-01-02", 4.0, tickers=("NVDA",))])
    assert df.empty


def test_empty_inputs_return_typed_empty_frames() -> None:
    agg = aggregate([])
    assert list(agg.per_ticker.columns) == PER_TICKER_COLS
    assert list(agg.market.columns) == MARKET_COLS
    assert agg.per_ticker.empty and agg.market.empty
    assert agg.per_ticker.index.names == ["date", "ticker"]


def test_separate_days_are_separate_rows() -> None:
    df = aggregate_per_ticker(
        [_rec("2024-01-02", 2.0, ("NVDA",)), _rec("2024-01-03", 3.0, ("NVDA",))]
    )
    assert len(df) == 2
    assert df["tone_mean"].tolist() == [2.0, 3.0]
