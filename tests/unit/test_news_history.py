"""Tests for the point-in-time news-history feature loader (Task 10).

Focus: scale-free buzz, the conservative publication lag, and the leakage
guarantee (a future news spike must not appear in an earlier feature row).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_agent.features.news_history import (
    NEWS_HISTORY_COLS,
    NewsStore,
    build_news_history_features,
)
from stock_agent.news.aggregate import MARKET_COLS, PER_TICKER_COLS


def _per_ticker(rows: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), t) for (d, t) in rows], names=["date", "ticker"]
    )
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in PER_TICKER_COLS}, index=idx)
    return pd.DataFrame(list(rows.values()), index=idx)[PER_TICKER_COLS]


def _market(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    idx = pd.Index([pd.Timestamp(d) for d in rows], name="date")
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in MARKET_COLS}, index=idx)
    return pd.DataFrame(list(rows.values()), index=idx)[MARKET_COLS]


def _row(ac: float, tone: float, pos: float, neg: float, std: float = 1.0) -> dict[str, float]:
    return {
        "article_count": ac,
        "tone_mean": tone,
        "tone_std": std,
        "pos_count": pos,
        "neg_count": neg,
    }


def _mrow(pol_c: float, pol_t: float, epu: float, pres_c: float, pres_t: float) -> dict[str, float]:
    return {
        "pol_article_count": pol_c,
        "pol_tone_mean": pol_t,
        "pol_tone_std": 4.0,
        "epu_count": epu,
        "pres_article_count": pres_c,
        "pres_tone_mean": pres_t,
    }


def test_unknown_ticker_yields_all_nan_columns() -> None:
    store = NewsStore(per_ticker=_per_ticker({}), market=_market({}))
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    out = build_news_history_features("NVDA", idx, store)
    assert list(out.columns) == NEWS_HISTORY_COLS
    assert out.isna().all().all()


def test_tone_and_fractions_aligned_with_one_day_lag() -> None:
    # News on 2024-01-10 should appear in the feature row for 2024-01-11 (lag=1).
    pt = _per_ticker({("2024-01-10", "NVDA"): _row(ac=10, tone=3.0, pos=6, neg=2)})
    store = NewsStore(per_ticker=pt, market=_market({}))
    idx = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"])
    out = build_news_history_features("NVDA", idx, store, lag_days=1, buzz_window=5)
    assert pd.isna(out.loc[idx[0], "news_tone"])  # same-day: not yet available
    assert out.loc[idx[1], "news_tone"] == 3.0  # next day: available
    assert out.loc[idx[1], "news_pos_frac"] == 0.6
    assert out.loc[idx[1], "news_neg_frac"] == 0.2
    assert out.loc[idx[2], "news_tone"] == 3.0  # ffill carries it forward


def test_buzz_is_a_scale_free_spike_ratio() -> None:
    # Flat 10/day for a month, then a 50-spike → buzz at the spike (next day) ≈ 5x.
    days = pd.date_range("2024-01-01", periods=31, freq="D")
    rows = {(d.strftime("%Y-%m-%d"), "AAPL"): _row(ac=10, tone=0.0, pos=3, neg=3) for d in days}
    spike_day = pd.Timestamp("2024-02-01")
    rows[(spike_day.strftime("%Y-%m-%d"), "AAPL")] = _row(ac=50, tone=0.0, pos=20, neg=5)
    store = NewsStore(per_ticker=_per_ticker(rows), market=_market({}))
    idx = pd.to_datetime(["2024-02-01", "2024-02-02"])
    out = build_news_history_features("AAPL", idx, store, lag_days=1, buzz_window=20)
    assert abs(out.loc[idx[1], "news_buzz"] - 5.0) < 1e-9  # 50 / baseline(10)


def test_leakage_future_spike_absent_from_earlier_rows() -> None:
    # A huge spike on 2024-01-20 must NOT influence the 2024-01-15 feature row.
    base = pd.date_range("2024-01-01", periods=14, freq="D")
    rows = {(d.strftime("%Y-%m-%d"), "MSFT"): _row(ac=5, tone=1.0, pos=2, neg=1) for d in base}
    rows[("2024-01-20", "MSFT")] = _row(ac=500, tone=-9.0, pos=0, neg=400)
    store = NewsStore(per_ticker=_per_ticker(rows), market=_market({}))
    idx = pd.to_datetime(["2024-01-15", "2024-01-21"])
    out = build_news_history_features("MSFT", idx, store, lag_days=1, buzz_window=10)
    assert out.loc[idx[0], "news_tone"] == 1.0  # pre-spike state, not the -9 spike
    assert out.loc[idx[1], "news_tone"] == -9.0  # spike visible the day after


def test_as_of_truncates_future_news() -> None:
    pt = _per_ticker(
        {
            ("2024-01-10", "NVDA"): _row(ac=10, tone=2.0, pos=5, neg=2),
            ("2024-01-20", "NVDA"): _row(ac=10, tone=-5.0, pos=1, neg=8),
        }
    )
    store = NewsStore(per_ticker=pt, market=_market({}))
    idx = pd.to_datetime(["2024-01-11", "2024-01-25"])
    # as_of before the second article → it must be invisible even for the 01-25 row.
    out = build_news_history_features("NVDA", idx, store, as_of=date(2024, 1, 15), lag_days=1)
    assert out.loc[idx[0], "news_tone"] == 2.0
    assert out.loc[idx[1], "news_tone"] == 2.0  # 01-20 article excluded by as_of


def test_market_stream_shared_features() -> None:
    mk = _market({"2024-03-01": _mrow(pol_c=100, pol_t=-1.0, epu=80, pres_c=20, pres_t=-0.5)})
    store = NewsStore(per_ticker=_per_ticker({}), market=mk)
    idx = pd.to_datetime(["2024-03-02"])
    out = build_news_history_features("ANY", idx, store, lag_days=1)
    assert out.loc[idx[0], "pol_tone"] == -1.0
    assert out.loc[idx[0], "pres_tone"] == -0.5
