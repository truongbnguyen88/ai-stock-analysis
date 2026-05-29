"""Feature engineering tests: point-in-time correctness and leakage prevention."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from stock_agent.features.assembler import THRESHOLDS, _target_col, build_training_matrix
from stock_agent.features.news_features import build_news_features
from stock_agent.features.price_features import (
    PRICE_FEATURE_COLS,
    build_price_feature_matrix,
)
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle


def _series(n: int = 120, trend: float = 0.001) -> PriceSeries:
    prices = [100.0 * (1 + trend) ** i for i in range(n)]
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=p,
            high=p + 0.5,
            low=p - 0.5,
            close=p,
        )
        for i, p in enumerate(prices)
    ]
    return PriceSeries(ticker="X", bars=bars)


# ---- price features ---------------------------------------------------------


def test_price_feature_matrix_has_all_cols() -> None:
    df = build_price_feature_matrix(_series())
    assert set(PRICE_FEATURE_COLS) == set(df.columns)


def test_price_features_index_matches_bar_dates() -> None:
    series = _series()
    df = build_price_feature_matrix(series)
    assert len(df) == len(series)
    assert df.index[0] == pd.Timestamp(series.dates[0])


def test_price_features_no_lookahead_by_construction() -> None:
    # Verify that no feature at row t depends on data from t+1 onwards.
    # Strategy: drop the last bar and check that all features except the last
    # row are identical. Any lookahead would change earlier rows.
    series = _series(60)
    df_full = build_price_feature_matrix(series)
    series_short = PriceSeries(ticker="X", bars=series.bars[:-1])
    df_short = build_price_feature_matrix(series_short)
    # All rows except the last of the full frame must match the short frame.
    common_idx = df_short.index
    pd.testing.assert_frame_equal(
        df_full.loc[common_idx].reset_index(drop=True),
        df_short.reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
    )


def test_early_rows_nan_for_long_lookbacks() -> None:
    # MA200 needs 200 bars; series has 120 → all ma50_to_ma200 are NaN.
    df = build_price_feature_matrix(_series(120))
    assert df["ma50_to_ma200"].isna().all()


# ---- assembler / leakage ----------------------------------------------------


def test_training_matrix_shape_and_no_target_overlap() -> None:
    series = _series(120)
    horizon = 20
    X, y = build_training_matrix(series, horizon=horizon)
    # Last `horizon` bars have no target → should be dropped.
    assert len(X) <= len(series) - horizon
    assert len(X) == len(y)
    assert set(y.columns) == {_target_col(t) for t in THRESHOLDS}


def test_training_matrix_targets_are_binary() -> None:
    X, y = build_training_matrix(_series(120), horizon=20)
    for col in y.columns:
        assert set(y[col].unique()).issubset({0.0, 1.0})


def test_training_matrix_insufficient_history_raises() -> None:
    with pytest.raises(ValueError, match="valid training rows"):
        build_training_matrix(_series(25), horizon=20, min_rows=30)


# ---- news features ----------------------------------------------------------


class FakeLLM:
    def __init__(self, scores: list[dict[str, object]]) -> None:
        self._payload = json.dumps({"scores": scores})

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        return self._payload


def _bundle(n: int = 3) -> NewsBundle:
    arts = [
        Article(
            title=f"NVDA earnings beat quarter {i}",
            url=f"https://x.com/{i}",
            source="s",
            published_at=datetime(2025, 1, 2 + i, tzinfo=UTC),
        )
        for i in range(n)
    ]
    return NewsBundle(ticker="NVDA", articles=arts)


def test_news_features_with_claude_scoring() -> None:
    bundle = _bundle(3)
    scores = [{"url": f"https://x.com/{i}", "score": [0.5, -0.3, 0.1][i]} for i in range(3)]
    feats = build_news_features(bundle, llm=FakeLLM(scores))
    assert feats["article_count"] == 3.0
    assert abs(feats["avg_sentiment"] - (0.5 - 0.3 + 0.1) / 3) < 1e-9
    assert feats["sentiment_coverage"] == 1.0
    assert feats["has_earnings"] == 1.0  # "earnings" in titles


def test_news_features_without_llm_defaults_neutral() -> None:
    feats = build_news_features(_bundle(3), llm=None)
    assert feats["avg_sentiment"] == 0.0
    assert feats["sentiment_coverage"] == 0.0
    assert feats["article_count"] == 3.0


def test_news_features_empty_bundle() -> None:
    feats = build_news_features(NewsBundle(ticker="X"), llm=None)
    assert all(v == 0.0 for v in feats.values())
