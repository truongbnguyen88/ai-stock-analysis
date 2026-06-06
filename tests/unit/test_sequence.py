"""LSTM sequence forecaster (Task 8 spike): train/forecast, determinism, fallback.

Skipped entirely when the optional ``[sequence]`` extra (torch) is absent — so the
core CI gate, which does not install torch, does not run or require these.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

pytest.importorskip("torch")

import pandas as pd  # noqa: E402

from stock_agent.features.news_history import NewsStore  # noqa: E402
from stock_agent.forecasting.base import ForecastModel  # noqa: E402
from stock_agent.forecasting.sequence import (  # noqa: E402
    SequenceForecaster,
    SequenceModel,
    train_sequence_model,
)
from stock_agent.news.aggregate import MARKET_COLS, PER_TICKER_COLS  # noqa: E402
from stock_agent.schemas.market import PriceBar, PriceSeries  # noqa: E402


def _series(seed: int, *, n: int = 300, ticker: str | None = None) -> PriceSeries:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    bars = [
        PriceBar(
            date=date(2019, 1, 1) + timedelta(days=i),
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
        )
        for i, c in enumerate(closes)
    ]
    return PriceSeries(ticker=ticker or f"T{seed}", bars=bars)


def _tiny_model() -> SequenceModel:
    universe = [_series(i) for i in range(6)]
    return train_sequence_model(
        universe, horizon=20, lookback=30, hidden=8, layers=1, epochs=2, seed=42
    )


def test_sequence_is_forecastmodel() -> None:
    assert isinstance(SequenceForecaster(_tiny_model()), ForecastModel)


def test_sequence_forecast_valid_and_deterministic() -> None:
    model = _tiny_model()
    target = _series(99, ticker="NVDA")
    fc1 = SequenceForecaster(model).forecast(target, horizon_days=20)
    fc2 = SequenceForecaster(model).forecast(target, horizon_days=20)
    assert fc1.model_name == "lstm_seq"
    assert abs(sum(b.probability for b in fc1.buckets) - 1.0) < 1e-6
    assert fc1 == fc2  # fixed seeds → bit-stable inference


def test_sequence_falls_back_on_short_history() -> None:
    model = _tiny_model()
    short = _series(7, n=25)  # fewer feature rows than lookback (30) → fallback
    fc = SequenceForecaster(model).forecast(short, horizon_days=20)
    assert fc.model_name == "lstm_seq"
    assert fc.notes is not None and "fallback" in fc.notes.lower()
    assert abs(sum(b.probability for b in fc.buckets) - 1.0) < 1e-6


def test_sequence_horizon_mismatch_raises() -> None:
    model = _tiny_model()  # trained at horizon 20
    with pytest.raises(ValueError, match="horizon"):
        SequenceForecaster(model).forecast(_series(1), horizon_days=30)


def _news_store(tickers: list[str], *, n: int = 320, seed: int = 0) -> NewsStore:
    """Synthetic daily news store covering the test series' calendar (2019-01-01 +n days)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")  # Timestamps, like the CSV loader
    rows = []
    for tk in tickers:
        for d in dates:
            cnt = float(rng.integers(0, 12))
            pos = float(rng.integers(0, int(cnt) + 1))
            rows.append((d, tk, cnt, float(rng.normal(0, 2)), 1.0, pos, max(0.0, cnt - pos)))
    pt = pd.DataFrame(rows, columns=["date", "ticker", *PER_TICKER_COLS])
    pt = pt.set_index(["date", "ticker"]).sort_index()
    mk = pd.DataFrame(
        {c: rng.normal(0, 1, n) if "tone" in c else rng.integers(1, 50, n) for c in MARKET_COLS},
        index=pd.Index(dates, name="date"),
    )
    empty_topics = pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in PER_TICKER_COLS},
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "topic"]),
    )
    return NewsStore(per_ticker=pt, market=mk, topics=empty_topics)


def test_sequence_news_widens_features_and_forecasts() -> None:
    tickers = [f"T{i}" for i in range(6)]
    universe = [_series(i, ticker=tickers[i]) for i in range(6)]
    store = _news_store(tickers)
    price_only = train_sequence_model(universe, horizon=20, lookback=30, hidden=8, epochs=2)
    with_news = train_sequence_model(
        universe, horizon=20, lookback=30, hidden=8, epochs=2, news_store=store
    )
    # News appends the active columns → wider feature dim + recorded for inference.
    assert with_news.news_cols  # non-empty (per-ticker + market cols have data)
    assert with_news.n_features == price_only.n_features + len(with_news.news_cols)

    fc = SequenceForecaster(with_news, news_store=store).forecast(universe[0], horizon_days=20)
    assert fc.model_name == "lstm_seq"
    assert abs(sum(b.probability for b in fc.buckets) - 1.0) < 1e-6
    assert fc.notes is not None and "price+news" in fc.notes


def test_sequence_news_model_without_store_falls_back() -> None:
    # A news-trained model used without a store would mismatch the feature dim → fallback.
    tickers = [f"T{i}" for i in range(6)]
    universe = [_series(i, ticker=tickers[i]) for i in range(6)]
    store = _news_store(tickers)
    with_news = train_sequence_model(
        universe, horizon=20, lookback=30, hidden=8, epochs=2, news_store=store
    )
    fc = SequenceForecaster(with_news).forecast(universe[0], horizon_days=20)  # no store passed
    assert fc.notes is not None and "fallback" in fc.notes.lower()
