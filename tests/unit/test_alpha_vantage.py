"""Alpha Vantage normalization + soft-error mapping (no network)."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.providers.alpha_vantage import _news_from_payload, _prices_from_payload
from stock_agent.providers.base import ProviderRateLimit, SymbolNotFound


def test_prices_normalize_clip_and_sort_ascending() -> None:
    payload = {
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "10",
                "2. high": "11",
                "3. low": "9",
                "4. close": "10.5",
                "5. volume": "1000",
            },
            "2024-01-02": {
                "1. open": "9",
                "2. high": "10",
                "3. low": "8",
                "4. close": "9.5",
                "5. volume": "900",
            },
            "2023-12-01": {
                "1. open": "5",
                "2. high": "6",
                "3. low": "4",
                "4. close": "5.5",
                "5. volume": "100",
            },
        }
    }
    series = _prices_from_payload("ABC", payload, date(2024, 1, 1), date(2024, 1, 31))
    assert len(series) == 2  # December clipped out of range
    assert series.dates[0] < series.dates[1]  # ascending despite AV newest-first


def test_prices_rate_limit_message() -> None:
    with pytest.raises(ProviderRateLimit):
        _prices_from_payload("ABC", {"Note": "limit"}, date(2024, 1, 1), date(2024, 1, 2))


def test_prices_error_message_is_symbol_not_found() -> None:
    with pytest.raises(SymbolNotFound):
        _prices_from_payload(
            "ABC", {"Error Message": "bad symbol"}, date(2024, 1, 1), date(2024, 1, 2)
        )


def test_news_normalizes_with_sentiment() -> None:
    payload = {
        "feed": [
            {
                "title": "T",
                "url": "https://x.com/a",
                "source": "S",
                "time_published": "20240115T133000",
                "overall_sentiment_score": 0.3,
                "summary": "sum",
            }
        ]
    }
    bundle = _news_from_payload("ABC", payload)
    assert len(bundle) == 1
    assert bundle.articles[0].sentiment == 0.3
    assert bundle.articles[0].published_at.year == 2024
