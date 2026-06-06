"""Finnhub news provider — company news (free tier: 60 req/min).

Primary news source. Finnhub's free tier does NOT include historical price
candles, so this provider intentionally implements only the news capability.

API: GET https://finnhub.io/api/v1/company-news?symbol=&from=&to=&token=
Each item: {datetime: <unix s>, headline, summary, source, url, category, ...}
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings

_NAME = "finnhub"
_URL = "https://finnhub.io/api/v1/company-news"


def _articles_from_payload(ticker: str, payload: list[dict[str, Any]]) -> NewsBundle:
    """Normalize Finnhub's news array into a ``NewsBundle``. Pure function."""
    articles: list[Article] = []
    for item in payload:
        url = item.get("url")
        headline = item.get("headline")
        ts = item.get("datetime")
        # Skip malformed rows rather than failing the whole batch.
        if not url or not headline or not ts:
            continue
        category = item.get("category")
        articles.append(
            Article(
                title=str(headline),
                url=str(url),
                source=str(item.get("source") or _NAME),
                published_at=datetime.fromtimestamp(int(ts), tz=UTC),
                summary=item.get("summary") or None,
                topics=[str(category)] if category else [],
            )
        )
    return NewsBundle(ticker=ticker, articles=articles)


class FinnhubProvider:
    """``NewsProvider`` backed by Finnhub company-news."""

    name = _NAME

    def __init__(self, settings: Settings, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(self._settings.finnhub_api_key)

    def _fetch(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        token = self._settings.require("finnhub_api_key", capability="Finnhub news")
        payload = self._http.get(
            _URL,
            params={
                "symbol": ticker.upper(),
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": token,
            },
        )
        # company-news returns a JSON array; anything else is unexpected.
        if not isinstance(payload, list):
            return NewsBundle(ticker=ticker.upper(), articles=[])
        return _articles_from_payload(ticker.upper(), payload)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache, key, NewsBundle, lambda: self._fetch(ticker, start, end),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
