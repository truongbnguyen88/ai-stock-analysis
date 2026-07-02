"""Tiingo — per-ticker financial news provider (free tier: 1,000 req/day, 3-month history).

Genuinely-free ticker-tagged news with real symbol tagging (Tiingo also tags on slang / company /
product mentions), so it strengthens the per-ticker MERGE chain alongside Finnhub / Marketaux. It
is the free replacement for FMP, whose news moved behind a paid plan. Keyed; skipped without a key.
Free tier serves ~3 months of history with real-time updates — fine for recency-focused news.

Note: some Tiingo accounts must one-time enable the news feed on the account page before the
endpoint returns data; a 403/empty result then simply falls through the MERGE chain.

API: GET https://api.tiingo.com/tiingo/news
     ?tickers=nvda&startDate=&endDate=&sortBy=publishedDate&limit=&token=
Response (top-level ARRAY): [{id, title, url, description, publishedDate, crawlDate, source,
          tickers: [...], tags: [...]}]
No per-article sentiment (numeric sentiment stays with Alpha Vantage / the models).
Official API (no scraping).
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "tiingo"
_URL = "https://api.tiingo.com/tiingo/news"
_MAX_LIMIT = 100


def _parse_dt(value: str) -> datetime | None:
    """Parse a Tiingo ISO-8601 ``publishedDate`` (trailing 'Z' → UTC); naive → assumed UTC."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _articles_from_payload(ticker: str, payload: Any, start: Date, end: Date) -> NewsBundle:
    """Normalize the Tiingo array into a ``NewsBundle``, filtered to [start, end]. Pure function.

    The window is enforced client-side (in addition to the server-side ``startDate``/``endDate``)
    so all providers apply identical inclusive [start, end] semantics regardless of API quirks.
    """
    if not isinstance(payload, list):
        return NewsBundle(ticker=ticker, articles=[])
    articles: list[Article] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        url, title, published = item.get("url"), item.get("title"), item.get("publishedDate")
        if not url or not title or not published:
            continue
        dt = _parse_dt(str(published))
        if dt is None or not (start <= dt.date() <= end):
            continue
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("source") or _NAME),
                published_at=dt,
                summary=str(item["description"]) if item.get("description") else None,
            )
        )
    return NewsBundle(ticker=ticker, articles=articles)


class TiingoProvider:
    """``NewsProvider`` backed by Tiingo's news feed (keyed)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(getattr(self._settings, "tiingo_api_key", None))

    def _fetch(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = self._settings.require("tiingo_api_key", capability="Tiingo company news")
        payload = self._http.get(
            _URL,
            params={
                # Tiingo ticker param is case-insensitive; lowercase matches their examples.
                "tickers": ticker.lower(),
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "sortBy": "publishedDate",  # newest-first by publication time
                "limit": _MAX_LIMIT,
                "token": key,
            },
        )
        return _articles_from_payload(ticker.upper(), payload, start, end)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        cache_key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache, cache_key, NewsBundle, lambda: self._fetch(ticker, start, end),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
