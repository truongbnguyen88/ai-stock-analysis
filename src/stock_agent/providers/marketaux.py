"""Marketaux news provider — secondary news source (free tier ~100 req/day).

Used to broaden coverage; the dedup step (Phase 3) removes overlap with Finnhub.

API: GET https://api.marketaux.com/v1/news/all
     ?symbols=&published_after=&language=en&api_token=
Response: {"data": [{title, description, snippet, url, source, published_at, ...}]}
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings

_NAME = "marketaux"
_URL = "https://api.marketaux.com/v1/news/all"


def _parse_dt(value: str) -> datetime:
    """Parse Marketaux ISO-8601 timestamps (handles trailing 'Z')."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _articles_from_payload(ticker: str, payload: dict[str, Any]) -> NewsBundle:
    """Normalize the Marketaux ``data`` array into a ``NewsBundle``. Pure function."""
    data = payload.get("data")
    if not isinstance(data, list):
        return NewsBundle(ticker=ticker, articles=[])

    articles: list[Article] = []
    for item in data:
        url = item.get("url")
        title = item.get("title")
        published = item.get("published_at")
        if not url or not title or not published:
            continue
        # Prefer the fuller description; fall back to the short snippet.
        summary = item.get("description") or item.get("snippet") or None
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("source") or _NAME),
                published_at=_parse_dt(str(published)),
                summary=summary,
            )
        )
    return NewsBundle(ticker=ticker, articles=articles)


class MarketauxProvider:
    """``NewsProvider`` backed by Marketaux."""

    name = _NAME

    def __init__(self, settings: Settings, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(self._settings.marketaux_api_key)

    def _fetch(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        token = self._settings.require("marketaux_api_key", capability="Marketaux news")
        payload = self._http.get(
            _URL,
            params={
                "symbols": ticker.upper(),
                "published_after": start.isoformat(),
                "published_before": end.isoformat(),
                "language": "en",
                "filter_entities": "true",
                "api_token": token,
            },
        )
        if not isinstance(payload, dict):
            return NewsBundle(ticker=ticker.upper(), articles=[])
        return _articles_from_payload(ticker.upper(), payload)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(self._cache, key, NewsBundle, lambda: self._fetch(ticker, start, end))

    def close(self) -> None:
        self._http.close()
