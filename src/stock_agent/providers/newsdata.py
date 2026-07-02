"""NewsData.io — general news provider covering BOTH per-ticker and theme news (free ~200/day).

The only new source whose free tier permits commercial use, so it's a good failover hedge in both
chains. Keyed; skipped without a key. The free ``latest`` endpoint has no date-range params (the
``archive`` endpoint is paid), so we fetch the latest matches for the query and filter to the
window client-side — fine for recency-focused news.

API: GET https://newsdata.io/api/1/latest?apikey=&q=<expr>&language=en[&category=business]
Response: {"status": "success", "results": [{title, link, source_id, description,
          pubDate: "2026-06-01 12:00:00"}], "nextPage": ...}
Official API (no scraping).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "newsdata"
_URL = "https://newsdata.io/api/1/latest"


def _build_query(keywords: Sequence[str]) -> str:
    """NewsData ``q`` expression: ``robotics OR "humanoid robot"`` (phrases quoted, OR-joined)."""
    terms = [f'"{k}"' if " " in k else k for k in keywords if k.strip()]
    if not terms:
        raise ProviderUnavailable(_NAME, "empty keyword set")
    return " OR ".join(terms)


def _parse_dt(value: str) -> datetime | None:
    """Parse a NewsData ``pubDate`` ('2026-06-01 12:00:00', UTC) as an aware UTC datetime."""
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _articles_from_payload(
    subject: str, payload: dict[str, Any], start: Date, end: Date
) -> NewsBundle:
    """Normalize the NewsData ``results`` array into a ``NewsBundle``, filtered to [start, end]."""
    results = payload.get("results")
    if not isinstance(results, list):
        return NewsBundle(ticker=subject, articles=[])
    articles: list[Article] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url, title, published = item.get("link"), item.get("title"), item.get("pubDate")
        if not url or not title or not published:
            continue
        dt = _parse_dt(str(published))
        if dt is None or not (start <= dt.date() <= end):
            continue
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("source_id") or _NAME),
                published_at=dt,
                summary=str(item["description"]) if item.get("description") else None,
            )
        )
    return NewsBundle(ticker=subject, articles=articles)


class NewsDataProvider:
    """``NewsProvider`` + ``TopicNewsProvider`` backed by NewsData.io (keyed)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(getattr(self._settings, "newsdata_api_key", None))

    def _fetch(
        self, subject: str, query: str, start: Date, end: Date, *, business: bool
    ) -> NewsBundle:
        key = self._settings.require("newsdata_api_key", capability="NewsData.io news")
        params: dict[str, Any] = {"apikey": key, "q": query, "language": "en"}
        if business:
            params["category"] = "business"
        payload = self._http.get(_URL, params=params)
        if not isinstance(payload, dict):
            return NewsBundle(ticker=subject, articles=[])
        return _articles_from_payload(subject, payload, start, end)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        # Ticker-scoped: query the symbol under the business category to bias toward finance news.
        cache_key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache, cache_key, NewsBundle,
            lambda: self._fetch(ticker.upper(), ticker.upper(), start, end, business=True),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        query = _build_query(keywords)
        cache_key = make_key(_NAME, "topic", query, start, end)
        # Themes are broader than finance, so no category filter on the topic path.
        return cached_model(
            self._cache, cache_key, NewsBundle,
            lambda: self._fetch(query, query, start, end, business=False),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
