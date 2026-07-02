"""TheNewsAPI — theme/keyword news provider (free tier: 100 req/day, 3 articles/request).

Production-friendly free tier (no dev-only clause) over 40k+ sources — a broad, keyed backup on the
topic FAILOVER chain after Guardian / NewsData. Serves the THEME path ("AI memory",
"semiconductors"), not ticker-scoped news. Keyed; skipped without a key.

Free-tier caveat: at most 3 articles per request (plan-capped). We omit ``limit`` so the plan's own
maximum applies (3 on free, higher on paid) and truncate to ``top_n`` client-side.

API: GET https://api.thenewsapi.com/v1/news/all
     ?search=<expr>&language=en&published_after=&published_before=&sort=published_at&api_token=
Response: {"meta": {...}, "data": [{uuid, title, description, snippet, url, published_at, source,
          categories: [...]}]}
Search syntax: OR = ``|``, phrase = quoted, AND = ``+`` (httpx URL-encodes automatically).
Official API (no scraping).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "thenewsapi"
_URL = "https://api.thenewsapi.com/v1/news/all"


def _build_query(keywords: Sequence[str]) -> str:
    """TheNewsAPI ``search`` expression: ``"AI memory" | robotics`` (phrases quoted, OR-joined).

    OR is ``|`` in TheNewsAPI's query grammar (cf. Guardian/NewsData ``OR``). httpx URL-encodes the
    special characters, so we build the raw expression here.
    """
    terms = [f'"{k}"' if " " in k else k for k in keywords if k.strip()]
    if not terms:
        raise ProviderUnavailable(_NAME, "empty keyword set")
    return " | ".join(terms)


def _parse_dt(value: str) -> datetime | None:
    """Parse a TheNewsAPI ``published_at`` ('2026-07-02T04:09:45.000000Z') as aware UTC."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _articles_from_payload(
    subject: str, payload: dict[str, Any], start: Date, end: Date, top_n: int
) -> NewsBundle:
    """Normalize the TheNewsAPI ``data`` array into a ``NewsBundle``, filtered to [start, end]."""
    data = payload.get("data")
    if not isinstance(data, list):
        return NewsBundle(ticker=subject, articles=[])
    articles: list[Article] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url, title, published = item.get("url"), item.get("title"), item.get("published_at")
        if not url or not title or not published:
            continue
        dt = _parse_dt(str(published))
        if dt is None or not (start <= dt.date() <= end):
            continue
        summary = item.get("description") or item.get("snippet")
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("source") or _NAME),
                published_at=dt,
                summary=str(summary) if summary else None,
            )
        )
        if len(articles) >= top_n:
            break
    return NewsBundle(ticker=subject, articles=articles)


class TheNewsApiProvider:
    """``TopicNewsProvider`` backed by TheNewsAPI (keyed)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(getattr(self._settings, "thenewsapi_api_key", None))

    def _fetch(self, query: str, start: Date, end: Date, top_n: int) -> NewsBundle:
        key = self._settings.require("thenewsapi_api_key", capability="TheNewsAPI topic news")
        payload = self._http.get(
            _URL,
            params={
                "search": query,
                "language": "en",
                "published_after": start.isoformat(),
                # published_before is exclusive of that instant; add a day so the full end date is
                # captured, then the client-side [start, end] filter trims anything past `end`.
                "published_before": (end + timedelta(days=1)).isoformat(),
                "sort": "published_at",  # newest-first
                "api_token": key,
            },
        )
        if not isinstance(payload, dict):
            return NewsBundle(ticker=query, articles=[])
        return _articles_from_payload(query, payload, start, end, top_n)

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        """Return newest-first theme articles for ``keywords`` within [start, end]."""
        query = _build_query(keywords)
        cache_key = make_key(_NAME, "topic", query, start, end, top_n)
        return cached_model(
            self._cache, cache_key, NewsBundle, lambda: self._fetch(query, start, end, top_n),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
