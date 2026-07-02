"""The Guardian Open Platform — theme/keyword news provider (free tier 5,000 req/day).

High-quality journalism with a generous, reliable free tier — the strong keyed backup for GDELT
on the topic chain (GDELT is keyless but rate-limit-flaky). English-only; general (not
finance-tagged) coverage, so it serves the THEME path ("AI memory", "semiconductors"), not
ticker-scoped news. The free developer key is non-commercial (fine for a research/education repo).

API: GET https://content.guardianapi.com/search
     ?q=<expr>&from-date=&to-date=&order-by=newest&page-size=&show-fields=trailText&api-key=
Response: {"response": {"status": "ok", "results": [{webTitle, webUrl, webPublicationDate,
          fields: {trailText}}]}}
Official API (no scraping).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date
from datetime import datetime
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "guardian"
_URL = "https://content.guardianapi.com/search"
_MAX_PAGE = 50  # Guardian page-size cap


def _build_query(keywords: Sequence[str]) -> str:
    """Guardian ``q`` expression: ``robotics OR "humanoid robot"`` (phrases quoted, OR-joined)."""
    terms = [f'"{k}"' if " " in k else k for k in keywords if k.strip()]
    if not terms:
        raise ProviderUnavailable(_NAME, "empty keyword set")
    return " OR ".join(terms)


def _parse_dt(value: str) -> datetime:
    """Parse a Guardian ISO-8601 ``webPublicationDate`` (trailing 'Z' → UTC offset)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _articles_from_payload(subject: str, payload: dict[str, Any], top_n: int) -> NewsBundle:
    """Normalize the Guardian ``response.results`` array into a ``NewsBundle``. Pure function."""
    response = payload.get("response")
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list):
        return NewsBundle(ticker=subject, articles=[])

    articles: list[Article] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url, title = item.get("webUrl"), item.get("webTitle")
        published = item.get("webPublicationDate")
        if not url or not title or not published:
            continue
        fields = item.get("fields")
        summary = fields.get("trailText") if isinstance(fields, dict) else None
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source="The Guardian",
                published_at=_parse_dt(str(published)),
                summary=str(summary) if summary else None,
            )
        )
        if len(articles) >= top_n:
            break
    return NewsBundle(ticker=subject, articles=articles)


class GuardianProvider:
    """``TopicNewsProvider`` backed by The Guardian Open Platform (keyed)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(getattr(self._settings, "guardian_api_key", None))

    def _fetch(self, query: str, start: Date, end: Date, top_n: int) -> NewsBundle:
        key = self._settings.require("guardian_api_key", capability="Guardian topic news")
        payload = self._http.get(
            _URL,
            params={
                "q": query,
                "from-date": start.isoformat(),
                "to-date": end.isoformat(),
                "order-by": "newest",
                "page-size": min(max(top_n, 1), _MAX_PAGE),
                "show-fields": "trailText",
                "api-key": key,
            },
        )
        if not isinstance(payload, dict):
            return NewsBundle(ticker=query, articles=[])
        return _articles_from_payload(query, payload, top_n)

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        """Return newest-first theme articles for ``keywords`` within [start, end]."""
        query = _build_query(keywords)
        key = make_key(_NAME, "topic", query, start, end, top_n)
        return cached_model(
            self._cache, key, NewsBundle, lambda: self._fetch(query, start, end, top_n),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
