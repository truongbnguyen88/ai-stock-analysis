"""Financial Modeling Prep (FMP) — per-ticker finance news provider (free tier 250 req/day).

Ticker-tagged financial news, so it deepens the per-ticker MERGE chain alongside Finnhub /
Marketaux. Free, keyed; skipped when the key is absent.

API: GET https://financialmodelingprep.com/api/v3/stock_news
     ?tickers=NVDA&from=&to=&limit=&apikey=
Response: [{symbol, publishedDate: "2026-06-01 12:00:00", title, image, site, text, url}, ...]

NOTE: FMP has reshuffled endpoints over time (v3 ``stock_news`` vs newer ``/stable/news/stock``).
If a live fetch 404s, confirm which news endpoint your plan exposes and update ``_URL`` — the
normalizer only depends on the field names above.
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "fmp"
_URL = "https://financialmodelingprep.com/api/v3/stock_news"
_MAX_LIMIT = 100


def _parse_dt(value: str) -> datetime | None:
    """Parse an FMP ``publishedDate`` ('2026-06-01 12:00:00', naive) as UTC; None if unparseable."""
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None


def _articles_from_payload(ticker: str, payload: Any, start: Date, end: Date) -> NewsBundle:
    """Normalize the FMP array into a ``NewsBundle``, filtered to [start, end]. Pure function.

    FMP's ``from``/``to`` params are honored server-side, but we also filter client-side so the
    window is enforced uniformly across providers (some plans ignore the date params).
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
                source=str(item.get("site") or _NAME),
                published_at=dt,
                summary=str(item["text"]) if item.get("text") else None,
            )
        )
    return NewsBundle(ticker=ticker, articles=articles)


class FmpProvider:
    """``NewsProvider`` backed by Financial Modeling Prep (keyed)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(getattr(self._settings, "fmp_api_key", None))

    def _fetch(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = self._settings.require("fmp_api_key", capability="FMP company news")
        payload = self._http.get(
            _URL,
            params={
                "tickers": ticker.upper(),
                "from": start.isoformat(),
                "to": end.isoformat(),
                "limit": _MAX_LIMIT,
                "apikey": key,
            },
        )
        return _articles_from_payload(ticker.upper(), payload, start, end)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache, key, NewsBundle, lambda: self._fetch(ticker, start, end),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
