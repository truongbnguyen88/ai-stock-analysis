"""GDELT DOC 2.0 topic-news provider — live, free, keyless, theme-aware.

Supplies ARTICLE-LEVEL news for a free-text/keyword query (Enhancement C), newest
first. This is the LIVE topic path — distinct from ``news/gdelt_ingest.py``, which
hits GDELT **BigQuery** for offline topic *sentiment features* (modeling), not
headlines. Official API, no scraping.

API: GET https://api.gdeltproject.org/api/v2/doc/doc
     ?query=<expr>&mode=ArtList&sort=DateDesc&format=json
     &startdatetime=<YYYYMMDDHHMMSS>&enddatetime=<...>&maxrecords=<n>
Response: {"articles": [{url, title, seendate, domain, language, sourcecountry, ...}]}

The DOC ArtList payload carries no per-article tone, so ``Article.sentiment`` is
left ``None`` (a tone-based sentiment is only populated if a future payload field
supplies it) — never inferred here, per the numbers-vs-narrative invariant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "gdelt_doc"
_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_MAX_RECORDS_CAP = 250  # GDELT DOC hard limit


def _build_query(keywords: Sequence[str]) -> str:
    """GDELT DOC query: ``(kw1 OR "kw two")`` — phrases quoted, OR-joined.

    Self-contained (no import from ``news.topics``) so the provider layer stays
    below the news layer per the dependency direction.
    """
    terms = [f'"{k}"' if " " in k else k for k in keywords if k.strip()]
    if not terms:
        raise ProviderUnavailable(_NAME, "empty keyword set")
    expr = " OR ".join(terms)
    return f"({expr})" if len(terms) > 1 else expr


def _parse_seendate(value: str) -> datetime | None:
    """Parse a GDELT ``seendate`` (``20260601T120000Z`` or ``20260601120000``) as UTC."""
    raw = value.strip().replace("T", "").rstrip("Z")
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _articles_from_payload(
    payload: dict[str, Any], *, language: str, top_n: int
) -> list[Article]:
    """Normalize the GDELT ``articles`` array into ``Article``s. Pure function.

    Filters to the target ``language`` (GDELT reports a language *name*, e.g.
    "English"); skips rows missing a url/title/parseable date. Assumes the API's
    ``sort=DateDesc`` (newest first) and truncates to ``top_n``.
    """
    raw = payload.get("articles")
    if not isinstance(raw, list):
        return []
    target = language.strip().lower()
    out: list[Article] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url, title, seen = item.get("url"), item.get("title"), item.get("seendate")
        if not url or not title or not seen:
            continue
        if target and str(item.get("language", "")).strip().lower() != target:
            continue
        published = _parse_seendate(str(seen))
        if published is None:
            continue
        out.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("domain") or _NAME),
                published_at=published,
            )
        )
        if len(out) >= top_n:
            break
    return out


def _tone_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate topic sentiment from a GDELT ``ToneChart`` histogram. Pure function.

    ToneChart returns ``{"tonechart": [{"bin": <tone>, "count": <n>}, ...]}``; we
    return the article-count-weighted mean tone (GDELT's ~[-100, +100] scale, where
    <0 is negative coverage). ``None`` if the histogram is empty/malformed. This is
    a DATA-derived number (not LLM), satisfying the numbers-vs-narrative invariant.
    """
    chart = payload.get("tonechart")
    if not isinstance(chart, list):
        return None
    total = 0.0
    weighted = 0.0
    for entry in chart:
        if not isinstance(entry, dict):
            continue
        try:
            tone = float(entry["bin"])
            count = float(entry.get("count", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if count <= 0:
            continue
        total += count
        weighted += tone * count
    if total <= 0:
        return None
    avg = weighted / total
    label = "net positive" if avg > 1 else "net negative" if avg < -1 else "roughly neutral"
    return {
        "avg_tone": round(avg, 3),
        "n_articles": int(total),
        "label": label,
        "scale": "GDELT article tone (~ -100..+100; <0 = negative coverage, >0 = positive)",
    }


class GdeltDocProvider:
    """``TopicNewsProvider`` backed by the GDELT DOC 2.0 API (keyless)."""

    name = _NAME

    def __init__(
        self,
        settings: Any,
        cache: DiskCache,
        http: HttpJson | None = None,
        *,
        language: str = "english",
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)
        self._language = language

    def available(self) -> bool:
        return True  # no API key required

    def _fetch(self, query: str, start: Date, end: Date, top_n: int) -> NewsBundle:
        payload = self._http.get(
            _URL,
            params={
                "query": query,
                "mode": "ArtList",
                "sort": "DateDesc",
                "format": "json",
                "startdatetime": f"{start.strftime('%Y%m%d')}000000",
                "enddatetime": f"{end.strftime('%Y%m%d')}235959",
                "maxrecords": min(max(top_n, 1), _MAX_RECORDS_CAP),
            },
        )
        if not isinstance(payload, dict):
            # GDELT returns text (not JSON) on a malformed query → treat as unavailable
            # so the registry can fall through to another topic provider.
            raise ProviderUnavailable(_NAME, "non-JSON response (malformed query?)")
        articles = _articles_from_payload(payload, language=self._language, top_n=top_n)
        return NewsBundle(ticker=query, articles=articles)

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        """Return newest-first topic articles for ``keywords`` within [start, end]."""
        query = _build_query(keywords)
        key = make_key(_NAME, "topic", query, start, end, top_n, self._language)
        return cached_model(
            self._cache,
            key,
            NewsBundle,
            lambda: self._fetch(query, start, end, top_n),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def _fetch_tone(self, query: str, start: Date, end: Date) -> dict[str, Any] | None:
        payload = self._http.get(
            _URL,
            params={
                "query": query,
                "mode": "ToneChart",
                "format": "json",
                "startdatetime": f"{start.strftime('%Y%m%d')}000000",
                "enddatetime": f"{end.strftime('%Y%m%d')}235959",
            },
        )
        if not isinstance(payload, dict):
            return None
        return _tone_from_payload(payload)

    def get_topic_tone(
        self, keywords: Sequence[str], start: Date, end: Date
    ) -> dict[str, Any] | None:
        """Aggregate topic sentiment (weighted mean article tone) for ``keywords``.

        Optional capability beyond ``TopicNewsProvider`` — the registry calls it
        only when present. Cached under the short news TTL (caches misses as well).
        """
        query = _build_query(keywords)
        key = make_key(_NAME, "tone", query, start, end)
        raw = self._cache.get(key, ttl_override=self._settings.cache_ttl_news_seconds)
        if raw is not None:
            cached: dict[str, Any] | None = json.loads(raw)
            return cached
        tone = self._fetch_tone(query, start, end)
        self._cache.set(key, json.dumps(tone))  # cache the miss too (null)
        return tone

    def close(self) -> None:
        self._http.close()
