"""Google News RSS — keyless news for BOTH per-ticker and theme queries ($0, no API key).

The zero-friction breadth/failover layer: no key, wide coverage, keyword search. Returns
HEADLINES + links only (no article body, no sentiment), so it's a last-resort source that pairs
with the existing dedup/rank + LLM summarizer. Parses the official Google News RSS feed with the
stdlib XML parser — an RSS feed, not HTML scraping (but an undocumented endpoint, so best-effort).

Feed: GET https://news.google.com/rss/search?q=<query>+when:<N>d&hl=en-US&gl=US&ceid=US:en
Items: <item><title/><link/><pubDate/(RFC-822)><source/></item>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import date as Date
from email.utils import parsedate_to_datetime
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.schemas.news import Article, NewsBundle

_NAME = "google_news_rss"
_URL = "https://news.google.com/rss/search"
_MAX_LOOKBACK_DAYS = 90  # cap the `when:Nd` relative window Google accepts


def _build_query(terms: Sequence[str], start: Date, end: Date) -> str:
    """Build the RSS ``q``: OR-joined terms plus a ``when:Nd`` window covering [start, today]."""
    parts = [f'"{t}"' if " " in t else t for t in terms if t.strip()]
    expr = " OR ".join(parts)
    lookback = min(max((Date.today() - start).days + 1, 1), _MAX_LOOKBACK_DAYS)
    return f"{expr} when:{lookback}d"


def _articles_from_xml(
    subject: str, xml_text: str, start: Date, end: Date, top_n: int
) -> NewsBundle:
    """Parse a Google News RSS document into a ``NewsBundle`` filtered to [start, end]. Pure."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return NewsBundle(ticker=subject, articles=[])
    channel = root.find("channel")
    if channel is None:
        return NewsBundle(ticker=subject, articles=[])

    articles: list[Article] = []
    for item in channel.findall("item"):
        title, link, pub = item.findtext("title"), item.findtext("link"), item.findtext("pubDate")
        if not title or not link or not pub:
            continue
        try:
            published = parsedate_to_datetime(pub)  # RFC-822 -> aware datetime
        except (TypeError, ValueError):
            continue
        if not (start <= published.date() <= end):
            continue
        source_el = item.find("source")
        source = source_el.text if source_el is not None and source_el.text else "Google News"
        articles.append(
            Article(title=str(title), url=str(link), source=str(source), published_at=published)
        )
        if len(articles) >= top_n:
            break
    return NewsBundle(ticker=subject, articles=articles)


class GoogleNewsRssProvider:
    """``NewsProvider`` + ``TopicNewsProvider`` backed by Google News RSS (keyless)."""

    name = _NAME

    def __init__(self, settings: Any, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return True  # keyless

    def _fetch(self, subject: str, query: str, start: Date, end: Date, top_n: int) -> NewsBundle:
        xml_text = self._http.get_text(
            _URL, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        return _articles_from_xml(subject, xml_text, start, end, top_n)

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        # "<TICKER> stock" biases the keyword feed toward the company rather than the bare symbol.
        query = _build_query([f"{ticker.upper()} stock"], start, end)
        key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache, key, NewsBundle,
            lambda: self._fetch(ticker.upper(), query, start, end, top_n=25),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        query = _build_query(keywords, start, end)
        key = make_key(_NAME, "topic", query, start, end, top_n)
        return cached_model(
            self._cache, key, NewsBundle,
            lambda: self._fetch(query, query, start, end, top_n),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
