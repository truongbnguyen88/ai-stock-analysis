"""Google News RSS provider — stdlib XML parsing, date filtering, keyless (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.google_news_rss import GoogleNewsRssProvider, _articles_from_xml
from stock_agent.settings import Settings

_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Google News</title>
<item>
  <title>AI memory boom - Reuters</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <pubDate>Tue, 02 Jun 2026 12:00:00 GMT</pubDate>
  <source url="https://reuters.com">Reuters</source>
</item>
<item>
  <title>old story - AP</title>
  <link>https://news.google.com/rss/articles/def</link>
  <pubDate>Fri, 01 May 2026 08:00:00 GMT</pubDate>
  <source url="https://ap.org">AP</source>
</item>
<item>
  <title>no link, dropped</title>
  <pubDate>Wed, 03 Jun 2026 08:00:00 GMT</pubDate>
</item>
</channel></rss>"""


def test_parses_rss_and_filters_window() -> None:
    bundle = _articles_from_xml("AI memory", _RSS, date(2026, 5, 25), date(2026, 6, 5), 25)
    assert len(bundle) == 1  # old (out of window) + link-less rows dropped
    art = bundle.articles[0]
    assert art.title == "AI memory boom - Reuters"
    assert art.source == "Reuters"
    assert str(art.url).endswith("/abc")
    assert art.published_at.date() == date(2026, 6, 2)
    assert art.sentiment is None


def test_malformed_xml_is_empty() -> None:
    assert len(_articles_from_xml("x", "not xml <<", date(2026, 5, 25), date(2026, 6, 5), 25)) == 0


def _provider(handler: Any) -> GoogleNewsRssProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleNewsRssProvider(
        Settings(_env_file=None), DiskCache(Path("/tmp/gnrss_cache"), 0),  # ttl 0 -> no caching
        HttpJson("google_news_rss", client=client),
    )


def test_topic_fetch_via_mock_transport(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, text=_RSS)

    provider = GoogleNewsRssProvider(
        Settings(_env_file=None), DiskCache(tmp_path, 100),
        HttpJson("google_news_rss", client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    bundle = provider.get_topic_news(["AI memory"], date(2026, 5, 25), date(2026, 6, 5))
    assert len(bundle) == 1
    q = seen["params"]["q"]
    assert "AI memory" in q and "when:" in q  # OR terms + relative window appended


def test_is_keyless_always_available(tmp_path: Path) -> None:
    provider = GoogleNewsRssProvider(Settings(_env_file=None), DiskCache(tmp_path, 100))
    assert provider.available() is True
