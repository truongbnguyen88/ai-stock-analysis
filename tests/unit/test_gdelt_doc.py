"""GDELT DOC normalization + fetch via httpx.MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.gdelt_doc import (
    GdeltDocProvider,
    _articles_from_payload,
    _parse_seendate,
)
from stock_agent.settings import Settings

_PAYLOAD: dict[str, Any] = {
    "articles": [
        {
            "url": "https://site.com/newest",
            "title": "Humanoid robot startup raises funding",
            "seendate": "20260601T120000Z",
            "domain": "site.com",
            "language": "English",
        },
        {
            "url": "https://site.com/older",
            "title": "Automation expands in factories",
            "seendate": "20260530120000",  # no T/Z variant
            "domain": "site.com",
            "language": "English",
        },
        {  # filtered out: non-English
            "url": "https://de.site.com/x",
            "title": "Roboter Nachrichten",
            "seendate": "20260601T100000Z",
            "domain": "de.site.com",
            "language": "German",
        },
        {"title": "missing url", "seendate": "20260601T120000Z", "language": "English"},
    ]
}


def test_parse_seendate_variants() -> None:
    assert _parse_seendate("20260601T120000Z").year == 2026  # type: ignore[union-attr]
    assert _parse_seendate("20260530120000").month == 5  # type: ignore[union-attr]
    assert _parse_seendate("garbage") is None


def test_normalizes_and_filters_language() -> None:
    arts = _articles_from_payload(_PAYLOAD, language="english", top_n=25)
    assert [a.title for a in arts] == [
        "Humanoid robot startup raises funding",
        "Automation expands in factories",
    ]  # German + url-less rows dropped
    assert arts[0].source == "site.com"
    assert arts[0].sentiment is None  # DOC ArtList has no tone


def test_top_n_truncates_preserving_order() -> None:
    arts = _articles_from_payload(_PAYLOAD, language="english", top_n=1)
    assert len(arts) == 1
    assert arts[0].title == "Humanoid robot startup raises funding"


def test_non_list_payload_is_empty() -> None:
    assert _articles_from_payload({"articles": "oops"}, language="english", top_n=5) == []


def _provider(tmp_path: Path, handler: Any) -> GdeltDocProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GdeltDocProvider(
        settings=Settings(_env_file=None),
        cache=DiskCache(tmp_path, ttl_seconds=100),
        http=HttpJson("gdelt_doc", client=client),
    )


def test_fetch_via_mock_transport(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    provider = _provider(tmp_path, handler)
    bundle = provider.get_topic_news(
        "(robotics OR \"humanoid robot\")", date(2026, 5, 25), date(2026, 6, 1), top_n=25
    )
    assert len(bundle) == 2
    assert seen["params"]["mode"] == "ArtList"
    assert seen["params"]["sort"] == "DateDesc"
    assert seen["params"]["startdatetime"] == "20260525000000"


def test_keyless_provider_is_available(tmp_path: Path) -> None:
    provider = GdeltDocProvider(Settings(_env_file=None), DiskCache(tmp_path, 100))
    assert provider.available() is True
