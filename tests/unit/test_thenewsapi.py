"""TheNewsAPI topic-news provider — normalization via httpx.MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.thenewsapi import TheNewsApiProvider, _build_query
from stock_agent.settings import Settings

_PAYLOAD: dict[str, Any] = {
    "meta": {"found": 42, "returned": 2, "limit": 3, "page": 1},
    "data": [
        {
            "uuid": "abc",
            "title": "AI memory demand surges",
            "description": "HBM shortages ripple through the supply chain.",
            "snippet": "High-bandwidth memory...",
            "url": "https://example.com/tech/a",
            "published_at": "2026-06-02T09:00:00.000000Z",
            "source": "example.com",
            "categories": ["business"],
        },
        {  # dropped: outside the window
            "uuid": "def",
            "title": "old memory news",
            "description": "stale",
            "url": "https://example.com/tech/old",
            "published_at": "2026-05-01T09:00:00.000000Z",
            "source": "example.com",
        },
        {  # dropped: missing url
            "uuid": "ghi",
            "title": "no url",
            "published_at": "2026-06-02T09:00:00.000000Z",
        },
    ],
}


def _provider(tmp_path: Path, handler: Any) -> TheNewsApiProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TheNewsApiProvider(
        Settings(_env_file=None, thenewsapi_api_key="k"),
        DiskCache(tmp_path, ttl_seconds=100),
        HttpJson("thenewsapi", client=client),
    )


def test_normalizes_and_filters_window(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_topic_news(
        ["AI memory", "high bandwidth memory"], date(2026, 5, 25), date(2026, 6, 2), top_n=25
    )
    assert len(bundle) == 1  # window + url-guard drop the other two
    art = bundle.articles[0]
    assert art.title == "AI memory demand surges"
    assert art.source == "example.com"
    assert art.summary is not None and art.summary.startswith("HBM shortages")
    assert art.sentiment is None
    # OR-joined via `|`, phrases quoted; published_before is end+1 day (exclusive-instant guard).
    assert seen["params"]["search"] == '"AI memory" | "high bandwidth memory"'
    assert seen["params"]["published_after"] == "2026-05-25"
    assert seen["params"]["published_before"] == "2026-06-03"
    assert seen["params"]["api_token"] == "k"


def test_top_n_truncates(tmp_path: Path) -> None:
    # Two in-window articles; top_n=1 keeps only the first.
    payload = {
        "data": [
            {
                "title": "a", "url": "https://x/a",
                "published_at": "2026-06-02T09:00:00.000000Z", "source": "x",
            },
            {
                "title": "b", "url": "https://x/b",
                "published_at": "2026-06-01T09:00:00.000000Z", "source": "x",
            },
        ]
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    bundle = _provider(tmp_path, handler).get_topic_news(
        ["AI memory"], date(2026, 5, 25), date(2026, 6, 2), top_n=1
    )
    assert len(bundle) == 1


def test_availability_requires_key(tmp_path: Path) -> None:
    assert TheNewsApiProvider(
        Settings(_env_file=None), DiskCache(tmp_path, 100)
    ).available() is False
    assert TheNewsApiProvider(
        Settings(_env_file=None, thenewsapi_api_key="k"), DiskCache(tmp_path, 100)
    ).available()


def test_build_query_or_joins_and_quotes_phrases() -> None:
    assert _build_query(["robotics", "humanoid robot"]) == 'robotics | "humanoid robot"'
