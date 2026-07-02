"""Guardian topic-news provider — normalization via httpx.MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.guardian import GuardianProvider, _build_query
from stock_agent.settings import Settings

_PAYLOAD: dict[str, Any] = {
    "response": {
        "status": "ok",
        "results": [
            {
                "webTitle": "AI memory demand surges",
                "webUrl": "https://theguardian.com/tech/a",
                "webPublicationDate": "2026-06-02T09:00:00Z",
                "fields": {"trailText": "HBM shortages ripple through the supply chain."},
            },
            {  # dropped: missing webUrl
                "webTitle": "no url",
                "webPublicationDate": "2026-06-01T09:00:00Z",
            },
        ],
    }
}


def _provider(tmp_path: Path, handler: Any, **kw: Any) -> GuardianProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GuardianProvider(
        Settings(_env_file=None, guardian_api_key="k", **kw),
        DiskCache(tmp_path, ttl_seconds=100),
        HttpJson("guardian", client=client),
    )


def test_normalizes_results(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_topic_news(
        ["AI memory", "high bandwidth memory"], date(2026, 5, 25), date(2026, 6, 2), top_n=25
    )
    assert len(bundle) == 1  # url-less row dropped
    art = bundle.articles[0]
    assert art.title == "AI memory demand surges"
    assert art.source == "The Guardian"
    assert art.summary.startswith("HBM shortages")
    assert art.sentiment is None  # Guardian supplies no sentiment
    # OR-joined query, phrases quoted; date window + api-key passed through.
    assert seen["params"]["q"] == '"AI memory" OR "high bandwidth memory"'
    assert seen["params"]["from-date"] == "2026-05-25"
    assert seen["params"]["api-key"] == "k"


def test_top_n_truncates(tmp_path: Path) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_topic_news(
        ["AI memory"], date(2026, 5, 25), date(2026, 6, 2), top_n=1
    )
    assert len(bundle) == 1


def test_availability_requires_key(tmp_path: Path) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PAYLOAD)

    assert _provider(tmp_path, handler).available() is True
    no_key = GuardianProvider(Settings(_env_file=None), DiskCache(tmp_path, 100))
    assert no_key.available() is False


def test_build_query_quotes_phrases() -> None:
    assert _build_query(["robotics", "humanoid robot"]) == 'robotics OR "humanoid robot"'
