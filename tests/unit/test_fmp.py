"""FMP per-ticker news — normalization + window filtering via MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.fmp import FmpProvider
from stock_agent.settings import Settings

_PAYLOAD: list[dict[str, Any]] = [
    {
        "symbol": "NVDA",
        "publishedDate": "2026-06-02 12:00:00",
        "title": "NVDA data-center revenue beats",
        "site": "reuters.com",
        "text": "Blackwell ramp drives the quarter.",
        "url": "https://reuters.com/nvda-a",
    },
    {  # dropped: outside the [start, end] window
        "symbol": "NVDA",
        "publishedDate": "2026-05-01 08:00:00",
        "title": "old news",
        "site": "reuters.com",
        "text": "stale",
        "url": "https://reuters.com/nvda-old",
    },
    {"symbol": "NVDA", "title": "no date", "url": "https://x.com/y"},  # dropped: missing date
]


def _provider(tmp_path: Path, handler: Any) -> FmpProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FmpProvider(
        Settings(_env_file=None, fmp_api_key="k"),
        DiskCache(tmp_path, ttl_seconds=100),
        HttpJson("fmp", client=client),
    )


def test_normalizes_and_filters_window(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_company_news(
        "nvda", date(2026, 5, 25), date(2026, 6, 3)
    )
    assert [a.title for a in bundle.articles] == [
        "NVDA data-center revenue beats"
    ]  # window + date-guard
    art = bundle.articles[0]
    assert art.source == "reuters.com"
    assert art.summary is not None and art.summary.startswith("Blackwell")
    assert art.published_at.tzinfo is not None  # naive FMP date coerced to aware UTC
    assert seen["params"]["symbols"] == "NVDA"


def test_non_list_payload_is_empty(tmp_path: Path) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "bad"})

    bundle = _provider(tmp_path, handler).get_company_news(
        "NVDA", date(2026, 5, 25), date(2026, 6, 3)
    )
    assert len(bundle) == 0


def test_availability_requires_key(tmp_path: Path) -> None:
    assert FmpProvider(Settings(_env_file=None), DiskCache(tmp_path, 100)).available() is False
    assert FmpProvider(
        Settings(_env_file=None, fmp_api_key="k"), DiskCache(tmp_path, 100)
    ).available()
