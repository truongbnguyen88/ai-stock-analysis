"""Tiingo per-ticker news — normalization + window filtering via MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.tiingo import TiingoProvider
from stock_agent.settings import Settings

_PAYLOAD: list[dict[str, Any]] = [
    {
        "id": 1,
        "title": "NVDA data-center revenue beats",
        "url": "https://tiingo.com/nvda-a",
        "description": "Blackwell ramp drives the quarter.",
        "publishedDate": "2026-06-02T12:00:00Z",  # trailing Z → UTC
        "source": "reuters.com",
        "tickers": ["nvda"],
        "tags": ["Technology"],
    },
    {  # dropped: outside the [start, end] window
        "id": 2,
        "title": "old news",
        "url": "https://tiingo.com/nvda-old",
        "description": "stale",
        "publishedDate": "2026-05-01T08:00:00Z",
        "source": "reuters.com",
    },
    {  # dropped: missing url
        "id": 3,
        "title": "no url",
        "publishedDate": "2026-06-02T09:00:00Z",
    },
]


def _provider(tmp_path: Path, handler: Any) -> TiingoProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TiingoProvider(
        Settings(_env_file=None, tiingo_api_key="k"),
        DiskCache(tmp_path, ttl_seconds=100),
        HttpJson("tiingo", client=client),
    )


def test_normalizes_and_filters_window(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_company_news(
        "nvda", date(2026, 5, 25), date(2026, 6, 3)
    )
    assert [a.title for a in bundle.articles] == ["NVDA data-center revenue beats"]
    art = bundle.articles[0]
    assert art.source == "reuters.com"
    assert art.summary is not None and art.summary.startswith("Blackwell")
    assert art.published_at.tzinfo is not None  # aware UTC
    assert art.sentiment is None  # Tiingo supplies no sentiment
    # token + window + lowercased ticker passed through
    assert seen["params"]["tickers"] == "nvda"
    assert seen["params"]["startDate"] == "2026-05-25"
    assert seen["params"]["token"] == "k"


def test_non_list_payload_is_empty(tmp_path: Path) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "error"})

    bundle = _provider(tmp_path, handler).get_company_news(
        "NVDA", date(2026, 5, 25), date(2026, 6, 3)
    )
    assert len(bundle) == 0


def test_availability_requires_key(tmp_path: Path) -> None:
    assert TiingoProvider(Settings(_env_file=None), DiskCache(tmp_path, 100)).available() is False
    assert TiingoProvider(
        Settings(_env_file=None, tiingo_api_key="k"), DiskCache(tmp_path, 100)
    ).available()
