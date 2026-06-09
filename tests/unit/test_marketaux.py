"""Marketaux normalization (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.marketaux import (
    MarketauxProvider,
    _articles_from_payload,
    _build_search,
    _parse_dt,
)
from stock_agent.settings import Settings


def test_normalizes_data_array() -> None:
    payload = {
        "data": [
            {
                "title": "T",
                "url": "https://x.com/a",
                "source": "Src",
                "published_at": "2024-01-15T13:30:00.000000Z",
                "description": "D",
                "snippet": "short",
            }
        ]
    }
    bundle = _articles_from_payload("AAPL", payload)
    assert len(bundle) == 1
    art = bundle.articles[0]
    assert art.summary == "D"  # description preferred over snippet
    assert art.published_at.year == 2024


def test_missing_data_key_returns_empty() -> None:
    assert len(_articles_from_payload("AAPL", {})) == 0


def test_parse_dt_handles_trailing_z() -> None:
    parsed = _parse_dt("2024-01-15T13:30:00Z")
    assert parsed.tzinfo is not None
    assert parsed.year == 2024


# ---- topic search (Enhancement C — secondary topic provider) ------------------
def test_build_search_ors_phrases() -> None:
    assert _build_search(["robotics", "humanoid robot"]) == 'robotics | "humanoid robot"'


def test_topic_fetch_via_mock_transport(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "title": "Robotics roundup",
                        "url": "https://x.com/r",
                        "source": "Src",
                        "published_at": "2026-06-01T10:00:00Z",
                        "description": "D",
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = MarketauxProvider(
        settings=Settings(_env_file=None, marketaux_api_key="k"),
        cache=DiskCache(tmp_path, ttl_seconds=100),
        http=HttpJson("marketaux", client=client),
    )
    bundle = provider.get_topic_news(
        ["robotics", "humanoid robot"], date(2026, 5, 25), date(2026, 6, 1)
    )
    assert len(bundle) == 1
    # Native Marketaux query: OR via '|', published-window params present.
    assert seen["params"]["search"] == 'robotics | "humanoid robot"'
    assert seen["params"]["published_after"] == "2026-05-25"
