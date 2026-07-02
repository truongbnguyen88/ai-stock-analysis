"""NewsData.io provider — company + topic normalization/filtering via MockTransport (no live)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.newsdata import NewsDataProvider
from stock_agent.settings import Settings

_PAYLOAD: dict[str, Any] = {
    "status": "success",
    "results": [
        {
            "title": "NVDA supply update",
            "link": "https://site.com/a",
            "pubDate": "2026-06-02 12:00:00",
            "source_id": "site",
            "description": "Supply chain note.",
        },
        {  # dropped: outside window
            "title": "old",
            "link": "https://site.com/old",
            "pubDate": "2026-05-01 00:00:00",
            "source_id": "site",
        },
    ],
}


def _provider(tmp_path: Path, handler: Any) -> NewsDataProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return NewsDataProvider(
        Settings(_env_file=None, newsdata_api_key="k"),
        DiskCache(tmp_path, ttl_seconds=100),
        HttpJson("newsdata", client=client),
    )


def test_company_news_uses_business_category(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    bundle = _provider(tmp_path, handler).get_company_news(
        "nvda", date(2026, 5, 25), date(2026, 6, 3)
    )
    assert [a.title for a in bundle.articles] == ["NVDA supply update"]  # window filter applied
    assert bundle.articles[0].source == "site"
    assert seen["params"]["q"] == "NVDA"
    assert seen["params"]["category"] == "business"  # finance bias on the ticker path


def test_topic_news_or_query_no_category(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    _provider(tmp_path, handler).get_topic_news(
        ["AI memory", "HBM"], date(2026, 5, 25), date(2026, 6, 3)
    )
    assert seen["params"]["q"] == '"AI memory" OR HBM'
    assert "category" not in seen["params"]  # themes are broader than finance


def test_availability_requires_key(tmp_path: Path) -> None:
    assert NewsDataProvider(Settings(_env_file=None), DiskCache(tmp_path, 100)).available() is False
