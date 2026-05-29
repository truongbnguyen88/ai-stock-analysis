"""Finnhub normalization + HTTP error mapping (via httpx.MockTransport)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderRateLimit
from stock_agent.providers.finnhub import FinnhubProvider, _articles_from_payload
from stock_agent.settings import Settings


def test_normalizes_payload() -> None:
    payload = [
        {
            "datetime": 1704196800,  # 2024-01-02 UTC
            "headline": "H",
            "summary": "S",
            "source": "Reuters",
            "url": "https://x.com/a",
            "category": "company",
        }
    ]
    bundle = _articles_from_payload("AAPL", payload)
    assert len(bundle) == 1
    art = bundle.articles[0]
    assert art.title == "H"
    assert art.topics == ["company"]
    assert art.published_at.year == 2024


def test_skips_malformed_rows() -> None:
    payload: list[dict[str, Any]] = [
        {"headline": "no url"},
        {"url": "https://x.com", "datetime": 1, "headline": "ok"},
    ]
    assert len(_articles_from_payload("AAPL", payload)) == 1


def _provider(tmp_path: Path, handler: Any) -> tuple[FinnhubProvider, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FinnhubProvider(
        settings=Settings(_env_file=None, finnhub_api_key="k"),
        cache=DiskCache(tmp_path, ttl_seconds=100),
        http=HttpJson("finnhub", client=client),
    )
    return provider, client


def test_fetch_via_mock_transport(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"datetime": 1704196800, "headline": "H", "url": "https://x.com/a", "source": "S"}
            ],
        )

    provider, client = _provider(tmp_path, handler)
    bundle = provider.get_company_news("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert len(bundle) == 1
    client.close()


def test_rate_limit_maps_to_typed_error(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    provider, client = _provider(tmp_path, handler)
    with pytest.raises(ProviderRateLimit):
        provider.get_company_news("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    client.close()


def test_unavailable_without_key(tmp_path: Path) -> None:
    provider = FinnhubProvider(settings=Settings(_env_file=None), cache=DiskCache(tmp_path, 100))
    assert provider.available() is False
