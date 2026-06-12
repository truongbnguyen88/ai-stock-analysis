"""SEC EDGAR provider (RAG P1) — pure parsers + fetch via MockTransport (offline)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import SymbolNotFound
from stock_agent.providers.sec_edgar import (
    SecEdgarProvider,
    _parse_cik_map,
    _parse_filings,
    _parse_form4_filings,
)
from stock_agent.schemas.documents import FilingRef
from stock_agent.settings import MissingSettingError, Settings

_CIK_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
}

_SUBMISSIONS = {
    "filings": {
        "recent": {
            # Form 4 / 4-A rows appended (older dates, so the 10-K/10-Q ordering tests
            # are unaffected); "4/A" is an amendment excluded by exact-match filtering.
            "form": ["10-K", "10-Q", "8-K", "10-K/A", "4", "4", "4/A"],
            "filingDate": [
                "2025-02-26", "2024-11-20", "2024-08-28", "2024-03-01",
                "2024-12-01", "2024-09-15", "2024-07-01",
            ],
            "accessionNumber": [
                "0001045810-25-000017",
                "0001045810-24-000316",
                "0001045810-24-000200",
                "0001045810-24-000099",
                "0001234567-24-000045",
                "0001234567-24-000030",
                "0001234567-24-000021",
            ],
            "primaryDocument": [
                "nvda-10k.htm", "nvda-10q.htm", "nvda-8k.htm", "nvda-amend.htm",
                "form4_a.xml", "form4_b.xml", "form4_amend.xml",
            ],
        }
    }
}


# ---- pure parsers ------------------------------------------------------------
def test_parse_cik_map_zero_pads() -> None:
    m = _parse_cik_map(_CIK_PAYLOAD)
    assert m["AAPL"] == "0000320193"
    assert m["NVDA"] == "0001045810"  # 10-digit zero-padded


def test_parse_cik_map_handles_garbage() -> None:
    assert _parse_cik_map("nope") == {}
    assert _parse_cik_map({"0": {"ticker": "X"}}) == {}  # missing cik_str


def test_parse_filings_filters_forms_and_excludes_amendments() -> None:
    refs = _parse_filings("NVDA", "0001045810", _SUBMISSIONS, forms=["10-K"], limit=10)
    assert [r.form for r in refs] == ["10-K"]  # 10-K/A excluded (exact match only)
    assert refs[0].accession_number == "0001045810-25-000017"


def test_parse_filings_newest_first_and_limit() -> None:
    refs = _parse_filings(
        "NVDA", "0001045810", _SUBMISSIONS, forms=["10-K", "10-Q", "8-K"], limit=2
    )
    assert len(refs) == 2
    assert refs[0].filing_date.isoformat() == "2025-02-26"  # newest first
    assert refs[1].filing_date.isoformat() == "2024-11-20"


def test_parse_filings_since_floor_excludes_older() -> None:
    # Floor at 2024-10-01 drops the 2024-08-28 8-K; keeps the 10-K + 10-Q (9d history window).
    refs = _parse_filings(
        "NVDA", "0001045810", _SUBMISSIONS,
        forms=["10-K", "10-Q", "8-K"], limit=10, since=date(2024, 10, 1),
    )
    dates = [r.filing_date.isoformat() for r in refs]
    assert dates == ["2025-02-26", "2024-11-20"]  # 2024-08-28 excluded by the floor


def test_filing_ref_url_and_document_id() -> None:
    ref = FilingRef(
        ticker="NVDA",
        cik="0001045810",
        form="10-K",
        filing_date=date(2025, 2, 26),
        accession_number="0001045810-25-000017",
        primary_document="nvda-10k.htm",
    )
    # CIK as bare int, accession dashes stripped:
    assert ref.url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000017/nvda-10k.htm"
    )
    assert ref.document_id == "NVDA:10-K:2025-02-26:0001045810-25-000017"


# ---- provider via MockTransport ----------------------------------------------
def _provider(tmp_path: Path, handler: Any) -> SecEdgarProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SecEdgarProvider(
        settings=Settings(_env_file=None, sec_user_agent="Tester test@example.com"),
        cache=DiskCache(tmp_path, ttl_seconds=100),
        http=HttpJson("sec_edgar", client=client),
        min_request_interval=0.0,  # no real sleeps in tests
    )


def _router(req: httpx.Request) -> httpx.Response:
    url = str(req.url)
    if "company_tickers.json" in url:
        return httpx.Response(200, json=_CIK_PAYLOAD)
    if "submissions" in url:
        return httpx.Response(200, json=_SUBMISSIONS)
    if "Archives/edgar/data" in url:
        if url.endswith(".xml"):
            return httpx.Response(200, text="<ownershipDocument><documentType>4</documentType>"
                                            "</ownershipDocument>")
        return httpx.Response(200, text="<html><body>10-K body</body></html>")
    return httpx.Response(404)


def test_get_cik(tmp_path: Path) -> None:
    p = _provider(tmp_path, _router)
    assert p.get_cik("nvda") == "0001045810"  # case-insensitive
    with pytest.raises(SymbolNotFound):
        p.get_cik("ZZZZ")


def test_list_filings_and_download(tmp_path: Path) -> None:
    p = _provider(tmp_path, _router)
    refs = p.list_filings("NVDA", ["10-K", "10-Q"], limit=5)
    assert {r.form for r in refs} == {"10-K", "10-Q"}
    html = p.download_filing(refs[0])
    assert "10-K body" in html


def test_parse_form4_filings_excludes_amendments_and_orders() -> None:
    refs = _parse_form4_filings("NVDA", "0001045810", _SUBMISSIONS, limit=10)
    # Two "4" filings (newest first); "4/A" amendment excluded by exact match.
    assert [r.filing_date.isoformat() for r in refs] == ["2024-12-01", "2024-09-15"]
    assert all(r.primary_document.endswith(".xml") for r in refs)


def test_parse_form4_filings_since_floor() -> None:
    refs = _parse_form4_filings(
        "NVDA", "0001045810", _SUBMISSIONS, limit=10, since=date(2024, 10, 1)
    )
    assert [r.filing_date.isoformat() for r in refs] == ["2024-12-01"]  # 09-15 dropped


def test_list_form4_filings_and_download(tmp_path: Path) -> None:
    p = _provider(tmp_path, _router)
    refs = p.list_form4_filings("NVDA")
    assert len(refs) == 2
    xml = p.download_form4(refs[0])
    assert "ownershipDocument" in xml  # fetched the XML branch, not the HTML body


def test_provider_builds_user_agent_header(tmp_path: Path) -> None:
    # The lazily-built client carries the SEC-required UA from settings (no network).
    p = SecEdgarProvider(
        Settings(_env_file=None, sec_user_agent="Tester test@example.com"),
        DiskCache(tmp_path, 100),
    )
    headers = p._client()._headers
    assert headers is not None
    assert headers["User-Agent"] == "Tester test@example.com"


def test_injected_http_headers_reach_the_wire(tmp_path: Path) -> None:
    # HttpJson forwards its headers on each request (so the provider's UA is sent).
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, json=_CIK_PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    http = HttpJson("sec_edgar", client=client, headers={"User-Agent": "UA contact@x.com"})
    p = SecEdgarProvider(
        Settings(_env_file=None), DiskCache(tmp_path, 100), http=http, min_request_interval=0.0
    )
    p.get_cik("AAPL")
    assert seen["ua"] == "UA contact@x.com"


def test_available_requires_user_agent() -> None:
    assert SecEdgarProvider(Settings(_env_file=None), DiskCache(Path("."), 1)).available() is False
    ua = Settings(_env_file=None, sec_user_agent="X y@z.com")
    assert SecEdgarProvider(ua, DiskCache(Path("."), 1)).available() is True


def test_missing_ua_raises_on_live_call(tmp_path: Path) -> None:
    # No injected http + no UA -> building the real client requires the UA setting.
    p = SecEdgarProvider(
        Settings(_env_file=None), DiskCache(tmp_path, 100), min_request_interval=0.0
    )
    with pytest.raises(MissingSettingError):
        p.get_cik("AAPL")
