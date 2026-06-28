"""documents/ download orchestration (RAG P1) — offline, fake provider."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from stock_agent.documents.download import (
    DEFAULT_FORMS,
    bulk_download,
    download_filings,
    filing_dir,
    manifest_path,
)
from stock_agent.documents.manifest import DownloadManifest
from stock_agent.documents.ticker_cik import load_universe, normalize_ticker
from stock_agent.schemas.documents import DocumentType, FilingRef


class FakeEdgar:
    """Stand-in for SecEdgarProvider: serves canned filings + counts downloads."""

    def __init__(self, refs: list[FilingRef]) -> None:
        self._refs = refs
        self.download_calls = 0

    def list_filings(
        self,
        ticker: str,
        forms: Sequence[DocumentType],
        *,
        limit: int = 10,
        since: date | None = None,
    ) -> list[FilingRef]:
        refs = [r for r in self._refs if r.form in set(forms)]
        if since is not None:
            refs = [r for r in refs if r.filing_date >= since]
        return refs[:limit]

    def download_filing(self, ref: FilingRef) -> str:
        self.download_calls += 1
        return f"<html>{ref.document_id} body</html>"


class MultiEdgar:
    """Bulk-test provider: synthesizes one 10-K per requested ticker; can fail chosen tickers."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.download_calls = 0

    def list_filings(
        self,
        ticker: str,
        forms: Sequence[DocumentType],
        *,
        limit: int = 10,
        since: date | None = None,
    ) -> list[FilingRef]:
        if ticker in self.fail:
            raise RuntimeError(f"CIK not found for {ticker}")  # whole-ticker failure
        return [
            FilingRef(
                ticker=ticker, cik="0000000000", form="10-K",
                filing_date=date(2025, 2, 26), accession_number=f"{ticker}-25-000001",
                primary_document=f"{ticker.lower()}-10k.htm",
            )
        ]

    def download_filing(self, ref: FilingRef) -> str:
        self.download_calls += 1
        return f"<html>{ref.document_id} body</html>"


def _refs() -> list[FilingRef]:
    return [
        FilingRef(
            ticker="NVDA", cik="0001045810", form="10-K",
            filing_date=date(2025, 2, 26), accession_number="0001045810-25-000017",
            primary_document="nvda-10k.htm",
        ),
        FilingRef(
            ticker="NVDA", cik="0001045810", form="10-Q",
            filing_date=date(2024, 11, 20), accession_number="0001045810-24-000316",
            primary_document="nvda-10q.htm",
        ),
    ]


def test_normalize_ticker() -> None:
    assert normalize_ticker(" nvda ") == "NVDA"


def test_download_writes_filing_metadata_and_manifest(tmp_path: Path) -> None:
    fake = FakeEdgar(_refs())
    r = download_filings("nvda", fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert r.ticker == "NVDA"
    assert len(r.downloaded) == 2 and not r.skipped and not r.errors
    assert fake.download_calls == 2

    # Files on disk in the expected layout.
    d = filing_dir(tmp_path, _refs()[0])
    assert (d / "filing.html").read_text().startswith("<html>NVDA:10-K")
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["ticker"] == "NVDA" and meta["document_type"] == "10-K"
    assert meta["source_url"].startswith("https://www.sec.gov/Archives/")

    # Manifest records both.
    man = DownloadManifest.load(manifest_path(tmp_path))
    assert len(man.entries) == 2
    assert man.has("NVDA:10-K:2025-02-26:0001045810-25-000017")


def test_rerun_is_idempotent_no_redownload(tmp_path: Path) -> None:
    fake = FakeEdgar(_refs())
    download_filings("NVDA", fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert fake.download_calls == 2
    # Second run: everything already on disk -> all skipped, no new downloads.
    r2 = download_filings("NVDA", fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert fake.download_calls == 2  # unchanged
    assert len(r2.skipped) == 2 and not r2.downloaded


def test_raw_not_overwritten(tmp_path: Path) -> None:
    fake = FakeEdgar(_refs())
    download_filings("NVDA", fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    d = filing_dir(tmp_path, _refs()[0])
    (d / "filing.html").write_text("EDITED LOCALLY")  # simulate a local edit
    download_filings("NVDA", fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert (d / "filing.html").read_text() == "EDITED LOCALLY"  # preserved, not clobbered


class _Boom(FakeEdgar):
    def download_filing(self, ref: FilingRef) -> str:
        raise RuntimeError("network down")


def test_per_filing_error_is_recorded_not_fatal(tmp_path: Path) -> None:
    r = download_filings("NVDA", _Boom(_refs()), documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert len(r.errors) == 2 and not r.downloaded
    assert DownloadManifest.load(manifest_path(tmp_path)).entries == {}


def test_default_forms_are_the_mvp_corpus() -> None:
    assert DEFAULT_FORMS == ("10-K", "10-Q", "8-K")


def test_load_universe(tmp_path: Path) -> None:
    p = tmp_path / "u.txt"
    p.write_text("# comment\nNVDA\n  msft \n\n# another\nAMD\n")
    assert load_universe(p) == ["NVDA", "MSFT", "AMD"]


def test_load_universe_strips_inline_comments(tmp_path: Path) -> None:
    # graph_universe.txt annotates each ticker with an inline sector comment; the loader must
    # strip everything from '#' onward (full-line and inline) and keep just the symbol.
    p = tmp_path / "u.txt"
    p.write_text("# header\nNVDA  # GPUs / AI accelerators\nMU\t# DRAM / HBM\n")
    assert load_universe(p) == ["NVDA", "MU"]


# ---- 9d: date-bounded history + bulk download --------------------------------


def test_download_since_floor_excludes_older_filings(tmp_path: Path) -> None:
    # _refs() has a 2025-02-26 10-K and a 2024-11-20 10-Q; floor at 2025-01-01 keeps only the 10-K.
    fake = FakeEdgar(_refs())
    r = download_filings(
        "NVDA", fake, documents_dir=tmp_path, limit=10, since=date(2025, 1, 1)  # type: ignore[arg-type]
    )
    assert len(r.downloaded) == 1 and fake.download_calls == 1
    assert DownloadManifest.load(manifest_path(tmp_path)).has(
        "NVDA:10-K:2025-02-26:0001045810-25-000017"
    )


def test_bulk_download_aggregates_across_tickers(tmp_path: Path) -> None:
    fake = MultiEdgar()
    res = bulk_download(["nvda", "MSFT"], fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert res.tickers == 2 and res.downloaded == 2 and not res.failed_tickers
    assert {r.ticker for r in res.per_ticker} == {"NVDA", "MSFT"}
    assert fake.download_calls == 2


def test_bulk_download_isolates_whole_ticker_failure(tmp_path: Path) -> None:
    fake = MultiEdgar(fail={"MSFT"})  # MSFT's listing raises (bad CIK)
    res = bulk_download(["NVDA", "MSFT"], fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert res.tickers == 2  # both attempted
    assert res.downloaded == 1  # NVDA still succeeded — the run was not aborted
    assert len(res.failed_tickers) == 1 and res.failed_tickers[0].startswith("MSFT:")
    assert [r.ticker for r in res.per_ticker] == ["NVDA"]


def test_bulk_download_is_idempotent(tmp_path: Path) -> None:
    fake = MultiEdgar()
    bulk_download(["NVDA"], fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    res2 = bulk_download(["NVDA"], fake, documents_dir=tmp_path, limit=5)  # type: ignore[arg-type]
    assert res2.downloaded == 0 and res2.skipped == 1  # already on disk -> skipped
    assert fake.download_calls == 1  # no re-fetch
