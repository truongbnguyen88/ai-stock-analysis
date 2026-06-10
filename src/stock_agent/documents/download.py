"""SEC filing download orchestration (RAG P1).

Lists filings via the EDGAR provider, downloads each primary document, and writes
it to the local store with a metadata sidecar — idempotently (raw is never
overwritten; already-downloaded filings are skipped via the manifest / filesystem).

Storage layout (under ``settings.documents_dir``):
    sec/{TICKER}/{FORM}/{filing_date}/filing.html
    sec/{TICKER}/{FORM}/{filing_date}/metadata.json
    sec/_manifest.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from stock_agent.documents.manifest import DownloadManifest, ManifestEntry, now_iso
from stock_agent.documents.ticker_cik import normalize_ticker
from stock_agent.logging_config import get_logger
from stock_agent.providers.sec_edgar import SecEdgarProvider
from stock_agent.schemas.documents import DocumentType, FilingRef

log = get_logger(__name__)

# Default MVP corpus + a small per-ticker cap (a 10-K is large; keep ingestion bounded).
DEFAULT_FORMS: tuple[DocumentType, ...] = ("10-K", "10-Q", "8-K")
_FILING_FILENAME = "filing.html"


class DownloadResult(BaseModel):
    """Summary of one ticker's download run (ids downloaded vs skipped)."""

    ticker: str
    downloaded: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)  # "document_id: message"


def filing_dir(base: Path, ref: FilingRef) -> Path:
    """Local directory for one filing: ``{base}/sec/{TICKER}/{FORM}/{filing_date}``."""
    return base / "sec" / ref.ticker / ref.form / ref.filing_date.isoformat()


def manifest_path(base: Path) -> Path:
    return base / "sec" / "_manifest.json"


def _write_metadata(directory: Path, ref: FilingRef) -> None:
    """Sidecar metadata.json next to the filing (read by the P2 parser)."""
    meta = {
        "ticker": ref.ticker,
        "document_type": ref.form,
        "source": "SEC",
        "source_url": ref.url,
        "filing_date": ref.filing_date.isoformat(),
        "document_id": ref.document_id,
        "accession_number": ref.accession_number,
        "primary_document": ref.primary_document,
        "section": None,
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def download_filings(
    ticker: str,
    provider: SecEdgarProvider,
    *,
    documents_dir: Path,
    forms: tuple[DocumentType, ...] = DEFAULT_FORMS,
    limit: int = 4,
) -> DownloadResult:
    """Download up to ``limit`` filings per requested form for ``ticker``.

    Idempotent: a filing already on disk (or in the manifest) is skipped, never
    re-fetched or overwritten. A per-filing failure is recorded and does not abort
    the rest of the run.
    """
    sym = normalize_ticker(ticker)
    result = DownloadResult(ticker=sym)
    mpath = manifest_path(documents_dir)
    manifest = DownloadManifest.load(mpath)

    refs = provider.list_filings(sym, forms, limit=limit)
    for ref in refs:
        directory = filing_dir(documents_dir, ref)
        html_path = directory / _FILING_FILENAME
        if html_path.exists() or manifest.has(ref.document_id):
            result.skipped.append(ref.document_id)
            continue
        try:
            text = provider.download_filing(ref)
        except Exception as exc:  # noqa: BLE001 — record + continue (one bad filing isn't fatal)
            log.warning("sec.download_failed", document_id=ref.document_id, error=str(exc))
            result.errors.append(f"{ref.document_id}: {exc}")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        html_path.write_text(text, encoding="utf-8")  # raw saved verbatim, never overwritten
        _write_metadata(directory, ref)
        manifest.add(
            ManifestEntry(
                document_id=ref.document_id,
                ticker=sym,
                form=ref.form,
                filing_date=ref.filing_date.isoformat(),
                accession_number=ref.accession_number,
                source_url=ref.url,
                path=str(html_path.relative_to(documents_dir)),
                downloaded_at=now_iso(),
            )
        )
        result.downloaded.append(ref.document_id)

    manifest.save(mpath)
    log.info(
        "sec.download",
        ticker=sym,
        downloaded=len(result.downloaded),
        skipped=len(result.skipped),
        errors=len(result.errors),
    )
    return result
