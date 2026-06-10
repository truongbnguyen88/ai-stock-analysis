"""Download manifest — idempotent record of what's already been fetched (RAG P1).

A small JSON file (under ``data/raw/sec/``) mapping ``document_id`` to a download
record, so re-running a download skips filings already on disk. Raw files are never
overwritten; the manifest is the queryable history of what exists.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from stock_agent.schemas.documents import DocumentType


class ManifestEntry(BaseModel):
    """One downloaded filing's provenance + local location."""

    document_id: str
    ticker: str
    form: DocumentType
    filing_date: str  # ISO date
    accession_number: str
    source_url: str
    path: str  # path to the saved primary document, relative to the documents dir
    downloaded_at: str  # ISO datetime


class DownloadManifest(BaseModel):
    """Map of ``document_id -> ManifestEntry`` with load/save + membership."""

    entries: dict[str, ManifestEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DownloadManifest:
        """Load the manifest at ``path`` (returns an empty manifest if absent)."""
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Persist the manifest (creates parent dirs); atomic temp+replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def has(self, document_id: str) -> bool:
        return document_id in self.entries

    def add(self, entry: ManifestEntry) -> None:
        self.entries[entry.document_id] = entry


def now_iso() -> str:
    """Local-timezone ISO timestamp for ``downloaded_at``."""
    return datetime.now().astimezone().isoformat()
