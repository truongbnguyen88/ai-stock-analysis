"""RAG corpus status helper (observability) — offline, fakes for store + manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

import stock_agent.rag.status as status_mod
from stock_agent.rag.status import corpus_status
from stock_agent.settings import Settings


class _Entry:
    def __init__(self, ticker: str, filing_date: str) -> None:
        self.ticker = ticker
        self.filing_date = filing_date


def _fake_manifest(entries: dict[str, _Entry]) -> type:
    class _Man:
        @staticmethod
        def load(_path: Path) -> _Man:
            m = _Man()
            m.entries = entries  # type: ignore[attr-defined]
            return m

    return _Man


def test_corpus_status_aggregates_embedder_collection_and_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Store:
        def count(self) -> int:
            return 42

    entries = {
        "a": _Entry("NVDA", "2024-01-01"),
        "b": _Entry("AAPL", "2025-02-02"),
        "c": _Entry("NVDA", "2023-06-06"),  # earliest; NVDA repeats -> 2 distinct tickers
    }
    monkeypatch.setattr(status_mod, "build_vector_store", lambda s: _Store())
    monkeypatch.setattr(status_mod, "DownloadManifest", _fake_manifest(entries))

    cs = corpus_status(
        Settings(_env_file=None, embedding_provider="voyage", documents_dir=tmp_path)
    )
    assert cs.provider == "voyage" and cs.embedder == "voyage-voyage-4"
    assert cs.collection == "filings-voyage-voyage-4" and cs.chunks == 42
    assert cs.filings == 3 and cs.tickers == 2
    assert cs.earliest == "2023-06-06" and cs.latest == "2025-02-02"
    assert "voyage-voyage-4" in cs.one_line and "42 chunks" in cs.one_line
    assert "fresh to 2025-02-02" in cs.one_line


def test_corpus_status_store_unavailable_is_graceful_and_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from structlog.testing import capture_logs

    class _BrokenStore:
        def count(self) -> int:
            raise RuntimeError("chromadb not installed")

    monkeypatch.setattr(status_mod, "build_vector_store", lambda s: _BrokenStore())
    monkeypatch.setattr(status_mod, "DownloadManifest", _fake_manifest({}))

    with capture_logs() as logs:
        cs = corpus_status(Settings(_env_file=None, documents_dir=tmp_path))
    assert cs.chunks == -1  # store failure -> sentinel, not an exception
    assert cs.filings == 0 and cs.earliest is None and cs.latest is None
    assert "unavailable" in cs.one_line
    # Observability: the swallowed cause must be logged (not silent) so a red dot is diagnosable.
    warnings = [e for e in logs if e["event"] == "corpus.store_unavailable"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert "chromadb not installed" in warnings[0]["error"]  # the real exception text is preserved


def test_corpus_status_local_provider_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The badge must reflect the *actual* provider, so a drift back to local is visible.
    class _Store:
        def count(self) -> int:
            return 7

    monkeypatch.setattr(status_mod, "build_vector_store", lambda s: _Store())
    monkeypatch.setattr(status_mod, "DownloadManifest", _fake_manifest({}))
    cs = corpus_status(Settings(_env_file=None, embedding_provider="local"))
    assert cs.provider == "local" and cs.embedder.startswith("local-")
    assert "voyage" not in cs.embedder
