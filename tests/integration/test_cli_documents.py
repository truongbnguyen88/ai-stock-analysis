"""CLI `documents download-sec` (RAG P1 + 9d bulk/history) — arg handling, no network."""

from __future__ import annotations

import importlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import typer

from stock_agent.documents.download import DEFAULT_FORMS, BulkDownloadResult, DownloadResult
from stock_agent.documents.ticker_cik import normalize_ticker
from stock_agent.rag.pipeline import BulkIngestResult
from stock_agent.rag.status import CorpusStatus
from stock_agent.settings import Settings

app = importlib.import_module("stock_agent.cli.app")


def _patch(monkeypatch: Any, tmp_path: Path, *, ua: str | None = "Tester t@e.com") -> list[Any]:
    """Patch settings/logging + capture the single bulk_download call; return the call log."""
    calls: list[Any] = []

    def fake_bulk(  # type: ignore[no-untyped-def]
        tickers: Any, provider: Any, *, documents_dir: Path, forms: Any, limit: int, since: Any
    ):
        calls.append({"tickers": list(tickers), "forms": forms, "limit": limit, "since": since})
        per = [
            DownloadResult(ticker=normalize_ticker(t), downloaded=["a", "b"], skipped=[])
            for t in tickers
        ]
        return BulkDownloadResult(
            tickers=len(per), downloaded=2 * len(per), skipped=0, errors=0,
            failed_tickers=[], per_ticker=per,
        )

    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: Settings(
            _env_file=None, sec_user_agent=ua, documents_dir=tmp_path, cache_dir=tmp_path / ".cache"
        ),
    )
    monkeypatch.setattr(app, "bulk_download", fake_bulk)
    return calls


def test_download_sec_normalizes_ticker_and_uses_defaults(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    calls = _patch(monkeypatch, tmp_path)
    app.download_sec(
        ticker="nvda", download_all=False, forms=None, limit=3, since=None, years=0,
        universe=Path("configs/u.txt"),
    )
    assert calls == [{"tickers": ["NVDA"], "forms": DEFAULT_FORMS, "limit": 3, "since": None}]
    out = capsys.readouterr().out
    assert "downloaded 2" in out and "total: 1 tickers" in out


def test_download_sec_missing_user_agent_exits(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path, ua=None)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker="NVDA", download_all=False, forms=None, limit=4, since=None, years=0,
            universe=Path("x"),
        )
    assert exc.value.exit_code == 1


def test_download_sec_rejects_unsupported_form(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker="NVDA", download_all=False, forms=["DEF 14A"], limit=4, since=None, years=0,
            universe=Path("x"),
        )
    assert exc.value.exit_code == 1


def test_download_sec_requires_target(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker=None, download_all=False, forms=None, limit=4, since=None, years=0,
            universe=Path("x"),
        )
    assert exc.value.exit_code == 1


def test_download_sec_all_iterates_universe(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "load_universe", lambda p: ["NVDA", "MSFT"])
    app.download_sec(
        ticker=None, download_all=True, forms=["10-K"], limit=2, since=None, years=0,
        universe=tmp_path / "u.txt",
    )
    assert calls[0]["tickers"] == ["NVDA", "MSFT"]
    assert calls[0]["forms"] == ("10-K",) and calls[0]["limit"] == 2


def test_download_sec_years_sets_date_floor(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _patch(monkeypatch, tmp_path)
    app.download_sec(
        ticker="NVDA", download_all=False, forms=None, limit=4, since=None, years=3,
        universe=Path("x"),
    )
    assert calls[0]["since"] == date.today() - timedelta(days=365 * 3)


def test_download_sec_explicit_since_wins_over_years(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _patch(monkeypatch, tmp_path)
    app.download_sec(
        ticker="NVDA", download_all=False, forms=None, limit=4, since="2023-01-01", years=5,
        universe=Path("x"),
    )
    assert calls[0]["since"] == date(2023, 1, 1)  # --since takes precedence


def test_download_sec_bad_since_exits(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker="NVDA", download_all=False, forms=None, limit=4, since="not-a-date", years=0,
            universe=Path("x"),
        )
    assert exc.value.exit_code == 1


# ---- 9e: documents refresh (download new + incremental ingest) ---------------


def _patch_refresh(
    monkeypatch: Any, tmp_path: Path, *, downloaded: dict[str, list[str]]
) -> dict[str, Any]:
    """Patch settings/logging + capture refresh's download + ingest calls. ``downloaded`` maps
    each ticker to its list of newly-downloaded filing ids (empty = nothing new)."""
    seen: dict[str, Any] = {}

    def fake_bulk_download(tickers: Any, provider: Any, *, documents_dir, forms, limit, since):  # type: ignore[no-untyped-def]
        seen["dl_tickers"] = list(tickers)
        seen["since"] = since
        per = [DownloadResult(ticker=t, downloaded=downloaded[t], skipped=[]) for t in tickers]
        return BulkDownloadResult(
            tickers=len(per), downloaded=sum(len(d) for d in downloaded.values()),
            skipped=0, errors=0, failed_tickers=[], per_ticker=per,
        )

    def fake_bulk_ingest(tickers: Any, **kw: Any) -> BulkIngestResult:
        seen["ing_tickers"] = list(tickers)
        seen["incremental"] = kw.get("incremental")
        return BulkIngestResult(
            tickers=len(list(tickers)), chunks=5, embed_tokens=100, skipped_existing=3,
        )

    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(
        app, "get_settings",
        lambda: Settings(
            _env_file=None, sec_user_agent="T t@e.com", documents_dir=tmp_path,
            cache_dir=tmp_path / ".cache",
        ),
    )
    monkeypatch.setattr(app, "bulk_download", fake_bulk_download)
    monkeypatch.setattr(app, "bulk_ingest", fake_bulk_ingest)
    monkeypatch.setattr(app, "build_embedder", lambda s: object())
    monkeypatch.setattr(app, "build_vector_store", lambda s: object())
    return seen


def test_refresh_ingests_only_changed_tickers_incrementally(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(app, "load_universe", lambda p: ["NVDA", "MSFT"])
    seen = _patch_refresh(monkeypatch, tmp_path, downloaded={"NVDA": ["d1"], "MSFT": []})
    app.refresh(
        ticker=None, refresh_all=True, months=6, limit=20, forms=None, universe=tmp_path / "u.txt"
    )
    assert seen["dl_tickers"] == ["NVDA", "MSFT"]  # download scans the whole universe
    assert seen["ing_tickers"] == ["NVDA"]  # but only NVDA got new filings -> only it is ingested
    assert seen["incremental"] is True  # and only NEW chunks are embedded


def test_refresh_up_to_date_skips_ingest(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    seen = _patch_refresh(monkeypatch, tmp_path, downloaded={"NVDA": []})  # nothing new
    app.refresh(
        ticker="NVDA", refresh_all=False, months=6, limit=20, forms=None, universe=Path("x")
    )
    assert "ing_tickers" not in seen  # bulk_ingest never called
    assert "up to date" in capsys.readouterr().out.lower()


# ---- rag status (observability: active embedder + collection + freshness) -----


def _status(**kw: Any) -> CorpusStatus:
    base = dict(
        provider="voyage", embedder="voyage-voyage-4", collection="filings-voyage-voyage-4",
        chunks=93109, filings=4461, tickers=104, earliest="2022-08-31", latest="2026-06-11",
    )
    base.update(kw)
    return CorpusStatus(**base)


def test_rag_status_reports_embedder_collection_and_freshness(
    monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(app, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(app, "corpus_status", lambda s: _status())
    app.rag_status()
    out = capsys.readouterr().out
    assert "voyage-voyage-4" in out and "provider=voyage" in out  # active embedder
    assert "filings-voyage-voyage-4" in out and "93,109 chunks" in out  # voyage collection
    assert "4,461 filings across 104 tickers" in out  # coverage
    assert "2022-08-31 → 2026-06-11" in out  # freshness range


def test_rag_status_empty_corpus(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(app, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(
        app, "corpus_status",
        lambda s: _status(chunks=0, filings=0, tickers=0, earliest=None, latest=None),
    )
    app.rag_status()
    out = capsys.readouterr().out
    assert "empty" in out.lower()  # 0-chunk hint
    assert "Downloaded : none" in out  # no-download hint
