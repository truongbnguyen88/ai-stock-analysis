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
