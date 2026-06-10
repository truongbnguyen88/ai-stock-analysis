"""CLI `documents download-sec` (RAG P1) — arg handling, no network."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import typer

from stock_agent.documents.download import DEFAULT_FORMS, DownloadResult
from stock_agent.settings import Settings

app = importlib.import_module("stock_agent.cli.app")


def _patch(monkeypatch: Any, tmp_path: Path, *, ua: str | None = "Tester t@e.com") -> list[Any]:
    """Patch settings/logging + capture download_filings calls; return the call log."""
    calls: list[Any] = []

    def fake_download(sym: str, provider: Any, *, documents_dir: Path, forms: Any, limit: int):  # type: ignore[no-untyped-def]
        calls.append({"sym": sym, "forms": forms, "limit": limit})
        return DownloadResult(ticker=sym, downloaded=["a", "b"], skipped=[])

    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(
        app,
        "get_settings",
        lambda: Settings(
            _env_file=None, sec_user_agent=ua, documents_dir=tmp_path, cache_dir=tmp_path / ".cache"
        ),
    )
    monkeypatch.setattr(app, "download_filings", fake_download)
    return calls


def test_download_sec_normalizes_ticker_and_uses_defaults(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    calls = _patch(monkeypatch, tmp_path)
    app.download_sec(
        ticker="nvda", download_all=False, forms=None, limit=3, universe=Path("configs/u.txt")
    )
    assert calls == [{"sym": "NVDA", "forms": DEFAULT_FORMS, "limit": 3}]
    assert "downloaded 2" in capsys.readouterr().out


def test_download_sec_missing_user_agent_exits(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path, ua=None)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker="NVDA", download_all=False, forms=None, limit=4, universe=Path("x")
        )
    assert exc.value.exit_code == 1


def test_download_sec_rejects_unsupported_form(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker="NVDA", download_all=False, forms=["DEF 14A"], limit=4, universe=Path("x")
        )
    assert exc.value.exit_code == 1


def test_download_sec_requires_target(monkeypatch: Any, tmp_path: Path) -> None:
    _patch(monkeypatch, tmp_path)
    with pytest.raises(typer.Exit) as exc:
        app.download_sec(
            ticker=None, download_all=False, forms=None, limit=4, universe=Path("x")
        )
    assert exc.value.exit_code == 1


def test_download_sec_all_iterates_universe(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "load_universe", lambda p: ["NVDA", "MSFT"])
    app.download_sec(
        ticker=None, download_all=True, forms=["10-K"], limit=2, universe=tmp_path / "u.txt"
    )
    assert [c["sym"] for c in calls] == ["NVDA", "MSFT"]
    assert all(c["forms"] == ("10-K",) for c in calls)
