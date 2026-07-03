"""Contract tests for the P2.0 api/ meta endpoints (/config, /corpus).

Deterministic: settings are injected via ``dependency_overrides`` (no real ``.env``) and
``corpus_status`` is monkeypatched to a canned snapshot (no vector store / manifest I/O).
Asserts the JSON shape the React sidebar/top-bar consume — the single source both frontends
read, so they cannot drift on what they display.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import stock_agent.api.routes.meta as meta
from stock_agent.agent.router import DOMAIN_NAMES
from stock_agent.api.app import create_app
from stock_agent.api.deps import settings_dep
from stock_agent.rag.status import CorpusStatus
from stock_agent.settings import Settings
from stock_agent.ui.routing import AUTO_MODE


def _client(settings: Settings) -> TestClient:
    """TestClient whose settings dependency is pinned (no env read)."""
    app = create_app()
    app.dependency_overrides[settings_dep] = lambda: settings
    return TestClient(app)


def test_config_surfaces_domains_auto_and_default_ticker() -> None:
    client = _client(Settings(anthropic_api_key="sk-test"))
    body = client.get("/config").json()

    assert body["default_ticker"] == meta.DEFAULT_TICKER == "NVDA"
    assert body["auto_mode"] == AUTO_MODE
    assert body["domains"] == list(DOMAIN_NAMES)  # exact set + order the sidebar offers


def test_config_key_status_encodes_present_and_required() -> None:
    # Anthropic present + required; Finnhub present + optional; others missing + optional.
    # All four set explicitly so a dev machine's real .env can't leak keys into the assertion.
    settings = Settings(
        anthropic_api_key="sk-test",
        finnhub_api_key="fk",
        marketaux_api_key=None,
        alpha_vantage_api_key=None,
    )
    body = _client(settings).get("/config").json()
    keys = {k["label"]: k for k in body["keys"]}

    assert keys["Anthropic"] == {"label": "Anthropic", "present": True, "required": True}
    assert keys["Finnhub"] == {"label": "Finnhub", "present": True, "required": False}
    assert keys["Marketaux"]["present"] is False and keys["Marketaux"]["required"] is False
    assert keys["Alpha Vantage"]["present"] is False


def test_config_anthropic_missing_flags_required_key_absent() -> None:
    body = _client(Settings(anthropic_api_key=None)).get("/config").json()
    anthropic = next(k for k in body["keys"] if k["label"] == "Anthropic")
    assert anthropic == {"label": "Anthropic", "present": False, "required": True}


def test_corpus_echoes_status_with_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = CorpusStatus(
        provider="voyage",
        embedder="voyage-voyage-4",
        collection="filings-voyage-voyage-4",
        chunks=1234,
        filings=10,
        tickers=3,
        earliest="2023-01-01",
        latest="2025-06-30",
    )
    monkeypatch.setattr(meta, "corpus_status", lambda _s: snapshot)
    body = _client(Settings()).get("/corpus").json()

    assert body["embedder"] == "voyage-voyage-4"
    assert body["chunks"] == 1234
    assert body["latest"] == "2025-06-30"
    assert body["one_line"] == snapshot.one_line  # computed property surfaced as a field


def test_corpus_unavailable_store_serializes_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    # chunks=-1 is the "store can't be opened" sentinel; must still serialize (never raise).
    snapshot = CorpusStatus(
        provider="fake",
        embedder="fake-embedder",
        collection="filings-fake",
        chunks=-1,
        filings=0,
        tickers=0,
        earliest=None,
        latest=None,
    )
    monkeypatch.setattr(meta, "corpus_status", lambda _s: snapshot)
    body = _client(Settings()).get("/corpus").json()

    assert body["chunks"] == -1
    assert body["latest"] is None
    assert "unavailable" in body["one_line"]
