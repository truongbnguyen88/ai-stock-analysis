"""Smoke tests: package imports, settings load offline, logging configures."""

from __future__ import annotations

from stock_agent import __version__
from stock_agent.logging_config import configure_logging, get_logger
from stock_agent.settings import MissingSettingError, Settings


def test_version() -> None:
    assert isinstance(__version__, str) and __version__


def test_settings_load_without_secrets() -> None:
    # Must construct with no .env and no API keys (offline / CI friendly).
    s = Settings(_env_file=None)
    assert s.env in {"dev", "prod"}
    assert s.price_priority == ["yfinance", "alpha_vantage"]
    assert s.news_priority[0] == "finnhub"


def test_require_raises_for_missing_key() -> None:
    s = Settings(_env_file=None)
    try:
        s.require("alpha_vantage_api_key", capability="price data")
    except MissingSettingError as exc:
        assert exc.attr == "alpha_vantage_api_key"
        assert "price data" in str(exc)
    else:
        raise AssertionError("expected MissingSettingError")


def test_require_returns_value_when_present() -> None:
    s = Settings(_env_file=None, alpha_vantage_api_key="abc123")
    assert s.require("alpha_vantage_api_key", capability="price data") == "abc123"


def test_logging_idempotent() -> None:
    s = Settings(_env_file=None)
    configure_logging(s)
    configure_logging(s)  # second call is a no-op
    get_logger("test").info("ok")
