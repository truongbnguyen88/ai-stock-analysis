"""RoutingChoice + chat-input placeholder — pure sidebar-selection logic (R1)."""

from __future__ import annotations

from stock_agent.ui.routing import RoutingChoice, chat_input_placeholder


def test_is_auto_iff_no_domain() -> None:
    assert RoutingChoice(ticker="NVDA").is_auto is True
    assert RoutingChoice(ticker="NVDA", domain="news").is_auto is False


def test_placeholder_filings() -> None:
    c = RoutingChoice(ticker="NVDA", domain="filings", variant="qa")
    assert chat_input_placeholder(c) == "Type your filing question…"


def test_placeholder_news_theme() -> None:
    c = RoutingChoice(ticker="NVDA", domain="news", variant="theme")
    assert chat_input_placeholder(c) == "Type a sector/theme (e.g. robotics)…"


def test_placeholder_named_domain_uses_ticker() -> None:
    c = RoutingChoice(ticker="NVDA", domain="technicals", variant="default")
    assert chat_input_placeholder(c) == "Press enter to run technicals on NVDA…"
    # Falls back to a generic noun when no ticker is set.
    c2 = RoutingChoice(ticker="", domain="technicals", variant="default")
    assert chat_input_placeholder(c2) == "Press enter to run technicals on a ticker…"


def test_placeholder_auto_is_open_ended() -> None:
    assert chat_input_placeholder(RoutingChoice(ticker="NVDA")) == "Ask anything about a stock…"
