"""RoutingChoice + chat-input placeholder — pure sidebar-selection logic (R1)."""

from __future__ import annotations

from stock_agent.ui.routing import RoutingChoice, chat_input_placeholder, context_chips

# The tones context_chips may emit must all be renderable by html.chip (allowed tone set).
_ALLOWED_TONES = {"ok", "accent", "off", "muted", "neutral"}


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


def test_context_chips_auto_ticker_and_mode() -> None:
    chips = context_chips(RoutingChoice(ticker="nvda"))
    # Ticker normalized + brass; mode chip 'Auto'; nothing else in open-ended mode.
    assert chips == [("NVDA", "accent"), ("Auto", "neutral")]


def test_context_chips_named_domain_with_variant_and_params() -> None:
    chips = context_chips(
        RoutingChoice(ticker="MSFT", domain="forecast", variant="ensemble", horizon=20)
    )
    assert ("MSFT", "accent") in chips
    assert ("forecast · ensemble", "neutral") in chips
    assert ("20d horizon", "muted") in chips
    assert not any(lbl.endswith("lookback") for lbl, _ in chips)  # days is None -> omitted


def test_context_chips_days_lookback_surfaced() -> None:
    chips = context_chips(RoutingChoice(ticker="AMD", domain="news", days=30))
    assert ("news", "neutral") in chips  # no variant -> bare domain
    assert ("30d lookback", "muted") in chips


def test_context_chips_missing_ticker_flagged() -> None:
    chips = context_chips(RoutingChoice(ticker="  "))
    assert chips[0] == ("no ticker", "muted")


def test_context_chips_tones_all_renderable() -> None:
    # Every tone must be one html.chip accepts, else the pill silently falls back to neutral.
    for choice in (
        RoutingChoice(ticker="NVDA"),
        RoutingChoice(ticker="", domain="filings", variant="qa"),
        RoutingChoice(ticker="AMD", domain="technicals", horizon=10, days=90),
    ):
        for _, tone in context_chips(choice):
            assert tone in _ALLOWED_TONES
