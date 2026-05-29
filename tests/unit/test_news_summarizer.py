"""Summarizer tests with a fake LLM (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news
from stock_agent.schemas.news import Article, NewsBundle


class FakeLLM:
    """Returns queued JSON strings in order (one per complete_json call)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out


def _bundle() -> NewsBundle:
    art = Article(
        title="Acme launches new chip",
        url="https://real.com/a",
        source="finnhub",
        published_at=datetime(2025, 1, 2, tzinfo=UTC),
        summary="big launch",
    )
    return NewsBundle(ticker="ACME", articles=[art])


def _payload(overview: str, sources: list[str]) -> str:
    return json.dumps(
        {
            "overview": overview,
            "key_themes": ["product"],
            "bullish": [{"point": "New chip launch", "sources": sources}],
            "bearish": [],
            "risks": [],
            "catalysts": [],
        }
    )


def test_valid_summary_parsed_and_citations_sanitized() -> None:
    raw = _payload("Acme launched a chip.", ["https://real.com/a", "https://fake.com/x"])
    summary = summarize_news(_bundle(), FakeLLM([raw]))
    assert summary.overview == "Acme launched a chip."
    # fabricated URL dropped, valid kept
    assert summary.bullish[0].sources == ["https://real.com/a"]


def test_forecast_then_clean_retry_succeeds() -> None:
    bad = _payload("There is a 70% chance shares rally.", ["https://real.com/a"])
    good = _payload("Acme launched a chip.", ["https://real.com/a"])
    llm = FakeLLM([bad, good])
    summary = summarize_news(_bundle(), llm)
    assert llm.calls == 2  # retried once
    assert summary.overview == "Acme launched a chip."


def test_persistent_forecast_raises() -> None:
    bad = _payload("Odds of a beat are high; price target $250.", ["https://real.com/a"])
    with pytest.raises(SummaryGuardError):
        summarize_news(_bundle(), FakeLLM([bad, bad]))


def test_lenient_json_parsing_tolerates_surrounding_prose() -> None:
    raw = "Here is the JSON:\n" + _payload("Clean overview.", []) + "\nThanks!"
    summary = summarize_news(_bundle(), FakeLLM([raw]))
    assert summary.overview == "Clean overview."
