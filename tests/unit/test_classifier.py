"""Haiku routing classifier — validation is fully deterministic via a canned-JSON fake LLM.

The classifier must FAIL SAFE (escalate) on anything it can't confidently + validly route, and it
must never raise. These tests pin every escalation trigger + the happy-path param extraction.
"""

from __future__ import annotations

from stock_agent.agent.classifier import Classification, RouteClassifier
from stock_agent.llm.client import LLMError


class _FakeJson:
    """A TextLLM that returns a fixed JSON string (or raises LLMError) — no network."""

    def __init__(self, payload: str, *, raise_error: bool = False) -> None:
        self._payload = payload
        self._raise = raise_error

    def complete_json(self, *, system: str, user: str, max_tokens: int = 256) -> str:
        if self._raise:
            raise LLMError("boom")
        return self._payload


def _classify(payload: str, *, min_confidence: float = 0.6, **kw: object) -> Classification:
    return RouteClassifier(_FakeJson(payload), min_confidence=min_confidence).classify(
        str(kw.get("q", "some question"))
    )


def test_dispatches_valid_route_with_params() -> None:
    d = _classify('{"route":"forecast","ticker":"nvda","horizon":30,"confidence":0.92}')
    assert not d.escalated
    assert d.route == "forecast"
    assert d.ticker == "NVDA"  # normalized/upper-cased
    assert d.horizon == 30


def test_theme_route_extracts_topic() -> None:
    d = _classify('{"route":"theme_news","topic":"AI memory","confidence":0.8}')
    assert d.route == "theme_news"
    assert d.topic == "AI memory"


def test_explicit_escalate() -> None:
    assert _classify('{"route":"escalate","confidence":0.3,"reason":"compound"}').escalated


def test_unknown_route_escalates() -> None:
    assert _classify('{"route":"does_not_exist","confidence":0.99}').escalated


def test_low_confidence_escalates() -> None:
    # A valid route but below the threshold → defer to the stronger model rather than guess.
    assert _classify('{"route":"news","ticker":"NVDA","confidence":0.4}').escalated


def test_missing_required_ticker_escalates() -> None:
    assert _classify('{"route":"news","ticker":null,"confidence":0.95}').escalated


def test_invalid_ticker_escalates_for_ticker_route() -> None:
    # "the stock" isn't a symbol → normalized to None → a ticker route can't run → escalate.
    assert _classify('{"route":"technicals","ticker":"the stock","confidence":0.9}').escalated


def test_missing_topic_escalates_for_theme_route() -> None:
    assert _classify('{"route":"theme_news","topic":null,"confidence":0.9}').escalated


def test_malformed_json_escalates_without_raising() -> None:
    assert _classify("sorry, I cannot answer").escalated
    assert _classify("").escalated


def test_llm_error_escalates() -> None:
    d = RouteClassifier(_FakeJson("", raise_error=True)).classify("x")
    assert d.escalated
    assert "failed" in d.reason


def test_empty_question_escalates_without_calling_llm() -> None:
    # Guard fires before the LLM; a payload that would otherwise route must be ignored.
    llm = _FakeJson('{"route":"news","ticker":"NVDA","confidence":0.9}')
    assert RouteClassifier(llm).classify("  ").escalated


def test_bare_integer_and_string_horizon_coerce() -> None:
    d = _classify('{"route":"forecast","ticker":"NVDA","horizon":"20","confidence":0.9}')
    assert d.horizon == 20  # numeric string coerced
    d2 = _classify('{"route":"forecast","ticker":"NVDA","horizon":-5,"confidence":0.9}')
    assert d2.horizon is None  # non-positive dropped → tool default applies
