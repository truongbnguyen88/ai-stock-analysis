"""Two-stage router (route='classify') — deterministic dispatch vs escalation, fully offline.

A fake classifier decides; a fake executor records the dispatched tool call; a fake tool-LLM stands
in for the Sonnet agent on escalation. No network, no real models.
"""

from __future__ import annotations

from typing import Any

import pytest

from stock_agent.agent.classifier import Classification
from stock_agent.agent.router import Router, RouterError
from stock_agent.agent.runtime import ToolResponse


class _FakeExecutor:
    """Records the (name, args) of each dispatched tool call and returns a canned result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"tool": name, **args}


class _FakeClassifier:
    def __init__(self, decision: Classification) -> None:
        self._decision = decision

    def classify(self, question: str) -> Classification:
        return self._decision


class _FinalToolLLM:
    """Stands in for the Sonnet agent: immediately returns a final answer (no tool calls)."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kw: Any) -> ToolResponse:
        self.calls += 1
        return ToolResponse(
            text="escalated answer", tool_uses=[], stop_reason="end_turn", assistant_content=[]
        )


def _router(decision: Classification, *, with_llm: bool = True) -> tuple[Router, _FakeExecutor]:
    ex = _FakeExecutor()
    tool_llm = _FinalToolLLM() if with_llm else None
    router = Router(ex, tool_llm=tool_llm, classifier=_FakeClassifier(decision))  # type: ignore[arg-type]
    return router, ex


def test_classify_dispatches_deterministic_route() -> None:
    router, ex = _router(Classification(route="technicals", ticker="NVDA", confidence=0.95))
    res = router.run("how's NVDA trending?", route="classify")
    assert res.mode == "deterministic"
    assert res.route == "technicals"
    assert ex.calls == [("compute_indicators", {"ticker": "NVDA"})]


def test_classify_theme_route_uses_extracted_topic_not_question() -> None:
    # The whole question would be a junk keyword query; the classifier's extracted topic is used.
    router, ex = _router(Classification(route="theme_news", topic="AI memory", confidence=0.9))
    router.run("pull recent news in the AI memory domain please", route="classify")
    name, args = ex.calls[0]
    assert name == "analyze_topic_news"
    assert args["topic"] == "AI memory"


def test_classify_escalates_to_agent() -> None:
    router, ex = _router(Classification(route=None, confidence=0.2, reason="compound"))
    res = router.run("compare NVDA and AMD filings and forecast both", route="classify")
    assert res.mode == "llm"
    assert res.text == "escalated answer"
    assert ex.calls == []  # no deterministic dispatch on escalation


def test_classify_without_classifier_raises() -> None:
    from stock_agent.agent.router import Router as R

    r = R(_FakeExecutor())  # type: ignore[arg-type]  # no classifier configured
    with pytest.raises(RouterError):
        r.run("anything", route="classify")
