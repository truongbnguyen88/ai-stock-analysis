"""Router streaming tests (P2.2): ``Router.run_events`` ordering + parity with ``Router.run``.

Verifies the router-owned events (``turn_start``/``route_decided``) frame both paths, that the
deterministic path emits ``tool_start``/``tool_finish``/``token``/``final`` around one dispatch,
that the LLM path delegates to ``run_agent_events`` (stamping ``turn_id`` on ``final``), and that
``run_events`` stays behavior-identical to ``run`` (text / tool_calls / structured). Same scripted
``FakeToolLLM`` + ``FakeProvider`` executor as the runtime tests — no network, no LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import cast

import pytest

from stock_agent.agent.classifier import RouteClassifier
from stock_agent.agent.events import (
    AgentEvent,
    Final,
    RouteDecided,
    Token,
    ToolFinish,
    ToolStart,
    TurnStart,
    hue_for,
)
from stock_agent.agent.router import Router, RouterError
from stock_agent.agent.runtime import ToolResponse, ToolUse
from stock_agent.agent.tools import ToolExecutor
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings


class FakeToolLLM:
    """Returns scripted ToolResponses in order (mirrors the runtime tests' fake)."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


class _FakeClassification:
    """Duck-typed stand-in for ``classifier.Classification`` (only the fields the router reads)."""

    def __init__(
        self,
        route: str | None,
        *,
        ticker: str | None = None,
        horizon: int | None = None,
        days: int | None = None,
        topic: str | None = None,
    ) -> None:
        self.route = route
        self.ticker = ticker
        self.horizon = horizon
        self.days = days
        self.topic = topic
        self.confidence = 0.9
        self.reason = "test"


class _FakeClassifier:
    def __init__(self, decision: _FakeClassification) -> None:
        self._decision = decision

    def classify(self, _request: str) -> _FakeClassification:
        return self._decision


def _executor(n_bars: int = 60) -> ToolExecutor:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=100.0 + i + 0.5,
            low=100.0 + i - 0.5,
            close=100.0 + i,
        )
        for i in range(n_bars)
    ]
    fake = FakeProvider("fake", prices=PriceSeries(ticker="NVDA", bars=bars))
    registry = ProviderRegistry([fake], Settings(_env_file=None, provider_price_priority="fake"))
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


def _tool_turn(*tools: ToolUse) -> ToolResponse:
    return ToolResponse(
        text="", tool_uses=list(tools), stop_reason="tool_use", assistant_content=[]
    )


def _forecast_script() -> list[ToolResponse]:
    """One tool turn (run_forecast) then a clean (grounded) final answer — the LLM-path fixture."""
    return [
        _tool_turn(
            ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA", "horizon_days": 20})
        ),
        _final("Analysis complete for NVDA over twenty trading days."),
    ]


def _types(events: Sequence[AgentEvent]) -> list[str]:
    return [e.type for e in events]


def test_deterministic_route_stream_order_and_payloads() -> None:
    router = Router(_executor())  # deterministic path needs no tool_llm
    events = list(
        router.run_events(
            "forecast it", route="forecast", ticker="nvda", horizon=20,
            thread_id="th", turn_id="tn",
        )
    )
    assert _types(events) == [
        "turn_start", "route_decided", "tool_start", "tool_finish", "token", "final",
    ]
    ts = events[0]
    assert isinstance(ts, TurnStart)
    assert ts.thread_id == "th" and ts.turn_id == "tn"
    assert ts.route == "forecast" and ts.ticker == "NVDA"  # ticker normalized (upper)

    rd = events[1]
    assert isinstance(rd, RouteDecided)
    assert rd.mode == "deterministic" and rd.route_name == "forecast"

    start, finish = events[2], events[3]
    assert isinstance(start, ToolStart) and isinstance(finish, ToolFinish)
    assert start.tool == finish.tool == "run_forecast"
    assert start.hue_key == hue_for("run_forecast")
    assert "ticker=NVDA" in start.input_summary and "horizon_days=20" in start.input_summary
    assert finish.ok is True and finish.elapsed_ms >= 0.0

    final = events[-1]
    assert isinstance(final, Final)
    assert final.tool_calls == ["run_forecast"] and final.iterations == 1
    assert final.turn_id == "tn"
    assert [inv.name for inv in final.tool_results] == ["run_forecast"]


def test_deterministic_stream_matches_run() -> None:
    # run_events must stay behavior-identical to run (the drift guard): same text + structured.
    router = Router(_executor())
    rr = router.run("forecast it", route="forecast", ticker="NVDA", horizon=20)
    events = list(
        router.run_events("forecast it", route="forecast", ticker="NVDA", horizon=20)
    )
    token = next(e for e in events if isinstance(e, Token))
    final = next(e for e in events if isinstance(e, Final))
    assert token.text == rr.text
    assert final.tool_calls == rr.tool_calls
    assert final.tool_results[0].result == rr.structured


def test_auto_route_delegates_to_runtime_and_stamps_turn_id() -> None:
    router = Router(_executor(), tool_llm=FakeToolLLM(_forecast_script()))
    events = list(router.run_events("forecast NVDA", route="auto", turn_id="tn"))
    assert _types(events) == [
        "turn_start", "route_decided", "tool_start", "tool_finish", "token", "final",
    ]
    rd = events[1]
    assert isinstance(rd, RouteDecided) and rd.mode == "auto" and rd.route_name == "auto"
    final = events[-1]
    # The runtime is turn-id agnostic; the router stamps it onto the terminal Final.
    assert isinstance(final, Final) and final.turn_id == "tn"


def test_auto_stream_matches_run() -> None:
    rr = Router(_executor(), tool_llm=FakeToolLLM(_forecast_script())).run(
        "forecast NVDA", route="auto"
    )
    events = list(
        Router(_executor(), tool_llm=FakeToolLLM(_forecast_script())).run_events(
            "forecast NVDA", route="auto"
        )
    )
    token = next(e for e in events if isinstance(e, Token))
    final = next(e for e in events if isinstance(e, Final))
    assert token.text == rr.text and final.tool_calls == rr.tool_calls


def test_classify_dispatch_streams_the_resolved_deterministic_route() -> None:
    decision = _FakeClassification("forecast", ticker="NVDA", horizon=20)
    router = Router(_executor(), classifier=cast(RouteClassifier, _FakeClassifier(decision)))
    events = list(router.run_events("will NVDA rally?", route="classify", turn_id="tn"))
    # classify resolves to a deterministic route → same shape as the deterministic path.
    assert _types(events) == [
        "turn_start", "route_decided", "tool_start", "tool_finish", "token", "final",
    ]
    rd = events[1]
    assert isinstance(rd, RouteDecided)
    assert rd.mode == "deterministic" and rd.route_name == "forecast"


def test_classify_escalation_delegates_to_the_agent_loop() -> None:
    router = Router(
        _executor(),
        tool_llm=FakeToolLLM(_forecast_script()),
        classifier=cast(RouteClassifier, _FakeClassifier(_FakeClassification(None))),  # → escalate
    )
    events = list(router.run_events("something compositional", route="classify"))
    rd = events[1]
    assert isinstance(rd, RouteDecided) and rd.mode == "auto"  # escalated to the LLM agent path


def test_missing_ticker_raises_router_error() -> None:
    router = Router(_executor())
    with pytest.raises(RouterError):
        list(router.run_events("forecast", route="forecast"))  # forecast needs a ticker
