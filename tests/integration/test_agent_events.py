"""Event-emitting runtime tests (P2.1): stream ordering + drain equivalence.

Verifies :func:`run_agent_events` yields the runtime-owned event subset in the §5 order, that
terminal failures surface as ``Error`` (not a raise inside the generator), and that
:func:`run_agent` is a behavior-identical drain of it. Uses the same scripted ``FakeToolLLM`` +
``FakeProvider`` executor as ``test_agent_runtime.py`` — no network, no LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from stock_agent.agent.events import (
    AgentEvent,
    Error,
    Final,
    Token,
    ToolFinish,
    ToolStart,
    hue_for,
)
from stock_agent.agent.runtime import ToolResponse, ToolUse, run_agent, run_agent_events
from stock_agent.agent.tools import ToolExecutor
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings


class FakeToolLLM:
    """Returns scripted ToolResponses in order (mirrors test_agent_runtime.FakeToolLLM)."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


class FakeStreamingLLM:
    """A streaming ToolLLM (P2.4): each turn is scripted as (text deltas, terminal ToolResponse).

    ``stream`` yields the deltas in order then the ToolResponse — the real ``AnthropicToolClient``
    contract. ``isinstance(FakeStreamingLLM(), StreamingToolLLM)`` is True (it has ``stream``), so
    ``stream_turn`` forwards its deltas; ``create`` exists only to satisfy the ``ToolLLM`` type.
    """

    def __init__(self, turns: list[tuple[list[str], ToolResponse]]) -> None:
        self._turns = turns
        self.calls = 0

    def stream(self, *, system, messages, tools, max_tokens):  # type: ignore[no-untyped-def]
        deltas, resp = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        yield from deltas
        yield resp

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        _, resp = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return resp


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


def _types(events: Sequence[AgentEvent]) -> list[str]:
    return [e.type for e in events]


def test_no_tool_answer_emits_token_then_final() -> None:
    events = list(
        run_agent_events("hello", llm=FakeToolLLM([_final("Hi there.")]), executor=_executor())
    )
    assert _types(events) == ["token", "final"]
    assert isinstance(events[0], Token) and events[0].text == "Hi there."
    final = events[-1]
    assert isinstance(final, Final)
    assert final.iterations == 1 and final.tool_calls == [] and final.tool_results == []


def test_single_tool_ordering_and_payloads() -> None:
    script = [
        _tool_turn(
            ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA", "horizon_days": 20})
        ),
        _final("Analysis complete for NVDA over twenty trading days."),  # no decimals/percents
    ]
    events = list(run_agent_events("forecast NVDA", llm=FakeToolLLM(script), executor=_executor()))

    # §5 subsequence the runtime owns: tool_start → tool_finish → token → final.
    assert _types(events) == ["tool_start", "tool_finish", "token", "final"]
    start, finish = events[0], events[1]
    assert isinstance(start, ToolStart) and isinstance(finish, ToolFinish)
    assert start.tool == finish.tool == "run_forecast"
    assert start.hue_key == hue_for("run_forecast")
    assert "ticker=NVDA" in start.input_summary and "horizon_days=20" in start.input_summary
    assert finish.ok is True and finish.elapsed_ms >= 0.0
    final = events[-1]
    assert isinstance(final, Final)
    assert final.tool_calls == ["run_forecast"] and final.iterations == 2
    assert [inv.name for inv in final.tool_results] == ["run_forecast"]


def test_multi_tool_turn_yields_paired_events_in_order() -> None:
    script = [
        _tool_turn(
            ToolUse(id="1", name="compute_indicators", input={"ticker": "NVDA"}),
            ToolUse(id="2", name="get_price_summary", input={"ticker": "NVDA"}),
        ),
        _final("Summary done for NVDA."),
    ]
    events = list(run_agent_events("brief NVDA", llm=FakeToolLLM(script), executor=_executor()))
    assert _types(events) == [
        "tool_start",
        "tool_finish",
        "tool_start",
        "tool_finish",
        "token",
        "final",
    ]
    starts = [e for e in events if isinstance(e, ToolStart)]
    assert [s.tool for s in starts] == ["compute_indicators", "get_price_summary"]


def test_grounding_rejection_emits_error_not_final_and_no_tokens() -> None:
    # Persistent fabrication (ungrounded on both attempts) → terminal Error(code="grounding");
    # crucially NO token event is ever emitted for the rejected prose.
    script = [_final("A 92.5% chance."), _final("Still 92.5% likely.")]
    events = list(run_agent_events("will it moon", llm=FakeToolLLM(script), executor=_executor()))
    assert _types(events) == ["error"]
    err = events[0]
    assert isinstance(err, Error) and err.code == "grounding"
    assert not any(isinstance(e, Token) for e in events)
    assert not any(isinstance(e, Final) for e in events)


def test_grounding_retry_then_success_emits_only_accepted_answer() -> None:
    # First answer is ungrounded → corrective retry (no token); second is clean → token + final.
    script = [
        _final("There is a 92.5% chance of a rally."),  # ungrounded, retried away
        _final("I cannot provide that figure without running the forecast."),
    ]
    events = list(
        run_agent_events("will NVDA go up?", llm=FakeToolLLM(script), executor=_executor())
    )
    assert _types(events) == ["token", "final"]
    token = events[0]
    assert isinstance(token, Token)
    assert "92.5%" not in token.text  # the fabricated attempt never streamed


def test_tool_error_sets_ok_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # When a tool returns an error dict (execute() never raises), ToolFinish carries ok=False.
    ex = _executor()

    def boom(_args: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("forced tool failure")

    monkeypatch.setattr(ex, "_tool_get_price_summary", boom)  # execute() traps it -> {"error": ...}
    script = [
        _tool_turn(ToolUse(id="1", name="get_price_summary", input={"ticker": "NVDA"})),
        _final("Could not retrieve data."),
    ]
    events = list(run_agent_events("price NVDA", llm=FakeToolLLM(script), executor=ex))
    finish = next(e for e in events if isinstance(e, ToolFinish))
    assert finish.ok is False


def test_run_agent_is_a_faithful_drain_of_the_generator() -> None:
    # The synchronous drain must reconstruct the same AgentResult the events describe.
    script = [
        _tool_turn(
            ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA", "horizon_days": 20})
        ),
        _final("Analysis complete for NVDA over twenty trading days."),
    ]
    ex = _executor()
    result = run_agent("forecast NVDA", llm=FakeToolLLM(script), executor=ex)

    ex2 = _executor()
    events = list(run_agent_events("forecast NVDA", llm=FakeToolLLM(script), executor=ex2))
    tokens = "".join(e.text for e in events if isinstance(e, Token))
    final = next(e for e in events if isinstance(e, Final))

    assert result.text == tokens
    assert result.tool_calls == final.tool_calls
    assert result.iterations == final.iterations
    assert [inv.name for inv in result.tool_results] == [inv.name for inv in final.tool_results]
    # messages include the final assistant turn for next-turn history (same as pre-refactor).
    assert any(m["role"] == "assistant" for m in result.messages)


# --- P2.4: true token streaming (multiple Token deltas, still grounding-safe) -------------------


def _stream_answer(deltas: list[str]) -> ToolResponse:
    """A terminal answer ToolResponse whose text is the concatenation of its streamed deltas."""
    return ToolResponse(
        text="".join(deltas), tool_uses=[], stop_reason="end_turn", assistant_content=[]
    )


def test_streaming_answer_emits_one_token_per_delta_preserving_boundaries() -> None:
    deltas = ["Over the next 20 ", "trading days the model ", "leans mildly up."]
    llm = FakeStreamingLLM([(deltas, _stream_answer(deltas))])
    events = list(run_agent_events("forecast NVDA", llm=llm, executor=_executor()))

    assert _types(events) == ["token", "token", "token", "final"]
    toks = [e.text for e in events if isinstance(e, Token)]
    assert toks == deltas  # the model's own chunk boundaries are preserved (true streaming)
    # Multi-chunk concatenation == the single-shot answer (the plan's P2.4 invariant).
    assert "".join(toks) == "Over the next 20 trading days the model leans mildly up."


def test_streaming_answer_still_grounds_on_the_full_text_no_ungrounded_delta_streams() -> None:
    # First streamed answer is ungrounded (92.5%) → corrective retry, emitting NO tokens; the
    # accepted second answer streams its own deltas. Grounding sees the WHOLE assembled attempt,
    # so an ungrounded figure never escapes even split across deltas ("92." + "5%").
    bad = ["There is a ", "92.", "5% chance."]
    good = ["I cannot provide that ", "without running the forecast."]
    llm = FakeStreamingLLM([(bad, _stream_answer(bad)), (good, _stream_answer(good))])
    events = list(run_agent_events("will NVDA moon?", llm=llm, executor=_executor()))

    assert _types(events) == ["token", "token", "final"]  # only the accepted answer's deltas
    assert not any("92.5%" in e.text for e in events if isinstance(e, Token))
    assert "".join(e.text for e in events if isinstance(e, Token)).startswith("I cannot provide")


def test_streaming_tool_turn_discards_preamble_deltas() -> None:
    # A tool turn may stream preamble text ("Let me check…") before the tool_use; that text is
    # discarded (never a Token), exactly as create-turn text was. Only the final answer streams.
    tool_turn = ToolResponse(
        text="Let me check the forecast.",
        tool_uses=[ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA"})],
        stop_reason="tool_use",
        assistant_content=[],
    )
    answer = ["Analysis complete for NVDA ", "over twenty trading days."]
    llm = FakeStreamingLLM(
        [(["Let me check ", "the forecast."], tool_turn), (answer, _stream_answer(answer))]
    )
    events = list(run_agent_events("forecast NVDA", llm=llm, executor=_executor()))

    assert _types(events) == ["tool_start", "tool_finish", "token", "token", "final"]
    assert not any("check" in e.text for e in events if isinstance(e, Token))  # preamble dropped


def test_run_agent_drains_streaming_deltas_into_the_full_answer() -> None:
    # The synchronous drain concatenates the Token deltas → identical text to a single-shot answer.
    deltas = ["Alpha ", "beta ", "gamma."]
    llm = FakeStreamingLLM([(deltas, _stream_answer(deltas))])
    result = run_agent("q", llm=llm, executor=_executor())
    assert result.text == "Alpha beta gamma."
