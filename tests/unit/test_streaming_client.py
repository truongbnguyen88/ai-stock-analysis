"""Streaming LLM client tests (P2.4): ``AnthropicToolClient.stream`` + ``stream_turn`` fallback.

No network — inject a fake Anthropic client whose ``messages.stream(...)`` returns a scripted
context manager (text deltas + a final assembled message). Asserts the plan's P2.4 contract:
the streamed text deltas concatenate to the terminal ``ToolResponse.text`` (multi-chunk ==
single-shot), tool_use blocks are parsed off the final message, sampling params match ``create``,
and create-only LLMs fall back to a single-delta turn (so pre-P2.4 behavior is preserved).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from stock_agent.agent.runtime import (
    AgentError,
    AnthropicToolClient,
    ToolResponse,
    ToolUse,
    stream_turn,
)
from stock_agent.settings import Settings


class _Block:
    """Duck-typed Anthropic content block (``type`` + attrs the parser reads)."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Message:
    def __init__(self, content: list[_Block], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason


class _StreamCtx:
    """Fake of the SDK's MessageStream context manager: scripted ``text_stream`` + final message."""

    def __init__(self, deltas: list[str], final: _Message, boom: bool = False) -> None:
        self._deltas = deltas
        self._final = final
        self._boom = boom

    def __enter__(self) -> _StreamCtx:
        return self

    def __exit__(self, *exc: object) -> None:
        return None  # do not suppress exceptions

    @property
    def text_stream(self) -> Iterator[str]:
        for d in self._deltas:
            if self._boom:
                raise RuntimeError("stream broke mid-flight")
            yield d

    def get_final_message(self) -> _Message:
        return self._final


class _StreamingAnthropic:
    """Minimal fake Anthropic client exposing ``messages.stream`` (records kwargs)."""

    class _Messages:
        def __init__(self, ctx: _StreamCtx) -> None:
            self._ctx = ctx
            self.kwargs: dict[str, Any] = {}

        def stream(self, **kwargs: Any) -> _StreamCtx:
            self.kwargs = kwargs
            return self._ctx

    def __init__(self, deltas: list[str], final: _Message, boom: bool = False) -> None:
        self.messages = _StreamingAnthropic._Messages(_StreamCtx(deltas, final, boom))


def _client(fake: _StreamingAnthropic, *, temperature: float | None = 0.0) -> AnthropicToolClient:
    settings = Settings(_env_file=None, agent_temperature=temperature)
    return AnthropicToolClient(settings, client=fake)


def _drain(client: AnthropicToolClient) -> list[str | ToolResponse]:
    return list(
        client.stream(
            system="s", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=16
        )
    )


def test_stream_yields_text_deltas_then_a_terminal_toolresponse() -> None:
    deltas = ["The model ", "leans ", "mildly up."]
    final = _Message([_Block(type="text", text="The model leans mildly up.")], "end_turn")
    items = _drain(_client(_StreamingAnthropic(deltas, final)))

    texts = [i for i in items if isinstance(i, str)]
    responses = [i for i in items if isinstance(i, ToolResponse)]
    assert texts == deltas  # deltas forwarded live, in order
    assert len(responses) == 1 and items[-1] is responses[0]  # exactly one, and it is LAST
    # The plan's P2.4 invariant: multi-chunk concatenation == the single-shot answer text.
    assert "".join(texts) == responses[0].text == "The model leans mildly up."
    assert responses[0].tool_uses == [] and responses[0].stop_reason == "end_turn"


def test_stream_parses_tool_use_off_the_final_message() -> None:
    # A tool turn: deltas may be empty; the assembled final message carries the tool_use block.
    final = _Message(
        [_Block(type="tool_use", id="t1", name="run_forecast", input={"ticker": "NVDA"})],
        "tool_use",
    )
    items = _drain(_client(_StreamingAnthropic([], final)))
    resp = items[-1]
    assert isinstance(resp, ToolResponse)
    assert resp.tool_uses == [ToolUse(id="t1", name="run_forecast", input={"ticker": "NVDA"})]
    assert resp.text == "" and resp.stop_reason == "tool_use"


def test_stream_pins_temperature_zero_by_default() -> None:
    fake = _StreamingAnthropic([], _Message([], "end_turn"))
    _drain(_client(fake, temperature=0.0))
    assert fake.messages.kwargs["temperature"] == 0.0


def test_stream_omits_temperature_when_none() -> None:
    # Models that reject sampling params (Opus 4.8/4.7, Fable 5) must not receive temperature.
    fake = _StreamingAnthropic([], _Message([], "end_turn"))
    _drain(_client(fake, temperature=None))
    assert "temperature" not in fake.messages.kwargs


def test_stream_normalizes_sdk_errors_to_agenterror() -> None:
    fake = _StreamingAnthropic(["partial"], _Message([], "end_turn"), boom=True)
    with pytest.raises(AgentError, match="stream failed"):
        _drain(_client(fake))


# --- stream_turn: routes streaming LLMs to .stream, create-only LLMs to a single-delta fallback ---


class _CreateOnly:
    """A create-only ToolLLM (like the test fakes) — must NOT be treated as streaming."""

    def __init__(self, resp: ToolResponse) -> None:
        self._resp = resp
        self.stream_calls = 0

    def create(self, *, system: str, messages: Any, tools: Any, max_tokens: int) -> ToolResponse:
        return self._resp


class _Streaming:
    def stream(
        self, *, system: str, messages: Any, tools: Any, max_tokens: int
    ) -> Iterator[str | ToolResponse]:
        yield "a"
        yield "b"
        yield ToolResponse(text="ab", tool_uses=[], stop_reason="end_turn", assistant_content=[])

    def create(self, *, system: str, messages: Any, tools: Any, max_tokens: int) -> ToolResponse:
        raise AssertionError("stream_turn must prefer .stream over .create for a streaming LLM")


def _turn(llm: Any) -> list[str | ToolResponse]:
    return list(stream_turn(llm, system="s", messages=[], tools=[], max_tokens=8))


def test_stream_turn_falls_back_to_single_delta_for_create_only_llm() -> None:
    resp = ToolResponse(
        text="full answer", tool_uses=[], stop_reason="end_turn", assistant_content=[]
    )
    items = _turn(_CreateOnly(resp))
    # Fallback yields NO text deltas, just the assembled ToolResponse → runtime emits one Token.
    assert items == [resp]
    assert not any(isinstance(i, str) for i in items)


def test_stream_turn_forwards_a_streaming_llms_deltas() -> None:
    items = _turn(_Streaming())
    assert [i for i in items if isinstance(i, str)] == ["a", "b"]
    assert isinstance(items[-1], ToolResponse) and items[-1].text == "ab"
