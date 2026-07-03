"""Agent runtime — the tool-use loop with numeric-grounding enforcement.

Drives a conversation: the model requests tools, we execute them, feed results
back, and repeat until it produces a final answer. Every number in tool results
is accumulated; the final answer is checked by the grounding guard. Ungrounded
figures trigger one corrective retry, then a refusal — the agent never knowingly
presents fabricated numbers.

``ToolLLM`` is the injection seam: ``AnthropicToolClient`` for production, a fake
for tests (so the suite never hits the network).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from stock_agent.agent.events import (
    AgentEvent,
    Error,
    Final,
    Token,
    ToolFinish,
    ToolStart,
    hue_for,
)
from stock_agent.agent.guards import NumberGrounding
from stock_agent.agent.prompts.agent import SYSTEM
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor
from stock_agent.logging_config import get_logger
from stock_agent.settings import Settings

log = get_logger(__name__)

_MAX_ITERATIONS = 8
_MAX_TOKENS = 4096  # headroom for comprehensive multi-section answers


class AgentError(RuntimeError):
    """The agent could not produce a valid answer."""


class AgentGroundingError(AgentError):
    """The agent stated figures not supported by any tool result."""


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResponse:
    """Normalized model turn: any text, any tool requests, and the raw blocks."""

    text: str
    tool_uses: list[ToolUse]
    stop_reason: str
    assistant_content: Any  # raw content blocks, appended back to the transcript


@dataclass
class ToolInvocation:
    """A single tool call made during a turn, with its structured result.

    Surfaced so a presentation layer (e.g. the Streamlit UI) can render charts
    from the numbers the tools already produced — without the LLM generating any
    chart data (the numbers-vs-narrative invariant).
    """

    name: str
    input: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResult:
    text: str
    tool_calls: list[str] = field(default_factory=list)
    iterations: int = 0
    # Full Anthropic-format message history for this turn (user query + tool
    # calls + final answer). Pass as ``history`` on the next turn to give the
    # agent memory of prior conversation.
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Structured tool results from THIS turn (not history), in call order — for
    # charting/inspection in the presentation layer.
    tool_results: list[ToolInvocation] = field(default_factory=list)


class ToolLLM(Protocol):
    """A tool-using chat model."""

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ToolResponse: ...


class AnthropicToolClient:
    """``ToolLLM`` backed by the Anthropic Messages API (tools + prompt caching)."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._model = settings.llm_model
        self._temperature = settings.agent_temperature  # None => omit (models that reject it)
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            key = self._settings.require("anthropic_api_key", capability="agent chat")
            self._client = anthropic.Anthropic(
                api_key=key,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        return self._client

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ToolResponse:
        client = self._ensure_client()
        # Omit temperature when None so models that reject sampling params (Opus 4.8/4.7, Fable 5)
        # don't 400; otherwise pin it (0.0 by default) for reproducible routing.
        extra: dict[str, Any] = {}
        if self._temperature is not None:
            extra["temperature"] = self._temperature
        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - normalize SDK/network errors
            raise AgentError(f"agent LLM request failed: {exc}") from exc

        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))
        return ToolResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=resp.stop_reason,
            assistant_content=resp.content,
        )


def _seed_grounding_from_history(grounding: NumberGrounding, history: list[dict[str, Any]]) -> None:
    """Ground numbers from prior turns' tool results in a continuing conversation.

    Grounding is otherwise reset each turn, which is correct for a single turn but
    too strict once the chat is stateful: a figure a tool produced earlier (a news
    stat, a prior forecast) is real and visible in the transcript, so referencing it
    in a later turn must not be treated as fabrication. We seed only from
    ``tool_result`` blocks — actual tool outputs, never the model's own prior text —
    so this cannot launder a hallucinated number into being "grounded".
    """
    for msg in history:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                grounding.add_from(block.get("content"))


def _input_summary(tool_input: dict[str, Any]) -> str:
    """Compact one-line label of a tool's salient args for the trace chip (presentation only).

    Picks a bounded set of well-known keys in a fixed order so the chip is stable and short;
    values are stringified and truncated. Never affects grounding or tool execution.
    """
    parts: list[str] = []
    for key in ("ticker", "tickers", "horizon_days", "days", "model", "topic", "question"):
        val = tool_input.get(key)
        if val in (None, "", []):
            continue
        text = ",".join(map(str, val)) if isinstance(val, list) else str(val)
        if len(text) > 40:
            text = text[:37] + "…"
        parts.append(f"{key}={text}")
    return " · ".join(parts)


def run_agent_events(
    query: str,
    *,
    llm: ToolLLM,
    executor: ToolExecutor,
    history: list[dict[str, Any]] | None = None,
    system: str = SYSTEM,
    tools: list[dict[str, Any]] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
    grounding_retries: int = 1,
) -> Iterator[AgentEvent]:
    """Run the tool-use loop as a generator, yielding events as it goes (plan §4).

    This is the single implementation of the loop; :func:`run_agent` is a thin synchronous
    drain of it, so the two cannot diverge in behavior. Yields the runtime-owned subset of the
    ``AgentEvent`` union — ``ToolStart``/``ToolFinish`` around each execution, a ``Token`` delta
    for the (grounded) final answer, then a terminal ``Final`` **or** ``Error``. It deliberately
    does NOT emit ``tiles``/``chart``/``sources`` (those are built by the api adapter from
    ``tool_results`` via pure ``ui``/``viz`` builders — the runtime never imports presentation
    code) nor ``turn_start``/``route_decided`` (router-owned).

    Grounding contract is unchanged from the pre-refactor loop: a final answer with ungrounded
    figures triggers one corrective retry (emitting NO tokens for the rejected attempt, so the
    client never sees ungrounded prose); a second failure yields ``Error(code="grounding")``
    instead of ``Final``. ``Token`` is emitted only for the accepted, post-guard answer — in P2.1
    a single delta carrying the whole answer; P2.4 swaps in true token streaming.

    See :func:`run_agent` for the ``history``/grounding-seeding semantics.
    """
    tools = tools if tools is not None else TOOL_SCHEMAS
    # Prepend prior turns, then append the new user message.
    messages: list[dict[str, Any]] = list(history) if history else []
    messages.append({"role": "user", "content": query})
    grounding = NumberGrounding()
    if history:
        # Stateful chat: let this turn reference figures earlier tools produced.
        _seed_grounding_from_history(grounding, history)
    tool_calls: list[str] = []
    tool_results: list[ToolInvocation] = []
    retries_used = 0

    for iteration in range(1, max_iterations + 1):
        resp = llm.create(system=system, messages=messages, tools=tools, max_tokens=_MAX_TOKENS)

        if resp.tool_uses:
            messages.append({"role": "assistant", "content": resp.assistant_content})
            results: list[dict[str, Any]] = []
            for tu in resp.tool_uses:
                tool_calls.append(tu.name)
                yield ToolStart(
                    tool=tu.name,
                    input_summary=_input_summary(tu.input),
                    hue_key=hue_for(tu.name),
                )
                start = time.perf_counter()
                result = executor.execute(tu.name, tu.input)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                # ``ok`` mirrors the tools' error convention (an "error" key means it failed).
                ok = isinstance(result, dict) and "error" not in result
                yield ToolFinish(tool=tu.name, ok=ok, elapsed_ms=elapsed_ms)
                grounding.add_from(result)
                tool_results.append(ToolInvocation(name=tu.name, input=tu.input, result=result))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        # Final answer: enforce numeric grounding.
        violations = grounding.ungrounded(resp.text)
        if violations and retries_used < grounding_retries:
            retries_used += 1
            log.warning("agent.grounding_retry", figures=violations)
            messages.append({"role": "assistant", "content": resp.assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "These figures are not supported by any tool result: "
                        f"{', '.join(violations)}. They will be rejected. Rewrite the answer so "
                        "EVERY percentage or decimal is one that a tool actually returned. For any "
                        "figure not in the tool outputs (e.g. a market-share or growth statistic "
                        "from general knowledge), do NOT guess a number — either call a tool that "
                        "produces it, or describe the effect QUALITATIVELY and directionally "
                        "(larger/smaller, tailwind/headwind, more/less concentrated) with no "
                        "invented figure. Keep all correctly-sourced numbers as they are."
                    ),
                }
            )
            continue
        if violations:
            # Terminal failure as an event (the drain re-raises AgentGroundingError). No tokens
            # were emitted for the rejected text, so nothing ungrounded ever reached the client.
            yield Error(
                code="grounding",
                message=f"unverified figures after retry: {', '.join(violations)}",
            )
            return

        # Accepted answer: append the assistant turn (so next-turn history is complete), stream
        # the (grounded) text as a token delta, then close with the terminal Final.
        messages.append({"role": "assistant", "content": resp.assistant_content})
        yield Token(text=resp.text)
        yield Final(
            tool_calls=tool_calls,
            iterations=iteration,
            messages=messages,
            tool_results=tool_results,
        )
        return

    yield Error(
        code="max_iterations",
        message=f"agent did not finish within {max_iterations} iterations",
    )


def run_agent(
    query: str,
    *,
    llm: ToolLLM,
    executor: ToolExecutor,
    history: list[dict[str, Any]] | None = None,
    system: str = SYSTEM,
    tools: list[dict[str, Any]] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
    grounding_retries: int = 1,
) -> AgentResult:
    """Return a grounded final answer from the tool-use loop (synchronous drain of the generator).

    Behavior-identical to the pre-P2.1 loop: this is now a thin ``_collect`` over
    :func:`run_agent_events` — it concatenates ``Token`` deltas into the answer text, reads the
    transcript / tool results off the terminal ``Final``, and re-raises the runtime exceptions on
    a terminal ``Error`` (so existing callers and ``pytest.raises`` tests are unaffected).

    Pass ``history`` (the ``messages`` field from a prior ``AgentResult``) to give the agent memory
    of previous turns. Grounding starts from this turn's tool calls, plus any numbers produced by
    tools in ``history`` (so a figure from an earlier turn can be referenced without re-fetching);
    it never trusts the model's own prior text.
    """
    text_parts: list[str] = []
    for event in run_agent_events(
        query,
        llm=llm,
        executor=executor,
        history=history,
        system=system,
        tools=tools,
        max_iterations=max_iterations,
        grounding_retries=grounding_retries,
    ):
        if isinstance(event, Token):
            text_parts.append(event.text)
        elif isinstance(event, Final):
            return AgentResult(
                text="".join(text_parts),
                tool_calls=event.tool_calls,
                iterations=event.iterations,
                messages=event.messages,
                tool_results=event.tool_results,
            )
        elif isinstance(event, Error):
            if event.code == "grounding":
                raise AgentGroundingError(event.message)
            raise AgentError(event.message)
    # The generator always terminates with Final or Error; reaching here is a contract violation.
    raise AgentError("agent produced no terminal event")
