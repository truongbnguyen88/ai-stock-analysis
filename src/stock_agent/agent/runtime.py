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
from dataclasses import dataclass, field
from typing import Any, Protocol

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
        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
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
    """Run the tool-use loop and return a grounded final answer.

    Pass ``history`` (the ``messages`` field from a prior ``AgentResult``) to
    give the agent memory of previous turns in the same conversation. Grounding
    starts from this turn's tool calls, plus any numbers produced by tools in
    ``history`` (so a figure from an earlier turn can be referenced without
    re-fetching); it never trusts the model's own prior text.
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
                result = executor.execute(tu.name, tu.input)
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
            raise AgentGroundingError(f"unverified figures after retry: {', '.join(violations)}")

        # Append the final assistant turn so the next call has the complete history.
        messages.append({"role": "assistant", "content": resp.assistant_content})
        return AgentResult(
            text=resp.text,
            tool_calls=tool_calls,
            iterations=iteration,
            messages=messages,
            tool_results=tool_results,
        )

    raise AgentError(f"agent did not finish within {max_iterations} iterations")
