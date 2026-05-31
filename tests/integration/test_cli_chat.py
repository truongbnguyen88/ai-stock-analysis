"""CLI chat loop: interactive mode threads conversation history across turns.

The runtime's history mechanism is tested in test_agent_runtime; this verifies the
``chat`` command actually carries each turn's transcript into the next call (the
statefulness that lets a follow-up like "combine the above" resolve against prior
turns). run_agent and the LLM clients are faked — no network.
"""

from __future__ import annotations

import builtins
import importlib
from typing import Any

from stock_agent.settings import Settings

# The stock_agent.cli package shadows the `app` submodule with the Typer instance,
# so grab the real module object (where get_settings/configure_logging/chat live).
app = importlib.import_module("stock_agent.cli.app")


class _FakeResult:
    def __init__(self, text: str, messages: list[dict[str, Any]]) -> None:
        self.text = text
        self.messages = messages


def test_interactive_chat_threads_history(monkeypatch: Any, capsys: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_agent(
        question: str, *, llm: Any, executor: Any, history: list[dict[str, Any]] | None = None
    ) -> _FakeResult:
        prior = list(history) if history else []
        calls.append({"question": question, "history_len": len(prior)})
        # Mimic run_agent: each turn appends a user + assistant message.
        transcript = prior + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"answer to {question}"},
        ]
        return _FakeResult(text=f"answer to {question}", messages=transcript)

    # Patch at the source modules: chat() imports these lazily at call time.
    monkeypatch.setattr("stock_agent.agent.runtime.run_agent", fake_run_agent)
    monkeypatch.setattr("stock_agent.agent.runtime.AnthropicToolClient", lambda *a, **k: object())
    monkeypatch.setattr("stock_agent.agent.tools.ToolExecutor", lambda *a, **k: object())
    monkeypatch.setattr("stock_agent.llm.client.AnthropicClient", lambda *a, **k: object())
    fake_settings = Settings(_env_file=None, anthropic_api_key="k")
    monkeypatch.setattr(app, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(app, "configure_logging", lambda s: None)

    turns = iter(
        [
            "extract and summarize news for DELL",
            "predict the upside/downside of DELL for the next 20 days",
            "combine the above predictions and news into an executive summary",
            "exit",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda *_: next(turns))

    app.chat(None)  # interactive (no one-shot message)

    # Three questions ran (the 4th was 'exit').
    assert [c["question"] for c in calls][:1] == ["extract and summarize news for DELL"]
    assert len(calls) == 3
    # History grows by one full turn (2 messages) each step: 0 -> 2 -> 4.
    assert [c["history_len"] for c in calls] == [0, 2, 4]


def test_reset_clears_conversation_context(monkeypatch: Any) -> None:
    calls: list[int] = []

    def fake_run_agent(
        question: str, *, llm: Any, executor: Any, history: list[dict[str, Any]] | None = None
    ) -> _FakeResult:
        prior = list(history) if history else []
        calls.append(len(prior))
        return _FakeResult(text="ok", messages=prior + [{"role": "user", "content": question}])

    monkeypatch.setattr("stock_agent.agent.runtime.run_agent", fake_run_agent)
    monkeypatch.setattr("stock_agent.agent.runtime.AnthropicToolClient", lambda *a, **k: object())
    monkeypatch.setattr("stock_agent.agent.tools.ToolExecutor", lambda *a, **k: object())
    monkeypatch.setattr("stock_agent.llm.client.AnthropicClient", lambda *a, **k: object())
    fake_settings = Settings(_env_file=None, anthropic_api_key="k")
    monkeypatch.setattr(app, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(app, "configure_logging", lambda s: None)

    turns = iter(["first question", "reset", "after reset", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(turns))

    app.chat(None)

    # 'reset' is not sent to the agent; the turn after it starts from empty history again.
    assert calls == [0, 0]
