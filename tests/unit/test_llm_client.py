"""AnthropicClient wrapper logic (no network; fake SDK client injected)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from stock_agent.llm.client import AnthropicClient, LLMError
from stock_agent.settings import Settings


class _FakeMessages:
    def __init__(self, text: str, exc: Exception | None = None) -> None:
        self._text = text
        self._exc = exc
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


class _FakeAnthropic:
    def __init__(self, text: str = "}", exc: Exception | None = None) -> None:
        self.messages = _FakeMessages(text, exc)


def _client(fake: _FakeAnthropic) -> AnthropicClient:
    return AnthropicClient(Settings(_env_file=None, anthropic_api_key="k"), client=fake)


def test_returns_model_text_verbatim() -> None:
    fake = _FakeAnthropic('{"overview": "x"}')
    out = _client(fake).complete_json(system="sys", user="usr")
    assert out == '{"overview": "x"}'


def test_system_block_cached_and_ends_with_user_message() -> None:
    fake = _FakeAnthropic("{}")
    _client(fake).complete_json(system="sys", user="usr")
    kwargs = fake.messages.last_kwargs
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Sonnet 4.x requires the conversation to end with a user message (no prefill).
    assert kwargs["messages"][-1] == {"role": "user", "content": "usr"}
    assert kwargs["temperature"] == 0.0


def test_sdk_errors_wrapped_in_llm_error() -> None:
    fake = _FakeAnthropic(exc=RuntimeError("boom"))
    with pytest.raises(LLMError):
        _client(fake).complete_json(system="s", user="u")


def test_client_constructed_with_timeout_and_retries(monkeypatch: Any) -> None:
    # When no client is injected, the wrapper builds anthropic.Anthropic with the
    # configured timeout/retries (resilience against transient stalls/drops).
    import anthropic

    captured: dict[str, Any] = {}

    def fake_ctor(**kwargs: Any) -> _FakeAnthropic:
        captured.update(kwargs)
        return _FakeAnthropic("{}")

    monkeypatch.setattr(anthropic, "Anthropic", fake_ctor)
    settings = Settings(
        _env_file=None, anthropic_api_key="k", llm_timeout_seconds=42.0, llm_max_retries=7
    )
    AnthropicClient(settings).complete_json(system="s", user="u")  # client=None -> constructs
    assert captured["timeout"] == 42.0
    assert captured["max_retries"] == 7
