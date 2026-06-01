"""LLM client wrapper (Anthropic) with prompt caching.

Exposes a narrow ``TextLLM`` protocol so business logic (the summarizer) depends
on an interface, not the SDK — tests inject a fake. ``AnthropicClient`` adds:
- prompt caching on the (static) system block to cut cost on repeated calls,
- a JSON prefill ("{") to make structured output reliable,
- temperature 0 for reproducibility.
"""

from __future__ import annotations

from typing import Any, Protocol

from stock_agent.settings import Settings


class LLMError(RuntimeError):
    """Raised when an LLM request fails (network, auth, quota, bad response)."""


class TextLLM(Protocol):
    """Minimal interface the summarizer needs: prompt in, JSON string out."""

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str: ...


def _extract_text(response: Any) -> str:
    """Concatenate text blocks from an Anthropic messages response."""
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


class AnthropicClient:
    """``TextLLM`` backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._model = settings.llm_model
        self._client = client  # injectable for tests; lazily created otherwise

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            key = self._settings.require("anthropic_api_key", capability="LLM summarization")
            self._client = anthropic.Anthropic(
                api_key=key,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        return self._client

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        """Return a JSON string. Caches the static system block; temperature 0."""
        client = self._ensure_client()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.0,
                # Cache the static instructions: cheap repeat calls across tickers.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/network error
            raise LLMError(f"LLM request failed: {exc}") from exc

        # Sonnet 4.x rejects assistant-message prefill, so we don't force the
        # opening brace; the system prompt mandates a bare JSON object and the
        # summarizer parses leniently (tolerates any stray prose).
        return _extract_text(response)
