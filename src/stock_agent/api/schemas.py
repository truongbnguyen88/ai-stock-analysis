"""JSON response models for the api/ layer (the React SPA's contract).

Pydantic models so FastAPI emits a stable, typed schema. These mirror what the Streamlit
sidebar already renders (routing modes, key chips, corpus status card) so the two frontends
cannot drift on what they show — both read the same underlying pure contracts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field

from stock_agent.rag.status import CorpusStatus


class ChatStreamRequest(BaseModel):
    """Body for ``POST /chat/stream`` — one turn's request (mirrors ``Router.run_events`` params).

    ``route`` selects the path: ``None``/``"auto"`` → LLM agent loop; a granular route name (e.g.
    ``"forecast"``) → deterministic dispatch; ``"classify"`` → two-stage classifier. ``ticker``/
    ``horizon``/``days``/``model`` are the structured params the deterministic routes consume;
    ``thread_id``/``turn_id`` echo back on ``turn_start``/``final`` so the client can correlate
    frames (threads are wired in P2.5). ``history`` is the prior transcript for the LLM path (P2.5).
    """

    message: str
    route: str | None = None
    ticker: str | None = None
    horizon: int | None = None
    days: int | None = None
    model: str | None = None
    thread_id: str = ""
    turn_id: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """Body for ``POST /export`` — render one turn's answer to a downloadable document.

    ``text`` is the assistant's final answer (markdown; it already carries the model figures — the
    exporter adds none, same non-advisory framing as the Streamlit export). ``charts`` are the same
    ``ChartSpec`` dicts the client rendered, embedded as figures in pdf/docx (markdown is text).
    ``fmt`` in {pdf, docx, md}; an unknown value is rejected 422.
    """

    text: str
    fmt: str
    title: str = "Stock Research Summary"
    charts: list[dict[str, Any]] = Field(default_factory=list)


class ThreadMetaResponse(BaseModel):
    """One thread descriptor for the sidebar list (mirrors ``chat.history.ThreadMeta``)."""

    id: str
    title: str
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC; the list is sorted by this, most-recent first


class ThreadResponse(ThreadMetaResponse):
    """A full thread: descriptor + display transcript (+ resumable agent history).

    ``display_messages`` is what the sidebar reopens (text + serialized chart/source dicts).
    ``agent_history`` is the resumable Anthropic-format transcript; empty for now because the
    SSE stream does not expose it yet (display-level resume is the P2.5b contract).
    """

    display_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_history: list[dict[str, Any]] = Field(default_factory=list)


class ThreadSaveRequest(BaseModel):
    """Body for ``POST /threads`` — create or update one thread (display-level persistence).

    An empty ``id`` mints a new thread; an existing ``id`` updates it in place with ``created_at``
    preserved. An empty ``title`` is derived from the first user message. ``agent_history`` is
    optional — the SSE stream does not surface it yet, so text+charts is the persisted contract.
    """

    id: str = ""
    title: str = ""
    display_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_history: list[dict[str, Any]] = Field(default_factory=list)


class KeyStatus(BaseModel):
    """One API-key row for the sidebar key chips (mirrors ``ui.html.keys_row`` tuples)."""

    label: str  # display name, e.g. "Anthropic"
    present: bool  # whether the key is configured in settings/.env
    required: bool  # Anthropic is required; providers are optional


class ConfigResponse(BaseModel):
    """Static app config the sidebar/top-bar need at load (``GET /config``)."""

    default_ticker: str  # seed value for the ticker field (matches Streamlit's "NVDA")
    auto_mode: str  # label for the LLM-router mode (vs. a named deterministic domain)
    domains: list[str]  # deterministic routing domains (``agent.router.DOMAIN_NAMES``)
    keys: list[KeyStatus]  # API-key availability (Anthropic required + optional providers)


class CorpusResponse(CorpusStatus):
    """Corpus status for the sidebar card (``GET /corpus``).

    Extends :class:`CorpusStatus` (the single source of truth) and re-exposes its ``one_line``
    summary as a **computed field** so it is serialized into the JSON (a plain ``@property`` on
    the parent is not emitted by ``model_dump``/FastAPI).
    """

    # Re-expose the parent's ``one_line`` property as a computed field so it serializes.
    @computed_field  # type: ignore[prop-decorator]  # decorating a property is the documented form
    @property
    def one_line(self) -> str:  # noqa: D102 — inherits CorpusStatus.one_line's contract
        # Reuse the parent getter (fget is untyped -> str() pins the return type for mypy).
        return str(CorpusStatus.one_line.fget(self))  # type: ignore[attr-defined]

    @classmethod
    def from_status(cls, status: CorpusStatus) -> CorpusResponse:
        """Build the response from the pure ``corpus_status`` snapshot (one_line auto-computes)."""
        return cls(**status.model_dump())
