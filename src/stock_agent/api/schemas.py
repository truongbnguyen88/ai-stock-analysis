"""JSON response models for the api/ layer (the React SPA's contract).

Pydantic models so FastAPI emits a stable, typed schema. These mirror what the Streamlit
sidebar already renders (routing modes, key chips, corpus status card) so the two frontends
cannot drift on what they show — both read the same underlying pure contracts.
"""

from __future__ import annotations

from pydantic import BaseModel, computed_field

from stock_agent.rag.status import CorpusStatus


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
