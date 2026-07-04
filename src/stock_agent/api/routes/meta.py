"""Read-only metadata endpoints: ``GET /config`` and ``GET /corpus`` (P2.0).

Thin — each handler just adapts an existing pure contract to a response model:
``/config`` surfaces routing domains + key availability + the default ticker (the same
facts the Streamlit sidebar shows); ``/corpus`` echoes ``rag.status.corpus_status``. No
agent calls, no LLM, no numbers computed here (invariant: numbers-from-tools only).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from stock_agent.agent.router import DOMAIN_NAMES
from stock_agent.api.deps import settings_dep
from stock_agent.api.schemas import ConfigResponse, CorpusResponse, KeyStatus
from stock_agent.rag.status import corpus_status
from stock_agent.settings import Settings
from stock_agent.ui.keys import key_statuses
from stock_agent.ui.routing import AUTO_MODE

# Default seed ticker for the sidebar field — matches the Streamlit sidebar's value="NVDA".
DEFAULT_TICKER = "NVDA"

router = APIRouter()


def _key_rows(settings: Settings) -> list[KeyStatus]:
    """API-key availability — the same set/order the Streamlit sidebar shows.

    Derived from ``ui.keys.key_statuses`` (single source of truth → the two frontends can't drift).
    """
    return [
        KeyStatus(label=lbl, present=present, required=req)
        for lbl, present, req in key_statuses(settings)
    ]


@router.get("/config", response_model=ConfigResponse)
def get_config(settings: Annotated[Settings, Depends(settings_dep)]) -> ConfigResponse:
    """Static app config for the sidebar/top-bar (routing modes, keys, default ticker)."""
    return ConfigResponse(
        default_ticker=DEFAULT_TICKER,
        auto_mode=AUTO_MODE,
        domains=list(DOMAIN_NAMES),
        keys=_key_rows(settings),
    )


@router.get("/corpus", response_model=CorpusResponse)
def get_corpus(settings: Annotated[Settings, Depends(settings_dep)]) -> CorpusResponse:
    """Corpus status card: active embedder/collection + coverage/freshness. Never raises."""
    return CorpusResponse.from_status(corpus_status(settings))
