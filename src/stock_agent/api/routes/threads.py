"""Chat-thread CRUD endpoints: ``/threads`` (Phase 2, P2.5b).

Display-level thread persistence for the React sidebar — create / list / open / delete over the
same ``ChatStore`` the Streamlit frontend uses (one JSON file per thread). Thin: no agent calls,
no numbers computed here (invariant: numbers-from-tools only). The SSE stream does not emit the
resumable ``agent_history`` yet, so a reopened thread restores its display transcript (text +
charts + sources); full agent-context resume is deferred to a later slice.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from stock_agent.api.deps import chat_store_dep
from stock_agent.api.schemas import ThreadMetaResponse, ThreadResponse, ThreadSaveRequest
from stock_agent.chat.history import ChatStore, ChatThread, derive_title, new_thread_id, now_iso

router = APIRouter()

# A load failure meaning "no such (or unreadable) thread" — missing file, corrupt JSON, or a JSON
# body missing required keys. Callers treat all three as absent (404 on GET, recreate on POST).
_LOAD_ERRORS = (OSError, json.JSONDecodeError, KeyError)


@router.get("/threads", response_model=list[ThreadMetaResponse])
def list_threads(
    store: Annotated[ChatStore, Depends(chat_store_dep)],
) -> list[ThreadMetaResponse]:
    """Thread descriptors, most-recently-updated first (age-pruned first; corrupt files skipped)."""
    store.prune()  # drop threads past the retention window before listing
    return [
        ThreadMetaResponse(
            id=m.id, title=m.title, created_at=m.created_at, updated_at=m.updated_at
        )
        for m in store.list_threads()
    ]


@router.post("/threads", response_model=ThreadResponse)
def save_thread(
    req: ThreadSaveRequest,
    store: Annotated[ChatStore, Depends(chat_store_dep)],
) -> ThreadResponse:
    """Create (empty id) or update (existing id) one thread; returns the saved thread.

    ``created_at`` is preserved across updates (loaded from the existing file); ``updated_at`` is
    stamped now. An empty title is derived from the first user message. An unknown id is not an
    error — it is simply created with that id (idempotent upsert).
    """
    now = now_iso()
    thread_id = req.id or new_thread_id()
    created_at = now
    if req.id:
        try:
            created_at = store.load(req.id).created_at  # preserve original creation time on update
        except _LOAD_ERRORS:
            created_at = now  # unknown/corrupt id -> create fresh under this id
    title = req.title.strip() or derive_title(req.display_messages)
    thread = ChatThread(
        id=thread_id,
        title=title,
        created_at=created_at,
        updated_at=now,
        display_messages=req.display_messages,
        agent_history=req.agent_history,
    )
    store.save(thread)
    return ThreadResponse(**thread.to_json())  # to_json coerces to JSON-safe dicts


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    store: Annotated[ChatStore, Depends(chat_store_dep)],
) -> ThreadResponse:
    """Load one full thread (display transcript + agent history); 404 if it does not exist."""
    try:
        thread = store.load(thread_id)
    except _LOAD_ERRORS as exc:
        raise HTTPException(status_code=404, detail=f"no such thread: {thread_id}") from exc
    return ThreadResponse(**thread.to_json())


@router.delete("/threads/{thread_id}", status_code=204)
def delete_thread(
    thread_id: str,
    store: Annotated[ChatStore, Depends(chat_store_dep)],
) -> None:
    """Delete one thread. Idempotent — deleting a missing thread is a no-op (still 204)."""
    store.delete(thread_id)
