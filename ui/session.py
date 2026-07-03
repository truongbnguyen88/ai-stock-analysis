"""Chat session state + thread persistence for the Streamlit app (redesign R1).

Streamlit-coupled glue (reads/writes ``st.session_state`` and the ``ChatStore``), so it
lives in the repo-root ``ui/`` layer rather than the type-checked package — same status
as ``chat_app.py``. Pure (de)serialization is delegated to ``stock_agent.ui.state``.

Extracted verbatim from ``ui/chat_app.py`` (R1 refactor) — behavior-preserving.
"""

from __future__ import annotations

import streamlit as st

from stock_agent.chat.history import (
    ChatStore,
    ChatThread,
    derive_title,
    new_thread_id,
    now_iso,
)
from stock_agent.ui.state import deserialize_messages, serialize_messages


def init_session_state() -> None:
    """Initialize per-session chat state before the sidebar (which lists threads) renders."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()
        st.session_state.thread_created_at = now_iso()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # display history (user/assistant text + charts)
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []  # Anthropic-format message history


def clear_transient_ui_state() -> None:
    """Drop per-turn UI offers that must not survive a thread switch.

    ``fallback_prompt`` (the bounce-to-Auto offer) belongs to the turn that produced it;
    leaving it set across a New chat / open / delete would render the button in a DIFFERENT
    thread and re-run the wrong question. ``force_auto_prompt`` is popped every run, but
    clear it too for safety.
    """
    st.session_state.pop("fallback_prompt", None)
    st.session_state.pop("force_auto_prompt", None)


def start_new_thread() -> None:
    """Begin a fresh, empty conversation."""
    st.session_state.thread_id = new_thread_id()
    st.session_state.thread_created_at = now_iso()
    st.session_state.messages = []
    st.session_state.agent_history = []
    clear_transient_ui_state()


def open_thread(store: ChatStore, thread_id: str) -> None:
    """Load a saved thread into session state for viewing/continuation."""
    t = store.load(thread_id)
    st.session_state.thread_id = t.id
    st.session_state.thread_created_at = t.created_at
    st.session_state.messages = deserialize_messages(t.display_messages)
    st.session_state.agent_history = t.agent_history
    clear_transient_ui_state()


def save_current_thread(store: ChatStore) -> None:
    """Persist the active conversation (skips an empty/unstarted thread)."""
    msgs = st.session_state.messages
    if not msgs:
        return
    store.save(
        ChatThread(
            id=st.session_state.thread_id,
            title=derive_title(msgs),
            created_at=st.session_state.thread_created_at,
            updated_at=now_iso(),
            display_messages=serialize_messages(msgs),
            agent_history=st.session_state.agent_history,
        )
    )
