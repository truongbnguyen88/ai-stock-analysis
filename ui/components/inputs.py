"""Chat composer: context chips + the message input (redesign R5).

Thin Streamlit wrapper — all logic is in the pure, tested ``stock_agent.ui`` layer:
``routing.context_chips`` decides what the chips say, ``html.context_row`` builds their
markup, ``routing.chat_input_placeholder`` supplies the hint text. This module only injects
the markup and returns the raw ``st.chat_input`` value; the turn/bounce logic stays in
``chat_app.py``.

Streamlit note: ``st.chat_input`` is docked to the bottom of the app, so the context chips
(rendered just before it) sit at the end of the scrollable content — visually *above* the
composer but not physically attached to it. That is the acknowledged Streamlit approximation
of the mockup's attached chips (docs/APP_REDESIGN.md §3 "last ~15%"); the focus-within brass
accent on the input is applied via CSS in ``theme._COMPONENTS``.
"""

from __future__ import annotations

import streamlit as st

from stock_agent.ui.html import context_row
from stock_agent.ui.routing import RoutingChoice, chat_input_placeholder, context_chips


def render_chat_input(choice: RoutingChoice) -> str | None:
    """Render the context-chip row + the chat input; return the submitted text (or ``None``).

    The chips surface the active ticker + routing mode ("on enter: NVDA · Auto"); the input
    placeholder is tailored to the mode. Returns exactly what ``st.chat_input`` returns so the
    caller keeps ownership of the pending / bounce-to-Auto fallbacks.
    """
    st.markdown(context_row(context_chips(choice)), unsafe_allow_html=True)
    return st.chat_input(chat_input_placeholder(choice))
