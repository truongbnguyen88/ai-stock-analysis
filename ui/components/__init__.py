"""Streamlit view components for the chat app (redesign R1).

Presentation-only render functions split out of ``ui/chat_app.py`` for maintainability.
Each takes explicit arguments and depends only on the typed ``stock_agent`` package
(plus ``ui.session`` for thread actions) — no sibling-component imports — so the
dependency graph stays a tree the orchestrator wires together.
"""
