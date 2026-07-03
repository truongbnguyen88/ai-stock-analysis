"""Empty-state capability showcase: animated typewriter + capability cards (redesign R1).

Extracted verbatim from ``ui/chat_app.py`` — behavior-preserving. Data-driven from
``CAPABILITIES`` (single source of truth): the typewriter is the visual, the cards are
the action. Clicking a card sets ``pending_prompt`` + reruns; the orchestrator submits it.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from stock_agent.ui.capabilities import CAPABILITIES


def render_typewriter() -> None:
    """Animated centerpiece: a pure client-side typewriter cycling capability headlines.

    Rendered in an iframe via ``components.html`` so it never triggers a server rerun or
    blocks the chat. The iframe does NOT inherit Streamlit's theme, so colors adapt via
    ``prefers-color-scheme``; motion honors ``prefers-reduced-motion`` (static first line).
    Fixed height reserves space so typed text never clips. The cards below are the
    actionable layer — this is the visual only (A.3).
    """
    phrases = [f"{cap.title}  {cap.icon}" for cap in CAPABILITIES]
    html = """
<div class="tw-wrap">
  <span class="tw-prefix">You can ask about&nbsp;</span><span id="tw-text"></span><span class="tw-caret">▌</span>
</div>
<style>
  .tw-wrap {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 1.5rem; font-weight: 600; text-align: center;
    height: 2.4rem; line-height: 2.4rem; color: #1f2937;
  }
  .tw-prefix { opacity: 0.6; }
  .tw-caret { animation: tw-blink 1s step-end infinite; opacity: 0.7; }
  @keyframes tw-blink { 50% { opacity: 0; } }
  @media (prefers-color-scheme: dark) { .tw-wrap { color: #e5e7eb; } }
  @media (prefers-reduced-motion: reduce) { .tw-caret { animation: none; } }
</style>
<script>
  const phrases = __PHRASES__;
  const el = document.getElementById("tw-text");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    el.textContent = phrases[0];  // static, no animation
  } else {
    let pi = 0, ci = 0, deleting = false;
    function tick() {
      const full = phrases[pi];
      ci += deleting ? -1 : 1;
      el.textContent = full.slice(0, ci);
      let delay = deleting ? 35 : 70;
      if (!deleting && ci === full.length) { delay = 1500; deleting = true; }
      else if (deleting && ci === 0) { deleting = false; pi = (pi + 1) % phrases.length; delay = 300; }
      setTimeout(tick, delay);
    }
    tick();
  }
</script>
"""
    html = html.replace("__PHRASES__", json.dumps(phrases))
    components.html(html, height=58)


def render_capability_hero(ticker: str) -> None:
    """Empty-state showcase: animated typewriter (A.3) over a grid of capability cards.

    Data-driven from ``CAPABILITIES`` (single source of truth) — the typewriter is the
    visual, the cards are the action. Clicking a card reuses the existing quick-starter
    plumbing — set ``pending_prompt`` + rerun, which the ``st.chat_input(...) or pending``
    line in the orchestrator submits.
    """
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        render_typewriter()
        st.markdown(
            "#### What can this agent do?\n"
            "Pick a capability to try it on **"
            f"{ticker or 'a ticker'}** — or just ask in the box below.",
        )
        n_cols = 2
        for row_start in range(0, len(CAPABILITIES), n_cols):
            cols = st.columns(n_cols)
            for col, cap in zip(cols, CAPABILITIES[row_start : row_start + n_cols]):
                with col:
                    if st.button(
                        f"{cap.icon}  **{cap.title}**\n\n{cap.blurb}",
                        key=f"cap_{row_start}_{cap.title}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_prompt = cap.render_example(ticker)
                        st.rerun()
