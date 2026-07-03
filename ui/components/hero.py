"""Empty-state capability showcase: token-themed typewriter + colored cards (redesign R3).

Data-driven from ``CAPABILITIES`` (single source of truth): the typewriter is the visual,
the cards are the action. Clicking a card's button sets ``pending_prompt`` + reruns; the
orchestrator submits it — behavior unchanged from R1. R3 restyle: the typewriter colors
come from the design tokens (``theme.iframe_colors``) with a brass accent word instead of
hardcoded hex, and each capability renders as a hue-coded card (``html.cap_card``) above
its action button.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from stock_agent.ui.capabilities import CAPABILITIES
from stock_agent.ui.html import cap_card, cap_hue, eyebrow
from stock_agent.ui.theme import iframe_colors


def render_typewriter() -> None:
    """Animated centerpiece: a pure client-side typewriter cycling capability headlines.

    Rendered in an iframe via ``components.html`` so it never triggers a server rerun or
    blocks the chat. The iframe does NOT inherit Streamlit's theme or our ``--sa-*`` vars,
    so colors are injected as literal hex from ``theme.iframe_colors`` (single source) and
    swap via the iframe's own ``prefers-color-scheme``; the typed phrase + caret use the
    brass accent. Motion honors ``prefers-reduced-motion`` (static first line). Fixed
    height reserves space so typed text never clips. The cards below are the actionable
    layer — this is the visual only.
    """
    phrases = [f"{cap.title}  {cap.icon}" for cap in CAPABILITIES]
    colors = iframe_colors()
    dark, light = colors["dark"], colors["light"]
    html = """
<div class="tw-wrap">
  <span class="tw-prefix">You can ask about&nbsp;</span><span id="tw-text"></span><span class="tw-caret">▌</span>
</div>
<style>
  .tw-wrap {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 1.5rem; font-weight: 600; text-align: center;
    height: 2.4rem; line-height: 2.4rem; color: __D_TEXT__;
  }
  .tw-prefix { color: __D_MUTED__; }
  #tw-text { color: __D_ACCENT__; }
  .tw-caret { color: __D_ACCENT__; animation: tw-blink 1s step-end infinite; }
  @keyframes tw-blink { 50% { opacity: 0; } }
  @media (prefers-color-scheme: light) {
    .tw-wrap { color: __L_TEXT__; }
    .tw-prefix { color: __L_MUTED__; }
    #tw-text { color: __L_ACCENT__; }
    .tw-caret { color: __L_ACCENT__; }
  }
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
    html = (
        html.replace("__PHRASES__", json.dumps(phrases))
        .replace("__D_TEXT__", dark["--sa-text"])
        .replace("__D_MUTED__", dark["--sa-muted"])
        .replace("__D_ACCENT__", dark["--sa-accent"])
        .replace("__L_TEXT__", light["--sa-text"])
        .replace("__L_MUTED__", light["--sa-muted"])
        .replace("__L_ACCENT__", light["--sa-accent"])
    )
    components.html(html, height=58)


def render_capability_hero(ticker: str) -> None:
    """Empty-state showcase: token-themed typewriter over a grid of hue-coded cards.

    Data-driven from ``CAPABILITIES`` (single source of truth) — the typewriter is the
    visual, the cards are the action. Each card is a hue-coded HTML visual
    (``html.cap_card``) with a sibling action button (injected HTML can't fire a Streamlit
    callback), which reuses the quick-starter plumbing: set ``pending_prompt`` + rerun,
    which the ``st.chat_input(...) or pending`` line in the orchestrator submits.
    """
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        render_typewriter()
        st.markdown(eyebrow("What this agent can do", tone="accent"), unsafe_allow_html=True)
        st.caption(
            f"Pick a capability to try it on **{ticker or 'a ticker'}** — or just ask in the box below."
        )
        cta = f"Try on {ticker} →" if ticker else "Try it →"
        n_cols = 2
        for row_start in range(0, len(CAPABILITIES), n_cols):
            cols = st.columns(n_cols)
            for offset, (col, cap) in enumerate(
                zip(cols, CAPABILITIES[row_start : row_start + n_cols])
            ):
                idx = row_start + offset
                with col:
                    st.markdown(
                        cap_card(
                            icon=cap.icon,
                            title=cap.title,
                            blurb=cap.blurb,
                            hue=cap_hue(idx),
                        ),
                        unsafe_allow_html=True,
                    )
                    if st.button(cta, key=f"cap_{idx}_{cap.title}", use_container_width=True):
                        st.session_state.pending_prompt = cap.render_example(ticker)
                        st.rerun()
