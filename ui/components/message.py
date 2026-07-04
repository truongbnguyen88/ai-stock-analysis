"""Assistant/user message rendering: tiles, charts, trace, SEC sources, export (R4).

Extracted from ``ui/chat_app.py`` (R1) and restyled in R4. Everything shown here traces
to tool output — stat tiles and charts are numbers the tools produced (never the LLM),
sources are the RAG tools' resolved citations, the trace chip row names the tools that ran.
All render alongside the answer text, never replacing it. The pure builders live in
``stock_agent.ui.html`` / ``stock_agent.ui.tiles``; this module only composes them via
Streamlit, so color/spacing stay in the one token source (theme.py, graceful degradation).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import streamlit as st

from stock_agent.reports.export import EXPORT_META, export_summary
from stock_agent.ui.html import chip, stat_tile, trace_bar
from stock_agent.viz.charts import ChartSpec
from stock_agent.viz.render import to_altair, to_png


def render_stat_tiles(tiles: Sequence[dict[str, Any]]) -> None:
    """Render headline stat tiles above the answer ("summary-before-detail", R4).

    ``tiles`` are display-ready dicts from ``stock_agent.ui.tiles`` (label/value/sub/tone);
    numbers were formatted upstream from tool results. Laid out in a compact grid (max 3
    per row) so a multi-tool turn reads as a scannable header. No-op when empty.
    """
    if not tiles:
        return
    per_row = 3
    for row_start in range(0, len(tiles), per_row):
        row = tiles[row_start : row_start + per_row]
        # Pad the final row so tiles keep a consistent width instead of stretching.
        cols = st.columns(per_row)
        for col, tile in zip(cols, row):
            with col:
                st.markdown(
                    stat_tile(
                        label=str(tile.get("label", "")),
                        value=str(tile.get("value", "")),
                        sub=str(tile.get("sub", "")),
                        tone=str(tile.get("tone", "accent")),
                        direction=tile.get("direction"),
                    ),
                    unsafe_allow_html=True,
                )


def render_chart(spec: ChartSpec) -> None:
    """Render one ChartSpec as an Altair chart alongside the agent's text."""
    st.altair_chart(to_altair(spec).properties(height=260), use_container_width=True)
    if spec.caption:
        st.caption(spec.caption)


def render_trace(tools: Sequence[str], *, is_auto: bool) -> None:
    """Render the tool trace as a colored chip row (Auto) or a direct-route chip (domain).

    Live-turn only (like the pre-R4 captions it replaces): Auto shows one hue-tinted chip
    per tool the agent ran; a deterministic route shows a single brass chip naming the tool
    plus the "no LLM routing call" note. No-op when no tool ran.
    """
    if not tools:
        return
    if is_auto:
        st.markdown(trace_bar(tools), unsafe_allow_html=True)
    else:
        # Deterministic path: exactly one tool; keep the "no routing LLM call" framing.
        st.markdown(
            '<div class="sa-trace">'
            + chip(tools[0], tone="accent", marker="→")
            + '<span class="sa-status__cov">direct route · no LLM routing call</span>'
            + "</div>",
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def _chart_pngs(charts_json: str) -> list[bytes]:
    """Render a turn's charts to PNG bytes (cached per chart set; vl-convert)."""
    if not charts_json or charts_json == "[]":
        return []
    specs = [ChartSpec.from_dict(d) for d in json.loads(charts_json)]
    return [to_png(s) for s in specs]


@st.cache_data(show_spinner=False)
def _export_bytes(text: str, fmt: str, charts_json: str) -> bytes:
    """Render an answer (text + chart figures) to document bytes; cached across reruns."""
    images = _chart_pngs(charts_json) if fmt in ("pdf", "docx") else None
    return export_summary(text, fmt, images=images)


def render_export(text: str, charts: list[ChartSpec], idx: int) -> None:
    """PDF / Word / Markdown download buttons for an assistant answer (with figures).

    R4: single ``Export ▾`` affordance via ``st.popover`` when available (Streamlit ≥1.31,
    within our ≥1.35 pin); degrades to the pre-R4 expander otherwise (principle #6). Same
    three formats + cached bytes — export behavior unchanged.
    """
    charts_json = json.dumps([c.to_dict() for c in charts])

    def _buttons(container: Any) -> None:
        labels = {"pdf": "PDF", "docx": "Word", "md": "Markdown"}
        for fmt in ("pdf", "docx", "md"):
            mime, ext = EXPORT_META[fmt]
            container.download_button(
                labels[fmt],
                data=_export_bytes(text, fmt, charts_json),
                file_name=f"stock_summary.{ext}",
                mime=mime,
                key=f"dl_{idx}_{fmt}",
                use_container_width=True,
            )

    if hasattr(st, "popover"):
        with st.popover("Export ▾", use_container_width=False):
            _buttons(st)
    else:  # graceful fallback on older Streamlit
        with st.expander("📄 Export this summary"):
            cols = st.columns(3)
            labels = {"pdf": "PDF", "docx": "Word", "md": "Markdown"}
            for col, fmt in zip(cols, ["pdf", "docx", "md"]):
                mime, ext = EXPORT_META[fmt]
                col.download_button(
                    labels[fmt],
                    data=_export_bytes(text, fmt, charts_json),
                    file_name=f"stock_summary.{ext}",
                    mime=mime,
                    key=f"dl_{idx}_{fmt}",
                    use_container_width=True,
                )


def render_sources(sources: list[dict]) -> None:  # type: ignore[type-arg]
    """Render a turn's resolved SEC citations as a compact 'Filing sources' expander.

    R4 restyle: each citation gets a mono marker badge + label (from tool output only);
    dedup/order are unchanged (the caller passes an already-deduped list). No-op when empty.
    """
    if not sources:
        return
    with st.expander(f"📑 Filing sources ({len(sources)})"):
        for s in sources:
            st.markdown(
                '<div class="sa-src">'
                + chip(str(s["marker"]), tone="accent")
                + f'<span class="sa-src__label">{_esc_label(s["label"])}</span>'
                + "</div>",
                unsafe_allow_html=True,
            )


def _esc_label(text: str) -> str:
    """Escape a citation label for inline HTML (labels come from tool output, but defensive)."""
    from html import escape

    return escape(str(text))
