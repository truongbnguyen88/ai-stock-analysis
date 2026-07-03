"""Assistant/user message rendering: charts, SEC sources, and export (redesign R1).

Extracted verbatim from ``ui/chat_app.py`` — behavior-preserving. Charts come from the
tool results (numbers the tools produced, never the LLM); sources are the RAG tools'
resolved citations. Both render alongside the answer text, never replacing it.
"""

from __future__ import annotations

import json

import streamlit as st

from stock_agent.reports.export import EXPORT_META, export_summary
from stock_agent.viz.charts import ChartSpec
from stock_agent.viz.render import to_altair, to_png


def render_chart(spec: ChartSpec) -> None:
    """Render one ChartSpec as an Altair chart alongside the agent's text."""
    st.altair_chart(to_altair(spec).properties(height=260), use_container_width=True)
    if spec.caption:
        st.caption(spec.caption)


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
    """PDF / Word / Markdown download buttons for an assistant answer (with figures)."""
    charts_json = json.dumps([c.to_dict() for c in charts])
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
    """Render a turn's resolved SEC citations as a compact 'Filing sources' expander."""
    if not sources:
        return
    with st.expander(f"📑 Filing sources ({len(sources)})"):
        for s in sources:
            st.caption(f"[{s['marker']}] {s['label']}")
