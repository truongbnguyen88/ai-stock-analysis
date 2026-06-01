"""Render a ChartSpec to an Altair chart (UI) or PNG bytes (document export).

Kept separate from ``viz.charts`` (the pure, pandas-only ChartSpec builders) so
that module stays dependency-light; altair and vl_convert are imported lazily here,
only when a chart is actually drawn. ``to_altair`` is the single source of truth for
how a ChartSpec looks, shared by the Streamlit UI and the PDF/DOCX export.
"""

from __future__ import annotations

from typing import Any

from stock_agent.viz.charts import ChartSpec


def to_altair(spec: ChartSpec) -> Any:
    """Build the Altair chart for a ChartSpec (title included; size left to the caller)."""
    import altair as alt
    import pandas as pd

    y_axis = alt.Axis(format="%") if spec.y_is_percent else alt.Axis()
    y_enc = alt.Y(f"{spec.y}:Q", axis=y_axis, title=None)
    x_sort: Any = list(spec.x_sort) if spec.x_sort else "ascending"

    if spec.kind == "reliability":
        # Predicted vs realized, with the y=x ideal as a dashed reference line.
        pts = alt.Chart(spec.data).mark_circle(size=90, color="#4c78a8").encode(
            x=alt.X("predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted"),
            y=alt.Y("realized:Q", scale=alt.Scale(domain=[0, 1]), title="Realized"),
        )
        ideal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[5, 5], color="gray")
            .encode(x="x:Q", y="y:Q")
        )
        chart = ideal + pts
    elif spec.kind == "grouped_bar":
        chart = alt.Chart(spec.data).mark_bar().encode(
            x=alt.X(f"{spec.x}:N", sort=x_sort, title=None),
            y=y_enc,
            color=alt.Color(f"{spec.color}:N", title=None),
            xOffset=f"{spec.color}:N",
            tooltip=list(spec.data.columns),
        )
    else:  # "bar"
        chart = alt.Chart(spec.data).mark_bar(color="#4c78a8").encode(
            x=alt.X(f"{spec.x}:N", sort=x_sort, title=None),
            y=y_enc,
            tooltip=list(spec.data.columns),
        )
    return chart.properties(title=spec.title)


def to_png(spec: ChartSpec, *, scale: float = 2.0) -> bytes:
    """Render a ChartSpec to PNG bytes via vl-convert (no browser/node)."""
    import vl_convert as vlc

    chart = to_altair(spec).properties(width=560, height=300)
    return bytes(vlc.vegalite_to_png(chart.to_json(), scale=scale))
