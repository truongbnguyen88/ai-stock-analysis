"""ChartSpec -> Altair / PNG rendering (vl-convert, no browser)."""

from __future__ import annotations

import pandas as pd

from stock_agent.viz.charts import ChartSpec
from stock_agent.viz.render import to_altair, to_png


def _bar() -> ChartSpec:
    return ChartSpec(
        title="Forecast buckets",
        kind="bar",
        data=pd.DataFrame({"bucket": ["< -10%", "> +10%"], "probability": [0.2, 0.8]}),
        x="bucket",
        y="probability",
        x_sort=("< -10%", "> +10%"),
        y_is_percent=True,
    )


def test_to_altair_carries_title_and_data() -> None:
    spec = to_altair(_bar()).to_dict()
    assert spec["title"] == "Forecast buckets"
    assert spec["mark"]["type"] == "bar"


def test_to_png_renders_valid_png() -> None:
    png = to_png(_bar())
    assert png[:4] == b"\x89PNG"  # PNG magic
    assert len(png) > 2000  # a real rasterized chart


def test_to_png_handles_reliability_layered_chart() -> None:
    spec = ChartSpec(
        title="Reliability",
        kind="reliability",
        data=pd.DataFrame({"predicted": [0.1, 0.5, 0.9], "realized": [0.08, 0.55, 0.85]}),
        x="predicted",
        y="realized",
    )
    assert to_png(spec)[:4] == b"\x89PNG"  # layered (ideal line + points) renders
