"""ChartSpec -> Altair / PNG rendering (vl-convert, no browser)."""

from __future__ import annotations

import pandas as pd

from stock_agent.ui.chart_theme import (
    CATEGORY_PALETTE,
    DOWN_COLOR,
    MARK_COLOR,
    UP_COLOR,
    altair_config,
)
from stock_agent.ui.theme import mono_font_stack
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


def _grouped() -> ChartSpec:
    return ChartSpec(
        title="Forecast comparison",
        kind="grouped_bar",
        data=pd.DataFrame(
            {
                "ticker": ["NVDA", "NVDA", "AMD", "AMD"],
                "metric": ["P(up)", "P(big move)", "P(up)", "P(big move)"],
                "probability": [0.6, 0.3, 0.55, 0.28],
            }
        ),
        x="ticker",
        y="probability",
        color="metric",
        y_is_percent=True,
    )


def test_to_altair_carries_title_and_data() -> None:
    spec = to_altair(_bar()).to_dict()
    assert spec["title"] == "Forecast buckets"
    assert spec["mark"]["type"] == "bar"


def test_single_bar_uses_brass_accent() -> None:
    mark = to_altair(_bar()).to_dict()["mark"]
    assert mark["color"] == MARK_COLOR  # brass single-series bar (semantic palette)


def test_direction_bar_colors_up_green_down_red_neutral_brass() -> None:
    # A "bar" with a `direction` column (e.g. the large-move up/down tail split) tints each bar
    # from the tool's own direction — not a flat fill — via an up/down/neutral color scale.
    spec = ChartSpec(
        title="Large-move breakdown",
        kind="bar",
        data=pd.DataFrame(
            {
                "outcome": ["Big up", "No big move", "Big down"],
                "probability": [0.2, 0.65, 0.15],
                "direction": ["up", "neutral", "down"],
            }
        ),
        x="outcome",
        y="probability",
        y_is_percent=True,
        direction="direction",
    )
    d = to_altair(spec).to_dict()
    color = d["encoding"]["color"]
    assert color["field"] == "direction"
    assert color["scale"]["domain"] == ["up", "down", "neutral"]
    assert color["scale"]["range"] == [UP_COLOR, DOWN_COLOR, MARK_COLOR]
    # Direction drives the fill via the color encoding, not a flat mark color.
    mark = d["mark"]
    assert mark == "bar" or (isinstance(mark, dict) and "color" not in mark)


def test_percent_axis_still_honored_under_theme() -> None:
    # y_is_percent must survive the theme application (percent tick format on the y axis).
    y_enc = to_altair(_bar()).to_dict()["encoding"]["y"]
    assert y_enc["axis"]["format"] == "%"


def test_theme_config_applied() -> None:
    # The shared config (mono fonts, faint grid, no view border) rides on every chart.
    # (Altair injects continuousWidth/Height defaults into config.view, so compare the
    # sections we own rather than the whole dict.)
    want = altair_config()
    cfg = to_altair(_bar()).to_dict()["config"]
    assert cfg["axis"] == want["axis"]
    assert cfg["legend"] == want["legend"]
    assert cfg["title"] == want["title"]
    assert cfg["range"] == want["range"]
    assert cfg["view"]["stroke"] is None
    assert cfg["axis"]["labelFont"] == mono_font_stack()


def test_grouped_bar_uses_semantic_palette() -> None:
    color_enc = to_altair(_grouped()).to_dict()["encoding"]["color"]
    assert color_enc["scale"]["range"] == list(CATEGORY_PALETTE)


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
