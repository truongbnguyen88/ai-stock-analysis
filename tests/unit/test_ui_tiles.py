"""Tool-result -> headline stat tiles (R4): golden per-tool + dedup/cap/error-skip.

Pure-data tests — the extractor has no Streamlit dependency; it turns the numbers the
tools already produced into display-ready tiles (numbers-from-tools invariant). The ``tone``
is a semantic stripe hue; a separate optional ``direction`` (up/down) tints the value green/red
from a deterministic sign read of the tool number (never the LLM) — see the direction tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stock_agent.ui.tiles import _MAX_TILES, stat_tiles_from_tool_results


@dataclass
class _Inv:
    """Minimal structural stand-in for agent.runtime.ToolInvocation (name + result)."""

    name: str
    result: dict[str, Any]


def _labels(tiles: list[dict[str, str]]) -> list[str]:
    return [t["label"] for t in tiles]


def _by_label(tiles: list[dict[str, str]], label: str) -> dict[str, str]:
    return next(t for t in tiles if t["label"] == label)


# ---- per-tool golden shapes -------------------------------------------------------
def test_price_summary_tile() -> None:
    inv = _Inv(
        "get_price_summary",
        {"last_close": 1234.5, "pct_change": 0.0812, "n_bars": 30},
    )
    tiles = stat_tiles_from_tool_results([inv])
    t = _by_label(tiles, "Last close")
    assert t["value"] == "$1,234.50"  # 2dp + thousands separator
    assert "+8.1% period" in t["sub"] and "30 bars" in t["sub"]
    assert t["tone"] == "sky"
    assert t["direction"] == "up"  # pct_change > 0


def test_forecast_tiles_pup_and_expected_return() -> None:
    inv = _Inv(
        "run_forecast",
        {
            "upside_prob": 0.58,
            "expected_return": -0.032,
            "horizon_days": 20,
            "model_name": "ensemble",
        },
    )
    tiles = stat_tiles_from_tool_results([inv])
    assert _labels(tiles) == ["P(up)", "Exp. return"]
    assert _by_label(tiles, "P(up)")["value"] == "58%"
    assert _by_label(tiles, "P(up)")["sub"] == "20d · ensemble"
    # Signed 1dp, negative sign preserved.
    assert _by_label(tiles, "Exp. return")["value"] == "-3.2%"
    assert _by_label(tiles, "P(up)")["tone"] == "indigo"
    assert _by_label(tiles, "Exp. return")["tone"] == "violet"
    # Tool-driven direction: P(up)=0.58 > 0.5 → up; expected_return=-0.032 < 0 → down.
    assert _by_label(tiles, "P(up)")["direction"] == "up"
    assert _by_label(tiles, "Exp. return")["direction"] == "down"


def test_large_move_tile_threshold_and_lean() -> None:
    inv = _Inv(
        "get_large_move",
        {"prob_large_move": 0.27, "threshold_pct": 10, "horizon_days": 30, "lean": "up"},
    )
    (t,) = stat_tiles_from_tool_results([inv])
    assert t["label"] == "P(±10%)"
    assert t["value"] == "27%"
    assert "30d" in t["sub"] and "lean up" in t["sub"]
    assert t["tone"] == "rose"
    assert t["direction"] == "up"  # from the tool's up/down tail lean


def test_large_move_tile_without_threshold_uses_generic_label() -> None:
    inv = _Inv("get_large_move", {"prob_large_move": 0.3})
    (t,) = stat_tiles_from_tool_results([inv])
    assert t["label"] == "P(big move)"


def test_sentiment_tile_signed_and_count() -> None:
    inv = _Inv(
        "get_news_sentiment",
        {"avg_sentiment": 0.146, "article_count": 12.0, "pct_positive": 0.5},
    )
    (t,) = stat_tiles_from_tool_results([inv])
    assert t["label"] == "Net sentiment"
    assert t["value"] == "+0.15"  # signed 2dp
    assert t["sub"] == "12 articles"
    assert t["tone"] == "teal"
    assert t["direction"] == "up"  # avg_sentiment > 0


# ---- executive brief: the consolidated 5-card header row --------------------------
def _brief_result() -> dict[str, Any]:
    """A research_summary payload shaped like the MU run (all 5 cards populated)."""
    return {
        "price_snapshot": {"last_close": 975.56, "pct_change": -0.0205, "window_days": 30},
        "news": {"avg_sentiment": 0.16, "article_count": 25},
        "forecasts": [
            {"horizon_days": 20, "prob_up": 0.67, "expected_return": 0.127},
            {"horizon_days": 60, "prob_up": 0.81, "expected_return": 0.519},
        ],
        "large_move": {
            "prob_large_move": 0.93, "threshold": 0.05, "horizon_days": 20, "lean": "balanced"
        },
    }


def test_research_summary_emits_full_five_card_row() -> None:
    tiles = stat_tiles_from_tool_results([_Inv("research_summary", _brief_result())])
    assert _labels(tiles) == ["Last close", "Net sentiment", "P(up)", "Exp. return", "P(±5%)"]
    assert _by_label(tiles, "Last close")["value"] == "$975.56"
    assert _by_label(tiles, "Last close")["direction"] == "down"  # pct_change < 0
    assert _by_label(tiles, "Net sentiment")["value"] == "+0.16"
    # P(up) / Exp. return come from the SHORTEST horizon (20d), matching the large-move card.
    assert _by_label(tiles, "P(up)")["value"] == "67%" and _by_label(tiles, "P(up)")["sub"] == "20d"
    assert _by_label(tiles, "Exp. return")["value"] == "+12.7%"
    assert _by_label(tiles, "P(±5%)")["value"] == "93%"
    assert "balanced" in _by_label(tiles, "P(±5%)")["sub"]
    assert "direction" not in _by_label(tiles, "P(±5%)")  # balanced lean → neutral


def test_research_summary_subtiles_noop_when_slices_absent() -> None:
    # A degraded brief (no news scores, no large-move) still yields the price + forecast cards.
    r = {"price_snapshot": {"last_close": 10.0, "pct_change": 0.0, "window_days": 30},
         "forecasts": [{"horizon_days": 20, "prob_up": 0.5, "expected_return": 0.0}]}
    tiles = stat_tiles_from_tool_results([_Inv("research_summary", r)])
    assert _labels(tiles) == ["Last close", "P(up)", "Exp. return"]


def test_research_summary_error_yields_no_tiles() -> None:
    assert stat_tiles_from_tool_results([_Inv("research_summary", {"error": "boom"})]) == []


# ---- tool-driven value direction (green/red) --------------------------------------
def test_direction_is_neutral_at_the_midpoint_and_zero() -> None:
    # P(up) exactly 0.5 and expected_return exactly 0.0 carry NO direction (neutral value),
    # so a coin-flip / flat forecast never shows a green or red bias.
    inv = _Inv("run_forecast", {"upside_prob": 0.5, "expected_return": 0.0, "horizon_days": 20})
    tiles = stat_tiles_from_tool_results([inv])
    assert "direction" not in _by_label(tiles, "P(up)")
    assert "direction" not in _by_label(tiles, "Exp. return")


def test_direction_follows_the_sign_down() -> None:
    # A bearish forecast: P(up) < 0.5 → down; negative sentiment → down.
    fc = _Inv("run_forecast", {"upside_prob": 0.42, "expected_return": -0.05, "horizon_days": 20})
    sent = _Inv("get_news_sentiment", {"avg_sentiment": -0.2, "article_count": 5})
    tiles = stat_tiles_from_tool_results([fc, sent])
    assert _by_label(tiles, "P(up)")["direction"] == "down"
    assert _by_label(tiles, "Exp. return")["direction"] == "down"
    assert _by_label(tiles, "Net sentiment")["direction"] == "down"


def test_large_move_without_lean_is_neutral() -> None:
    # Magnitude-only (no up/down tail lean from the tool) → no direction (the tile stays neutral).
    inv = _Inv("get_large_move", {"prob_large_move": 0.3, "threshold_pct": 10})
    (t,) = stat_tiles_from_tool_results([inv])
    assert "direction" not in t


# ---- aggregation semantics --------------------------------------------------------
def test_error_results_skipped() -> None:
    inv = _Inv("run_forecast", {"error": "boom"})
    assert stat_tiles_from_tool_results([inv]) == []


def test_unknown_tools_yield_no_tiles() -> None:
    # Comparisons / RAG surface as charts / sources, not tiles.
    inv = _Inv("compare_forecasts", {"rows": [{"ticker": "NVDA", "upside_prob": 0.6}]})
    assert stat_tiles_from_tool_results([inv]) == []


def test_missing_scalar_yields_no_tile() -> None:
    inv = _Inv("get_price_summary", {"n_bars": 30})  # no last_close
    assert stat_tiles_from_tool_results([inv]) == []


def test_dedup_by_label_first_seen_wins() -> None:
    a = _Inv("run_forecast", {"upside_prob": 0.60, "horizon_days": 20})
    b = _Inv("run_forecast", {"upside_prob": 0.40, "horizon_days": 60})
    tiles = stat_tiles_from_tool_results([a, b])
    # Only one "P(up)" tile, from the first invocation.
    pups = [t for t in tiles if t["label"] == "P(up)"]
    assert len(pups) == 1
    assert pups[0]["value"] == "60%"


def test_tiles_capped_at_max() -> None:
    # Many distinct-labelled tiles get capped; build via several tools.
    invs = [
        _Inv("get_price_summary", {"last_close": 10.0, "pct_change": 0.1, "n_bars": 5}),
        _Inv(
            "run_forecast",
            {"upside_prob": 0.5, "expected_return": 0.01, "horizon_days": 20},
        ),
        _Inv("get_large_move", {"prob_large_move": 0.2, "threshold_pct": 5, "horizon_days": 20}),
        _Inv("get_news_sentiment", {"avg_sentiment": 0.1, "article_count": 3}),
    ]
    tiles = stat_tiles_from_tool_results(invs)
    assert len(tiles) <= _MAX_TILES


def test_empty_input() -> None:
    assert stat_tiles_from_tool_results([]) == []
