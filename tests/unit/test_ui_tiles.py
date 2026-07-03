"""Tool-result -> headline stat tiles (R4): golden per-tool + dedup/cap/error-skip.

Pure-data tests — the extractor has no Streamlit dependency; it turns the numbers the
tools already produced into display-ready tiles (numbers-from-tools invariant). Tones are
semantic hues only (never chart up/down red-green).
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
