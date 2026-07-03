"""Tool-result -> headline stat tiles for the chat UI (redesign R4).

Pure, typed, import-safe (no Streamlit, no I/O) — mirrors ``stock_agent.viz.charts``:
turns the *numbers the tools already computed* into a small, bounded set of summary tiles
shown above the answer ("summary-before-detail"). Every value traces to a tool result,
never to the LLM, preserving the numbers-vs-narrative invariant (APP_REDESIGN §6).

Dispatch is by tool ``name`` (like ``charts_for``) so only well-known scalar-bearing
results produce tiles; multi-ticker comparisons and RAG tools intentionally yield none
(they surface as charts / filing sources instead). Error results are skipped. Tones are
semantic secondary hues (teal/sky/indigo/violet/rose) — never chart up/down red-green,
which is reserved for marks (§2 signaling rule).

Each tile is a plain ``dict[str, str]`` (``label``/``value``/``sub``/``tone``): directly
JSON-serializable for the chat store and directly consumable by ``ui.html.stat_tile``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

# One tile is a display-ready bag of strings (formatting already applied here).
Tile = dict[str, str]

# Cap the row so a multi-tool Auto turn can't flood the header; first-seen order wins.
_MAX_TILES = 6


class _HasResult(Protocol):
    """Structural view of a tool call (matches ``agent.runtime.ToolInvocation``)."""

    name: str
    result: dict[str, Any]


def _ok(result: dict[str, Any]) -> bool:
    """True when the tool succeeded (mirrors ``charts_for``'s error convention)."""
    return isinstance(result, dict) and "error" not in result


def _pct(x: Any, *, signed: bool = False, dp: int = 0) -> str | None:
    """Format a fraction as a percent string (``0.58`` -> ``"58%"``); None-safe."""
    if not isinstance(x, (int, float)):
        return None
    return f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"


def _money(x: Any) -> str | None:
    """Format a price as ``$1,234.56`` (2 dp, thousands separator); None-safe."""
    if not isinstance(x, (int, float)):
        return None
    return f"${x:,.2f}"


def _horizon_sub(result: dict[str, Any], extra: str = "") -> str:
    """Compose a compact sub-label from horizon (+ optional model / extra note)."""
    parts: list[str] = []
    h = result.get("horizon_days")
    if isinstance(h, int):
        parts.append(f"{h}d")
    model = result.get("model_name")
    if isinstance(model, str) and model:
        parts.append(model)
    if extra:
        parts.append(extra)
    return " · ".join(parts)


# ---- per-tool tile builders (each returns 0..2 tiles) ------------------------
def _price_tiles(r: dict[str, Any]) -> list[Tile]:
    """Last close + period change from ``get_price_summary``."""
    close = _money(r.get("last_close"))
    if close is None:
        return []
    chg = _pct(r.get("pct_change"), signed=True, dp=1)
    n = r.get("n_bars")
    parts = [
        f"{chg} period" if chg else "",
        f"{n} bars" if isinstance(n, int) else "",
    ]
    sub = " · ".join(p for p in parts if p)
    return [{"label": "Last close", "value": close, "sub": sub, "tone": "sky"}]


def _forecast_tiles(r: dict[str, Any]) -> list[Tile]:
    """P(up) + expected return from ``run_forecast``."""
    tiles: list[Tile] = []
    up = _pct(r.get("upside_prob"))
    if up is not None:
        tiles.append(
            {"label": "P(up)", "value": up, "sub": _horizon_sub(r), "tone": "indigo"}
        )
    exp = _pct(r.get("expected_return"), signed=True, dp=1)
    if exp is not None:
        tiles.append(
            {"label": "Exp. return", "value": exp, "sub": _horizon_sub(r), "tone": "violet"}
        )
    return tiles


def _large_move_tiles(r: dict[str, Any]) -> list[Tile]:
    """P(|r|>k) magnitude signal from ``get_large_move``."""
    p = _pct(r.get("prob_large_move"))
    if p is None:
        return []
    k = r.get("threshold_pct")
    label = f"P(±{k}%)" if isinstance(k, int) else "P(big move)"
    lean = r.get("lean")
    sub = _horizon_sub(r, extra=f"lean {lean}" if isinstance(lean, str) else "")
    return [{"label": label, "value": p, "sub": sub, "tone": "rose"}]


def _sentiment_tiles(r: dict[str, Any]) -> list[Tile]:
    """Net average sentiment from ``get_news_sentiment`` (article count as sub)."""
    avg = r.get("avg_sentiment")
    if not isinstance(avg, (int, float)):
        return []
    n = r.get("article_count")
    n_int = int(n) if isinstance(n, (int, float)) else None
    sub = f"{n_int} articles" if n_int is not None else ""
    return [{"label": "Net sentiment", "value": f"{avg:+.2f}", "sub": sub, "tone": "teal"}]


# Tool name -> tile builder. Only these tools produce tiles; everything else (comparisons,
# RAG, raw news, earnings) surfaces through charts / sources instead.
_BUILDERS = {
    "get_price_summary": _price_tiles,
    "run_forecast": _forecast_tiles,
    "get_large_move": _large_move_tiles,
    "get_news_sentiment": _sentiment_tiles,
}


def stat_tiles_from_tool_results(tool_results: Sequence[_HasResult]) -> list[Tile]:
    """Collect headline stat tiles from recognized tool results (numbers-from-tools).

    Dispatches by tool ``name``; skips error results and unknown tools. Deduped by
    ``label`` (first-seen wins) and capped at :data:`_MAX_TILES` so a many-tool Auto turn
    stays a compact header row. Returns display-ready dicts (see module docstring).
    """
    seen: set[str] = set()
    tiles: list[Tile] = []
    for inv in tool_results:
        builder = _BUILDERS.get(inv.name)
        if builder is None or not _ok(inv.result):
            continue
        for tile in builder(inv.result):
            if tile["label"] in seen:
                continue
            seen.add(tile["label"])
            tiles.append(tile)
            if len(tiles) >= _MAX_TILES:
                return tiles
    return tiles
