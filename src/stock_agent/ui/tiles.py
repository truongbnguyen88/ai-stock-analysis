"""Tool-result -> headline stat tiles for the chat UI (redesign R4).

Pure, typed, import-safe (no Streamlit, no I/O) — mirrors ``stock_agent.viz.charts``:
turns the *numbers the tools already computed* into a small, bounded set of summary tiles
shown above the answer ("summary-before-detail"). Every value traces to a tool result,
never to the LLM, preserving the numbers-vs-narrative invariant (APP_REDESIGN §6).

Dispatch is by tool ``name`` (like ``charts_for``) so only well-known scalar-bearing
results produce tiles; multi-ticker comparisons and RAG tools intentionally yield none
(they surface as charts / filing sources instead). Error results are skipped. The ``tone``
is a semantic secondary hue (teal/sky/indigo/violet/rose) for the category stripe; a separate
**optional** ``direction`` (``up``/``down``) tints the *value* green/red. Direction is set
only from a **deterministic sign read of the tool number** (never the LLM), so the §2 signaling
rule still holds — the color comes from the figure, not the narrative. Absent ``direction`` →
neutral value.

Each tile is a plain ``dict[str, str]`` (``label``/``value``/``sub``/``tone`` [+ ``direction``]):
directly JSON-serializable for the chat store and directly consumable by ``ui.html.stat_tile``.
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


def _sign_dir(x: Any) -> str | None:
    """Value direction from a signed tool number: ``>0`` → ``"up"``, ``<0`` → ``"down"``, else None.

    Deterministic sign read — NEVER an LLM/heuristic call — so tinting the tile value green/red
    honors the §2 signaling rule (direction from the figure, not the narrative).
    """
    if not isinstance(x, (int, float)):
        return None
    return "up" if x > 0 else "down" if x < 0 else None


def _prob_dir(x: Any, *, mid: float = 0.5) -> str | None:
    """Direction from which side of ``mid`` a probability sits (``P(up) > 0.5`` → ``"up"``).

    A deterministic threshold read of the model's probability mass; not a recommendation, and
    never LLM-assigned. ``mid`` is the no-lean point (0.5 for a binary up/down probability).
    """
    if not isinstance(x, (int, float)):
        return None
    return "up" if x > mid else "down" if x < mid else None


def _dir(tile: Tile, direction: str | None) -> Tile:
    """Attach a tool-driven ``direction`` to a tile only when present (omit → neutral value)."""
    if direction is not None:
        tile["direction"] = direction
    return tile


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
    # Value tinted by the sign of the period change (up/down over the window), not the LLM.
    return [
        _dir(
            {"label": "Last close", "value": close, "sub": sub, "tone": "sky"},
            _sign_dir(r.get("pct_change")),
        )
    ]


def _forecast_tiles(r: dict[str, Any]) -> list[Tile]:
    """P(up) + expected return from ``run_forecast``."""
    tiles: list[Tile] = []
    up = _pct(r.get("upside_prob"))
    if up is not None:
        tiles.append(
            _dir(
                {"label": "P(up)", "value": up, "sub": _horizon_sub(r), "tone": "indigo"},
                _prob_dir(r.get("upside_prob")),  # >0.5 leans up, <0.5 leans down
            )
        )
    exp = _pct(r.get("expected_return"), signed=True, dp=1)
    if exp is not None:
        tiles.append(
            _dir(
                {"label": "Exp. return", "value": exp, "sub": _horizon_sub(r), "tone": "violet"},
                _sign_dir(r.get("expected_return")),
            )
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
    # Direction from the tool's own up/down tail lean (not the LLM); magnitude otherwise neutral.
    direction = lean if lean in ("up", "down") else None
    return [_dir({"label": label, "value": p, "sub": sub, "tone": "rose"}, direction)]


def _sentiment_tiles(r: dict[str, Any]) -> list[Tile]:
    """Net average sentiment from ``get_news_sentiment`` (article count as sub)."""
    avg = r.get("avg_sentiment")
    if not isinstance(avg, (int, float)):
        return []
    n = r.get("article_count")
    n_int = int(n) if isinstance(n, (int, float)) else None
    sub = f"{n_int} articles" if n_int is not None else ""
    return [
        _dir(
            {"label": "Net sentiment", "value": f"{avg:+.2f}", "sub": sub, "tone": "teal"},
            _sign_dir(avg),
        )
    ]


def _research_tiles(r: dict[str, Any]) -> list[Tile]:
    """The full header row for the executive brief (``research_summary``) — up to 5 cards.

    Mirrors the cards the individual tools produce, read from the brief's consolidated
    payload: Last close (price snapshot), Net sentiment (news block), P(up) + Exp. return
    (shortest-horizon forecast), and P(±k%) (large-move tail split). Every value is a tool
    number (never the LLM); each sub-tile no-ops when its slice is absent.
    """
    if not _ok(r):
        return []
    tiles: list[Tile] = []

    ps = r.get("price_snapshot")
    if isinstance(ps, dict) and (close := _money(ps.get("last_close"))) is not None:
        chg = _pct(ps.get("pct_change"), signed=True, dp=1)
        sub = f"{chg} · {ps.get('window_days')}d" if chg else f"{ps.get('window_days')}d"
        tiles.append(
            _dir(
                {"label": "Last close", "value": close, "sub": sub, "tone": "sky"},
                _sign_dir(ps.get("pct_change")),
            )
        )

    news = r.get("news")
    if isinstance(news, dict) and isinstance(news.get("avg_sentiment"), (int, float)):
        avg = news["avg_sentiment"]
        n = news.get("article_count")
        sub = f"{int(n)} articles" if isinstance(n, (int, float)) else ""
        tiles.append(
            _dir(
                {"label": "Net sentiment", "value": f"{avg:+.2f}", "sub": sub, "tone": "teal"},
                _sign_dir(avg),
            )
        )

    fcs = r.get("forecasts")
    if isinstance(fcs, list) and fcs:
        # Shortest horizon = the primary directional signal (matches the large-move card).
        fc = min(
            (f for f in fcs if isinstance(f, dict)),
            key=lambda f: f.get("horizon_days", 1_000_000),
            default=None,
        )
        if fc is not None:
            h = fc.get("horizon_days")
            hsub = f"{h}d" if isinstance(h, int) else ""
            up = _pct(fc.get("prob_up"))
            if up is not None:
                tiles.append(
                    _dir(
                        {"label": "P(up)", "value": up, "sub": hsub, "tone": "indigo"},
                        _prob_dir(fc.get("prob_up")),
                    )
                )
            exp = _pct(fc.get("expected_return"), signed=True, dp=1)
            if exp is not None:
                tiles.append(
                    _dir(
                        {"label": "Exp. return", "value": exp, "sub": hsub, "tone": "violet"},
                        _sign_dir(fc.get("expected_return")),
                    )
                )

    lm = r.get("large_move")
    if isinstance(lm, dict) and (p := _pct(lm.get("prob_large_move"))) is not None:
        thr = lm.get("threshold")
        label = f"P(±{round(thr * 100)}%)" if isinstance(thr, (int, float)) else "P(big move)"
        lean = lm.get("lean")
        h = lm.get("horizon_days")
        sub = f"{h}d · lean {lean}" if isinstance(h, int) and isinstance(lean, str) else ""
        tiles.append(
            _dir(
                {"label": label, "value": p, "sub": sub, "tone": "rose"},
                lean if lean in ("up", "down") else None,
            )
        )
    return tiles


# Tool name -> tile builder. Only these tools produce tiles; everything else (comparisons,
# RAG, raw news, earnings) surfaces through charts / sources instead. The executive brief
# (research_summary) consolidates the individual cards into one full header row.
_BUILDERS = {
    "get_price_summary": _price_tiles,
    "run_forecast": _forecast_tiles,
    "get_large_move": _large_move_tiles,
    "get_news_sentiment": _sentiment_tiles,
    "research_summary": _research_tiles,
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
