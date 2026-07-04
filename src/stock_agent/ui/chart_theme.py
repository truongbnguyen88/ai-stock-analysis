"""Shared Altair/Vega-Lite chart theme for the chat UI (redesign R5).

Pure, typed, import-safe (**no Altair import**): returns plain data (a Vega-Lite ``config``
dict + color constants) that ``stock_agent.viz.render`` applies to a chart. Keeping it Altair-
free means it is strict-typed and unit-tested like the rest of ``stock_agent.ui`` — the render
module (which already imports Altair lazily) is the only place charts are actually built.

Colors and the mono font are **derived from the design tokens** (``stock_agent.ui.theme``),
so the chart palette can never drift from the CSS token set (docs/APP_REDESIGN.md §5.2). We
use the *dark-theme* token hexes as the single mark palette because one bar fill must read on
both the dark app surface and the white PDF/DOCX export page; the secondary hues are mid-tone
and legible on both. Text/label *colors* are intentionally left unset so Streamlit's adaptive
Vega theme supplies AA-contrast text per active theme in-app (and Vega's default handles the
white export page) — this module only fixes the theme-independent look: fonts, faint grid,
and the categorical palette.

Signaling rule (§2): chart up/down green-red is reserved for direction *marks*. Current
``ChartSpec``s carry no direction flag (by design — the LLM/heuristic must not assign
financial semantics), so bars use the brass accent (single series) or the neutral secondary
palette (grouped). The up/down token colors live in the theme for a future direction-aware
spec (Phase 2); they are deliberately not applied as chrome here.
"""

from __future__ import annotations

from typing import Any

from stock_agent.ui.theme import dark_token, mono_font_stack

# Monospace stack (same source as the CSS labels) — Vega falls back gracefully if a specific
# face is absent (e.g. vl-convert's font set for PNG export).
_MONO: str = mono_font_stack()

# Categorical palette for grouped-bar series, ordered for colorblind separability: the first
# two series (the common case — e.g. P(up)/P(big move), Positive/Negative) get sky vs. rose,
# which differ in both hue and lightness and avoid a red-green pairing. Derived from §5.2.
CATEGORY_PALETTE: tuple[str, ...] = tuple(
    dark_token(t) for t in ("--sa-sky", "--sa-rose", "--sa-indigo", "--sa-teal", "--sa-violet")
)

# Single-series bars: the restrained brass primary accent (one accent, §2).
MARK_COLOR: str = dark_token("--sa-accent")

# Reliability-diagram points (calibration): sky, distinct from the gray y=x reference line.
POINT_COLOR: str = dark_token("--sa-sky")

# The y=x "perfect calibration" reference line: muted gray at higher alpha than --sa-grid so
# it reads as a guide without competing with the data. RGB is the --sa-muted token (138,147,163).
REFERENCE_COLOR: str = "rgba(138,147,163,0.55)"

# Directional bar marks (green up / red down / neutral brass). Applied ONLY when a ChartSpec
# carries a `direction` column set from tool numbers (e.g. the get_large_move up-/down-tail
# split) — so the §2 rule holds: the color reflects the figure, not an LLM/heuristic judgment.
# Dark-token hexes for both themes (same single-palette rule as the categorical marks above).
UP_COLOR: str = dark_token("--sa-up")
DOWN_COLOR: str = dark_token("--sa-down")

# Faint gridlines / axis rules (the --sa-grid token value).
_GRID: str = "rgba(138,147,163,0.14)"


def altair_config() -> dict[str, Any]:
    """Return the Vega-Lite ``config`` dict applied to every chart (pure, deterministic).

    Fixes only the theme-independent look — mono label/title fonts, a faint grid + axis rules,
    no view border, and the categorical range — while leaving text *colors* unset so the
    in-app Streamlit theme (adaptive, AA both themes) or Vega's export default supplies them.
    Applied via ``chart.configure(**altair_config())`` in ``stock_agent.viz.render``.
    """
    return {
        "view": {"stroke": None},  # drop the default outer chart border
        "axis": {
            "grid": True,
            "gridColor": _GRID,
            "gridWidth": 1,
            "domainColor": _GRID,
            "tickColor": _GRID,
            "labelFont": _MONO,
            "titleFont": _MONO,
            "labelFontSize": 11,
            "titleFontSize": 12,
        },
        "legend": {
            "labelFont": _MONO,
            "titleFont": _MONO,
            "labelFontSize": 11,
            "titleFontSize": 11,
        },
        "title": {"font": _MONO, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
        "range": {"category": list(CATEGORY_PALETTE)},
    }
