"""Pure HTML sidebar builders (R2): freshness logic + fragment well-formedness/escaping.

Pure-string tests — the module has no Streamlit dependency, so these assert the fragment
contract the view layer composes without rendering anything.
"""

from __future__ import annotations

from datetime import date, timedelta

from stock_agent.ui.capabilities import CAPABILITIES
from stock_agent.ui.html import (
    brand_block,
    cap_card,
    cap_hue,
    chip,
    corpus_freshness,
    eyebrow,
    keys_row,
    status_card,
    status_dot,
)

_TODAY = date(2026, 7, 2)


# ---- corpus_freshness -------------------------------------------------------------
def test_freshness_unavailable_when_store_closed() -> None:
    # chunks < 0 signals the vector store couldn't be opened (per corpus_status).
    assert corpus_freshness("2026-06-01", -1, today=_TODAY) == "unavailable"


def test_freshness_unavailable_when_no_corpus() -> None:
    assert corpus_freshness(None, 0, today=_TODAY) == "unavailable"


def test_freshness_unavailable_on_bad_date() -> None:
    # A malformed date must not raise — degrade to unavailable.
    assert corpus_freshness("not-a-date", 100, today=_TODAY) == "unavailable"


def test_freshness_fresh_and_stale_around_threshold() -> None:
    # Boundary at stale_after_days (inclusive = fresh).
    at = (_TODAY - timedelta(days=120)).isoformat()
    just_past = (_TODAY - timedelta(days=121)).isoformat()
    assert corpus_freshness(at, 100, today=_TODAY, stale_after_days=120) == "fresh"
    assert corpus_freshness(just_past, 100, today=_TODAY, stale_after_days=120) == "stale"


def test_freshness_recent_is_fresh() -> None:
    recent = (_TODAY - timedelta(days=10)).isoformat()
    assert corpus_freshness(recent, 5000, today=_TODAY) == "fresh"


# ---- status_dot -------------------------------------------------------------------
def test_status_dot_valid_states() -> None:
    for state in ("fresh", "stale", "unavailable"):
        frag = status_dot(state)
        assert f"sa-dot--{state}" in frag
        assert frag.startswith("<span") and frag.endswith("</span>")


def test_status_dot_unknown_falls_back() -> None:
    assert "sa-dot--unavailable" in status_dot("bogus")


# ---- brand_block ------------------------------------------------------------------
def test_brand_block_contains_parts_and_classes() -> None:
    frag = brand_block(mark="📈", name="Stock Research Agent", tagline="not advice")
    assert "sa-brand__mark" in frag and "📈" in frag
    assert "sa-brand__name" in frag and "Stock Research Agent" in frag
    assert "sa-brand__tag" in frag and "not advice" in frag


# ---- status_card ------------------------------------------------------------------
def test_status_card_unavailable_chunks() -> None:
    frag = status_card(state="unavailable", embedder="voyage-4", chunks=-1, tickers=0, latest=None)
    assert "unavailable" in frag  # chunk text
    assert "no filings ingested yet" in frag
    assert "voyage-4" in frag
    assert "sa-dot--unavailable" in frag  # dot embedded


def test_status_card_formats_chunks_and_coverage() -> None:
    frag = status_card(
        state="fresh", embedder="voyage-voyage-4", chunks=12345, tickers=7, latest="2026-06-15"
    )
    assert "12,345 chunks" in frag  # thousands separator
    assert "7 tickers · fresh to 2026-06-15" in frag
    assert "sa-dot--fresh" in frag


# ---- chip -------------------------------------------------------------------------
def test_chip_tone_and_marker() -> None:
    frag = chip("Anthropic", tone="ok", marker="✓")
    assert "sa-chip--ok" in frag
    assert "✓" in frag and "Anthropic" in frag


def test_chip_unknown_tone_falls_back_to_neutral() -> None:
    assert "sa-chip--neutral" in chip("x", tone="bogus")


def test_chip_escapes_text() -> None:
    # Defensive: any interpolated text is HTML-escaped.
    assert "<script>" not in chip("<script>")
    assert "&lt;script&gt;" in chip("<script>")


# ---- keys_row ---------------------------------------------------------------------
def test_keys_row_encodes_state_by_tone_and_marker() -> None:
    frag = keys_row(
        [
            ("Anthropic", True, True),  # present
            ("Finnhub", False, False),  # optional missing
            ("Marketaux", False, True),  # required missing
        ]
    )
    assert frag.count("sa-chip ") == 3  # one chip per key
    assert "sa-chip--ok" in frag and "✓" in frag
    assert "sa-chip--muted" in frag and "·" in frag
    assert "sa-chip--off" in frag and "✕" in frag
    assert "Marketaux (required)" in frag  # required-missing annotates


# ---- eyebrow ----------------------------------------------------------------------
def test_eyebrow_wraps_text() -> None:
    frag = eyebrow("Routing")
    assert 'class="sa-eyebrow"' in frag and "Routing" in frag


def test_eyebrow_accent_tone_adds_modifier() -> None:
    assert "sa-eyebrow--accent" in eyebrow("Hero", tone="accent")
    # Default is muted: no accent modifier.
    assert "sa-eyebrow--accent" not in eyebrow("Sidebar")


# ---- cap_hue / cap_card (R3) ------------------------------------------------------
def test_cap_hue_cycles_five_hues() -> None:
    hues = [cap_hue(i) for i in range(5)]
    assert hues == ["teal", "sky", "indigo", "violet", "rose"]
    # Wraps deterministically past the 5th.
    assert cap_hue(5) == "teal"
    assert cap_hue(11) == cap_hue(1)


def test_cap_card_renders_parts_and_hue_var() -> None:
    frag = cap_card(icon="📊", title="Technical analysis", blurb="Trend, momentum", hue="teal")
    assert "sa-cap__badge" in frag and "📊" in frag
    assert "sa-cap__title" in frag and "Technical analysis" in frag
    assert "sa-cap__blurb" in frag and "Trend, momentum" in frag
    assert "--cap-hue: var(--sa-teal)" in frag  # hue wired as a CSS variable


def test_cap_card_unknown_hue_falls_back_to_accent() -> None:
    assert "var(--sa-accent)" in cap_card(icon="x", title="t", blurb="b", hue="chartreuse")


def test_cap_card_escapes_text() -> None:
    frag = cap_card(icon="x", title="<b>t</b>", blurb="b", hue="sky")
    assert "<b>t</b>" not in frag
    assert "&lt;b&gt;t&lt;/b&gt;" in frag


def test_cap_card_renders_every_capability() -> None:
    # Every real capability produces a well-formed card with its assigned hue.
    for i, cap in enumerate(CAPABILITIES):
        frag = cap_card(icon=cap.icon, title=cap.title, blurb=cap.blurb, hue=cap_hue(i))
        assert frag.startswith('<div class="sa-cap"') and frag.endswith("</div>")
        assert cap.icon in frag
