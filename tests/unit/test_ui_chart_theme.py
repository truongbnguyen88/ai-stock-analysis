"""Shared Altair chart theme: config shape + palette derived from tokens (pure, R5)."""

from __future__ import annotations

import re

from stock_agent.ui.chart_theme import (
    CATEGORY_PALETTE,
    MARK_COLOR,
    POINT_COLOR,
    REFERENCE_COLOR,
    altair_config,
)
from stock_agent.ui.theme import dark_token, mono_font_stack

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_RGBA = re.compile(r"^rgba\(\d+,\s*\d+,\s*\d+,\s*[0-9.]+\)$")


def test_config_pure_and_deterministic() -> None:
    # No hidden state: same dict every call (safe to apply per render).
    assert altair_config() == altair_config()


def test_config_has_expected_sections() -> None:
    cfg = altair_config()
    assert set(cfg) == {"view", "axis", "legend", "title", "range"}
    # Faint grid on, view border dropped, categorical range carried for export parity.
    assert cfg["axis"]["grid"] is True
    assert cfg["view"]["stroke"] is None
    assert cfg["range"]["category"] == list(CATEGORY_PALETTE)


def test_fonts_are_mono_no_webfont() -> None:
    cfg = altair_config()
    mono = mono_font_stack()
    assert cfg["axis"]["labelFont"] == mono
    assert cfg["axis"]["titleFont"] == mono
    assert cfg["title"]["font"] == mono
    assert cfg["legend"]["labelFont"] == mono
    # Mono stack, no CDN webfont (CSP would block it, silent fallback would break the look).
    assert "monospace" in mono
    assert "url(" not in mono and "@import" not in mono


def test_text_colors_left_unset_for_adaptive_theme() -> None:
    # Label/title COLORS are intentionally omitted so Streamlit's adaptive theme (in-app)
    # and Vega's default (export) supply AA-contrast text per background.
    cfg = altair_config()
    assert "labelColor" not in cfg["axis"]
    assert "titleColor" not in cfg["axis"]


def test_palette_derives_from_dark_tokens_no_drift() -> None:
    # Single source of truth: every chart color is a real design token (§5.2), so the
    # palette can never drift from the CSS. Mirrors the iframe-colors drift guard.
    assert list(CATEGORY_PALETTE) == [
        dark_token(t) for t in ("--sa-sky", "--sa-rose", "--sa-indigo", "--sa-teal", "--sa-violet")
    ]
    assert dark_token("--sa-accent") == MARK_COLOR
    assert dark_token("--sa-sky") == POINT_COLOR


def test_palette_wellformed_and_distinct() -> None:
    assert len(CATEGORY_PALETTE) == 5
    assert len(set(CATEGORY_PALETTE)) == 5  # no duplicate hues
    for hexval in (*CATEGORY_PALETTE, MARK_COLOR, POINT_COLOR):
        assert _HEX.match(hexval), hexval
    assert _RGBA.match(REFERENCE_COLOR)


def test_no_red_green_pair_in_first_two_series() -> None:
    # Colorblind/signaling rule: the first two grouped series (the common case) must not be
    # the chart up/down green-red pair — those are reserved for direction marks (§2).
    up, down = dark_token("--sa-up"), dark_token("--sa-down")
    assert CATEGORY_PALETTE[0] not in (up, down)
    assert CATEGORY_PALETTE[1] not in (up, down)
