"""Token-bridge drift guard (Phase 2, §6): web/src/tokens.css == theme.web_tokens_css().

The Python ``--sa-*`` tokens are the single source of truth; the committed CSS is their
generated mirror for the React SPA. If someone edits a token in ``theme.py`` (or the CSS by
hand) without regenerating, this fails — the fix is ``python scripts/gen_web_tokens.py``.
Also pins the structural properties the bridge relies on (both theme blocks, full token
coverage, system-only fonts).
"""

from __future__ import annotations

from pathlib import Path

from stock_agent.ui.theme import COLOR_TOKENS, STRUCTURAL_TOKENS, web_tokens_css

# Repo-relative target — kept in lockstep with scripts/gen_web_tokens.py::TOKENS_PATH.
TOKENS_PATH = Path(__file__).resolve().parents[2] / "web" / "src" / "tokens.css"


def test_committed_css_matches_generator() -> None:
    assert TOKENS_PATH.exists(), "tokens.css missing — run: python scripts/gen_web_tokens.py"
    assert TOKENS_PATH.read_text(encoding="utf-8") == web_tokens_css(), (
        "web/src/tokens.css is stale — run: python scripts/gen_web_tokens.py"
    )


def test_emits_both_theme_blocks() -> None:
    css = web_tokens_css()
    assert ":root {" in css  # dark default
    assert ':root[data-theme="light"] {' in css  # light override (the real Phase-2 toggle)
    assert "color-scheme: dark;" in css
    assert "color-scheme: light;" in css


def test_every_color_token_present_in_both_themes() -> None:
    css = web_tokens_css()
    # Split at the light block so we can assert per-theme coverage (no token missing on one side).
    dark_block, _, light_block = css.partition(':root[data-theme="light"] {')
    for token in COLOR_TOKENS:
        assert f"{token}:" in dark_block, f"{token} missing from dark :root"
        assert f"{token}:" in light_block, f"{token} missing from light override"


def test_structural_tokens_dark_only() -> None:
    # Fonts/radii/spacing are theme-independent: declared once (dark :root), not re-emitted light.
    css = web_tokens_css()
    _, _, light_block = css.partition(':root[data-theme="light"] {')
    for token in STRUCTURAL_TOKENS:
        assert f"{token}:" in css
        assert f"{token}:" not in light_block, f"{token} must not be re-declared in light block"


def test_no_webfont_url_in_tokens() -> None:
    # System font stacks only (CSP + silent-fallback risk) — no @import / url() sneaking in.
    css = web_tokens_css()
    assert "url(" not in css
    assert "@import" not in css
