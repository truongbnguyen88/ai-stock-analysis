"""Design-system theme tokens: CSS well-formedness + dark/light parity (no UI runtime).

Pure-string tests — the theme module has no Streamlit dependency, so these assert the
token contract the later redesign phases (R2+) rely on without rendering anything.
"""

from __future__ import annotations

from stock_agent.ui.theme import (
    COLOR_TOKENS,
    STRUCTURAL_TOKENS,
    iframe_colors,
    theme_style_tag,
)


def test_returns_wrapped_style_block() -> None:
    css = theme_style_tag()
    assert css.startswith("<style>")
    assert css.rstrip().endswith("</style>")
    assert len(css) > 500  # a real token block, not a stub


def test_pure_and_deterministic() -> None:
    # No hidden state / randomness: identical output every call (safe to inject per rerun).
    assert theme_style_tag() == theme_style_tag()


def test_both_theme_blocks_present() -> None:
    css = theme_style_tag()
    assert ":root {" in css
    assert "@media (prefers-color-scheme: light)" in css
    assert "color-scheme: dark;" in css
    assert "color-scheme: light;" in css


def test_color_tokens_defined_in_both_themes() -> None:
    # Every color token must be redefined under BOTH dark (:root) and light (media) —
    # a token missing from one theme would render with the wrong value there.
    css = theme_style_tag()
    for name in COLOR_TOKENS:
        assert css.count(f"{name}:") >= 2, f"{name} not defined in both themes"


def test_structural_tokens_declared_once() -> None:
    # Fonts/radii/spacing are theme-independent: declared once in the dark :root block.
    css = theme_style_tag()
    for name in STRUCTURAL_TOKENS:
        assert f"{name}:" in css, f"{name} missing"


def test_braces_balanced() -> None:
    css = theme_style_tag()
    assert css.count("{") == css.count("}")


def test_brass_accent_in_both_themes() -> None:
    # The signature accent: brass in dark, its darker AA-safe variant in light.
    css = theme_style_tag()
    assert "#E8A13A" in css  # dark accent
    assert "#B7791F" in css  # light accent


def test_font_stacks_are_system_only() -> None:
    # No webfont: CSP blocks font CDNs and a silent fallback would break the look.
    css = theme_style_tag()
    assert "ui-monospace" in css
    assert "@import" not in css
    assert "url(" not in css
    assert "googleapis" not in css


def test_iframe_colors_both_themes_and_synced_to_tokens() -> None:
    # The typewriter iframe can't inherit --sa-* vars, so it embeds literal hex; this
    # accessor is the single source. Both themes carry the same 3 keys, and the values
    # must equal the real token hex (no drift) — check via presence in the CSS block.
    colors = iframe_colors()
    keys = {"--sa-text", "--sa-muted", "--sa-accent"}
    assert set(colors["dark"]) == keys
    assert set(colors["light"]) == keys
    css = theme_style_tag()
    for theme in ("dark", "light"):
        for value in colors[theme].values():
            assert value in css  # the exposed hex is a real token value used in the CSS
    # Brass accent differs across themes (dark brass vs. AA-safe light brass).
    assert colors["dark"]["--sa-accent"] != colors["light"]["--sa-accent"]
