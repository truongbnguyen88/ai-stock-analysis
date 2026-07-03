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


def test_dark_only_no_light_media() -> None:
    css = theme_style_tag()
    assert ":root {" in css
    assert "color-scheme: dark;" in css
    # Regression guard (render-check finding): our --sa-* tokens keyed off the OS
    # `prefers-color-scheme` while Streamlit's chrome is pinned dark by config.toml, so a
    # light-preference viewer got dark chrome + light tokens (white cards, dark-on-dark
    # text). The override must NOT be emitted; real light mode is R6 (Streamlit's signal).
    assert "prefers-color-scheme: light" not in css
    assert "color-scheme: light;" not in css


def test_color_tokens_defined_dark() -> None:
    # Every color token is defined once, in the dark :root block (dark-only emission).
    css = theme_style_tag()
    for name in COLOR_TOKENS:
        assert f"{name}:" in css, f"{name} missing from the dark token set"


def test_structural_tokens_declared_once() -> None:
    # Fonts/radii/spacing are theme-independent: declared once in the dark :root block.
    css = theme_style_tag()
    for name in STRUCTURAL_TOKENS:
        assert f"{name}:" in css, f"{name} missing"


def test_braces_balanced() -> None:
    css = theme_style_tag()
    assert css.count("{") == css.count("}")


def test_brass_accent_dark_only() -> None:
    # The signature accent is emitted in its dark form; the AA-safe light brass is retained
    # as data (_LIGHT / iframe_colors) for R6 but is NOT emitted in the dark-only style tag.
    css = theme_style_tag()
    assert "#E8A13A" in css  # dark accent emitted
    assert "#B7791F" not in css  # light accent not emitted (data-only)


def test_reduced_motion_and_responsive_rules_present() -> None:
    # R6 a11y/responsive contract: motion is opt-out (honors prefers-reduced-motion) and the
    # mono stat-tile value has a narrow-viewport guard so it stays legible if Streamlit keeps
    # columns side-by-side on a phone. Both are graceful enhancements, but their absence is a
    # regression, so pin the rules by presence (not by rendered layout, which we can't test here).
    css = theme_style_tag()
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (max-width: 640px)" in css
    assert "overflow-wrap: anywhere" in css  # tile value/label break rather than overflow


def test_font_stacks_are_system_only() -> None:
    # No webfont: CSP blocks font CDNs and a silent fallback would break the look.
    css = theme_style_tag()
    assert "ui-monospace" in css
    assert "@import" not in css
    assert "url(" not in css
    assert "googleapis" not in css


def test_iframe_colors_expose_palette_dark_emitted() -> None:
    # The typewriter iframe can't inherit --sa-* vars, so it embeds literal hex; this
    # accessor is the single source. It still exposes both palettes (light = R6 source),
    # but only the DARK values are the ones actually emitted / rendered (dark-only).
    colors = iframe_colors()
    keys = {"--sa-text", "--sa-muted", "--sa-accent"}
    assert set(colors["dark"]) == keys
    assert set(colors["light"]) == keys  # retained for R6 light mode
    css = theme_style_tag()
    for value in colors["dark"].values():
        assert value in css  # the exposed dark hex is a real token value used in the CSS
    # Brass accent differs across the two palettes (dark brass vs. AA-safe light brass).
    assert colors["dark"]["--sa-accent"] != colors["light"]["--sa-accent"]
