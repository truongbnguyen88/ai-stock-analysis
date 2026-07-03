"""Design-system tokens + base styling for the Streamlit frontend (redesign R0).

Pure, typed, import-safe (no Streamlit, no I/O) — mirrors ``stock_agent.ui.capabilities``
so it is type-checked and unit-tested like the rest of the package. The repo-root
``ui/chat_app.py`` injects the returned ``<style>`` block once at app top via
``st.markdown(theme_style_tag(), unsafe_allow_html=True)``.

Design direction — "brass on ink" (see docs/APP_REDESIGN.md §2/§5): a cool near-black
slate ground with one restrained brass/amber primary accent and a secondary hue family
used *semantically*. Dark is the primary treatment (matches ``.streamlit/config.toml``
``base = "dark"``); a ``prefers-color-scheme: light`` block provides a coherent light
palette for users whose OS/Streamlit theme is light.

Separation of concerns:
- ``.streamlit/config.toml`` owns Streamlit's *native* chrome colors (background, text,
  primary, secondary background) — the robust, version-stable layer.
- This module owns the ``--sa-*`` *design tokens* our own components consume in later
  phases (R2+), plus base typography. Any rule here that targets Streamlit's internal
  DOM is intentionally conservative and **degrades gracefully**: if a selector goes
  stale on a Streamlit upgrade, the app still works (config.toml colors still apply).
"""

from __future__ import annotations

# --- design tokens (dark = primary) -------------------------------------------------
# Kept as data so tests can assert completeness without parsing CSS. Values are lifted
# verbatim from the approved mockup (docs/APP_REDESIGN.md §5.2).
_DARK: dict[str, str] = {
    "--sa-bg": "#0E1116",
    "--sa-surface": "#161B22",
    "--sa-surface-2": "#1C222B",
    "--sa-surface-3": "#222A35",
    "--sa-border": "#262D38",
    "--sa-border-strong": "#323B48",
    "--sa-text": "#E6EAF0",
    "--sa-muted": "#8A93A3",
    "--sa-faint": "#616B7A",
    "--sa-accent": "#E8A13A",
    "--sa-accent-weak": "rgba(232,161,58,0.12)",
    "--sa-accent-line": "rgba(232,161,58,0.35)",
    "--sa-teal": "#35B0A7",
    "--sa-sky": "#4FA8E8",
    "--sa-indigo": "#7C82F0",
    "--sa-violet": "#B38BEA",
    "--sa-rose": "#E5709B",
    "--sa-up": "#3FB950",
    "--sa-down": "#E5534B",
    "--sa-grid": "rgba(138,147,163,0.14)",
    "--sa-shadow": "0 1px 2px rgba(0,0,0,0.40), 0 8px 24px rgba(0,0,0,0.28)",
}

# Light overrides — only the tokens whose value changes between themes. Structural
# tokens (fonts, radii, spacing) are theme-independent and declared once in _BASE.
_LIGHT: dict[str, str] = {
    "--sa-bg": "#F5F6F8",
    "--sa-surface": "#FFFFFF",
    "--sa-surface-2": "#FBFBFC",
    "--sa-surface-3": "#F1F3F6",
    "--sa-border": "#E4E7EC",
    "--sa-border-strong": "#D3D8DF",
    "--sa-text": "#1A1F27",
    "--sa-muted": "#5B6472",
    "--sa-faint": "#8A93A3",
    "--sa-accent": "#B7791F",
    "--sa-accent-weak": "rgba(183,121,31,0.10)",
    "--sa-accent-line": "rgba(183,121,31,0.40)",
    "--sa-teal": "#0E9384",
    "--sa-sky": "#2E7FC4",
    "--sa-indigo": "#5B60D6",
    "--sa-violet": "#8B54C9",
    "--sa-rose": "#C6417A",
    "--sa-up": "#1A7F37",
    "--sa-down": "#C4392F",
    "--sa-grid": "rgba(91,100,114,0.14)",
    "--sa-shadow": "0 1px 2px rgba(16,22,30,0.06), 0 10px 30px rgba(16,22,30,0.08)",
}

# Theme-independent structural tokens (declared once, in the dark :root block).
# No webfont: Artifact/Streamlit CSP + silent-fallback risk -> system stacks only.
_STRUCTURAL: dict[str, str] = {
    "--sa-font-sans": '-apple-system, "Segoe UI", Inter, Roboto, system-ui, sans-serif',
    "--sa-font-mono": 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace',
    "--sa-r": "10px",
    "--sa-r-sm": "7px",
    "--sa-space-1": "4px",
    "--sa-space-2": "8px",
    "--sa-space-3": "12px",
    "--sa-space-4": "16px",
    "--sa-space-5": "24px",
    "--sa-space-6": "32px",
}

# Full set of token names every theme block that redefines colors must provide. Used by
# tests to assert dark/light parity (no token silently missing from one theme).
COLOR_TOKENS: tuple[str, ...] = tuple(_DARK)
STRUCTURAL_TOKENS: tuple[str, ...] = tuple(_STRUCTURAL)


def _render_vars(tokens: dict[str, str], *, indent: str) -> str:
    """Render ``name: value;`` declarations, one per line, in declaration order (stable)."""
    return "\n".join(f"{indent}{name}: {value};" for name, value in tokens.items())


# Conservative base styling. Rules that reference Streamlit's internal DOM are grouped
# here and commented; they are enhancements, not load-bearing (graceful degradation).
_BASE = """
/* Prose in the app's sans stack; code/monospace surfaces in the mono stack. */
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: var(--sa-font-sans);
}
[data-testid="stAppViewContainer"] code,
[data-testid="stAppViewContainer"] pre,
[data-testid="stAppViewContainer"] kbd { font-family: var(--sa-font-mono); }

/* Headings: slightly tighter tracking + balanced wrapping for calmer hierarchy. */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
  letter-spacing: -0.01em; text-wrap: balance;
}

/* Captions read as secondary text. */
[data-testid="stCaptionContainer"] { color: var(--sa-muted); }
"""


def theme_style_tag() -> str:
    """Return the full ``<style>…</style>`` block to inject once at the top of the app.

    Pure and deterministic (same output every call). Emits the dark ``:root`` token set
    (primary treatment) + structural tokens, a ``prefers-color-scheme: light`` override
    that redefines every color token for parity, and conservative base typography.
    """
    dark_vars = _render_vars({**_DARK, **_STRUCTURAL}, indent="  ")
    light_vars = _render_vars(_LIGHT, indent="    ")
    css = (
        ":root {\n"
        "  color-scheme: dark;\n"
        f"{dark_vars}\n"
        "}\n"
        "@media (prefers-color-scheme: light) {\n"
        "  :root {\n"
        "    color-scheme: light;\n"
        f"{light_vars}\n"
        "  }\n"
        "}\n"
        f"{_BASE}"
    )
    return f"<style>\n{css}\n</style>"
