"""Design-system tokens + base styling for the Streamlit frontend (redesign R0).

Pure, typed, import-safe (no Streamlit, no I/O) — mirrors ``stock_agent.ui.capabilities``
so it is type-checked and unit-tested like the rest of the package. The repo-root
``ui/chat_app.py`` injects the returned ``<style>`` block once at app top via
``st.markdown(theme_style_tag(), unsafe_allow_html=True)``.

Design direction — "brass on ink" (see docs/APP_REDESIGN.md §2/§5): a cool near-black
slate ground with one restrained brass/amber primary accent and a secondary hue family
used *semantically*. **Dark-only**, matching the ``.streamlit/config.toml`` ``base = "dark"``
pin (an OS ``prefers-color-scheme`` swap would desync from Streamlit's config-pinned chrome —
see :func:`theme_style_tag`). First-class light mode is R6 (``_LIGHT`` is retained for it).

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

# Light palette — RETAINED AS DATA, not currently emitted (see theme_style_tag). It is the
# single source for the proper R6 light mode (which must switch on Streamlit's own theme
# signal, not the OS preference). Only the tokens whose value changes between themes; the
# structural tokens (fonts, radii, spacing) are theme-independent (declared once in _BASE).
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


# Tokens the empty-state typewriter iframe needs as literal hex. A sandboxed
# ``components.html`` iframe does NOT inherit the app's ``--sa-*`` custom properties, so
# the hero must embed real hex; sourcing them here keeps a single source of truth (no
# re-hardcoded colors drifting from the token set). See ui/components/hero.py.
_IFRAME_KEYS: tuple[str, ...] = ("--sa-text", "--sa-muted", "--sa-accent")


def dark_token(name: str) -> str:
    """Return the dark-theme value for a ``--sa-*`` token (single source for chart colors).

    Used by ``stock_agent.ui.chart_theme`` (R5) so the Altair palette derives from the same
    token hexes as the CSS — no drifting hardcoded chart colors. Raises ``KeyError`` on an
    unknown token (a typo should fail loudly, not silently pick a wrong color).
    """
    return _DARK[name]


def mono_font_stack() -> str:
    """Return the monospace font stack (single source for CSS + the Altair chart theme, R5)."""
    return _STRUCTURAL["--sa-font-mono"]


def iframe_colors() -> dict[str, dict[str, str]]:
    """Return ``{"dark": {token: hex}, "light": {token: hex}}`` for the typewriter iframe.

    Pure. Only the subset in :data:`_IFRAME_KEYS` (text / muted / brass accent). The hero
    currently renders the iframe **dark-only** (consistent with the config-pinned dark
    chrome); the ``light`` half is retained as the R6 light-mode source (same rationale as
    :func:`theme_style_tag`).
    """
    return {
        "dark": {k: _DARK[k] for k in _IFRAME_KEYS},
        "light": {k: _LIGHT[k] for k in _IFRAME_KEYS},
    }


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

# Redesign component styling (R2+). Classes consumed by ``stock_agent.ui.html`` builders,
# injected via ``st.markdown(unsafe_allow_html=True)``. All rules reference --sa-* tokens
# so the two themes stay in sync; if a class ever goes unstyled the fragment still reads
# (graceful degradation, principle #6). No ``url(...)`` here (keeps the webfont test green).
_COMPONENTS = """
/* Section eyebrow: the redesign's mono uppercase structural marker (replaces headings). */
.sa-eyebrow {
  font-family: var(--sa-font-mono);
  font-size: 11px; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--sa-muted); margin: var(--sa-space-4) 0 var(--sa-space-2);
}
/* Brand block: product mark + name (mono) + faint tagline. */
.sa-brand { display: flex; align-items: center; gap: var(--sa-space-3);
  padding: var(--sa-space-2) 0 var(--sa-space-3); }
.sa-brand__mark { font-size: 22px; line-height: 1; }
.sa-brand__text { display: flex; flex-direction: column; }
.sa-brand__name { font-family: var(--sa-font-mono); font-weight: 600;
  font-size: 15px; letter-spacing: -0.01em; color: var(--sa-text); }
.sa-brand__tag { font-size: 11px; color: var(--sa-faint); }
/* Surface card + corpus status card. */
.sa-card { background: var(--sa-surface-2); border: 1px solid var(--sa-border);
  border-radius: var(--sa-r-sm); padding: var(--sa-space-3); box-shadow: var(--sa-shadow); }
.sa-status { display: flex; align-items: flex-start; gap: var(--sa-space-2); }
.sa-status__body { font-size: 12px; color: var(--sa-muted); line-height: 1.5; }
.sa-status__label { font-family: var(--sa-font-mono); text-transform: uppercase;
  letter-spacing: 0.06em; font-size: 10px; color: var(--sa-faint); }
.sa-status__embedder { font-family: var(--sa-font-mono); color: var(--sa-text); font-size: 12px; }
.sa-status__cov { color: var(--sa-faint); }
/* Status dot: green fresh / brass stale / red unavailable, each with a soft halo. */
.sa-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px;
  flex: 0 0 auto; background: var(--sa-faint); }
.sa-dot--fresh { background: var(--sa-up); box-shadow: 0 0 0 3px rgba(63,185,80,0.16); }
.sa-dot--stale { background: var(--sa-accent); box-shadow: 0 0 0 3px var(--sa-accent-weak); }
.sa-dot--unavailable { background: var(--sa-down); box-shadow: 0 0 0 3px rgba(229,83,75,0.16); }
/* Chip row (keys, tags): mono pills; tone carries meaning, marker gives a redundant glyph. */
.sa-chips { display: flex; flex-wrap: wrap; gap: var(--sa-space-2); margin-top: var(--sa-space-1); }
.sa-chip { display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--sa-font-mono); font-size: 11px; line-height: 1;
  padding: 4px 8px; border-radius: 999px; border: 1px solid var(--sa-border-strong);
  color: var(--sa-muted); background: var(--sa-surface-3); }
.sa-chip__mk { font-weight: 700; }
.sa-chip--ok { color: var(--sa-teal); border-color: var(--sa-teal); }
.sa-chip--accent { color: var(--sa-accent); border-color: var(--sa-accent-line); }
.sa-chip--off { color: var(--sa-down); border-color: var(--sa-down); }
.sa-chip--muted { color: var(--sa-faint); }
/* Hero eyebrow variant: brass instead of muted (empty-state section label). */
.sa-eyebrow--accent { color: var(--sa-accent); }
/* Capability card (empty-state hero): hue top-rail + tinted icon badge; hover lift.
   --cap-hue is set inline per card; children derive their color from it. */
.sa-cap {
  border: 1px solid var(--sa-border); border-top: 2px solid var(--cap-hue);
  border-radius: var(--sa-r); background: var(--sa-surface);
  padding: var(--sa-space-4); min-height: 118px;
  transition: transform 0.08s ease, box-shadow 0.08s ease, border-color 0.08s ease;
}
.sa-cap:hover { transform: translateY(-2px); box-shadow: var(--sa-shadow);
  border-color: var(--sa-border-strong); }
.sa-cap__badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 34px; height: 34px; border-radius: var(--sa-r-sm); font-size: 18px;
  margin-bottom: var(--sa-space-2);
  background: color-mix(in srgb, var(--cap-hue) 16%, transparent);
}
.sa-cap__title { font-weight: 600; font-size: 15px; color: var(--sa-text); margin-bottom: 2px; }
.sa-cap__blurb { font-size: 12px; color: var(--sa-muted); line-height: 1.45; }
@media (prefers-reduced-motion: reduce) {
  .sa-cap { transition: none; }
  .sa-cap:hover { transform: none; }
}
/* Subtle hover nudge on sidebar buttons (enhancement; native styling if selector stales). */
[data-testid="stSidebar"] .stButton > button { transition: transform 0.06s ease; }
[data-testid="stSidebar"] .stButton > button:hover { transform: translateX(2px); }
/* Stat tile (chat message header): colored category stripe + mono label + big value (R4).
   --tile-hue is set inline per tile; the left stripe + label derive their color from it. */
.sa-tile {
  border: 1px solid var(--sa-border); border-left: 3px solid var(--tile-hue);
  border-radius: var(--sa-r-sm); background: var(--sa-surface-2);
  padding: var(--sa-space-3) var(--sa-space-4); min-height: 74px;
}
.sa-tile__label {
  font-family: var(--sa-font-mono); font-size: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--tile-hue);
}
.sa-tile__value {
  font-family: var(--sa-font-mono); font-variant-numeric: tabular-nums;
  font-size: 22px; font-weight: 600; color: var(--sa-text); line-height: 1.25;
}
.sa-tile__sub { font-size: 11px; color: var(--sa-faint); }
/* Narrow viewports (R6 responsive): Streamlit owns the column *count* (st.columns reflow) —
   we don't force-stack via a column-container selector (that would stale on a version bump, §10).
   We only guard the mono tile value from overflowing if 3 tiles stay side-by-side on a phone:
   shrink the value + let long tokens break. Pure enhancement; no fragile Streamlit selector. */
@media (max-width: 640px) {
  .sa-tile { min-height: 64px; padding: var(--sa-space-2) var(--sa-space-3); }
  .sa-tile__value { font-size: 18px; overflow-wrap: anywhere; }
  .sa-tile__label { overflow-wrap: anywhere; }
}
/* Tool-trace chip row (Auto turn): one mono pill per tool, tinted by --tc-hue. */
.sa-trace { display: flex; flex-wrap: wrap; gap: var(--sa-space-2); margin: var(--sa-space-2) 0; }
.sa-tchip {
  display: inline-flex; align-items: center; font-family: var(--sa-font-mono);
  font-size: 11px; line-height: 1; padding: 4px 9px; border-radius: 999px;
  color: var(--tc-hue); border: 1px solid var(--tc-hue);
  background: color-mix(in srgb, var(--tc-hue) 12%, transparent);
}
/* Filing source row: mono marker badge + citation label (R4). */
.sa-src { display: flex; align-items: baseline; gap: var(--sa-space-2);
  margin-bottom: var(--sa-space-1); }
.sa-src__label { font-size: 13px; color: var(--sa-muted); }
/* Composer context chips (R5): 'on enter' lead + ticker/mode pills, above the input. */
.sa-context { display: flex; flex-wrap: wrap; align-items: center; gap: var(--sa-space-2);
  margin: var(--sa-space-2) 0 var(--sa-space-1); }
.sa-context__lead { font-family: var(--sa-font-mono); font-size: 10px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--sa-faint); }
/* Composer focus-within brass accent (R5). Enhancement targeting Streamlit's chat-input
   container; if the selector stales on an upgrade the input still works (graceful #6). */
[data-testid="stChatInput"] { transition: border-color 0.12s ease, box-shadow 0.12s ease; }
[data-testid="stChatInput"]:focus-within {
  border-color: var(--sa-accent-line);
  box-shadow: 0 0 0 2px var(--sa-accent-weak);
}
"""


def theme_style_tag() -> str:
    """Return the full ``<style>…</style>`` block to inject once at the top of the app.

    Pure and deterministic. Emits the **dark** ``:root`` token set + structural tokens and
    conservative base typography — dark-only, matching the ``base = "dark"`` pin in
    ``.streamlit/config.toml``.

    Deliberately does **not** emit a ``prefers-color-scheme: light`` override. Our ``--sa-*``
    tokens would switch on the *OS/browser* preference while Streamlit's own chrome is pinned
    by config, so a light-preference viewer got dark chrome + light component tokens (white
    cards on the dark ground, dark-on-dark brand text). Real light mode must switch on the
    *same* signal Streamlit uses — deferred to R6 (see docs/APP_REDESIGN.md); ``_LIGHT`` is
    retained as that work's single source.
    """
    dark_vars = _render_vars({**_DARK, **_STRUCTURAL}, indent="  ")
    css = (
        ":root {\n"
        "  color-scheme: dark;\n"
        f"{dark_vars}\n"
        "}\n"
        f"{_BASE}"
        f"{_COMPONENTS}"
    )
    return f"<style>\n{css}\n</style>"
