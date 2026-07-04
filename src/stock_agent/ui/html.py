"""Pure HTML component builders for the Streamlit sidebar/redesign (R2+).

Pure, typed, import-safe (no Streamlit, no I/O) — the repo-root view modules inject the
returned strings via ``st.markdown(..., unsafe_allow_html=True)``. Every builder emits a
small fragment that references the ``--sa-*`` design tokens (defined in
``stock_agent.ui.theme``) through CSS classes, so color/spacing stay in one source of
truth and the fragments degrade gracefully to unstyled-but-legible if the class block
ever goes stale (design principle #6).

Split rationale (docs/APP_REDESIGN.md §4): pure builders live under ``src`` so they are
strict-typed and unit-tested; the Streamlit-coupled composition lives in ``ui/``. All
interpolated text is HTML-escaped defensively even though inputs are internal.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from html import escape as _esc
from zlib import crc32

# Freshness states for the corpus status dot. "fresh" = corpus has a recent filing;
# "stale" = corpus loaded but its latest filing is older than the quarterly window;
# "unavailable" = the vector store can't be opened or the corpus is empty.
_DOT_STATES: tuple[str, ...] = ("fresh", "stale", "unavailable")

# Allowed chip tones -> the CSS modifier class suffix (styled in theme._COMPONENTS).
_CHIP_TONES: frozenset[str] = frozenset({"ok", "accent", "off", "muted", "neutral"})

# Semantic secondary-hue cycle for capability cards (one hue per card, cycled by index).
# Order matches the token family in theme.py (§5.2); "accent" (brass) is the fallback.
_CAP_HUES: tuple[str, ...] = ("teal", "sky", "indigo", "violet", "rose")
_HUE_TOKENS: frozenset[str] = frozenset(_CAP_HUES) | {"accent"}


def _hue_token(tone: str) -> str:
    """Resolve a caller-supplied tone to a valid ``--sa-*`` hue name (accent fallback)."""
    return tone if tone in _HUE_TOKENS else "accent"


def corpus_freshness(
    latest: str | None,
    chunks: int,
    *,
    today: date,
    stale_after_days: int = 120,
) -> str:
    """Derive the status-dot state from corpus coverage — pure (``today`` injected).

    ``latest`` is the most recent filing_date (ISO ``YYYY-MM-DD``) or ``None``; ``chunks``
    is the vector count (``< 0`` when the store can't be opened, per ``corpus_status``).
    Returns one of :data:`_DOT_STATES`. Threshold defaults to 120 days: SEC 10-Q cadence
    is ~90 days, so a corpus with no filing in ~4 months is missing a recent quarter.
    """
    if chunks < 0 or not latest:
        return "unavailable"
    try:
        latest_date = date.fromisoformat(latest)
    except ValueError:
        return "unavailable"
    age_days = (today - latest_date).days
    return "fresh" if age_days <= stale_after_days else "stale"


def status_dot(state: str) -> str:
    """Return a small colored status dot ``<span>`` for a freshness ``state``.

    Unknown states fall back to the neutral ``unavailable`` styling rather than raising,
    so a future state name can't blank the sidebar.
    """
    cls = state if state in _DOT_STATES else "unavailable"
    return f'<span class="sa-dot sa-dot--{cls}" aria-hidden="true"></span>'


def eyebrow(text: str, *, tone: str = "muted") -> str:
    """Uppercase mono section label — the redesign's structural section marker.

    ``tone="accent"`` renders it in brass (used for the empty-state hero eyebrow);
    the default muted tone is used for sidebar section labels.
    """
    mod = " sa-eyebrow--accent" if tone == "accent" else ""
    return f'<div class="sa-eyebrow{mod}">{_esc(text)}</div>'


def cap_hue(index: int) -> str:
    """Return the semantic hue name for capability-card ``index`` (cycles the 5 hues)."""
    return _CAP_HUES[index % len(_CAP_HUES)]


def cap_card(*, icon: str, title: str, blurb: str, hue: str) -> str:
    """Capability card: hue rail + tinted icon badge + title + blurb (visual only).

    ``hue`` is a token name (``teal|sky|indigo|violet|rose|accent``) set on the element
    as ``--cap-hue`` so the rail/badge tint derive from one variable; unknown hues fall
    back to the brass accent. The *action* is a sibling Streamlit button in the view
    (injected HTML can't fire a callback), so this fragment is presentation-only.
    """
    hue_name = _hue_token(hue)
    return (
        f'<div class="sa-cap" style="--cap-hue: var(--sa-{hue_name});">'
        f'<div class="sa-cap__badge">{_esc(icon)}</div>'
        f'<div class="sa-cap__title">{_esc(title)}</div>'
        f'<div class="sa-cap__blurb">{_esc(blurb)}</div>'
        "</div>"
    )


def brand_block(*, mark: str, name: str, tagline: str) -> str:
    """Product mark + name + tagline for the sidebar header."""
    return (
        '<div class="sa-brand">'
        f'<span class="sa-brand__mark">{_esc(mark)}</span>'
        '<span class="sa-brand__text">'
        f'<span class="sa-brand__name">{_esc(name)}</span>'
        f'<span class="sa-brand__tag">{_esc(tagline)}</span>'
        "</span>"
        "</div>"
    )


def status_card(
    *,
    state: str,
    embedder: str,
    chunks: int,
    tickers: int,
    latest: str | None,
) -> str:
    """Corpus status card: dot + which embedder answers filing questions + freshness.

    Content mirrors the pre-redesign captions exactly (embedder namespace, chunk count,
    ticker count, latest filing date) so no information is lost — only the presentation
    changes. ``chunks < 0`` renders as "unavailable" to match ``CorpusStatus.one_line``.
    """
    chunk_txt = "unavailable" if chunks < 0 else f"{chunks:,} chunks"
    coverage = (
        f"{tickers} tickers · fresh to {_esc(latest)}"
        if latest
        else "no filings ingested yet"
    )
    return (
        '<div class="sa-card sa-status">'
        f"{status_dot(state)}"
        '<div class="sa-status__body">'
        '<span class="sa-status__label">Filing search</span> '
        f'<span class="sa-status__embedder">{_esc(embedder)}</span>'
        f"<br>{_esc(chunk_txt)}"
        f'<br><span class="sa-status__cov">{coverage}</span>'
        "</div>"
        "</div>"
    )


def chip(text: str, *, tone: str = "neutral", marker: str = "") -> str:
    """Inline pill in one of the allowed :data:`_CHIP_TONES`.

    ``marker`` is an optional short status glyph (e.g. ``✓``) shown before the label.
    Unknown tones fall back to ``neutral`` (no raise) so a typo can't break the row.
    """
    tone_cls = tone if tone in _CHIP_TONES else "neutral"
    mk = f'<span class="sa-chip__mk">{_esc(marker)}</span>' if marker else ""
    return f'<span class="sa-chip sa-chip--{tone_cls}">{mk}{_esc(text)}</span>'


def stat_tile(
    *, label: str, value: str, sub: str = "", tone: str = "accent", direction: str | None = None
) -> str:
    """Metric tile: colored category stripe + mono uppercase label + big value + sub (R4).

    ``value``/``sub`` are already-formatted display strings (the number formatting lives in
    ``stock_agent.ui.tiles`` so this stays presentation-only). ``tone`` is a semantic hue
    token (``teal|sky|indigo|violet|rose|accent``) driving the left stripe via ``--tile-hue``;
    unknown tones fall back to brass. The optional ``direction`` (``up``/``down``) tints the
    *value* green/red — supplied by ``ui.tiles`` from a deterministic sign read of the tool
    number (never the LLM), so the §2 signaling rule holds (the color is the figure, not the
    narrative); absent → neutral. Presentation only: numbers come from tool results upstream.
    """
    hue_name = _hue_token(tone)
    val_mod = f" sa-tile__value--{direction}" if direction in ("up", "down") else ""
    sub_html = f'<div class="sa-tile__sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="sa-tile" style="--tile-hue: var(--sa-{hue_name});">'
        f'<div class="sa-tile__label">{_esc(label)}</div>'
        f'<div class="sa-tile__value{val_mod}">{_esc(value)}</div>'
        f"{sub_html}"
        "</div>"
    )


def tool_hue(name: str) -> str:
    """Deterministic, stable hue for a tool ``name`` in the trace chip row.

    Hashing on the name (not call order) keeps a given tool the same color across turns.
    Uses ``crc32`` (deterministic across processes — unlike salted ``hash()``) mod the hue
    count, so it is unit-testable and spreads similar tool names across the hue family.
    """
    bucket = crc32(name.encode("utf-8")) % len(_CAP_HUES)
    return _CAP_HUES[bucket]


def trace_bar(tools: Sequence[str]) -> str:
    """Per-tool colored chip row for an Auto turn's tool trace (empty -> empty string).

    Each distinct tool becomes a mono chip tinted by :func:`tool_hue`. Order is preserved;
    the caller is expected to have deduped (``dict.fromkeys``) already, but a defensive
    dedup here keeps the row clean regardless.
    """
    seen: set[str] = set()
    chips = []
    for name in tools:
        if name in seen:
            continue
        seen.add(name)
        hue = tool_hue(name)
        chips.append(
            f'<span class="sa-tchip" style="--tc-hue: var(--sa-{hue});">{_esc(name)}</span>'
        )
    if not chips:
        return ""
    return '<div class="sa-trace">' + "".join(chips) + "</div>"


def context_row(chips: Sequence[tuple[str, str]]) -> str:
    """Composer context-chip row from ``(label, tone)`` pairs — 'on enter' lead + pills (R5).

    Renders what pressing enter will do (from :func:`stock_agent.ui.routing.context_chips`).
    Reuses :func:`chip` for each pill (tone falls back to ``neutral`` on a typo). Empty input
    yields an empty string (no stray lead label).
    """
    if not chips:
        return ""
    pills = "".join(chip(text, tone=tone) for text, tone in chips)
    return (
        '<div class="sa-context">'
        '<span class="sa-context__lead">on enter</span>'
        f"{pills}"
        "</div>"
    )


def keys_row(keys: Sequence[tuple[str, bool, bool]]) -> str:
    """Render API-key status as a chip row from ``(label, present, required)`` tuples.

    Tone/marker encode state: present → ``ok`` + ✓; missing & required → ``off`` + ✕;
    missing & optional → ``muted`` + ·. Preserves the exact key set the caption listed.
    """
    chips = []
    for label, present, required in keys:
        if present:
            chips.append(chip(label, tone="ok", marker="✓"))
        elif required:
            chips.append(chip(f"{label} (required)", tone="off", marker="✕"))
        else:
            chips.append(chip(label, tone="muted", marker="·"))
    return '<div class="sa-chips">' + "".join(chips) + "</div>"
