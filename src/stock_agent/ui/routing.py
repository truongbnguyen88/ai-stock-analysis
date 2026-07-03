"""Sidebar routing selection — pure data + placeholder copy (redesign R1).

No Streamlit, no I/O. The sidebar component resolves its controls into a
:class:`RoutingChoice`; the turn handler and input bar read it back. Distinct from
``stock_agent.agent.router`` (the actual dispatcher) — this only captures *what the user
selected* in the UI. Extracted verbatim from ``ui/chat_app.py`` (R1 refactor).
"""

from __future__ import annotations

from dataclasses import dataclass

# Sidebar label for the LLM-router mode (vs. a named deterministic domain).
AUTO_MODE = "🤖 Auto (LLM router)"


@dataclass(frozen=True)
class RoutingChoice:
    """The routing controls' resolved state for the current turn.

    ``domain is None`` ⇔ Auto mode: the LLM router picks and composes tools. A named
    ``domain`` runs that one capability deterministically (no routing LLM call). ``horizon``
    and ``days`` are surfaced only for the domains that use them (``None`` ⇒ tool default).
    """

    ticker: str
    domain: str | None = None
    variant: str | None = None
    horizon: int | None = None
    days: int | None = None

    @property
    def is_auto(self) -> bool:
        """True when the LLM router should choose tools (no deterministic domain picked)."""
        return self.domain is None


def chat_input_placeholder(choice: RoutingChoice) -> str:
    """Hint text for the chat input, tailored to the active routing mode.

    Mirrors the pre-R1 inline logic: a filings/theme domain asks for its specific input;
    any other named domain says it will run on the current ticker; Auto is open-ended.
    """
    if choice.domain == "filings":
        return "Type your filing question…"
    if choice.domain == "news" and choice.variant == "theme":
        return "Type a sector/theme (e.g. robotics)…"
    if choice.domain is not None:
        return f"Press enter to run {choice.domain} on {choice.ticker or 'a ticker'}…"
    return "Ask anything about a stock…"


def context_chips(choice: RoutingChoice) -> list[tuple[str, str]]:
    """Chips summarizing what pressing enter will do — ticker + mode (+ params) (R5).

    Pure: returns ``(label, tone)`` pairs the composer renders above the input ("on enter:
    NVDA · Auto"), surfacing the same routing state the placeholder describes. Tones are the
    allowed chip tones (``accent`` for the active ticker, ``neutral`` for the mode, ``muted``
    for the optional horizon/lookback params); a missing ticker is flagged rather than hidden.
    """
    chips: list[tuple[str, str]] = []
    ticker = choice.ticker.strip().upper() if choice.ticker else ""
    chips.append((ticker, "accent") if ticker else ("no ticker", "muted"))

    if choice.is_auto:
        chips.append(("Auto", "neutral"))
    else:
        label = choice.domain or "route"
        if choice.variant:
            label = f"{label} · {choice.variant}"
        chips.append((label, "neutral"))

    # Only the domains that use them set horizon/days (see RoutingChoice); surface when present.
    if choice.horizon is not None:
        chips.append((f"{choice.horizon}d horizon", "muted"))
    if choice.days is not None:
        chips.append((f"{choice.days}d lookback", "muted"))
    return chips
