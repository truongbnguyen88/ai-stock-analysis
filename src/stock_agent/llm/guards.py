"""LLM output schema and the invariant guards (Role A).

The summarizer's job is qualitative interpretation only. These guards enforce the
numbers-vs-narrative invariant at the output boundary:

- ``find_forecast_violations``: rejects probabilistic/forecast language the LLM
  must never produce (probabilities, odds, price targets, numeric forecasts).
  Patterns require a forecast keyword, so reported facts like "shares fell 5%"
  are NOT flagged — only fabricated forward-looking numbers.
- citation guards: every cited URL must come from the provided article set;
  fabricated URLs are detected and stripped (hallucination control).
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from stock_agent.news.clean import canonical_url


class CitedPoint(BaseModel):
    """A qualitative point with the article URLs that support it."""

    point: str
    sources: list[str] = Field(default_factory=list)


class NewsSummary(BaseModel):
    """Structured qualitative news synthesis produced by the LLM (Role A).

    Contains NO probabilities or forecasts by design (enforced by guards).
    """

    overview: str = ""
    key_themes: list[str] = Field(default_factory=list)
    bullish: list[CitedPoint] = Field(default_factory=list)
    bearish: list[CitedPoint] = Field(default_factory=list)
    risks: list[CitedPoint] = Field(default_factory=list)
    catalysts: list[CitedPoint] = Field(default_factory=list)


# Patterns that indicate LLM-invented probability/forecast language.
# Each requires an explicit forward-looking cue so legitimately *reported* facts
# (e.g. "revenue up 20%", "analyst raised price target to $300") are NOT flagged.
#
# Key distinction: "Truist raised its price target to $300" is a reported news fact.
# "My price target is $300" or "the stock will reach $300" is an LLM-invented forecast.
_FORECAST_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Explicit probability/odds language.
    re.compile(r"\b\d+(\.\d+)?\s*%?\s*(chance|probability|odds|likelihood)\b", re.I),
    re.compile(r"\b(probability|chance|odds|likelihood)\s+(of|that)\b", re.I),
    # LLM asserting its own price target (not reporting an analyst's).
    # "my/our/I set a price target" but NOT "analyst raised price target".
    re.compile(r"\b(my|our|I\s+set|we\s+set)\s+(a\s+)?price\s+target\b", re.I),
    # Numeric return/price forecasts explicitly attributed to the model's own view.
    re.compile(
        r"\b(expected|projected|forecast(ed)?|predicted)\s+(return|price|gain|loss)\b", re.I
    ),
    re.compile(r"\b\d+(\.\d+)?\s*%\s*(upside|downside)\s+(potential|target|from here)\b", re.I),
    re.compile(r"\b(will|expected to|likely to)\s+\w+\s+(by\s+)?\d+(\.\d+)?\s*%", re.I),
)


def _iter_text(summary: NewsSummary) -> list[str]:
    """All free-text fragments in a summary (for scanning)."""
    fragments: list[str] = [summary.overview, *summary.key_themes]
    for group in (summary.bullish, summary.bearish, summary.risks, summary.catalysts):
        fragments.extend(point.point for point in group)
    return [f for f in fragments if f]


def find_forecast_violations(summary: NewsSummary) -> list[str]:
    """Return matched snippets that look like forecasts/probabilities (empty == clean)."""
    violations: list[str] = []
    for fragment in _iter_text(summary):
        for pattern in _FORECAST_PATTERNS:
            match = pattern.search(fragment)
            if match:
                violations.append(fragment.strip())
                break
    return violations


def _all_cited(summary: NewsSummary) -> list[str]:
    urls: list[str] = []
    for group in (summary.bullish, summary.bearish, summary.risks, summary.catalysts):
        for point in group:
            urls.extend(point.sources)
    return urls


def find_invalid_citations(summary: NewsSummary, allowed_urls: set[str]) -> list[str]:
    """Return cited URLs not present in the provided article set (canonicalized)."""
    return [url for url in _all_cited(summary) if canonical_url(url) not in allowed_urls]


def sanitize_citations(summary: NewsSummary, allowed_urls: set[str]) -> NewsSummary:
    """Drop fabricated source URLs, keeping the point text (hallucination control)."""

    def clean_group(group: list[CitedPoint]) -> list[CitedPoint]:
        return [
            point.model_copy(
                update={"sources": [u for u in point.sources if canonical_url(u) in allowed_urls]}
            )
            for point in group
        ]

    return summary.model_copy(
        update={
            "bullish": clean_group(summary.bullish),
            "bearish": clean_group(summary.bearish),
            "risks": clean_group(summary.risks),
            "catalysts": clean_group(summary.catalysts),
        }
    )


# ---------------------------------------------------------------------------
# Numeric grounding (shared by the chat agent and the synthesizer)
# ---------------------------------------------------------------------------
# An LLM that *reports* model/tool numbers (the agent, the synthesizer) may only
# state figures that came from those inputs — it must never invent or revise a
# probability/return. NumberGrounding collects numbers from the inputs and flags
# any decimal/percent in the LLM's text that doesn't trace back to one.

# Numbers inside input data (structured or text): ints and decimals.
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Figures to verify in the LLM's text: percentages (int or decimal) or bare
# decimals. Bare integers (years, day counts) are intentionally not checked.
_CHECK_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%|-?\d+\.\d+")


def _normalized_forms(value: float) -> set[float]:
    """Rounded fraction- and percent-scaled forms used for matching."""
    if not math.isfinite(value):
        return set()
    # No 0-digit rounding — it collides distinct values (0.71 and 0.92 both round
    # to 1.0), which would mask fabricated figures.
    forms: set[float] = set()
    for scaled in (value, value * 100.0):
        for digits in (1, 2, 4):
            forms.add(round(scaled, digits))
    return forms


class NumberGrounding:
    """Accumulates numbers from inputs and checks an LLM's text against them."""

    def __init__(self) -> None:
        self._grounded: set[float] = set()

    def add_value(self, value: float) -> None:
        self._grounded |= _normalized_forms(value)

    def add_from(self, obj: Any) -> None:
        """Recursively collect numbers from an input (dicts/lists/strings)."""
        if isinstance(obj, bool):
            return  # bool is an int subclass; ignore
        if isinstance(obj, int | float):
            self.add_value(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                self.add_from(v)
        elif isinstance(obj, list | tuple):
            for item in obj:
                self.add_from(item)
        elif isinstance(obj, str):
            # Ground numbers embedded in text (e.g. "revenue up 20%").
            for match in _ANY_NUMBER.finditer(obj):
                try:
                    self.add_value(float(match.group(0)))
                except ValueError:
                    continue

    def ungrounded(self, text: str) -> list[str]:
        """Return distinct numeric tokens in ``text`` not traceable to an input."""
        violations: list[str] = []
        seen: set[str] = set()
        for match in _CHECK_NUMBER.finditer(text):
            token = match.group(0)
            raw = token[:-1] if token.endswith("%") else token
            try:
                value = float(raw)
            except ValueError:
                continue
            # A percent token "3%" may also mean the fraction 0.03; check both.
            candidate = _normalized_forms(value) | _normalized_forms(value / 100.0)
            if not (candidate & self._grounded) and token not in seen:
                violations.append(token)
                seen.add(token)
        return violations
