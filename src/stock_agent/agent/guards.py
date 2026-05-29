"""Numeric-grounding guard for the chat agent (Role B).

The agent is a router: it may *report* quantitative figures (probabilities,
returns, VaR, prices, indicator values) but only ones produced by tools — it must
never compute or invent numbers. This guard collects every number appearing in
tool results (both structured values and numbers embedded in tool text, so the
agent may quote news facts) and flags any decimal/percent in the agent's answer
that does not trace back to one of them.

Design choices:
- Only decimals and percentages are checked. Bare integers (years, day counts,
  article counts) are ignored to avoid false positives — fabricated probabilities
  and returns essentially always carry a decimal point or a % sign.
- Grounded values are stored in several normalized forms (fraction and percent,
  at a few rounding precisions) so 0.0312, "3.1%", and "3.12%" all match.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Numbers inside tool output (structured or text): ints and decimals.
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Figures to verify in the agent's answer: percentages (int or decimal) or bare
# decimals. Bare integers are intentionally not checked.
_CHECK_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%|-?\d+\.\d+")


def _normalized_forms(value: float) -> set[float]:
    """Rounded fraction- and percent-scaled forms used for matching."""
    if not math.isfinite(value):
        return set()
    # Note: no 0-digit rounding — it collides distinct values (0.71 and 0.92
    # both round to 1.0), which would mask fabricated figures.
    forms: set[float] = set()
    for scaled in (value, value * 100.0):
        for digits in (1, 2, 4):
            forms.add(round(scaled, digits))
    return forms


class NumberGrounding:
    """Accumulates numbers from tool results and checks the agent's answer."""

    def __init__(self) -> None:
        self._grounded: set[float] = set()

    def add_value(self, value: float) -> None:
        self._grounded |= _normalized_forms(value)

    def add_from(self, obj: Any) -> None:
        """Recursively collect numbers from a tool result (dicts/lists/strings)."""
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
            # Ground numbers embedded in text (e.g. a news summary's "revenue up 20%").
            for match in _ANY_NUMBER.finditer(obj):
                try:
                    self.add_value(float(match.group(0)))
                except ValueError:
                    continue

    def ungrounded(self, text: str) -> list[str]:
        """Return distinct numeric tokens in ``text`` not traceable to a tool result."""
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
