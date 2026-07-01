"""Two-stage routing — the cheap Haiku classifier (prototype, gated by ``use_llm_classifier``).

A LIGHT classifier that maps a natural-language question to ONE deterministic capability (a
``router.ROUTES`` name) with extracted params, or decides to ESCALATE to the full Sonnet agent for
compositional / multi-hop / uncertain questions.

Scope discipline (the whole point of this design):
- Haiku is used ONLY for the routing decision here. It emits **only** a small JSON object
  ``{route, ticker, horizon, days, topic, confidence, escalate, reason}`` — never prose, never a
  number. Invariant #1 (numbers come from tools, not the LLM) is untouched.
- ALL content — news analysis, synthesis, RAG, the final memo — runs on ``llm_model`` (Sonnet),
  whether we dispatch a deterministic route (its own guarded Sonnet call) or escalate (the Sonnet
  agent loop). The classifier picks the door; Sonnet does the work behind it.

Determinism: the backing client runs at temperature 0 and we force a JSON object, so the routing
decision is reproducible and parseable. Any parse/validation failure fails SAFE — escalate, never
guess (a confident misroute is worse than an escalation to the stronger model).

Injection seam: ``RouteClassifier`` depends only on the narrow ``TextLLM`` (``complete_json``), so
tests inject a ``FakeJsonLLM`` and never hit the network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from stock_agent.agent.router import ROUTE_NAMES, ROUTES
from stock_agent.llm.client import LLMError, TextLLM
from stock_agent.logging_config import get_logger

log = get_logger(__name__)

# A plausible ticker symbol: 1–6 uppercase letters, optional dotted class suffix (e.g. BRK.B).
_TICKER_RE = re.compile(r"^[A-Z]{1,6}(?:\.[A-Z]{1,2})?$")
_ESCALATE = "escalate"  # sentinel value the model may return for the route


@dataclass(frozen=True)
class Classification:
    """The classifier's decision for one question.

    ``route`` is a ``ROUTES`` name to dispatch deterministically, or ``None`` meaning ESCALATE to
    the full agent. The param facets are what the classifier could extract from the text (already
    validated/normalized); missing ones are ``None`` and the target tool applies its own default.
    """

    route: str | None
    ticker: str | None = None
    horizon: int | None = None
    days: int | None = None
    topic: str | None = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def escalated(self) -> bool:
        return self.route is None

    @classmethod
    def escalate(cls, reason: str, *, confidence: float = 0.0) -> Classification:
        """A fail-safe escalation decision (no route, defer to the Sonnet agent)."""
        return cls(route=None, confidence=confidence, reason=reason)


def _route_catalog() -> str:
    """Render the deterministic route menu (name · needs · blurb) for the classifier prompt.

    Built from the single source of truth (``ROUTES``) so the classifier's action space can never
    drift from what the router can actually dispatch. Stable order → a stable, cacheable prompt.
    """
    lines = []
    for name in ROUTE_NAMES:
        spec = ROUTES[name]
        needs = f" (needs: {', '.join(spec.needs)})" if spec.needs else ""
        lines.append(f"- {name}{needs}: {spec.blurb}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """The classifier's system instructions. Pure + deterministic (cached across calls)."""
    return (
        "You are a strict ROUTER for a stock-research assistant. Choose the SINGLE best capability "
        "to answer the user's question, or escalate to a stronger general agent.\n\n"
        "Capabilities:\n"
        f"{_route_catalog()}\n\n"
        "Rules:\n"
        f'- Pick exactly ONE capability name the question maps to, or set route to "{_ESCALATE}".\n'
        f'- ESCALATE (route "{_ESCALATE}") when the question needs MORE THAN ONE capability, a '
        "comparison or multi-hop across companies/periods/entities, is compound (\"X and Y\"), or "
        "you are not confident a single capability fully answers it.\n"
        "- Extract only parameters STATED in the question: ticker (stock symbol), horizon "
        "(forecast horizon in trading days), days (news/price lookback), topic (a short "
        "sector/theme phrase, e.g. 'AI memory'). Use null if not stated — never invent a ticker.\n"
        "- You NEVER answer the question, summarize, or produce any number. You only route.\n"
        "- confidence is your 0..1 certainty the chosen single capability fully answers the "
        "question.\n\n"
        "Output ONLY a JSON object with exactly these keys: "
        '{"route": string, "ticker": string|null, "horizon": integer|null, "days": integer|null, '
        '"topic": string|null, "confidence": number, "reason": string}. No prose.'
    )


def _coerce_int(value: Any) -> int | None:
    """Best-effort positive-int coercion (accepts int or numeric string); else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        n = int(value)
        return n if n > 0 else None
    return None


def _norm_ticker(value: Any) -> str | None:
    """Uppercase/strip and validate a ticker symbol; None if absent or implausible."""
    if not isinstance(value, str):
        return None
    t = value.upper().strip()
    return t if _TICKER_RE.match(t) else None


class RouteClassifier:
    """Classifies a question into one deterministic route or an escalation. Never raises."""

    def __init__(self, llm: TextLLM, *, min_confidence: float = 0.6, max_tokens: int = 256) -> None:
        self._llm = llm
        self._min_confidence = min_confidence
        self._max_tokens = max_tokens
        self._system = build_system_prompt()

    def classify(self, question: str) -> Classification:
        """Return the routing decision; escalate on any failure or below-threshold confidence."""
        q = question.strip()
        if not q:
            return Classification.escalate("empty question")
        try:
            raw = self._llm.complete_json(
                system=self._system, user=q, max_tokens=self._max_tokens
            )
        except LLMError as exc:
            log.warning("classifier.llm_failed", error=str(exc))
            return Classification.escalate(f"classifier LLM failed: {exc}")
        return self._validate(raw, q)

    def _validate(self, raw: str, question: str) -> Classification:
        """Parse + validate the model JSON into a safe ``Classification`` (fail-safe = escalate)."""
        obj = _parse_json_object(raw)
        if obj is None:
            return Classification.escalate("unparseable classifier output")

        confidence = obj.get("confidence")
        conf = float(confidence) if isinstance(confidence, int | float) else 0.0
        reason = str(obj.get("reason", ""))[:200]
        route = obj.get("route")

        # Explicit escalate, unknown/absent route → defer to the agent.
        if not isinstance(route, str) or route == _ESCALATE or route not in ROUTE_NAMES:
            return Classification.escalate(reason or "no single-capability match", confidence=conf)
        # Low confidence → escalate rather than dispatch a shaky single route.
        if conf < self._min_confidence:
            return Classification.escalate(
                reason or f"low confidence {conf:.2f}", confidence=conf
            )

        ticker = _norm_ticker(obj.get("ticker"))
        horizon = _coerce_int(obj.get("horizon"))
        days = _coerce_int(obj.get("days"))
        topic = obj.get("topic")
        topic = topic.strip() if isinstance(topic, str) and topic.strip() else None

        # A route can't run without its required facet — escalate instead of guessing.
        needs = ROUTES[route].needs
        if "ticker" in needs and ticker is None:
            return Classification.escalate(
                f"route '{route}' needs a ticker, none stated", confidence=conf
            )
        if "topic" in needs and topic is None:
            return Classification.escalate(
                f"route '{route}' needs a theme/topic, none stated", confidence=conf
            )

        return Classification(
            route=route,
            ticker=ticker,
            horizon=horizon,
            days=days,
            topic=topic,
            confidence=conf,
            reason=reason,
        )


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Lenient parse: the first ``{...}`` block in ``raw``, or None. Tolerates stray prose."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
