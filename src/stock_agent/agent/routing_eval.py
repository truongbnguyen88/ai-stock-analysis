"""Routing-eval harness — measure the Haiku classifier before trusting it (prototype).

Given labeled ``(question, expected_route)`` examples, run a classifier over them and report the
metrics that decide whether two-stage routing is safe to enable:

- **exact accuracy** — fraction where the predicted decision equals the label (a route name, or
  ``"escalate"``). The headline number.
- **misroute rate** — the DANGEROUS class: predicted a *confident single route that is wrong*
  (not an escalation). A misroute silently answers with the wrong capability; an over-escalation
  merely costs a Sonnet call. Keep this near zero.
- **escalation rate** vs **expected escalation rate** — how often we defer to Sonnet vs how often
  we should. Over-escalation erodes the cost saving; under-escalation risks misroutes on hard Qs.

Pure over an injected ``ClassifierLike`` (anything with ``classify(question) -> Classification``),
so tests use a deterministic fake and the live eval (``python -m stock_agent eval-routing``) uses a
Haiku-backed ``RouteClassifier``. ``EXPECTED_ESCALATE`` is the label for questions that *should*
go to the full agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from stock_agent.agent.classifier import Classification

EXPECTED_ESCALATE = "escalate"  # label for questions that SHOULD go to the full Sonnet agent


class ClassifierLike(Protocol):
    def classify(self, question: str) -> Classification: ...


@dataclass(frozen=True)
class RoutingExample:
    """One labeled routing case: the question and the route it should resolve to (or 'escalate')."""

    question: str
    expected: str  # a ROUTES name, or EXPECTED_ESCALATE


@dataclass
class RoutingEvalReport:
    """Aggregate routing metrics + the specific errors, for a labeled run."""

    n: int
    correct: int
    escalations: int
    expected_escalations: int
    misroutes: list[tuple[str, str, str]] = field(default_factory=list)  # (question, expected, got)
    over_escalations: list[tuple[str, str]] = field(default_factory=list)  # (question, expected)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def misroute_rate(self) -> float:
        return len(self.misroutes) / self.n if self.n else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.n if self.n else 0.0

    @property
    def expected_escalation_rate(self) -> float:
        return self.expected_escalations / self.n if self.n else 0.0

    def summary(self) -> str:
        """One-block human-readable report (used by the CLI)."""
        lines = [
            f"Routing eval — n={self.n}",
            f"  accuracy         : {self.accuracy:.1%} ({self.correct}/{self.n})",
            f"  misroute rate    : {self.misroute_rate:.1%} ({len(self.misroutes)})  "
            "(confident WRONG route — the dangerous case)",
            f"  escalation rate  : {self.escalation_rate:.1%} "
            f"(expected {self.expected_escalation_rate:.1%})",
        ]
        if self.misroutes:
            lines.append("  misroutes:")
            lines += [f"    - [{exp} -> {got}] {q}" for q, exp, got in self.misroutes]
        if self.over_escalations:
            lines.append("  over-escalations (should have routed):")
            lines += [f"    - [{exp}] {q}" for q, exp in self.over_escalations]
        return "\n".join(lines)


def evaluate_routing(
    classifier: ClassifierLike, examples: list[RoutingExample]
) -> RoutingEvalReport:
    """Run ``classifier`` over ``examples`` and tally accuracy / misroute / escalation metrics."""
    correct = escalations = expected_escalations = 0
    misroutes: list[tuple[str, str, str]] = []
    over_escalations: list[tuple[str, str]] = []

    for ex in examples:
        decision = classifier.classify(ex.question)
        predicted = EXPECTED_ESCALATE if decision.escalated else decision.route
        assert predicted is not None  # route is non-None when not escalated
        if decision.escalated:
            escalations += 1
        if ex.expected == EXPECTED_ESCALATE:
            expected_escalations += 1

        if predicted == ex.expected:
            correct += 1
        elif predicted == EXPECTED_ESCALATE:
            over_escalations.append((ex.question, ex.expected))  # should have routed, deferred
        else:
            misroutes.append((ex.question, ex.expected, predicted))  # confident wrong route

    return RoutingEvalReport(
        n=len(examples),
        correct=correct,
        escalations=escalations,
        expected_escalations=expected_escalations,
        misroutes=misroutes,
        over_escalations=over_escalations,
    )


def load_examples(path: Path) -> list[RoutingExample]:
    """Load labeled examples from a JSONL file (``{"question": ..., "expected": ...}`` per line)."""
    out: list[RoutingExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        out.append(RoutingExample(question=str(d["question"]), expected=str(d["expected"])))
    return out


# A curated starter set covering every route + the escalation cases. Grow it in
# configs/routing_eval.jsonl (loaded via `load_examples`) as real misroutes surface.
DEFAULT_EXAMPLES: list[RoutingExample] = [
    # --- single-capability (should dispatch deterministically) ---
    RoutingExample("Show me NVDA's technical indicators", "technicals"),
    RoutingExample("What are AMD's RSI and MACD right now?", "technicals"),
    RoutingExample("Recent price stats for MSFT over the last 30 days", "price"),
    RoutingExample("Give me a 30-day forecast for NVDA", "forecast"),
    RoutingExample("Forecast TSLA 20 trading days out", "forecast"),
    RoutingExample("What's the chance of a big move in NVDA over the next month?", "big_move"),
    RoutingExample("Just the latest NVDA headlines", "headlines"),
    RoutingExample("Summarize NVDA news — themes, risks, catalysts", "news"),
    RoutingExample("Analyze recent news about AI memory", "theme_news"),
    RoutingExample("What's the news on the semiconductors theme?", "theme_news"),
    RoutingExample("What risk factors does NVDA disclose in its 10-K?", "filings"),
    RoutingExample("What does MSFT management say about cloud margins in its filings?", "filings"),
    RoutingExample("Give me the full executive brief on NVDA", "research_brief"),
    # --- compositional / multi-hop / uncertain (should escalate) ---
    RoutingExample("Compare NVDA's and AMD's risk factors across their filings", EXPECTED_ESCALATE),
    RoutingExample(
        "Analyze NVDA news, forecast 30 days, and summarize its filing risks", EXPECTED_ESCALATE
    ),
    RoutingExample(
        "Which of NVDA's named suppliers flag the same supply risk?", EXPECTED_ESCALATE
    ),
    RoutingExample("How did TSLA's risk disclosures change from 2023 to 2025?", EXPECTED_ESCALATE),
    RoutingExample(
        "Pull AI-memory news and tell me which stocks it affects and how", EXPECTED_ESCALATE
    ),
    RoutingExample("Compare the 20-day forecasts for NVDA, MSFT, and AMD", EXPECTED_ESCALATE),
]
