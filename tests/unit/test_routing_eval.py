"""Routing-eval harness — metrics are computed from a deterministic fake classifier (no network)."""

from __future__ import annotations

from pathlib import Path

from stock_agent.agent.classifier import Classification
from stock_agent.agent.routing_eval import (
    DEFAULT_EXAMPLES,
    EXPECTED_ESCALATE,
    RoutingExample,
    evaluate_routing,
    load_examples,
)


class _FakeClassifier:
    """Maps a question to a pre-decided Classification (route name, or None to escalate)."""

    def __init__(self, decisions: dict[str, str | None]) -> None:
        self._decisions = decisions

    def classify(self, question: str) -> Classification:
        route = self._decisions.get(question)
        return Classification(route=route, confidence=1.0)


def test_report_tallies_correct_misroute_and_over_escalation() -> None:
    examples = [
        RoutingExample("a", "news"),  # correct
        RoutingExample("b", "forecast"),  # misroute (predicts technicals)
        RoutingExample("c", EXPECTED_ESCALATE),  # correct escalation
        RoutingExample("d", "price"),  # over-escalation (predicts escalate)
    ]
    fake = _FakeClassifier({"a": "news", "b": "technicals", "c": None, "d": None})
    report = evaluate_routing(fake, examples)

    assert report.n == 4
    assert report.correct == 2  # 'a' and 'c'
    assert report.accuracy == 0.5
    assert report.misroutes == [("b", "forecast", "technicals")]  # confident WRONG route
    assert report.over_escalations == [("d", "price")]
    assert report.escalations == 2  # c, d predicted escalate
    assert report.expected_escalations == 1  # only c should escalate
    assert report.misroute_rate == 0.25


def test_perfect_run_has_zero_error_lists() -> None:
    examples = [RoutingExample("x", "news"), RoutingExample("y", EXPECTED_ESCALATE)]
    fake = _FakeClassifier({"x": "news", "y": None})
    report = evaluate_routing(fake, examples)
    assert report.accuracy == 1.0
    assert not report.misroutes
    assert not report.over_escalations


def test_load_examples_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '# a comment line\n'
        '{"question": "summarize NVDA news", "expected": "news"}\n'
        '\n'  # blank line skipped
        '{"question": "compare NVDA and AMD", "expected": "escalate"}\n',
        encoding="utf-8",
    )
    examples = load_examples(path)
    assert [(e.question, e.expected) for e in examples] == [
        ("summarize NVDA news", "news"),
        ("compare NVDA and AMD", "escalate"),
    ]


def test_default_dataset_is_well_formed() -> None:
    # Labels must be a real route or the escalate sentinel — guards against typos in the dataset.
    from stock_agent.agent.router import ROUTE_NAMES

    valid = set(ROUTE_NAMES) | {EXPECTED_ESCALATE}
    assert len(DEFAULT_EXAMPLES) >= 15
    assert all(ex.expected in valid for ex in DEFAULT_EXAMPLES)
    assert any(ex.expected == EXPECTED_ESCALATE for ex in DEFAULT_EXAMPLES)
