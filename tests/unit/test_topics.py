"""Topic registry resolution + GDELT query building (Enhancement C, pure)."""

from __future__ import annotations

from stock_agent.news.topics import (
    TOPIC_REGISTRY,
    gdelt_query_expression,
    list_topics,
    resolve_topic,
)


def test_known_topic_resolves_from_registry() -> None:
    r = resolve_topic("robotics")
    assert r.known is True
    assert r.topic == "robotics"
    assert r.keywords == TOPIC_REGISTRY["robotics"].keywords


def test_alias_resolves_to_canonical_slug() -> None:
    for alias in ("EVs", "electric vehicles"):
        r = resolve_topic(alias)
        assert r.known is True
        assert r.topic == "ev"
    assert resolve_topic("AI memory").topic == "ai_memory"


def test_case_and_separator_insensitive() -> None:
    assert resolve_topic("AI-Infra").topic == "ai_infra"
    assert resolve_topic(" ai_infra ").topic == "ai_infra"


def test_freeform_fallback_for_unlisted_theme() -> None:
    r = resolve_topic("quantum computing")
    assert r.known is False
    assert r.keywords == ("quantum computing",)
    assert r.label == "quantum computing"


def test_query_expression_quotes_phrases_and_ors() -> None:
    r = resolve_topic("ai_memory")  # ("HBM", "high bandwidth memory", "AI memory")
    expr = gdelt_query_expression(r)
    assert expr.startswith("(") and expr.endswith(")")
    assert " OR " in expr
    assert '"high bandwidth memory"' in expr  # multi-word quoted
    assert "HBM" in expr  # single token bare-ish (no surrounding spaces required)


def test_single_keyword_expression_has_no_parens() -> None:
    expr = gdelt_query_expression(resolve_topic("blockchain"))  # free-form single phrase
    assert expr == '"blockchain"' or expr == "blockchain"
    assert " OR " not in expr


def test_list_topics_nonempty_and_curated() -> None:
    topics = list_topics()
    assert {"robotics", "ev", "ai_memory", "semiconductors"} <= set(topics)
