"""Tests for the numeric-grounding guard."""

from __future__ import annotations

from stock_agent.agent.guards import NumberGrounding


def test_grounded_percent_from_fraction_passes() -> None:
    g = NumberGrounding()
    g.add_from({"expected_return": 0.0312, "upside_prob": 0.71})
    # Agent restates as percentages — should be considered grounded.
    assert g.ungrounded("Expected return is +3.12% with a 71% chance of gains.") == []


def test_fabricated_decimal_is_flagged() -> None:
    g = NumberGrounding()
    g.add_from({"upside_prob": 0.71})
    violations = g.ungrounded("There is a 92.5% chance of a rally.")
    assert "92.5%" in violations


def test_numbers_inside_tool_text_are_grounded() -> None:
    g = NumberGrounding()
    g.add_from({"overview": "Revenue rose 20% and shares fell 5.2% on the news."})
    assert g.ungrounded("Revenue grew 20% while the stock dropped 5.2%.") == []


def test_bare_integers_are_not_flagged() -> None:
    g = NumberGrounding()  # nothing grounded
    # Years, day counts, article counts must not trip the guard.
    assert g.ungrounded("Over the past 5 years and 20 trading days, across 8 articles.") == []


def test_grounded_price_decimal_passes() -> None:
    g = NumberGrounding()
    g.add_from({"last_close": 311.58})
    assert g.ungrounded("The last close was 311.58.") == []


def test_grouping_comma_in_llm_text_is_grounded() -> None:
    # Regression: a tool returns the float 1043.27; the LLM writes "1,043.27".
    # The grouping comma must not split the token into the spurious "043.27".
    g = NumberGrounding()
    g.add_from({"figure": 1043.27})
    assert g.ungrounded("The figure is 1,043.27 for the period.") == []


def test_nested_grouping_commas_are_grounded() -> None:
    g = NumberGrounding()
    g.add_from({"revenue_musd": 1043000.50})
    assert g.ungrounded("Revenue was 1,043,000.50 million.") == []


def test_grouping_comma_in_tool_string_is_grounded() -> None:
    # Symmetric case: a tool returns the figure preformatted as a string.
    g = NumberGrounding()
    g.add_from({"note": "market cap of $1,043.27"})
    assert g.ungrounded("Market cap is 1,043.27.") == []


def test_list_style_commas_do_not_merge_into_a_number() -> None:
    # "1,2,3" is not a grouped number; the normalizer must leave it alone so we
    # don't fabricate a grounded "123" that could mask a real violation.
    g = NumberGrounding()
    g.add_from({"note": "items 1,2,3 in the set"})
    assert g.ungrounded("The ratio was 123.0.") == ["123.0"]
