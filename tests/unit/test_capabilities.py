"""Showcase capability data: well-formedness + template correctness (no UI)."""

from __future__ import annotations

import re

from stock_agent.ui.capabilities import CAPABILITIES, TICKER_PLACEHOLDER, Capability

# Any unfilled {placeholder} left after rendering would surface as literal text.
_BRACE_TOKEN = re.compile(r"\{[^}]*\}")


def test_capabilities_non_empty() -> None:
    assert len(CAPABILITIES) >= 6  # curated showcase, not a thin list
    assert all(isinstance(c, Capability) for c in CAPABILITIES)


def test_every_field_present() -> None:
    for c in CAPABILITIES:
        assert c.icon.strip(), c
        assert c.title.strip(), c
        assert c.blurb.strip(), c
        assert c.example.strip(), c


def test_titles_unique() -> None:
    titles = [c.title for c in CAPABILITIES]
    assert len(titles) == len(set(titles))


def test_render_example_substitutes_ticker_and_leaves_no_placeholder() -> None:
    for c in CAPABILITIES:
        rendered = c.render_example("NVDA")
        assert TICKER_PLACEHOLDER not in rendered, c
        # No stray template tokens of any kind survive into the submitted prompt.
        assert not _BRACE_TOKEN.search(rendered), c
        if TICKER_PLACEHOLDER in c.example:
            assert "NVDA" in rendered, c


def test_cross_enhancement_capabilities_present() -> None:
    # The showcase advertises the multi-ticker (B) and topic-news (C) features.
    titles = " ".join(c.title.lower() for c in CAPABILITIES)
    assert "compare multiple tickers" in titles
    assert "theme" in titles
