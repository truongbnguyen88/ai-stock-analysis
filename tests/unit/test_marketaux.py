"""Marketaux normalization (no network)."""

from __future__ import annotations

from stock_agent.providers.marketaux import _articles_from_payload, _parse_dt


def test_normalizes_data_array() -> None:
    payload = {
        "data": [
            {
                "title": "T",
                "url": "https://x.com/a",
                "source": "Src",
                "published_at": "2024-01-15T13:30:00.000000Z",
                "description": "D",
                "snippet": "short",
            }
        ]
    }
    bundle = _articles_from_payload("AAPL", payload)
    assert len(bundle) == 1
    art = bundle.articles[0]
    assert art.summary == "D"  # description preferred over snippet
    assert art.published_at.year == 2024


def test_missing_data_key_returns_empty() -> None:
    assert len(_articles_from_payload("AAPL", {})) == 0


def test_parse_dt_handles_trailing_z() -> None:
    parsed = _parse_dt("2024-01-15T13:30:00Z")
    assert parsed.tzinfo is not None
    assert parsed.year == 2024
