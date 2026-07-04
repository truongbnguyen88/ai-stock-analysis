"""Unit tests for ``ui.keys.key_statuses`` — the shared KEYS-row source of truth.

Hermetic: every ``Settings`` is built with ``_env_file=None`` so a dev machine's real ``.env``
(which may set ``SEC_USER_AGENT`` / ``VOYAGE_API_KEY`` / ``EMBEDDING_PROVIDER=voyage``) can't
leak into the assertions.
"""

from __future__ import annotations

from stock_agent.settings import Settings
from stock_agent.ui.keys import key_statuses


def _labels(settings: Settings) -> list[str]:
    return [label for label, _present, _required in key_statuses(settings)]


def test_anthropic_is_the_only_required_key() -> None:
    rows = key_statuses(Settings(_env_file=None))
    required = [label for label, _present, req in rows if req]
    assert required == ["Anthropic"]


def test_present_flag_reflects_settings() -> None:
    rows = key_statuses(
        Settings(
            _env_file=None,
            anthropic_api_key="sk",
            finnhub_api_key="fk",
            marketaux_api_key=None,
            alpha_vantage_api_key=None,
            sec_user_agent="Tester test@example.com",
        )
    )
    present = {label: p for label, p, _req in rows}
    assert present["Anthropic"] is True
    assert present["Finnhub"] is True
    assert present["Marketaux"] is False
    assert present["Alpha Vantage"] is False
    assert present["SEC EDGAR"] is True  # sec_user_agent gates SEC-filing downloads


def test_sec_edgar_always_listed_as_optional() -> None:
    # SEC UA is surfaced regardless of embedder, and is optional (app runs without filings).
    rows = key_statuses(Settings(_env_file=None, embedding_provider="local"))
    sec = next(r for r in rows if r[0] == "SEC EDGAR")
    assert sec[2] is False  # optional


def test_voyage_row_only_when_voyage_is_active_embedder() -> None:
    # Voyage is the query-time embedder only under embedding_provider="voyage" — hide it
    # for local/openai users so an irrelevant key doesn't clutter the row.
    assert "Voyage" not in _labels(Settings(_env_file=None, embedding_provider="local"))
    assert "Voyage" not in _labels(Settings(_env_file=None, embedding_provider="openai"))

    voyage_rows = key_statuses(Settings(_env_file=None, embedding_provider="voyage"))
    voyage = next(r for r in voyage_rows if r[0] == "Voyage")
    assert voyage == ("Voyage", False, False)  # shown, missing (no key), optional

    with_key = key_statuses(
        Settings(_env_file=None, embedding_provider="voyage", voyage_api_key="vk")
    )
    assert ("Voyage", True, False) in with_key


def test_order_is_stable_for_the_chip_row() -> None:
    rows = key_statuses(Settings(_env_file=None, embedding_provider="voyage"))
    assert [label for label, _p, _r in rows] == [
        "Anthropic",
        "Finnhub",
        "Marketaux",
        "Alpha Vantage",
        "SEC EDGAR",
        "Voyage",
    ]
