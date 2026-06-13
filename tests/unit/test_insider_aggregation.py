"""Re-engineered insider aggregation + fetch orchestration (offline, fake provider).

The signal separates the informative BUY channel from sells, weights buys by
Δ-ownership conviction + senior (CEO/CFO) role, and excludes Rule 10b5-1
(pre-scheduled, non-discretionary) trades.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_agent.data.insider import (
    ACTIVITY_COLS,
    aggregate_transactions,
    fetch_insider_activity,
)
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.schemas.insider import InsiderFilingRef, InsiderTransaction

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "form4_sample.xml"


def _txn(
    code: str, ad: str, shares: float, fdate: date, *,
    owned_after: float | None = None, senior: bool = False, planned: bool = False,
) -> InsiderTransaction:
    return InsiderTransaction(
        ticker="NVDA", filing_date=fdate, transaction_date=fdate, code=code,
        acquired_disposed=ad, shares=shares, price_per_share=100.0, is_derivative=False,
        shares_owned_after=owned_after, officer_title="CEO" if senior else None,
        is_officer=senior, is_planned_10b5_1=planned,
    )


def test_buy_and_sell_channels_are_separate_never_netted() -> None:
    # A buy and a sell on the same day must NOT cancel — they land in different columns.
    txns = [
        _txn("P", "A", 1000, date(2024, 5, 9), owned_after=11000),  # prior 10000 → +0.10
        _txn("S", "D", 2000, date(2024, 5, 9), owned_after=8000),   # prior 10000 → |−0.20|
    ]
    df = aggregate_transactions(txns)
    assert list(df.columns) == ACTIVITY_COLS
    row = df.loc["2024-05-09"]
    assert row["buy_conviction"] == pytest.approx(0.10)   # not diluted by the sell
    assert row["sell_pressure"] == pytest.approx(0.20)    # tracked independently
    assert row["senior_buy_n"] == 0.0


def test_excludes_grants_exercises_and_10b5_1() -> None:
    txns = [
        _txn("A", "A", 5000, date(2024, 5, 9)),                       # grant → excluded
        _txn("M", "A", 2000, date(2024, 5, 9)),                       # option exercise → excluded
        _txn("S", "D", 400, date(2024, 5, 9), owned_after=9600, planned=True),  # 10b5-1 → excluded
    ]
    assert aggregate_transactions(txns).empty


def test_senior_buy_counted_and_conviction_capped() -> None:
    txns = [
        # Senior (CEO) buy establishing position (no prior holdings) → conviction capped at 1.0.
        _txn("P", "A", 500, date(2024, 6, 1), owned_after=None, senior=True),
    ]
    row = aggregate_transactions(txns).loc["2024-06-01"]
    assert row["senior_buy_n"] == 1.0
    assert row["buy_conviction"] == pytest.approx(1.0)  # unknown prior → max conviction, capped


def test_aggregate_empty_when_no_discretionary_trades() -> None:
    assert aggregate_transactions([_txn("A", "A", 5000, date(2024, 5, 9))]).empty


class _FakeProvider:
    """Minimal stand-in exposing the two methods data/insider.py calls."""

    def __init__(self, refs: list[InsiderFilingRef], xml: str, *, fail_download: bool = False):
        self._refs = refs
        self._xml = xml
        self._fail_download = fail_download

    def list_form4_filings(
        self, ticker: str, *, since: object = None, limit: int = 200  # noqa: ARG002
    ) -> list[InsiderFilingRef]:
        return self._refs

    def download_form4(self, ref: InsiderFilingRef) -> str:  # noqa: ARG002
        if self._fail_download:
            raise ProviderUnavailable("sec_edgar", "boom")
        return self._xml


def _ref(fdate: date, acc: str) -> InsiderFilingRef:
    return InsiderFilingRef(
        ticker="NVDA", cik="0001045810", filing_date=fdate,
        accession_number=acc, primary_document="form4.xml",
    )


def test_fetch_insider_activity_parses_fixture() -> None:
    provider = _FakeProvider([_ref(date(2024, 5, 10), "0001-24-1")], _FIXTURE.read_text())
    df = fetch_insider_activity(provider, "NVDA")  # type: ignore[arg-type]
    row = df.loc["2024-05-10"]
    # Fixture: CFO buys 1000 (prior 25000 → +0.04); the sell is a 10b5-1 plan → excluded.
    assert row["buy_conviction"] == pytest.approx(1000 / 25000)
    assert row["senior_buy_n"] == 1.0          # CFO is senior
    assert row["sell_pressure"] == pytest.approx(0.0)  # 10b5-1 sell filtered out


def test_fetch_insider_activity_download_failure_is_graceful() -> None:
    provider = _FakeProvider([_ref(date(2024, 5, 10), "x")], "", fail_download=True)
    df = fetch_insider_activity(provider, "NVDA")  # type: ignore[arg-type]
    assert df.empty  # failed download skipped → no transactions → empty frame
