"""Insider activity aggregation + fetch orchestration (offline, fake provider)."""

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


def _txn(code: str, ad: str, shares: float, price: float | None, fdate: date) -> InsiderTransaction:
    return InsiderTransaction(
        ticker="NVDA", filing_date=fdate, transaction_date=fdate, code=code,
        acquired_disposed=ad, shares=shares, price_per_share=price, is_derivative=False,
    )


def test_aggregate_keeps_only_open_market_and_sums_signed() -> None:
    txns = [
        _txn("P", "A", 1000, 100.0, date(2024, 5, 9)),   # +100k buy
        _txn("S", "D", 400, 110.0, date(2024, 5, 9)),    # −44k sell (same day)
        _txn("A", "A", 5000, 0.0, date(2024, 5, 9)),     # grant → excluded
        _txn("M", "A", 2000, 50.0, date(2024, 5, 9)),    # option exercise → excluded
        _txn("P", "A", 200, 90.0, date(2024, 6, 1)),     # +18k buy, later day
    ]
    df = aggregate_transactions(txns)
    assert list(df.columns) == ACTIVITY_COLS
    assert len(df) == 2  # two distinct filing dates
    may9 = df.loc["2024-05-09"]
    assert may9["net_value"] == pytest.approx(1000 * 100.0 - 400 * 110.0)
    assert may9["n_buys"] == 1 and may9["n_sells"] == 1
    jun1 = df.loc["2024-06-01"]
    assert jun1["net_value"] == pytest.approx(200 * 90.0)
    assert jun1["n_buys"] == 1 and jun1["n_sells"] == 0


def test_aggregate_empty_when_no_open_market_trades() -> None:
    df = aggregate_transactions([_txn("A", "A", 5000, 0.0, date(2024, 5, 9))])
    assert df.empty and list(df.columns) == ACTIVITY_COLS


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
    # Fixture has one P (1000@120.50) and one S (400@121.00) open-market trade.
    assert df.loc["2024-05-10", "net_value"] == pytest.approx(1000 * 120.50 - 400 * 121.00)
    assert df.loc["2024-05-10", "n_buys"] == 1
    assert df.loc["2024-05-10", "n_sells"] == 1


def test_fetch_insider_activity_download_failure_is_graceful() -> None:
    provider = _FakeProvider([_ref(date(2024, 5, 10), "x")], "", fail_download=True)
    df = fetch_insider_activity(provider, "NVDA")  # type: ignore[arg-type]
    assert df.empty  # failed download skipped → no transactions → empty frame
