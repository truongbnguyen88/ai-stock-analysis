"""Golden tests for the Form 4 ownership-XML parser (recorded fixture, no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_agent.documents.form4 import parse_form4_xml
from stock_agent.schemas.insider import InsiderFilingRef, InsiderTransaction

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "form4_sample.xml"
_FILING_DATE = date(2024, 5, 10)


def _parse() -> list[InsiderTransaction]:
    return parse_form4_xml(_FIXTURE.read_text(), ticker="NVDA", filing_date=_FILING_DATE)


def test_parses_all_transactions_skips_holdings() -> None:
    txns = _parse()
    # 3 non-derivative transactions (P, S, A) + 1 derivative (M); the holding row is skipped.
    assert len(txns) == 4
    codes = sorted(t.code for t in txns)
    assert codes == ["A", "M", "P", "S"]


def test_carries_filing_date_and_owner() -> None:
    t = _parse()[0]
    assert t.filing_date == _FILING_DATE  # from the index, not the XML
    assert t.transaction_date == date(2024, 5, 9)
    assert t.owner_name == "Doe Jane"


def test_signed_value_sign_and_magnitude() -> None:
    by_code = {t.code: t for t in _parse()}
    assert by_code["P"].signed_value == pytest.approx(1000 * 120.50)   # acquired → +
    assert by_code["S"].signed_value == pytest.approx(-400 * 121.00)   # disposed → −
    assert by_code["A"].signed_value == pytest.approx(0.0)             # price 0 → 0 value


def test_derivative_flag() -> None:
    by_code = {t.code: t for t in _parse()}
    assert by_code["M"].is_derivative is True
    assert by_code["P"].is_derivative is False


def test_namespaced_xml_is_handled() -> None:
    ns = (
        '<ownershipDocument xmlns="http://www.sec.gov/edgar/ownership">'
        "<nonDerivativeTable><nonDerivativeTransaction>"
        "<transactionDate><value>2024-01-02</value></transactionDate>"
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>"
        "<transactionAmounts>"
        "<transactionShares><value>10</value></transactionShares>"
        "<transactionPricePerShare><value>5</value></transactionPricePerShare>"
        "<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>"
        "</transactionAmounts>"
        "</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"
    )
    txns = parse_form4_xml(ns, ticker="X", filing_date=_FILING_DATE)
    assert len(txns) == 1 and txns[0].signed_value == pytest.approx(50.0)


def test_malformed_xml_returns_empty_not_raises() -> None:
    assert parse_form4_xml("<not-xml", ticker="X", filing_date=_FILING_DATE) == []


def test_insider_filing_ref_url() -> None:
    ref = InsiderFilingRef(
        ticker="NVDA", cik="0001045810", filing_date=_FILING_DATE,
        accession_number="0001234567-24-000045", primary_document="wk-form4.xml",
    )
    assert ref.url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000123456724000045/wk-form4.xml"
    )
    assert ref.filing_id == "NVDA:4:2024-05-10:0001234567-24-000045"
