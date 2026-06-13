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


def test_owner_identity_and_role() -> None:
    t = _parse()[0]
    assert t.owner_cik == "0001234567"  # for distinct-insider (cluster) counts
    assert t.is_officer is True
    assert t.is_director is False
    assert t.officer_title == "CFO"
    assert t.is_senior is True  # CFO → senior insider


def test_post_transaction_holdings_and_conviction() -> None:
    by_code = {t.code: t for t in _parse()}
    p = by_code["P"]  # bought 1000, holds 26000 after → prior 25000
    assert p.shares_owned_after == pytest.approx(26000.0)
    assert p.prior_holdings == pytest.approx(25000.0)
    assert p.ownership_change_fraction == pytest.approx(1000 / 25000)  # +4% conviction
    s = by_code["S"]  # sold 400, holds 25600 after → prior 26000
    assert s.prior_holdings == pytest.approx(26000.0)
    assert s.ownership_change_fraction == pytest.approx(-400 / 26000)  # signed negative
    # Rows without post-transaction holdings → conviction undefined (None), not 0.
    assert by_code["A"].shares_owned_after is None
    assert by_code["A"].ownership_change_fraction is None


def test_10b5_1_detected_from_footnote() -> None:
    by_code = {t.code: t for t in _parse()}
    assert by_code["S"].is_planned_10b5_1 is True   # footnote F1 names a 10b5-1 plan
    assert by_code["P"].is_planned_10b5_1 is False  # opportunistic buy, no plan footnote


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


def test_insider_filing_ref_url_strips_xsl_render_prefix() -> None:
    # EDGAR's primaryDocument for Form 4 is the rendered HTML path; the URL must
    # de-render it to the raw XML (strip the xslF.../ directory).
    ref = InsiderFilingRef(
        ticker="MSFT", cik="0000789019", filing_date=_FILING_DATE,
        accession_number="0000789019-26-000109", primary_document="xslF345X06/form4.xml",
    )
    assert ref.url == (
        "https://www.sec.gov/Archives/edgar/data/789019/000078901926000109/form4.xml"
    )
