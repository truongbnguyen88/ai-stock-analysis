"""Insider-transaction domain models (SEC Form 4).

Form 4 reports changes in beneficial ownership by corporate insiders (officers,
directors, 10%+ holders). Unlike the RAG corpus filings (10-K/10-Q/8-K), Form 4
is structured *ownership XML*, not narrative text — so it has its own ref/record
types here and never touches ``schemas.documents.DocumentType`` (the RAG enum).

Point-in-time note: a Form 4 must be filed within two business days of the trade,
so the information becomes public on ``filing_date`` (>= ``transaction_date``).
Downstream features key off ``filing_date`` to stay leakage-safe.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel

_SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

# Open-market, discretionary, information-bearing transaction codes. Compensation /
# non-discretionary codes (A=grant/award, M=option exercise, F=tax withholding,
# G=gift, etc.) are deliberately excluded from the signal — they carry little
# directional information. See data/insider.py for how these are aggregated.
OPEN_MARKET_BUY = "P"  # open-market or private purchase
OPEN_MARKET_SELL = "S"  # open-market or private sale


class InsiderFilingRef(BaseModel):
    """Pointer to one Form 4 filing from the EDGAR submissions index (pre-download).

    Mirrors ``schemas.documents.FilingRef`` but with ``form`` fixed to "4" and no
    ``DocumentType`` constraint, keeping the RAG corpus enum unpolluted.
    """

    ticker: str
    cik: str
    filing_date: Date
    accession_number: str
    primary_document: str  # the ownership XML, e.g. "wk-form4_1715...xml"

    @property
    def url(self) -> str:
        """Absolute URL of the primary (XML) document on EDGAR."""
        acc = self.accession_number.replace("-", "")
        return f"{_SEC_ARCHIVES}/{int(self.cik)}/{acc}/{self.primary_document}"

    @property
    def filing_id(self) -> str:
        """Stable id for this filing (ticker:4:date:accession)."""
        return f"{self.ticker}:4:{self.filing_date.isoformat()}:{self.accession_number}"


class InsiderTransaction(BaseModel):
    """One non-derivative/derivative transaction line parsed from a Form 4.

    ``shares`` and ``price_per_share`` are as reported; ``acquired_disposed`` is
    "A" (acquired) or "D" (disposed); ``code`` is the SEC transaction code (P/S/A/M/…).
    ``filing_date`` is the point-in-time public date (carried from the submissions
    index, not the XML).
    """

    ticker: str
    filing_date: Date
    transaction_date: Date | None
    code: str
    acquired_disposed: str  # "A" or "D"
    shares: float
    price_per_share: float | None
    is_derivative: bool
    owner_name: str | None = None

    @property
    def signed_value(self) -> float:
        """Signed dollar value: + for acquisitions, − for disposals (0 if no price).

        Uses the acquired/disposed code for sign so it is robust across transaction
        codes; callers restrict to open-market P/S before summing if they want the
        discretionary-only signal.
        """
        if self.price_per_share is None:
            return 0.0
        sign = 1.0 if self.acquired_disposed == "A" else -1.0
        return sign * self.shares * self.price_per_share
