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

import re
from datetime import date as Date

from pydantic import BaseModel

_SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
# EDGAR's submissions index lists the Form 4 primaryDocument as the XSL-RENDERED
# view (e.g. "xslF345X06/form4.xml") — that path serves HTML, not parseable XML.
# The raw ownershipDocument XML lives at the same filename with this rendering
# directory stripped. We de-render in the URL so downloads get the machine XML.
_XSL_RENDER_DIR = re.compile(r"^xslF[0-9A-Z]+/")

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
        """Absolute URL of the RAW ownership-XML document on EDGAR.

        Strips the ``xslF.../`` rendering-directory prefix EDGAR puts on the Form 4
        primaryDocument, so the fetched file is the machine-readable XML (not the
        rendered HTML view). No-op when the document is already un-prefixed.
        """
        acc = self.accession_number.replace("-", "")
        doc = _XSL_RENDER_DIR.sub("", self.primary_document)
        return f"{_SEC_ARCHIVES}/{int(self.cik)}/{acc}/{doc}"

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
    # --- Enrichment for sharper feature engineering (Phase 1.6 re-engineering) ---
    owner_cik: str | None = None  # reporting-owner CIK → distinct-insider (cluster) counts
    is_officer: bool = False
    is_director: bool = False
    is_ten_pct_owner: bool = False
    officer_title: str | None = None  # e.g. "CFO", "Chief Executive Officer"
    shares_owned_after: float | None = None  # post-transaction holdings → conviction (Δ-ownership)
    is_planned_10b5_1: bool = False  # Rule 10b5-1 scheduled trade (non-discretionary → filter out)

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

    @property
    def prior_holdings(self) -> float | None:
        """Shares held immediately BEFORE this trade (post ∓ traded), or None if unknown."""
        if self.shares_owned_after is None:
            return None
        delta = self.shares if self.acquired_disposed == "A" else -self.shares
        return self.shares_owned_after - delta

    @property
    def ownership_change_fraction(self) -> float | None:
        """Fractional change in the insider's own position from this trade (conviction).

        ``± shares / prior_holdings`` (sign from acquired/disposed). This is the
        scale-free "how much did the insider move their *own* stake" measure that
        the dollar-value features miss. None when prior holdings are unknown or
        non-positive (e.g. a brand-new position: prior = 0 → ratio undefined).
        """
        prior = self.prior_holdings
        if prior is None or prior <= 0:
            return None
        sign = 1.0 if self.acquired_disposed == "A" else -1.0
        return sign * self.shares / prior

    @property
    def is_senior(self) -> bool:
        """CEO/CFO (or equivalent) by officer title — the most predictive insiders.

        Matches the two roles the insider-trading literature finds most informative;
        kept deliberately narrow (not every VP/"officer").
        """
        if not self.officer_title:
            return False
        t = self.officer_title.lower()
        return any(
            k in t for k in ("chief executive", "ceo", "chief financial", "cfo")
        )
