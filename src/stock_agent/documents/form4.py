"""Pure parser for SEC Form 4 ownership XML → ``InsiderTransaction`` records.

Form 4 follows the SEC ``ownershipDocument`` schema: non-derivative and derivative
transaction tables, each row carrying a transaction code, date, share count, price,
and an acquired/disposed flag. This module is I/O-free and deterministic (testable
against a recorded fixture), mirroring the ``sec_edgar`` pure-parser convention.

Robustness: real-world Form 4s vary (missing prices on gifts, holdings rows with no
transaction, occasional XML namespaces). We skip rows lacking the minimal fields
rather than raise, so one malformed line never drops an entire filing.
"""

from __future__ import annotations

import re
from datetime import date as Date
from xml.etree import ElementTree as ET

from stock_agent.logging_config import get_logger
from stock_agent.schemas.insider import InsiderTransaction

log = get_logger(__name__)

_TRUTHY = {"1", "true", "yes", "y"}
# Rule 10b5-1 disclosure varies by filing vintage: a structured flag (post-2023,
# tag containing "10b5") or — more commonly, historically — a footnote whose text
# names the plan. We detect both; matches "10b5-1", "10b5 1", "10b51".
_RULE_10B5_1 = re.compile(r"10b5[\s\-]?1", re.IGNORECASE)


def _strip_ns(root: ET.Element) -> None:
    """Drop XML namespaces in-place (some filers emit a namespaced ownershipDocument).

    Local tag names are all we match on, so collapsing ``{ns}tag`` → ``tag`` keeps
    the find-paths simple and namespace-agnostic.
    """
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.rsplit("}", 1)[1]


def _text(el: ET.Element | None, path: str) -> str | None:
    """``el.findtext(path)`` trimmed to None if empty/missing."""
    if el is None:
        return None
    val = el.findtext(path)
    val = val.strip() if val is not None else None
    return val or None


def _float(el: ET.Element | None, path: str) -> float | None:
    raw = _text(el, path)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bool(el: ET.Element | None, path: str) -> bool:
    """Parse a Form 4 boolean ('1'/'0'/'true'/'false'); False if missing/empty."""
    val = _text(el, path)
    return val is not None and val.strip().lower() in _TRUTHY


def _parse_date(raw: str | None) -> Date | None:
    if not raw:
        return None
    try:
        return Date.fromisoformat(raw[:10])
    except ValueError:
        return None


class _Owner:
    """Reporting-owner attributes parsed once per filing (one owner per Form 4)."""

    __slots__ = ("name", "cik", "is_officer", "is_director", "is_ten_pct", "title")

    def __init__(self, root: ET.Element) -> None:
        rel = "reportingOwner/reportingOwnerRelationship/"
        self.name = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
        self.cik = _text(root, "reportingOwner/reportingOwnerId/rptOwnerCik")
        self.is_officer = _bool(root, rel + "isOfficer")
        self.is_director = _bool(root, rel + "isDirector")
        self.is_ten_pct = _bool(root, rel + "isTenPercentOwner")
        self.title = _text(root, rel + "officerTitle")


def _footnote_map(root: ET.Element) -> dict[str, str]:
    """Map footnote id → text (``<footnote id="F1">…</footnote>``)."""
    out: dict[str, str] = {}
    for fn in root.findall("footnotes/footnote"):
        fid = fn.get("id")
        if fid:
            out[fid] = (fn.text or "").strip()
    return out


def _is_10b5_1(tx: ET.Element, footnotes: dict[str, str], *, doc_planned: bool) -> bool:
    """True if this transaction was made under a Rule 10b5-1 plan (non-discretionary).

    Document-level structured flag (newer filings) OR any footnote referenced by
    this transaction whose text names a 10b5-1 plan (the historical disclosure).
    """
    if doc_planned:
        return True
    for ref in tx.findall(".//footnoteId"):
        fid = ref.get("id")
        if fid and _RULE_10B5_1.search(footnotes.get(fid, "")):
            return True
    return False


def _doc_level_10b5_1(root: ET.Element) -> bool:
    """Best-effort document-level 10b5-1 flag (a truthy element whose tag names it)."""
    for el in root.iter():
        if "10b5" in el.tag.lower() and (el.text or "").strip().lower() in _TRUTHY:
            return True
    return False


def _parse_transaction(
    tx: ET.Element,
    *,
    ticker: str,
    filing_date: Date,
    owner: _Owner,
    is_derivative: bool,
    footnotes: dict[str, str],
    doc_planned: bool,
) -> InsiderTransaction | None:
    """Build one ``InsiderTransaction`` from a transaction element (None if unusable)."""
    code = _text(tx, "transactionCoding/transactionCode")
    shares = _float(tx, "transactionAmounts/transactionShares/value")
    ad = _text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
    # A transaction line is only meaningful with a code, a share count, and a
    # direction; holdings rows (no transaction) lack these and are skipped.
    if code is None or shares is None or ad is None:
        return None
    return InsiderTransaction(
        ticker=ticker,
        filing_date=filing_date,
        transaction_date=_parse_date(_text(tx, "transactionDate/value")),
        code=code,
        acquired_disposed=ad,
        shares=shares,
        price_per_share=_float(tx, "transactionAmounts/transactionPricePerShare/value"),
        is_derivative=is_derivative,
        owner_name=owner.name,
        owner_cik=owner.cik,
        is_officer=owner.is_officer,
        is_director=owner.is_director,
        is_ten_pct_owner=owner.is_ten_pct,
        officer_title=owner.title,
        shares_owned_after=_float(
            tx, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        ),
        is_planned_10b5_1=_is_10b5_1(tx, footnotes, doc_planned=doc_planned),
    )


def parse_form4_xml(xml_text: str, *, ticker: str, filing_date: Date) -> list[InsiderTransaction]:
    """Parse Form 4 ownership XML into transaction records.

    ``filing_date`` is supplied by the caller (from the submissions index) because
    it is the point-in-time public date and is not reliably present in the XML.
    Returns an empty list on unparseable XML (logged), never raises.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("form4.parse_failed", ticker=ticker, error=str(exc))
        return []
    _strip_ns(root)

    owner = _Owner(root)
    footnotes = _footnote_map(root)
    doc_planned = _doc_level_10b5_1(root)
    out: list[InsiderTransaction] = []
    for path, is_deriv in (
        ("nonDerivativeTable/nonDerivativeTransaction", False),
        ("derivativeTable/derivativeTransaction", True),
    ):
        for tx in root.findall(path):
            rec = _parse_transaction(
                tx, ticker=ticker, filing_date=filing_date, owner=owner,
                is_derivative=is_deriv, footnotes=footnotes, doc_planned=doc_planned,
            )
            if rec is not None:
                out.append(rec)
    return out
