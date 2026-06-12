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

from datetime import date as Date
from xml.etree import ElementTree as ET

from stock_agent.logging_config import get_logger
from stock_agent.schemas.insider import InsiderTransaction

log = get_logger(__name__)


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


def _parse_date(raw: str | None) -> Date | None:
    if not raw:
        return None
    try:
        return Date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_transaction(
    tx: ET.Element, *, ticker: str, filing_date: Date, owner: str | None, is_derivative: bool
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
        owner_name=owner,
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

    owner = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    out: list[InsiderTransaction] = []
    for path, is_deriv in (
        ("nonDerivativeTable/nonDerivativeTransaction", False),
        ("derivativeTable/derivativeTransaction", True),
    ):
        for tx in root.findall(path):
            rec = _parse_transaction(
                tx, ticker=ticker, filing_date=filing_date, owner=owner, is_derivative=is_deriv
            )
            if rec is not None:
                out.append(rec)
    return out
