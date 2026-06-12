"""SEC EDGAR provider — official-API client for filings (RAG P1).

Uses SEC's **official, free, keyless** endpoints (NOT scraping):
- ticker -> CIK : https://www.sec.gov/files/company_tickers.json
- filing list   : https://data.sec.gov/submissions/CIK##########.json
- filing doc    : https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}

SEC fair-access policy requires a descriptive ``User-Agent`` (name + contact) and
caps traffic at ~10 requests/second, so every call sets the UA from settings and
passes through a throttle. JSON indexes are disk-cached; raw filing documents are
persisted by the ``documents`` layer (raw is never re-fetched once on disk).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import date as Date

from stock_agent.providers._cache import DiskCache, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderUnavailable, SymbolNotFound
from stock_agent.schemas.documents import DocumentType, FilingRef
from stock_agent.schemas.insider import InsiderFilingRef
from stock_agent.settings import Settings

_NAME = "sec_edgar"
_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_SEC_MAX_RPS = 10.0  # SEC fair-access cap


def _parse_cik_map(payload: object) -> dict[str, str]:
    """Normalize ``company_tickers.json`` into ``{TICKER -> 10-digit CIK}``. Pure.

    Payload shape: ``{"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}``.
    """
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        ticker, cik = entry.get("ticker"), entry.get("cik_str")
        if ticker is None or cik is None:
            continue
        try:
            out[str(ticker).strip().upper()] = f"{int(cik):010d}"
        except (TypeError, ValueError):
            continue
    return out


def _parse_filings(
    ticker: str,
    cik: str,
    payload: object,
    *,
    forms: Sequence[str],
    limit: int,
    since: Date | None = None,
) -> list[FilingRef]:
    """Extract newest-first ``FilingRef``s for the requested forms. Pure.

    The submissions index stores filings as PARALLEL arrays under
    ``filings.recent`` (form[i], filingDate[i], accessionNumber[i],
    primaryDocument[i]). We keep only exact form matches (so "10-K/A" amendments
    are excluded when "10-K" is requested), skip malformed rows, **drop filings older
    than ``since``** (inclusive date floor; ``None`` = no floor), sort newest-first,
    and cap to ``limit`` (a safety ceiling within the date window).

    Note: only ``filings.recent`` (the most-recent ~1000 filings) is read — more than
    enough for a 2–3yr 10-K/10-Q/8-K window; complete deep history would need EDGAR's
    older ``filings.files`` shards (out of scope).
    """
    recent: object = payload
    for key in ("filings", "recent"):
        recent = recent.get(key, {}) if isinstance(recent, dict) else {}
    if not isinstance(recent, dict):
        return []
    form_arr = recent.get("form") or []
    date_arr = recent.get("filingDate") or []
    acc_arr = recent.get("accessionNumber") or []
    doc_arr = recent.get("primaryDocument") or []
    wanted = set(forms)

    refs: list[FilingRef] = []
    for form, fdate, acc, doc in zip(form_arr, date_arr, acc_arr, doc_arr, strict=False):
        if form not in wanted or not (fdate and acc and doc):
            continue
        try:
            filing_date = Date.fromisoformat(str(fdate))
        except ValueError:
            continue
        if since is not None and filing_date < since:
            continue  # outside the requested history window
        refs.append(
            FilingRef(
                ticker=ticker,
                cik=cik,
                form=form,  # filtered to wanted ⊆ DocumentType (validated by pydantic)
                filing_date=filing_date,
                accession_number=str(acc),
                primary_document=str(doc),
            )
        )
    refs.sort(key=lambda r: r.filing_date, reverse=True)
    return refs[:limit]


def _parse_form4_filings(
    ticker: str,
    cik: str,
    payload: object,
    *,
    limit: int,
    since: Date | None = None,
) -> list[InsiderFilingRef]:
    """Extract newest-first Form 4 (``form == "4"``) filing refs. Pure.

    Same parallel-array shape as ``_parse_filings`` but yields ``InsiderFilingRef``
    (form fixed to "4", outside the RAG ``DocumentType``). Excludes amendments
    ("4/A") by exact match; applies the inclusive ``since`` floor and ``limit``.
    """
    recent: object = payload
    for key in ("filings", "recent"):
        recent = recent.get(key, {}) if isinstance(recent, dict) else {}
    if not isinstance(recent, dict):
        return []
    form_arr = recent.get("form") or []
    date_arr = recent.get("filingDate") or []
    acc_arr = recent.get("accessionNumber") or []
    doc_arr = recent.get("primaryDocument") or []

    refs: list[InsiderFilingRef] = []
    for form, fdate, acc, doc in zip(form_arr, date_arr, acc_arr, doc_arr, strict=False):
        if form != "4" or not (fdate and acc and doc):
            continue
        try:
            filing_date = Date.fromisoformat(str(fdate))
        except ValueError:
            continue
        if since is not None and filing_date < since:
            continue
        refs.append(
            InsiderFilingRef(
                ticker=ticker,
                cik=cik,
                filing_date=filing_date,
                accession_number=str(acc),
                primary_document=str(doc),
            )
        )
    refs.sort(key=lambda r: r.filing_date, reverse=True)
    return refs[:limit]


class SecEdgarProvider:
    """Official EDGAR client: ticker->CIK, filing list, and filing download."""

    name = _NAME

    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        http: HttpJson | None = None,
        *,
        min_request_interval: float = 1.0 / _SEC_MAX_RPS,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http  # injected in tests; built lazily (with UA) otherwise
        self._min_interval = min_request_interval
        self._last_request = 0.0

    def available(self) -> bool:
        return bool(self._settings.sec_user_agent)

    def _client(self) -> HttpJson:
        if self._http is None:
            ua = self._settings.require("sec_user_agent", capability="SEC EDGAR downloads")
            self._http = HttpJson(
                _NAME, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
            )
        return self._http

    def _throttle(self) -> None:
        """Block until at least ``min_request_interval`` has elapsed since the last call."""
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _cik_map(self) -> dict[str, str]:
        key = make_key(_NAME, "cik_map")
        raw = self._cache.get(key)
        if raw is not None:
            cached: dict[str, str] = json.loads(raw)
            return cached
        self._throttle()
        payload = self._client().get(_CIK_MAP_URL, params={})
        mapping = _parse_cik_map(payload)
        if not mapping:
            raise ProviderUnavailable(_NAME, "empty or unparseable company_tickers.json")
        self._cache.set(key, json.dumps(mapping))
        return mapping

    def get_cik(self, ticker: str) -> str:
        """Return the 10-digit CIK for ``ticker`` (raises ``SymbolNotFound`` if unknown)."""
        sym = ticker.strip().upper()
        cik = self._cik_map().get(sym)
        if cik is None:
            raise SymbolNotFound(_NAME, f"no CIK on file for ticker {sym!r}")
        return cik

    def list_filings(
        self,
        ticker: str,
        forms: Sequence[DocumentType],
        *,
        limit: int = 10,
        since: Date | None = None,
    ) -> list[FilingRef]:
        """Return newest-first filings of the requested ``forms`` for ``ticker``.

        ``since`` is an inclusive date floor (e.g. 3 years ago for a bulk history pull);
        ``limit`` caps the count within that window.
        """
        sym = ticker.strip().upper()
        cik = self.get_cik(sym)
        payload = self._submissions(cik)
        return _parse_filings(sym, cik, payload, forms=forms, limit=limit, since=since)

    def list_form4_filings(
        self, ticker: str, *, limit: int = 200, since: Date | None = None
    ) -> list[InsiderFilingRef]:
        """Return newest-first Form 4 (insider-transaction) filings for ``ticker``.

        Reuses the cached submissions index. ``limit`` defaults high (insiders file
        frequently) so a multi-year ``since`` window isn't truncated prematurely.
        """
        sym = ticker.strip().upper()
        cik = self.get_cik(sym)
        payload = self._submissions(cik)
        return _parse_form4_filings(sym, cik, payload, limit=limit, since=since)

    def _submissions(self, cik: str) -> object:
        """Fetch (or cache-hit) the EDGAR submissions index for ``cik``."""
        key = make_key(_NAME, "submissions", cik)
        raw = self._cache.get(key)
        if raw is not None:
            return json.loads(raw)
        self._throttle()
        payload = self._client().get(_SUBMISSIONS_URL.format(cik10=cik), params={})
        self._cache.set(key, json.dumps(payload))
        return payload

    def download_filing(self, ref: FilingRef) -> str:
        """Fetch the raw primary-document text (HTML/TXT) for ``ref``."""
        self._throttle()
        return self._client().get_text(ref.url)

    def download_form4(self, ref: InsiderFilingRef) -> str:
        """Fetch the raw Form 4 ownership-XML text for ``ref`` (disk-cached by URL).

        Form 4 XML is immutable once filed, so caching is safe and avoids re-downloading
        the same documents on every walk-forward as-of (insiders file frequently).
        """
        key = make_key(_NAME, "form4_xml", ref.url)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self._throttle()
        text = self._client().get_text(ref.url)
        self._cache.set(key, text)
        return text

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
