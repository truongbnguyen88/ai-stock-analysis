"""Insider-activity aggregation: Form 4 filings → point-in-time daily net signal.

Pipeline per ticker: list Form 4 filings (EDGAR) → download + parse each ownership
XML → keep only open-market discretionary trades (codes P/S) → aggregate by
``filing_date`` into a daily activity frame. ``filing_date`` (not transaction_date)
is the public/point-in-time anchor, so trailing-window features over this frame are
leakage-safe. Every step degrades gracefully: a failed ticker or filing yields an
empty/partial frame rather than breaking training.

The frame is fetched ONCE over a span (like VIX/SPY) and reindexed per fold by the
feature layer — Form 4 XML is never re-downloaded inside the fold loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import date as Date
from functools import partial
from typing import TypeVar

import pandas as pd

from stock_agent.documents.form4 import parse_form4_xml
from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError, SymbolNotFound
from stock_agent.providers.sec_edgar import SecEdgarProvider
from stock_agent.schemas.insider import OPEN_MARKET_BUY, OPEN_MARKET_SELL, InsiderTransaction
from stock_agent.settings import Settings

log = get_logger(__name__)

_T = TypeVar("_T")


def build_sec_provider(settings: Settings) -> SecEdgarProvider | None:
    """Construct a ``SecEdgarProvider`` for insider fetches, or None if unconfigured.

    SEC requires a descriptive User-Agent; without it we cannot make EDGAR calls, so
    callers should treat None as "insider data unavailable" (features become NaN).
    Per-request client — fine for low-volume single-ticker inference fetches. For
    BULK fetches (training / warm over a whole universe) use
    ``build_hardened_sec_provider``, which pools connections to avoid throttling.
    """
    from stock_agent.providers._cache import DiskCache

    provider = SecEdgarProvider(settings, DiskCache(settings.cache_dir, settings.cache_ttl_seconds))
    return provider if provider.available() else None


def build_hardened_sec_provider(
    settings: Settings, *, rps: float = 5.0, timeout: float = 30.0
) -> SecEdgarProvider | None:
    """SEC provider for BULK Form 4 fetches: pooled keep-alive client + polite rate.

    A fresh TLS handshake per request (the per-call-client default) trips EDGAR's
    fair-access throttling at universe scale (thousands of downloads). A single
    reused connection pool eliminates that. ``rps`` stays under SEC's 10 req/s
    ceiling. Returns None if SEC is unconfigured. CALLER MUST ``close()`` it.
    """
    import httpx

    from stock_agent.providers._cache import DiskCache
    from stock_agent.providers._http import HttpJson
    from stock_agent.providers.sec_edgar import _NAME

    ua = settings.sec_user_agent
    if not ua:
        return None
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    client = httpx.Client(
        timeout=timeout,
        headers=headers,
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )
    http = HttpJson(_NAME, client=client, headers=headers)
    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    return SecEdgarProvider(settings, cache, http=http, min_request_interval=1.0 / rps)


def with_retry(
    fn: Callable[[], _T], *, what: str, retries: int, base_backoff: float = 1.0
) -> _T:
    """Call ``fn`` with exponential backoff on transient ProviderError; reraise if exhausted.

    ``SymbolNotFound`` is not transient (e.g. an ETF with no CIK) and is reraised
    immediately. Cached calls (a warm cache hit) never fail, so this is a no-op there.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except SymbolNotFound:
            raise
        except ProviderError as exc:
            if attempt == retries:
                raise
            sleep = base_backoff * (2**attempt)
            log.warning("insider.retry", what=what, attempt=attempt + 1, error=str(exc))
            time.sleep(sleep)
    raise RuntimeError("unreachable")

# Columns of the daily activity frame (indexed by filing_date). The re-engineered
# signal (Phase 1.6) separates the BUY channel from the SELL channel — never nets
# them — because insider *buys* are informative while *sells* are mostly liquidity:
#   buy_conviction = Σ Δ-ownership of opportunistic buys (conviction, not dollars)
#   senior_buy_n   = count of opportunistic buys by CEO/CFO (the highest-signal subset)
#   sell_pressure  = Σ |Δ-ownership| of opportunistic sells (kept separate)
# Rule 10b5-1 (pre-scheduled, non-discretionary) trades are excluded entirely.
ACTIVITY_COLS: list[str] = ["buy_conviction", "senior_buy_n", "sell_pressure"]

# Cap per-trade Δ-ownership at 1.0 (a 100% increase / full exit) so a tiny prior
# holding can't produce an outlier fraction that dominates the pooled signal.
_CONVICTION_CAP = 1.0


def _conviction(t: InsiderTransaction) -> float:
    """Outlier-capped |Δ-ownership| for one trade; 1.0 when prior holdings unknown.

    An unknown prior (e.g. a brand-new position, prior = 0) is treated as maximal
    conviction rather than dropped — establishing a stake is a strong signal.
    """
    frac = t.ownership_change_fraction
    if frac is None:
        return _CONVICTION_CAP
    return min(abs(frac), _CONVICTION_CAP)


def aggregate_transactions(transactions: Sequence[InsiderTransaction]) -> pd.DataFrame:
    """Aggregate parsed transactions into a per-``filing_date`` activity frame. Pure.

    Keeps only open-market, **discretionary** trades: codes P (buy) / S (sell) that
    are NOT Rule 10b5-1 scheduled. Buys and sells go to separate columns (never
    netted). Returns an empty (typed) frame when no qualifying trades exist.
    """
    rows: list[dict[str, object]] = []
    for t in transactions:
        if t.is_planned_10b5_1:
            continue  # pre-scheduled, non-discretionary → not information-bearing
        if t.code == OPEN_MARKET_BUY and t.acquired_disposed == "A":
            rows.append({
                "filing_date": t.filing_date,
                "buy_conviction": _conviction(t),
                "senior_buy_n": 1.0 if t.is_senior else 0.0,
                "sell_pressure": 0.0,
            })
        elif t.code == OPEN_MARKET_SELL and t.acquired_disposed == "D":
            rows.append({
                "filing_date": t.filing_date,
                "buy_conviction": 0.0,
                "senior_buy_n": 0.0,
                "sell_pressure": _conviction(t),
            })
    if not rows:
        return _empty_activity()
    df = pd.DataFrame(rows)
    grouped = df.groupby("filing_date", as_index=True)[ACTIVITY_COLS].sum()
    grouped.index = pd.DatetimeIndex(grouped.index)
    return grouped.sort_index()


def _empty_activity() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in ACTIVITY_COLS},
        index=pd.DatetimeIndex([]),
    )


def fetch_insider_activity(
    provider: SecEdgarProvider,
    ticker: str,
    *,
    since: Date | None = None,
    limit: int = 200,
    retries: int = 0,
) -> pd.DataFrame:
    """Fetch + parse + aggregate one ticker's Form 4 history into a daily frame.

    Empty frame (never raises) on any provider/parse failure, so dependent features
    simply become NaN/0 rather than breaking the pipeline. ``retries`` retries each
    transient EDGAR call with backoff (use for bulk train/warm; default 0 = inference,
    where a miss just degrades that one forecast).
    """
    try:
        refs = with_retry(
            partial(provider.list_form4_filings, ticker, since=since, limit=limit),
            what=f"list:{ticker}", retries=retries,
        )
    except ProviderError as exc:
        log.warning("insider.list_failed", ticker=ticker, error=str(exc))
        return _empty_activity()

    transactions: list[InsiderTransaction] = []
    for ref in refs:
        try:
            xml = with_retry(
                partial(provider.download_form4, ref), what=f"dl:{ref.filing_id}",
                retries=retries,
            )
        except ProviderError as exc:
            log.warning("insider.download_failed", filing=ref.filing_id, error=str(exc))
            continue
        transactions.extend(
            parse_form4_xml(xml, ticker=ref.ticker, filing_date=ref.filing_date)
        )
    return aggregate_transactions(transactions)


def fetch_insider_by_ticker(
    provider: SecEdgarProvider,
    tickers: Sequence[str],
    *,
    since: Date | None = None,
    retries: int = 0,
) -> dict[str, pd.DataFrame]:
    """Per-ticker insider activity frames for a universe (empty frames where absent)."""
    return {
        t.upper(): fetch_insider_activity(provider, t, since=since, retries=retries)
        for t in tickers
    }
