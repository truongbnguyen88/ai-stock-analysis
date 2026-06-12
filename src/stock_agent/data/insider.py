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

from collections.abc import Sequence
from datetime import date as Date

import pandas as pd

from stock_agent.documents.form4 import parse_form4_xml
from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.sec_edgar import SecEdgarProvider
from stock_agent.schemas.insider import OPEN_MARKET_BUY, OPEN_MARKET_SELL, InsiderTransaction
from stock_agent.settings import Settings

log = get_logger(__name__)


def build_sec_provider(settings: Settings) -> SecEdgarProvider | None:
    """Construct a ``SecEdgarProvider`` for insider fetches, or None if unconfigured.

    SEC requires a descriptive User-Agent; without it we cannot make EDGAR calls, so
    callers should treat None as "insider data unavailable" (features become NaN).
    """
    from stock_agent.providers._cache import DiskCache

    provider = SecEdgarProvider(settings, DiskCache(settings.cache_dir, settings.cache_ttl_seconds))
    return provider if provider.available() else None

# Columns of the daily activity frame (indexed by filing_date).
ACTIVITY_COLS: list[str] = ["net_value", "n_buys", "n_sells"]


def aggregate_transactions(transactions: Sequence[InsiderTransaction]) -> pd.DataFrame:
    """Aggregate parsed transactions into a per-``filing_date`` activity frame. Pure.

    Restricts to open-market purchases/sales (codes P/S) — the discretionary,
    information-bearing trades — and sums their signed dollar value plus buy/sell
    counts. Returns an empty (typed) frame when there are no qualifying trades.
    """
    rows = [
        {
            "filing_date": t.filing_date,
            "net_value": t.signed_value,
            "n_buys": 1 if t.code == OPEN_MARKET_BUY else 0,
            "n_sells": 1 if t.code == OPEN_MARKET_SELL else 0,
        }
        for t in transactions
        if t.code in (OPEN_MARKET_BUY, OPEN_MARKET_SELL)
    ]
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
    provider: SecEdgarProvider, ticker: str, *, since: Date | None = None, limit: int = 200
) -> pd.DataFrame:
    """Fetch + parse + aggregate one ticker's Form 4 history into a daily frame.

    Empty frame (never raises) on any provider/parse failure, so dependent features
    simply become NaN/0 rather than breaking the pipeline.
    """
    try:
        refs = provider.list_form4_filings(ticker, since=since, limit=limit)
    except ProviderError as exc:
        log.warning("insider.list_failed", ticker=ticker, error=str(exc))
        return _empty_activity()

    transactions: list[InsiderTransaction] = []
    for ref in refs:
        try:
            xml = provider.download_form4(ref)
        except ProviderError as exc:
            log.warning("insider.download_failed", filing=ref.filing_id, error=str(exc))
            continue
        transactions.extend(
            parse_form4_xml(xml, ticker=ref.ticker, filing_date=ref.filing_date)
        )
    return aggregate_transactions(transactions)


def fetch_insider_by_ticker(
    provider: SecEdgarProvider, tickers: Sequence[str], *, since: Date | None = None
) -> dict[str, pd.DataFrame]:
    """Per-ticker insider activity frames for a universe (empty frames where absent)."""
    return {t.upper(): fetch_insider_activity(provider, t, since=since) for t in tickers}
