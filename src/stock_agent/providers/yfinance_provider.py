"""yfinance price provider — keyless historical daily prices.

yfinance pulls from Yahoo Finance's JSON endpoints (not HTML scraping) and needs
no API key, which makes it the default primary price source. We request
unadjusted OHLC plus the adjusted close so both ``close`` and ``adj_close`` are
available downstream.

Network access is confined to ``_download``; the DataFrame -> PriceSeries
normalization is a pure function (``_frame_to_series``) so it can be unit-tested
without hitting the network.
"""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import timedelta

import pandas as pd
import yfinance as yf

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers.base import ProviderError, SymbolNotFound
from stock_agent.schemas.market import PriceBar, PriceSeries

_NAME = "yfinance"


def _frame_to_series(ticker: str, frame: pd.DataFrame) -> PriceSeries:
    """Normalize a yfinance OHLCV DataFrame into a ``PriceSeries``.

    Expects columns Open/High/Low/Close/Volume and optionally 'Adj Close',
    indexed by date. Pure function — no I/O.
    """
    if frame is None or frame.empty:
        raise SymbolNotFound(_NAME, f"no price data for '{ticker}'")

    has_adj = "Adj Close" in frame.columns
    bars: list[PriceBar] = []
    for idx, row in frame.iterrows():
        # idx may be a Timestamp; normalize to a date.
        bar_date: Date = pd.Timestamp(idx).date()
        # Yahoo occasionally emits NaN rows (e.g. halted days); skip them.
        if pd.isna(row["Open"]) or pd.isna(row["Close"]):
            continue
        op, cl = float(row["Open"]), float(row["Close"])
        hi_raw, lo_raw = float(row["High"]), float(row["Low"])
        # Split/dividend adjustment occasionally rounds open/close a cent outside the
        # reported [low, high] (real, harmless artifact). Widen the band to bound them
        # so the bar is internally consistent (PriceBar requires low <= open,close <= high).
        high = max(op, cl, hi_raw, lo_raw)
        low = min(op, cl, hi_raw, lo_raw)
        bars.append(
            PriceBar(
                date=bar_date,
                open=op,
                high=high,
                low=low,
                close=cl,
                adj_close=float(row["Adj Close"]) if has_adj else None,
                volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            )
        )

    if not bars:
        raise SymbolNotFound(_NAME, f"no usable price rows for '{ticker}'")
    return PriceSeries(ticker=ticker, bars=bars)


class YFinanceProvider:
    """``PriceProvider`` backed by yfinance."""

    name = _NAME

    def __init__(self, cache: DiskCache) -> None:
        self._cache = cache

    def available(self) -> bool:
        # Keyless source: always usable (network failures surface per-call).
        return True

    def _download(self, ticker: str, start: Date, end: Date) -> pd.DataFrame:
        """Fetch raw OHLCV from yfinance (the only network call)."""
        # yfinance treats `end` as exclusive, so add a day to include it.
        # auto_adjust=False keeps raw OHLC and adds an 'Adj Close' column.
        try:
            frame = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any yfinance/network error
            raise ProviderError(_NAME, f"download failed: {exc}") from exc

        # yfinance may return multi-indexed columns for a single ticker; flatten.
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame

    def get_daily_prices(self, ticker: str, start: Date, end: Date) -> PriceSeries:
        key = make_key(_NAME, "prices", ticker.upper(), start, end)
        return cached_model(
            self._cache,
            key,
            PriceSeries,
            lambda: _frame_to_series(ticker.upper(), self._download(ticker, start, end)),
        )

    def get_earnings_dates(self, ticker: str) -> list[Date]:
        """Earnings announcement dates (past + upcoming) from yfinance, ascending.

        Cached as ISO date strings (the disk cache stores JSON). Requires ``lxml``
        (yfinance scrapes the earnings table). Returns [] if none are available.
        """
        key = make_key(_NAME, "earnings_dates", ticker.upper())
        cached = self._cache.get(key)
        if cached is not None:
            return [Date.fromisoformat(d) for d in json.loads(cached)]

        try:
            df = yf.Ticker(ticker.upper()).get_earnings_dates(limit=40)
        except Exception as exc:  # noqa: BLE001 - normalize any yfinance/network error
            raise ProviderError(_NAME, f"earnings fetch failed: {exc}") from exc

        dates: list[Date] = []
        if df is not None and not df.empty:
            dates = sorted({pd.Timestamp(idx).date() for idx in df.index})
        self._cache.set(key, json.dumps([d.isoformat() for d in dates]))
        return dates
