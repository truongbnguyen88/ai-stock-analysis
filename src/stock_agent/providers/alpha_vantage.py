"""Alpha Vantage provider — price fallback + news with sentiment.

Free tier is limited to ~25 requests/day, so this sits LAST in both chains and is
only active when ALPHA_VANTAGE_API_KEY is set. It is kept because its
NEWS_SENTIMENT endpoint uniquely returns per-article sentiment scores, useful as
model features later (Phase 5).

Endpoints (function= query param on https://www.alphavantage.co/query):
  - TIME_SERIES_DAILY  -> {"Time Series (Daily)": {date: {"1. open", ... "5. volume"}}}
  - NEWS_SENTIMENT     -> {"feed": [{title, url, time_published, overall_sentiment_score, ...}]}
Rate-limit / error responses arrive as 200 with a "Note"/"Information"/"Error Message" key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from typing import Any

from stock_agent.providers._cache import DiskCache, cached_model, make_key
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderRateLimit, SymbolNotFound
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings

_NAME = "alpha_vantage"
_URL = "https://www.alphavantage.co/query"


def _raise_for_av_messages(payload: dict[str, Any]) -> None:
    """Translate Alpha Vantage's soft-error payloads into typed errors."""
    if "Note" in payload or "Information" in payload:
        # AV returns these when the daily/minute call quota is exceeded.
        raise ProviderRateLimit(_NAME, str(payload.get("Note") or payload.get("Information")))
    if "Error Message" in payload:
        raise SymbolNotFound(_NAME, str(payload["Error Message"]))


def _prices_from_payload(
    ticker: str, payload: dict[str, Any], start: Date, end: Date
) -> PriceSeries:
    """Normalize TIME_SERIES_DAILY into a date-ascending ``PriceSeries`` within range."""
    _raise_for_av_messages(payload)
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        raise SymbolNotFound(_NAME, f"no price data for '{ticker}'")

    bars: list[PriceBar] = []
    for day_str, ohlc in series.items():
        day = Date.fromisoformat(day_str)
        if day < start or day > end:  # AV returns a long history; clip to window
            continue
        bars.append(
            PriceBar(
                date=day,
                open=float(ohlc["1. open"]),
                high=float(ohlc["2. high"]),
                low=float(ohlc["3. low"]),
                close=float(ohlc["4. close"]),
                adj_close=None,  # adjusted close is a premium endpoint
                volume=int(float(ohlc["5. volume"])),
            )
        )
    if not bars:
        raise SymbolNotFound(_NAME, f"no price rows in range for '{ticker}'")
    bars.sort(key=lambda b: b.date)  # AV lists newest-first; schema wants ascending
    return PriceSeries(ticker=ticker, bars=bars)


def _news_from_payload(ticker: str, payload: dict[str, Any]) -> NewsBundle:
    """Normalize NEWS_SENTIMENT into a ``NewsBundle`` (with article sentiment)."""
    _raise_for_av_messages(payload)
    feed = payload.get("feed")
    if not isinstance(feed, list):
        return NewsBundle(ticker=ticker, articles=[])

    articles: list[Article] = []
    for item in feed:
        url = item.get("url")
        title = item.get("title")
        published = item.get("time_published")  # format: YYYYMMDDTHHMMSS
        if not url or not title or not published:
            continue
        score = item.get("overall_sentiment_score")
        articles.append(
            Article(
                title=str(title),
                url=str(url),
                source=str(item.get("source") or _NAME),
                published_at=datetime.strptime(str(published), "%Y%m%dT%H%M%S").replace(tzinfo=UTC),
                summary=item.get("summary") or None,
                sentiment=float(score) if score is not None else None,
            )
        )
    return NewsBundle(ticker=ticker, articles=articles)


class AlphaVantageProvider:
    """``PriceProvider`` + ``NewsProvider`` backed by Alpha Vantage (key-gated)."""

    name = _NAME

    def __init__(self, settings: Settings, cache: DiskCache, http: HttpJson | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._http = http or HttpJson(_NAME)

    def available(self) -> bool:
        return bool(self._settings.alpha_vantage_api_key)

    def _query(self, params: dict[str, Any]) -> dict[str, Any]:
        token = self._settings.require("alpha_vantage_api_key", capability="Alpha Vantage")
        result = self._http.get(_URL, params={**params, "apikey": token})
        return result if isinstance(result, dict) else {}

    def get_daily_prices(self, ticker: str, start: Date, end: Date) -> PriceSeries:
        key = make_key(_NAME, "prices", ticker.upper(), start, end)
        return cached_model(
            self._cache,
            key,
            PriceSeries,
            lambda: _prices_from_payload(
                ticker.upper(),
                # outputsize=compact (last ~100 trading days) is the free-tier limit;
                # "full" is premium. AV is only the price *fallback* (yfinance is
                # primary and serves longer histories), so compact is sufficient.
                self._query(
                    {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": ticker.upper(),
                        "outputsize": "compact",
                    }
                ),
                start,
                end,
            ),
        )

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        key = make_key(_NAME, "news", ticker.upper(), start, end)
        return cached_model(
            self._cache,
            key,
            NewsBundle,
            lambda: _news_from_payload(
                ticker.upper(),
                self._query(
                    {
                        "function": "NEWS_SENTIMENT",
                        "tickers": ticker.upper(),
                        "time_from": start.strftime("%Y%m%dT0000"),
                        "time_to": end.strftime("%Y%m%dT2359"),
                    }
                ),
            ),
            ttl=self._settings.cache_ttl_news_seconds,
        )

    def close(self) -> None:
        self._http.close()
