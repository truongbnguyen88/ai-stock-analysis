"""Point-in-time news-history features from the GDELT daily-sentiment store (Task 10).

Turns the raw daily aggregates (``outputs/news_sentiment/``, produced by
``news.gdelt_ingest``) into **scale-free, leakage-safe** model features aligned to a
ticker's price dates. Two groups, mirroring the price/VIX feature design:

  - **per-ticker** (cross-sectional): ``news_buzz`` (today's article count vs the
    ticker's OWN trailing-window mean — a spike ratio, so a mega-cap's higher base
    coverage cancels out and cross-ticker pooling stays valid), ``news_tone`` (daily
    mean tone), ``news_pos_frac`` / ``news_neg_frac``.
  - **market-wide** (same for every ticker on a date, like VIX): ``pol_tone``
    (political coverage tone), ``epu_buzz`` (economic-policy-uncertainty volume
    spike), ``pres_tone`` (presidential coverage tone).

Leakage discipline (the correctness requirement for Task 10):
  - All daily series are filtered to ``<= cutoff`` (``as_of`` in backtests) BEFORE
    any rolling stat, and the buzz baseline uses ``.shift(1)`` (prior days only).
  - A conservative **publication lag** (default 1 day) shifts each day's news to be
    "available" only the NEXT day, then values are ffill-aligned onto price dates —
    so the feature at price date ``t`` reflects news through ``t - lag`` only.
  - Raw counts are never used cross-ticker; only the own-history buzz ratio is.

Missing data degrades gracefully to NaN (handled like any absent feature: LightGBM
natively, scikit-learn via the persisted imputer), exactly like the VIX path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd

from stock_agent.news.aggregate import MARKET_COLS, PER_TICKER_COLS
from stock_agent.news.gdelt_ingest import DEFAULT_STORE_DIR

# Topic/sector macro streams (shared across tickers on a date, like the market stream).
# Must match the topics produced by news.gdelt_ingest (TOPIC_THEMES + TOPIC_NAMES).
NEWS_TOPICS: list[str] = ["tech", "healthcare", "energy", "ai", "ai_infra"]

# Model feature columns contributed by news history (kept stable across train/infer).
NEWS_HISTORY_COLS: list[str] = [
    # per-ticker (scale-free / bounded)
    "news_buzz",
    "news_tone",
    "news_pos_frac",
    "news_neg_frac",
    # market-wide (shared across tickers on a date, like VIX)
    "pol_tone",
    "epu_buzz",
    "pres_tone",
    # topic/sector macro streams: tone + volume-spike per topic
    *(f"{t}_tone" for t in NEWS_TOPICS),
    *(f"{t}_buzz" for t in NEWS_TOPICS),
]

DEFAULT_BUZZ_WINDOW = 60  # trailing calendar days for the buzz baseline
DEFAULT_LAG_DAYS = 1  # conservative publication lag (news of day D available at D+1)


@dataclass(frozen=True)
class NewsStore:
    """Loaded daily news-sentiment store (per-ticker + market + topic streams)."""

    per_ticker: pd.DataFrame  # (date, ticker) MultiIndex → PER_TICKER_COLS
    market: pd.DataFrame  # date index → MARKET_COLS
    topics: pd.DataFrame  # (date, topic) MultiIndex → PER_TICKER_COLS

    def tickers(self) -> set[str]:
        if self.per_ticker.empty:
            return set()
        return set(self.per_ticker.index.get_level_values("ticker"))

    def topics_present(self) -> set[str]:
        if self.topics.empty:
            return set()
        return set(self.topics.index.get_level_values("topic"))

    @property
    def is_empty(self) -> bool:
        return bool(self.per_ticker.empty and self.market.empty and self.topics.empty)


def _empty_keyed(level: str) -> pd.DataFrame:
    midx = pd.MultiIndex.from_arrays([[], []], names=["date", level])
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PER_TICKER_COLS}, index=midx)


def _empty_market() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in MARKET_COLS}, index=pd.Index([], name="date")
    )


def load_news_store(store_dir: Path = DEFAULT_STORE_DIR) -> NewsStore:
    """Load the per-ticker + market + topics CSVs (empty typed frames if absent)."""
    store_dir = Path(store_dir)

    def _keyed(name: str, level: str) -> pd.DataFrame:
        p = store_dir / f"{name}.csv"
        if not p.exists():
            return _empty_keyed(level)
        return pd.read_csv(p, parse_dates=["date"]).set_index(["date", level]).sort_index()

    mk_path = store_dir / "market.csv"
    mk = (
        pd.read_csv(mk_path, parse_dates=["date"]).set_index("date").sort_index()
        if mk_path.exists()
        else _empty_market()
    )
    return NewsStore(
        per_ticker=_keyed("per_ticker", "ticker"),
        market=mk,
        topics=_keyed("topics", "topic"),
    )


def _buzz(count: pd.Series, window: int) -> pd.Series:
    """Spike ratio: today's count vs the trailing-window mean of PRIOR days.

    ``.shift(1)`` makes the baseline use only days strictly before ``t`` (leakage-safe).
    Scale-free, so a heavily-covered mega-cap and a quiet mid-cap are comparable.
    NaN early (baseline not yet defined) and where the baseline is 0.
    """
    baseline = count.rolling(window, min_periods=max(5, window // 4)).mean().shift(1)
    ratio = count / baseline
    return ratio.replace([float("inf"), -float("inf")], pd.NA).astype("float64")


def _continuous(daily: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Reindex a sparse date-indexed frame onto a continuous daily calendar up to cutoff.

    Missing days are real "no coverage" days: counts become 0 (so buzz reflects the
    lull), tone/frac stay NaN (no sentiment to report).
    """
    cal = pd.date_range(daily.index.min(), cutoff, freq="D")
    return daily.reindex(cal)


def _lag_align(feat: pd.DataFrame, price_index: pd.DatetimeIndex, lag_days: int) -> pd.DataFrame:
    """Shift each day's features forward by ``lag_days`` then ffill onto price dates.

    After the shift, the most recent value at price date ``t`` is day ``t - lag``'s —
    the conservative point-in-time alignment.
    """
    shifted = feat.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    return shifted.reindex(price_index, method="ffill")


def build_news_history_features(
    ticker: str,
    price_index: pd.DatetimeIndex,
    store: NewsStore,
    *,
    as_of: Date | None = None,
    lag_days: int = DEFAULT_LAG_DAYS,
    buzz_window: int = DEFAULT_BUZZ_WINDOW,
) -> pd.DataFrame:
    """Return a ``price_index``-aligned DataFrame of ``NEWS_HISTORY_COLS`` for ``ticker``.

    Leakage-safe: only news dated ``<= min(as_of, price_index.max())`` and then lagged
    by ``lag_days`` influences any row. Columns are NaN where the store has no data
    (unknown ticker, dates before coverage begins) — handled like any missing feature.
    """
    price_index = pd.DatetimeIndex(price_index)
    out = pd.DataFrame(index=price_index, columns=NEWS_HISTORY_COLS, dtype="float64")
    if len(price_index) == 0:
        return out
    cutoff = pd.Timestamp(as_of) if as_of is not None else price_index.max()

    # --- per-ticker stream ---
    if ticker in store.tickers():
        sub = store.per_ticker.xs(ticker, level="ticker")
        sub = sub[sub.index <= cutoff]
        if not sub.empty:
            d = _continuous(sub, cutoff)
            count = d["article_count"].fillna(0.0)
            feat = pd.DataFrame(index=d.index)
            feat["news_buzz"] = _buzz(count, buzz_window)
            feat["news_tone"] = d["tone_mean"]
            feat["news_pos_frac"] = (d["pos_count"] / count).where(count > 0)
            feat["news_neg_frac"] = (d["neg_count"] / count).where(count > 0)
            # Sentiment LEVELS carry forward over quiet days (last known read); only
            # ``news_buzz`` is genuinely 0 on a no-coverage day, so it is left as-is.
            feat[["news_tone", "news_pos_frac", "news_neg_frac"]] = feat[
                ["news_tone", "news_pos_frac", "news_neg_frac"]
            ].ffill()
            aligned = _lag_align(feat, price_index, lag_days)
            for col in ("news_buzz", "news_tone", "news_pos_frac", "news_neg_frac"):
                out[col] = aligned[col]

    # --- market stream (shared) ---
    mk = store.market[store.market.index <= cutoff]
    if not mk.empty:
        d = _continuous(mk, cutoff)
        feat = pd.DataFrame(index=d.index)
        feat["pol_tone"] = d["pol_tone_mean"]
        feat["pres_tone"] = d["pres_tone_mean"]
        feat["epu_buzz"] = _buzz(d["epu_count"].fillna(0.0), buzz_window)
        feat[["pol_tone", "pres_tone"]] = feat[["pol_tone", "pres_tone"]].ffill()
        aligned = _lag_align(feat, price_index, lag_days)
        for col in ("pol_tone", "epu_buzz", "pres_tone"):
            out[col] = aligned[col]

    # --- topic / sector macro streams (shared across tickers, like market) ---
    present = store.topics_present()
    for topic in NEWS_TOPICS:
        if topic not in present:
            continue  # columns stay NaN (e.g. AI topics not pulled)
        sub = store.topics.xs(topic, level="topic")
        sub = sub[sub.index <= cutoff]
        if sub.empty:
            continue
        d = _continuous(sub, cutoff)
        feat = pd.DataFrame(index=d.index)
        feat[f"{topic}_tone"] = d["tone_mean"].ffill()
        feat[f"{topic}_buzz"] = _buzz(d["article_count"].fillna(0.0), buzz_window)
        aligned = _lag_align(feat, price_index, lag_days)
        out[f"{topic}_tone"] = aligned[f"{topic}_tone"]
        out[f"{topic}_buzz"] = aligned[f"{topic}_buzz"]

    return out
