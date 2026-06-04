"""Source-agnostic aggregation of GDELT GKG records into daily news features.

This is the **reference semantics** for the daily news-sentiment store (Task 10,
news-as-model-feature). The primary ingest path (BigQuery) performs the
equivalent ``GROUP BY`` *server-side* so only the small aggregated daily rows are
downloaded — never article text (satisfies the "store features, not news"
constraint). This module implements the identical aggregation for the raw-file /
DOC-API fallback path AND is the tested oracle pinning those semantics.

Two daily streams come out (mirroring the VIX design in ``features``):
  - **per-ticker** — cross-sectional company sentiment, keyed by (date, ticker).
  - **market** — market-wide political / policy sentiment (same for every ticker
    on a date, like VIX), built from articles tagged with political / economic-
    policy-uncertainty / presidential themes.

Point-in-time discipline: each record carries the GDELT *monitoring* datetime —
when GDELT first observed the article, i.e. when the information became available.
We bucket by the UTC calendar day of that datetime. Leakage-safe alignment to
trading dates (and the conservative 1-day publication lag) happens downstream in
``features/news_history.py`` — this module only aggregates.

Raw daily aggregates (counts + tone) are stored as-is; scale-free, pooling-valid
features (e.g. a ``news_buzz`` spike ratio vs the ticker's own trailing mean) are
derived later, because they need each ticker's history and a trailing window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# GDELT V2Tone is in [-100, +100] but practically lands in [-10, +10]. Treat a
# small band around zero as neutral so pos/neg counts reflect genuinely-toned
# coverage, not rounding noise. Named so the SQL path can use the same cut.
NEUTRAL_TONE_BAND: float = 1.0

# Store schema (raw daily aggregates — reproducible, not yet scale-free).
PER_TICKER_COLS: list[str] = ["article_count", "tone_mean", "tone_std", "pos_count", "neg_count"]
MARKET_COLS: list[str] = [
    "pol_article_count",
    "pol_tone_mean",
    "pol_tone_std",
    "epu_count",
    "pres_article_count",
    "pres_tone_mean",
]


@dataclass(frozen=True, slots=True)
class GkgRecord:
    """One normalized GDELT GKG article at the aggregation boundary.

    ``tone`` is the V2Tone tone component (avg positive minus avg negative word
    score). ``tickers`` are the universe tickers this article was mapped to from
    its tagged organizations (may be empty → contributes only to the market
    stream if a theme flag is set). The boolean theme flags drive the market
    stream and are mutually non-exclusive.
    """

    dt: datetime
    tone: float
    tickers: tuple[str, ...] = ()
    political: bool = False
    epu: bool = False  # Economic Policy Uncertainty theme present
    presidential: bool = False


def _empty(cols: list[str], index: pd.Index) -> pd.DataFrame:
    """Typed empty frame with the given columns and (named) index."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}, index=index)


def _frame(records: list[GkgRecord]) -> pd.DataFrame:
    """Flatten records into a per-row DataFrame (one row per article)."""
    rows = [
        {
            "date": r.dt.date(),
            "tone": float(r.tone),
            "tickers": r.tickers,
            "political": r.political,
            "epu": r.epu,
            "presidential": r.presidential,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        # Pre-compute tone sign once (shared by both streams). Neutral band → 0.
        df["is_pos"] = (df["tone"] > NEUTRAL_TONE_BAND).astype(float)
        df["is_neg"] = (df["tone"] < -NEUTRAL_TONE_BAND).astype(float)
    return df


def _agg_block(g: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Aggregate one already-filtered group of articles into tone stats.

    ``tone_std`` uses sample std (ddof=1); a single-article group has no
    dispersion → 0.0 (not NaN), so downstream models see a clean number.
    """
    n = len(g)
    std = float(g["tone"].std(ddof=1)) if n > 1 else 0.0
    return {
        f"{prefix}article_count": float(n),
        f"{prefix}tone_mean": float(g["tone"].mean()),
        f"{prefix}tone_std": std,
        f"{prefix}pos_count": float(g["is_pos"].sum()),
        f"{prefix}neg_count": float(g["is_neg"].sum()),
    }


def aggregate_per_ticker(records: list[GkgRecord]) -> pd.DataFrame:
    """Daily per-ticker tone aggregates, indexed by a (date, ticker) MultiIndex.

    An article tagged with multiple tickers contributes to each of them (it is
    real coverage of each). Articles with no ticker mapping are dropped from this
    stream. Columns: ``PER_TICKER_COLS``.
    """
    midx = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    df = _frame(records)
    if df.empty:
        return _empty(PER_TICKER_COLS, midx)

    # Explode the per-article ticker tuple → one (article, ticker) row each.
    exploded = df.explode("tickers").dropna(subset=["tickers"])
    exploded = exploded.rename(columns={"tickers": "ticker"})
    if exploded.empty:
        return _empty(PER_TICKER_COLS, midx)

    out: dict[tuple[object, str], dict[str, float]] = {}
    for (day, ticker), g in exploded.groupby(["date", "ticker"], sort=True):
        out[(day, str(ticker))] = _agg_block(g, prefix="")
    result = pd.DataFrame.from_dict(out, orient="index")[PER_TICKER_COLS]
    result.index = pd.MultiIndex.from_tuples(result.index, names=["date", "ticker"])
    return result


def aggregate_market(records: list[GkgRecord]) -> pd.DataFrame:
    """Daily market-wide political / policy sentiment, indexed by date.

    Three sub-streams over the same days:
      - ``pol_*``  — all articles flagged ``political``.
      - ``epu_count`` — count of Economic-Policy-Uncertainty-flagged articles.
      - ``pres_*`` — articles flagged ``presidential`` (e.g. presidential remarks).

    A day with no political coverage at all is simply absent (downstream ffill
    carries the last known value). Columns: ``MARKET_COLS``.
    """
    idx: pd.Index = pd.Index([], name="date")
    df = _frame(records)
    if df.empty:
        return _empty(MARKET_COLS, idx)

    days = sorted({d for d in df["date"]})
    rows: dict[object, dict[str, float]] = {}
    for day in days:
        day_df = df[df["date"] == day]
        pol = day_df[day_df["political"]]
        pres = day_df[day_df["presidential"]]
        # Emit a row if the day had ANY macro-flagged coverage (political, EPU, or
        # presidential). A day with none is omitted → downstream ffill carries the
        # last known value. Each sub-stream's stats are 0 when that flag is absent.
        if pol.empty and pres.empty and not bool(day_df["epu"].any()):
            continue
        rows[day] = {
            "pol_article_count": float(len(pol)),
            "pol_tone_mean": float(pol["tone"].mean()) if not pol.empty else 0.0,
            "pol_tone_std": float(pol["tone"].std(ddof=1)) if len(pol) > 1 else 0.0,
            "epu_count": float(int(day_df["epu"].sum())),
            "pres_article_count": float(len(pres)),
            "pres_tone_mean": float(pres["tone"].mean()) if not pres.empty else 0.0,
        }
    if not rows:
        return _empty(MARKET_COLS, idx)
    result = pd.DataFrame.from_dict(rows, orient="index")[MARKET_COLS]
    result.index = pd.Index(list(rows.keys()), name="date")
    return result


@dataclass(frozen=True, slots=True)
class DailyAggregates:
    """Both daily streams produced from one batch of GKG records."""

    per_ticker: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    market: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


def aggregate(records: list[GkgRecord]) -> DailyAggregates:
    """Convenience: build both daily streams from one record batch."""
    return DailyAggregates(
        per_ticker=aggregate_per_ticker(records),
        market=aggregate_market(records),
    )
