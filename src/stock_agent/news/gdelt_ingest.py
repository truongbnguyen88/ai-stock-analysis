"""GDELT GKG → daily news-sentiment store, via Google BigQuery (Task 10 ingest).

Why BigQuery: the GDELT 2.0 Global Knowledge Graph (article-level tone + theme +
organization tags, 2015→present) is hosted free on BigQuery. We do the daily
``GROUP BY`` **server-side**, so only the small aggregated daily rows are
downloaded — never article text. This satisfies the project constraint of
storing news *features*, not news.

Cost discipline (the whole reason we use the partitioned table):
  - Query ``gdelt-bq.gdeltv2.gkg_partitioned`` and ALWAYS constrain
    ``_PARTITIONTIME`` to the requested window. BigQuery prunes to those
    partitions; without it every query scans the multi-TB full table.
  - Select only the columns each stream needs (billing is columns × partitions).
  - ``estimate_bytes`` runs a dry-run so the caller sees bytes-to-scan (and can
    chunk by year / spread across calendar months to stay under the 1 TiB/month
    free tier) BEFORE spending anything.

Two streams, matching ``news.aggregate`` semantics exactly (that module is the
tested oracle for what this SQL must compute):
  - **per-ticker** — articles whose tagged organizations match a universe ticker
    (word-boundary alias match), optionally disambiguated by requiring a
    business/economics theme co-occurrence (kills "Apple"-the-fruit etc.).
  - **market** — political / economic-policy-uncertainty / presidential coverage,
    market-wide (same for every ticker on a date, like VIX).

The BigQuery client is imported lazily (optional ``[gdelt]`` extra) so the core
package installs and tests run without it.
"""

from __future__ import annotations

import json
import re
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_agent.logging_config import get_logger
from stock_agent.news.aggregate import MARKET_COLS, NEUTRAL_TONE_BAND, PER_TICKER_COLS

log = get_logger(__name__)

GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# --- Theme taxonomies for the market stream (lowercased substring patterns) ---
# GDELT V2Themes is a long ';'-delimited string. These are tunable: re-running is
# cheap (dry-run first). See http://data.gdeltproject.org/documentation/ for the
# full GKG theme list.
POLITICAL_THEMES: tuple[str, ...] = (
    "uspec_politics",
    "general_government",
    "leader",
    "elect",
    "legislat",
    "democracy",
    "econ_",  # economic policy / trade / tariffs — primary market movers
    "epu_",
)
EPU_THEMES: tuple[str, ...] = ("epu_",)  # Economic Policy Uncertainty family
PRESIDENTIAL_THEMES: tuple[str, ...] = ("tax_fncact_president",)  # "president" as actor

# Disambiguation for per-ticker matches: require a business/econ context theme so
# a bare organization-name hit in unrelated coverage is dropped.
BUSINESS_THEMES: tuple[str, ...] = ("econ_", "business", "bus_", "epu_", "tax_", "wb_")

# --- Topic / sector macro streams (daily sentiment per topic, shared across tickers) ---
# Topic → V2Themes regex patterns. These fold into the SAME V2Themes scan as the
# market query (cheap). Patterns chosen from a GDELT theme-frequency probe (high
# volume, clean tags): see docs/NEWS_INGEST.md. ``wb_.*health`` etc. are regex.
TOPIC_THEMES: dict[str, tuple[str, ...]] = {
    "tech": (
        "information_and_communication_technolog",
        "soc_innovation",
        "soc_emergingtech",
        "tech_automation",
        "wb_376_innovation_technology",
        "wb_377_firm_innovation",
    ),
    "healthcare": (
        "general_health",
        "medical",
        "wb_.*health",
        "tax_disease",
        "pharmaceutic",
        "drug_trade",
        "ungp_healthcare",
        "epu_cats_healthcare",
    ),
    "energy": (
        "energy_and_extractives",
        "renewable_energy",
        "env_oil",
        "env_naturalgas",
        "env_coal",
        "env_solar",
        "nuclear_energy",
        "power_systems",
        "energy_efficiency",
        "env_mining",
    ),
}
# Topic → AllNames entity patterns. GDELT has NO clean "AI" theme, so AI topics are
# captured by named-entity keyword match (probe-confirmed volume). This requires
# scanning the AllNames column → extra bytes (opt-in; see include_names).
TOPIC_NAMES: dict[str, tuple[str, ...]] = {
    "ai": (
        "artificial intelligence",
        "machine learning",
        "generative ai",
        "large language model",
        "neural network",
        "deep learning",
    ),
    "ai_infra": (
        "data center",  # also matches "data centers"
        "data centre",  # also matches "data centres"
        "cloud computing",
        "semiconductor",
        "hyperscale",
    ),
}

# Default output store (gitignored; features only).
DEFAULT_STORE_DIR = Path("outputs/news_sentiment")
DEFAULT_ALIAS_PATH = Path("configs/ticker_aliases.json")


# --------------------------------------------------------------------------- #
# Alias handling (org-name → ticker)
# --------------------------------------------------------------------------- #
def load_alias_map(path: Path = DEFAULT_ALIAS_PATH) -> dict[str, list[str]]:
    """Load the ``{ticker: [name aliases]}`` mapping from JSON.

    Tickers with an empty alias list (e.g. ETFs) are dropped — they have no
    company-name footprint in GDELT organizations and rely on the market stream.
    """
    raw = json.loads(Path(path).read_text())
    return {
        t: [a for a in aliases if a.strip()]
        for t, aliases in raw.items()
        if not t.startswith("_") and isinstance(aliases, list) and aliases
    }


# RE2 (BigQuery's regex engine) metacharacters that must be escaped in a normal
# (non-class) context. NOTE: we deliberately do NOT use Python's ``re.escape`` — it
# escapes spaces as ``\ ``, which RE2 rejects ("invalid escape sequence"), and
# company aliases are full of spaces. Spaces / ``&`` / ``-`` / ``'`` are literal in
# RE2 and pass through unescaped.
_RE2_META = re.compile(r"([.^$*+?()\[\]{}|\\])")


def _re2_escape(s: str) -> str:
    """Escape only RE2 metacharacters (keeps spaces literal — see _RE2_META note)."""
    return _RE2_META.sub(r"\\\1", s)


def alias_regex_rows(alias_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Return ``[(ticker, combined word-boundary regex)]`` — one row per ticker.

    All of a ticker's aliases collapse into a single alternation so an article
    matches a given ticker **at most once** (prevents double-counting when two
    aliases of the same company both appear). Case-insensitive; aliases are
    RE2-escaped. Matched against the lowercased ``V2Organizations`` string.
    """
    rows: list[tuple[str, str]] = []
    for ticker in sorted(alias_map):
        aliases = alias_map[ticker]
        if not aliases:
            continue
        alt = "|".join(_re2_escape(a.lower()) for a in sorted(set(aliases), key=len, reverse=True))
        rows.append((ticker, rf"\b({alt})\b"))
    return rows


# --------------------------------------------------------------------------- #
# SQL builders (pure string construction — unit-tested without BigQuery)
# --------------------------------------------------------------------------- #
def _ts(d: Date) -> str:
    """Render a date as a BigQuery TIMESTAMP literal (date is trusted, not user text)."""
    return f'TIMESTAMP("{d.isoformat()}")'


def _theme_predicate(col: str, patterns: tuple[str, ...]) -> str:
    """OR of REGEXP_CONTAINS on a lowercased theme/org column for the given patterns."""
    alt = "|".join(patterns)
    return f"REGEXP_CONTAINS({col}, r'(?i)({alt})')"


def _partition_window(start: Date, end: Date) -> str:
    """``_PARTITIONTIME`` pruning clause — the cost-critical predicate (end exclusive)."""
    return f"_PARTITIONTIME >= {_ts(start)} AND _PARTITIONTIME < {_ts(end)}"


def build_market_query(start: Date, end: Date) -> str:
    """SQL for the daily market (political/EPU/presidential) stream over [start, end)."""
    pol = _theme_predicate("LOWER(V2Themes)", POLITICAL_THEMES)
    epu = _theme_predicate("LOWER(V2Themes)", EPU_THEMES)
    pres = _theme_predicate("LOWER(V2Themes)", PRESIDENTIAL_THEMES)
    return f"""
SELECT
  DIV(DATE, 1000000) AS day,
  COUNTIF(is_pol) AS pol_article_count,
  AVG(IF(is_pol, tone, NULL)) AS pol_tone_mean,
  STDDEV_SAMP(IF(is_pol, tone, NULL)) AS pol_tone_std,
  COUNTIF(is_epu) AS epu_count,
  COUNTIF(is_pres) AS pres_article_count,
  AVG(IF(is_pres, tone, NULL)) AS pres_tone_mean
FROM (
  SELECT
    DATE,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    {pol} AS is_pol,
    {epu} AS is_epu,
    {pres} AS is_pres
  FROM `{GKG_TABLE}`
  WHERE {_partition_window(start, end)}
    AND V2Tone IS NOT NULL AND V2Themes IS NOT NULL
)
WHERE is_pol OR is_epu OR is_pres
GROUP BY day
ORDER BY day
""".strip()  # noqa: E501 — generated SQL; readability favored over line length


def build_per_ticker_query(
    start: Date,
    end: Date,
    alias_rows: list[tuple[str, str]],
    *,
    require_business_theme: bool = True,
) -> str:
    """SQL for the daily per-ticker stream over [start, end).

    ``alias_rows`` (from :func:`alias_regex_rows`) become an inline ``aliases``
    CTE; each matched (article, ticker) pair is grouped to daily tone stats. With
    ``require_business_theme`` (default) a business/econ theme must co-occur,
    cutting false-positive name hits.
    """
    if not alias_rows:
        raise ValueError("alias_rows is empty — load_alias_map / alias_regex_rows first")
    # Pattern goes in a TRIPLE-quoted raw string: aliases legitimately contain
    # single quotes (e.g. "mcdonald's", "lowe's"), which would otherwise terminate
    # a normal '...' literal and produce a BigQuery 400. Patterns always end in
    # ``\b`` so the char before the closing ''' is never a backslash.
    values = " UNION ALL\n  ".join(
        f"SELECT '{ticker}' AS ticker, r'''(?i){pattern}''' AS pattern"
        for ticker, pattern in alias_rows
    )
    biz = (
        f"  WHERE {_theme_predicate('m.themes', BUSINESS_THEMES)}\n"
        if require_business_theme
        else ""
    )
    band = NEUTRAL_TONE_BAND
    return f"""
WITH aliases AS (
  {values}
),
matched AS (
  SELECT
    DIV(DATE, 1000000) AS day,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    LOWER(V2Organizations) AS orgs,
    LOWER(V2Themes) AS themes
  FROM `{GKG_TABLE}`
  WHERE {_partition_window(start, end)}
    AND V2Tone IS NOT NULL AND V2Organizations IS NOT NULL AND V2Themes IS NOT NULL
)
SELECT
  a.ticker AS ticker,
  m.day AS day,
  COUNT(*) AS article_count,
  AVG(m.tone) AS tone_mean,
  STDDEV_SAMP(m.tone) AS tone_std,
  COUNTIF(m.tone > {band}) AS pos_count,
  COUNTIF(m.tone < -{band}) AS neg_count
FROM matched m
JOIN aliases a ON REGEXP_CONTAINS(m.orgs, a.pattern)
{biz}GROUP BY ticker, day
ORDER BY day, ticker
""".strip()  # noqa: E501 — generated SQL


def topic_names(*, include_names: bool = True) -> list[str]:
    """Topics emitted by the topics stream (theme-based always; AllNames-based if on)."""
    return list(TOPIC_THEMES) + (list(TOPIC_NAMES) if include_names else [])


def build_topics_query(start: Date, end: Date, *, include_names: bool = True) -> str:
    """SQL for the daily topic/sector macro stream over [start, end).

    One scan of the base table computes a boolean flag per topic, then an in-memory
    ``UNNEST`` unpivots to (topic, day) rows — so an article counts toward every
    topic it matches (e.g. an "AI data-center power" article hits ai + ai_infra +
    energy) without re-scanning. Theme topics read ``V2Themes`` (same column the
    market query scans → cheap); ``include_names`` adds AI topics from ``AllNames``
    (an extra column scan, ~doubles the bytes).
    """
    flags = [
        f"    {_theme_predicate('LOWER(V2Themes)', pats)} AS is_{topic}"
        for topic, pats in TOPIC_THEMES.items()
    ]
    extra = ""
    if include_names:
        flags += [
            f"    {_theme_predicate('LOWER(AllNames)', pats)} AS is_{topic}"
            for topic, pats in TOPIC_NAMES.items()
        ]
        extra = " AND AllNames IS NOT NULL"
    structs = ",\n  ".join(
        f"STRUCT('{t}' AS topic, is_{t} AS hit)" for t in topic_names(include_names=include_names)
    )
    flags_sql = ",\n".join(flags)
    band = NEUTRAL_TONE_BAND
    return f"""
WITH base AS (
  SELECT
    DIV(DATE, 1000000) AS day,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
{flags_sql}
  FROM `{GKG_TABLE}`
  WHERE {_partition_window(start, end)}
    AND V2Tone IS NOT NULL AND V2Themes IS NOT NULL{extra}
)
SELECT
  u.topic AS topic,
  base.day AS day,
  COUNT(*) AS article_count,
  AVG(base.tone) AS tone_mean,
  STDDEV_SAMP(base.tone) AS tone_std,
  COUNTIF(base.tone > {band}) AS pos_count,
  COUNTIF(base.tone < -{band}) AS neg_count
FROM base, UNNEST([
  {structs}
]) u
WHERE u.hit
GROUP BY topic, day
ORDER BY day, topic
""".strip()  # noqa: E501 — generated SQL


# --------------------------------------------------------------------------- #
# BigQuery execution (lazy import; optional [gdelt] extra)
# --------------------------------------------------------------------------- #
def _client(project: str | None) -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Install the optional extra:\n"
            '    pip install -e ".[gdelt]"\n'
            "and authenticate once with:  gcloud auth application-default login"
        ) from exc
    return bigquery.Client(project=project)


def estimate_bytes(sql: str, *, project: str | None = None) -> int:
    """Dry-run the query; return bytes that *would* be scanned (no rows read, no cost)."""
    from google.cloud import bigquery

    client = _client(project)
    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed)


def run_query(sql: str, *, project: str | None = None) -> pd.DataFrame:
    """Execute ``sql`` on BigQuery and return the (small, aggregated) result frame."""
    client = _client(project)
    return client.query(sql).result().to_dataframe()


# --------------------------------------------------------------------------- #
# Result normalization → store schema (matches news.aggregate columns)
# --------------------------------------------------------------------------- #
def _day_to_date(day: Any) -> Date:
    """YYYYMMDD int (from ``DIV(DATE, 1000000)``) → ``date``."""
    return datetime.strptime(str(int(day)), "%Y%m%d").date()


def normalize_per_ticker_result(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery per-ticker rows → (date, ticker)-indexed frame with PER_TICKER_COLS."""
    midx = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in PER_TICKER_COLS}, index=midx)
    out = df.copy()
    out["date"] = out["day"].map(_day_to_date)
    out = out.set_index(["date", "ticker"]).sort_index()
    out["tone_std"] = out["tone_std"].fillna(0.0)  # STDDEV_SAMP of 1 row → NULL → 0.0
    return out[PER_TICKER_COLS].astype("float64")


def normalize_market_result(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery market rows → date-indexed frame with MARKET_COLS."""
    idx = pd.Index([], name="date")
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in MARKET_COLS}, index=idx)
    out = df.copy()
    out["date"] = out["day"].map(_day_to_date)
    out = out.set_index("date").sort_index()
    # NULL means "no article in that sub-stream that day" → 0 count / 0.0 tone.
    fills = {c: 0.0 for c in MARKET_COLS}
    return out.reindex(columns=MARKET_COLS).fillna(fills).astype("float64")


def normalize_topics_result(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery topic rows → (date, topic)-indexed frame (same cols as per-ticker)."""
    midx = pd.MultiIndex.from_arrays([[], []], names=["date", "topic"])
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in PER_TICKER_COLS}, index=midx)
    out = df.copy()
    out["date"] = out["day"].map(_day_to_date)
    out = out.set_index(["date", "topic"]).sort_index()
    out["tone_std"] = out["tone_std"].fillna(0.0)
    return out[PER_TICKER_COLS].astype("float64")


# --------------------------------------------------------------------------- #
# Store IO (CSV — small, dependency-free, inspectable; features only)
# --------------------------------------------------------------------------- #
ALL_STREAMS: tuple[str, ...] = ("per_ticker", "market", "topics")
STREAM_FILE: dict[str, str] = {s: f"{s}.csv" for s in ALL_STREAMS}


def stream_query(
    stream: str,
    start: Date,
    end: Date,
    *,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    require_business_theme: bool = True,
    include_topic_names: bool = True,
) -> str:
    """Build the SQL for one named stream (per_ticker | market | topics)."""
    if stream == "market":
        return build_market_query(start, end)
    if stream == "topics":
        return build_topics_query(start, end, include_names=include_topic_names)
    if stream == "per_ticker":
        rows = alias_regex_rows(load_alias_map(alias_path))
        return build_per_ticker_query(start, end, rows, require_business_theme=require_business_theme)  # noqa: E501
    raise ValueError(f"unknown stream: {stream!r} (expected one of {ALL_STREAMS})")


def _normalize(stream: str, df: pd.DataFrame) -> pd.DataFrame:
    if stream == "market":
        return normalize_market_result(df)
    if stream == "topics":
        return normalize_topics_result(df)
    return normalize_per_ticker_result(df)


def ingest(
    start: Date,
    end: Date,
    *,
    project: str | None = None,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    store_dir: Path = DEFAULT_STORE_DIR,
    require_business_theme: bool = True,
    streams: tuple[str, ...] = ("per_ticker", "market"),
    include_topic_names: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the selected stream queries over [start, end), normalize, write CSVs.

    Only the requested streams are queried and written — so a later run (e.g.
    ``streams=("topics",)`` next month) does NOT clobber the existing per-ticker /
    market files. End is exclusive. Returns ``{stream: frame}``.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for stream in streams:
        sql = stream_query(
            stream, start, end, alias_path=alias_path,
            require_business_theme=require_business_theme,
            include_topic_names=include_topic_names,
        )
        frame = _normalize(stream, run_query(sql, project=project))
        frame.to_csv(store_dir / STREAM_FILE[stream])
        out[stream] = frame
        log.info("news_store.written", stream=stream, rows=len(frame), store_dir=str(store_dir))
    return out
