"""Unit tests for the GDELT ingest module (Task 10).

Covers the pure, BigQuery-free pieces: alias-map loading + regex assembly, SQL
structure (especially the cost-critical _PARTITIONTIME pruning clause), and
result→store normalization. The BigQuery client itself is not exercised here
(it needs credentials + the optional [gdelt] extra).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_agent.news.aggregate import MARKET_COLS, PER_TICKER_COLS
from stock_agent.news.gdelt_ingest import (
    DEFAULT_ALIAS_PATH,
    GKG_TABLE,
    alias_regex_rows,
    build_market_query,
    build_per_ticker_query,
    load_alias_map,
    normalize_market_result,
    normalize_per_ticker_result,
)


def test_load_real_alias_map_skips_comments_and_empty_etfs() -> None:
    amap = load_alias_map(DEFAULT_ALIAS_PATH)
    assert "_comment" not in amap
    assert "SPY" not in amap  # ETF with empty aliases dropped
    assert amap["NVDA"] == ["nvidia"]


def test_alias_regex_one_row_per_ticker_word_boundary() -> None:
    amap = {"AMD": ["advanced micro devices", "amd inc"], "KO": ["coca-cola"]}
    rows = dict(alias_regex_rows(amap))
    assert set(rows) == {"AMD", "KO"}
    # Longer aliases first (alternation order), word-boundary wrapped.
    assert rows["AMD"].startswith(r"\b(") and rows["AMD"].endswith(r")\b")
    # Spaces stay literal (RE2 rejects "\ "); dots/parens would be escaped.
    assert "advanced micro devices" in rows["AMD"]


def test_alias_regex_escapes_re2_meta_but_not_spaces_or_apostrophes() -> None:
    rows = dict(alias_regex_rows({"AMZN": ["amazon.com"], "MCD": ["mcdonald's"]}))
    assert r"amazon\.com" in rows["AMZN"]  # '.' escaped for RE2
    assert "mcdonald's" in rows["MCD"]  # apostrophe left literal (triple-quoted in SQL)


def test_market_query_has_partition_pruning_and_table() -> None:
    sql = build_market_query(date(2016, 1, 1), date(2020, 1, 1))
    assert GKG_TABLE in sql
    # The cost-critical clause MUST be present and bound to the window.
    assert '_PARTITIONTIME >= TIMESTAMP("2016-01-01")' in sql
    assert '_PARTITIONTIME < TIMESTAMP("2020-01-01")' in sql
    assert "pol_tone_mean" in sql and "epu_count" in sql and "pres_tone_mean" in sql


def test_per_ticker_query_embeds_alias_cte_and_pruning() -> None:
    rows = alias_regex_rows({"NVDA": ["nvidia"]})
    sql = build_per_ticker_query(date(2018, 1, 1), date(2019, 1, 1), rows)
    assert "WITH aliases AS" in sql
    assert "SELECT 'NVDA' AS ticker" in sql
    assert "_PARTITIONTIME >=" in sql
    assert "REGEXP_CONTAINS(m.orgs, a.pattern)" in sql
    # Disambiguation theme filter present by default.
    assert "m.themes" in sql


def test_per_ticker_query_can_disable_business_theme_filter() -> None:
    rows = alias_regex_rows({"NVDA": ["nvidia"]})
    sql = build_per_ticker_query(
        date(2018, 1, 1), date(2019, 1, 1), rows, require_business_theme=False
    )
    assert "m.themes" not in sql  # disambiguation clause omitted


def test_normalize_per_ticker_result_shapes_and_fills_std() -> None:
    raw = pd.DataFrame(
        {
            "day": [20240102, 20240103],
            "ticker": ["NVDA", "NVDA"],
            "article_count": [3.0, 1.0],
            "tone_mean": [1.5, -2.0],
            "tone_std": [0.8, None],  # single-article day → NULL std
            "pos_count": [2.0, 0.0],
            "neg_count": [1.0, 1.0],
        }
    )
    out = normalize_per_ticker_result(raw)
    assert list(out.columns) == PER_TICKER_COLS
    assert out.index.names == ["date", "ticker"]
    assert out.loc[(date(2024, 1, 3), "NVDA"), "tone_std"] == 0.0
    assert out.loc[(date(2024, 1, 2), "NVDA"), "article_count"] == 3.0


def test_normalize_market_result_shapes_and_fills() -> None:
    raw = pd.DataFrame(
        {
            "day": [20240102],
            "pol_article_count": [10.0],
            "pol_tone_mean": [-1.2],
            "pol_tone_std": [2.0],
            "epu_count": [3.0],
            "pres_article_count": [0.0],
            "pres_tone_mean": [None],  # no presidential article that day
        }
    )
    out = normalize_market_result(raw)
    assert list(out.columns) == MARKET_COLS
    assert out.index.name == "date"
    assert out.loc[date(2024, 1, 2), "pres_tone_mean"] == 0.0


def test_normalize_empty_results_are_typed() -> None:
    pt = normalize_per_ticker_result(pd.DataFrame())
    mk = normalize_market_result(pd.DataFrame())
    assert list(pt.columns) == PER_TICKER_COLS and pt.empty
    assert list(mk.columns) == MARKET_COLS and mk.empty
