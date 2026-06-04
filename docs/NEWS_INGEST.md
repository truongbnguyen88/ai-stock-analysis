# News-feature ingest (GDELT → daily sentiment store)

Task 10 (news-as-model-feature). Pulls **daily news-sentiment features** from the
GDELT 2.0 Global Knowledge Graph via BigQuery. Aggregation runs **server-side**, so
only small daily feature rows are downloaded — **never article text**. Output lands
in `outputs/news_sentiment/` (gitignored; features only).

> This produces the *data*. Wiring the features into the pooled model + the OOS
> price-only-vs-price+news backtest are the following slices.

## What you get

Three independent streams, each its own CSV (so you can pull one without re-pulling
the others — see `--streams`):

| File | Grain | Columns |
|---|---|---|
| `per_ticker.csv` | (date, ticker) | `article_count, tone_mean, tone_std, pos_count, neg_count` |
| `market.csv` | date | `pol_article_count, pol_tone_mean, pol_tone_std, epu_count, pres_article_count, pres_tone_mean` |
| `topics.csv` | (date, topic) | `article_count, tone_mean, tone_std, pos_count, neg_count` |

- **per-ticker** — company coverage tone, matched from GDELT organization tags to
  `configs/ticker_aliases.json` (word-boundary, business-theme-disambiguated).
- **market** — market-wide political / economic-policy-uncertainty / presidential
  tone (same for every ticker on a date, like VIX). The "things presidents say /
  policy" macro signal.
- **topics** — sector/topic macro sentiment (same for every ticker on a date). Topics:
  `tech, healthcare, energy` (from GDELT `V2Themes` — cheap, same column the market
  query scans) + `ai, ai_infra` (from `AllNames` entity match — GDELT has no clean
  "AI" theme, so these need keyword matching; `--no-topic-names` skips them). An
  article counts toward every topic it matches.

Tone is GDELT's lexicon (bag-of-words) score in ≈[-10, +10]; crude but
deterministic and reproducible. Raw daily aggregates are stored; scale-free,
pooling-valid features (`news_buzz`/`{topic}_buzz` spike vs trailing mean,
`{topic}_tone`, …) are derived downstream by `features/news_history.py`.

Tone is GDELT's lexicon (bag-of-words) score in ≈[-10, +10]; crude but
deterministic and reproducible. Raw daily aggregates are stored; scale-free,
pooling-valid features (e.g. a `news_buzz` spike vs the ticker's own trailing
mean) are derived downstream so cross-ticker pooling stays valid.

## One-time setup (Google Cloud — free tier)

1. Create a Google Cloud project (free): <https://console.cloud.google.com/projectcreate>.
2. Enable the **BigQuery API** for that project.
3. Authenticate locally:
   ```bash
   gcloud auth application-default login
   ```
4. Install the optional extra:
   ```bash
   pip install -e ".[gdelt]"
   ```

GDELT is a free public dataset; you pay only for **query bytes scanned** —
**1 TiB/month is free**, then ~$5/TiB. The dry-run below shows the cost before you
spend anything.

## Run it

Default window is **~5 years** (`2021-01-01 → present`) — matches the price-training
lookback, gives enough regime variety (2022 bear, 2023–24 AI run, 2025 vol), and
roughly halves the BigQuery scan vs going back to 2016.

**Always dry-run first** (no rows read, no spend):

```bash
python -m stock_agent ingest-news --start 2021-01-01 --end 2026-06-03 \
    --project YOUR_GCP_PROJECT --dry-run
```

It prints GiB-to-scan per stream and whether you're inside the free tier. The text
columns (`V2Themes`, `V2Organizations`) are large, so if even 5 years exceeds 1 TiB,
**chunk by year and spread across calendar months**:

```bash
# Each call ~one year; run a few per month to stay under the free 1 TiB.
python -m stock_agent ingest-news --start 2021-01-01 --end 2022-01-01 --project YOUR_GCP_PROJECT
python -m stock_agent ingest-news --start 2022-01-01 --end 2023-01-01 --project YOUR_GCP_PROJECT
# … (each run appends/overwrites the store; see note below)
```

Flags:
- `--streams` — comma list of `per_ticker,market,topics` (default `per_ticker,market`).
  Each writes its own CSV, so you can add a stream later without re-pulling the others.
- `--no-topic-names` — topics: themes only (`tech/healthcare/energy`), skip the
  `ai`/`ai_infra` AllNames scan (cheaper).
- `--no-business-filter` — drop the per-ticker business/econ theme co-occurrence
  filter (more recall, more false-positive name hits).
- `--dry-run` — estimate only. `--project` — the GCP project billed.

### Adding the topics stream later (free)
The three streams are independent files, so once you have `per_ticker` + `market`
you can add topics in a separate run — the full topics stream (incl. AI) is ~553 GiB,
**within the free tier on its own**:

```bash
python -m stock_agent ingest-news --start 2023-01-01 --end 2026-06-03 \
    --project YOUR_GCP_PROJECT --streams topics --dry-run    # confirm ~553 GiB
python -m stock_agent ingest-news --start 2023-01-01 --end 2026-06-03 \
    --project YOUR_GCP_PROJECT --streams topics              # writes topics.csv
```

> **Note (chunked runs):** each invocation *overwrites* the CSV(s) for the streams it
> pulls (others untouched). Multi-window accumulation (append + dedup) is a small
> follow-up; until then, run one wide window per stream, or merge the per-window CSVs.

## Tuning the topical scope

- **Company → ticker aliases:** `configs/ticker_aliases.json`. Add distinctive
  company *names* (not bare ticker symbols). ETFs are intentionally empty.
- **Political / EPU / presidential themes**, the **business disambiguation set**, and
  the **topic definitions** (`TOPIC_THEMES` = sector V2Themes patterns; `TOPIC_NAMES` =
  AI AllNames keywords) are all constants at the top of
  [`src/stock_agent/news/gdelt_ingest.py`](../src/stock_agent/news/gdelt_ingest.py).
  They were chosen from a GDELT theme/entity frequency probe (high-volume, clean tags;
  AI has no theme so it uses entity keywords). Edit + re-run (dry-run first).

## Leakage / point-in-time

GDELT stamps each article at its **monitoring time** (when first observed) and the
historical archive is fixed → genuinely reconstructable point-in-time and
reproducible. Aggregation buckets by the UTC day of that timestamp. Leakage-safe
alignment to trading dates (with a conservative 1-day publication lag) happens in
the feature loader, not here.
