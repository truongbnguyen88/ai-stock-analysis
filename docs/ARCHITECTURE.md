# Architecture — AI Stock Research Assistant

> Research / education only. **NOT financial advice.** No automated buy/sell signals.

## 1. Positioning

A modular Python tool that produces structured equity research reports combining:

- **Deterministic quantitative analysis** (indicators, fundamentals)
- **Model-generated probabilistic forecasts** (scenario probabilities, VaR, CIs)
- **LLM-assisted *qualitative* news interpretation** (themes, signals, citations)

### Core invariant — numbers vs. narrative

| Concern | Owner | Never owned by |
|---|---|---|
| Prices, indicators, fundamentals | Deterministic compute | LLM |
| Probabilities, forecasts, VaR, CIs | Statistical / ML models | LLM |
| News summarization, signal *labeling*, narration | LLM | — |
| Report assembly | Templates + LLM (narrative only) | — |

The LLM is a **summarizer, router, and explainer** — never a **forecaster**. This invariant propagates into module boundaries, prompt design, output guards, and tests. The report schema contains **no recommendation field** — non-advisory by construction, not just by disclaimer.

## 2. Layered architecture

Dependencies flow downward only.

```
┌──────────────────────────────────────────────────────────────┐
│  Front-ends:   CLI            │            Chat Agent           │
│  (scriptable, reproducible)   │   (NL → tool calls → narration) │
├──────────────────────────────────────────────────────────────┤
│  agent/   tool-use loop · tool schemas · numeric-grounding guard│
├──────────────────────────────────────────────────────────────┤
│  pipelines/   analyze · forecast · backtest  (thin orchestration)│
├───────────┬───────────┬───────────┬───────────┬───────────────┤
│ indicators│ forecasting│ features  │ news+llm  │ reports        │
├───────────┴───────────┴───────────┴───────────┴───────────────┤
│  backtesting / evaluation  (calibration, walk-forward)         │
├──────────────────────────────────────────────────────────────┤
│  provider abstraction layer  (Protocol-based interfaces)        │
├───────────┬───────────┬───────────┬────────────────────────────┤
│AlphaVantage│ Finnhub  │ yfinance  │ Marketaux ...               │
├──────────────────────────────────────────────────────────────┤
│  cross-cutting:  config · logging · caching · schemas           │
└──────────────────────────────────────────────────────────────┘
```

Key choices:

- **Pydantic models** are the contract between all layers; validation at boundaries.
- **Providers normalize raw JSON → domain objects** at the boundary; business logic is provider-agnostic.
- **Indicators / feature engineering are pure functions** over DataFrames → testable, leakage-free by construction when point-in-time discipline holds.
- **Forecasting models share one interface** (`ForecastModel` Protocol: a `name` + `forecast(series, *, horizon_days, as_of) -> ScenarioForecast`) so baseline, Monte Carlo, and ML are swappable and directly comparable in backtests.
- **Two front-ends, one core.** The CLI and the chat agent are independent entry points over the *same* `pipelines/` and `forecasting/` logic.
- **A parallel RAG stack** (SEC filings → grounded research memo) follows the same downward-only dependency rule and the same numbers-vs-narrative invariant; it is reachable from both front-ends (`research` CLI; `search_filings` / `research_summary` agent tools). Full design in **§13**.

## 3. Three LLM roles (keep separate)

| | Role A — News Summarizer | Role B — Orchestrating Agent | Role C — Synthesizer |
|---|---|---|---|
| Job | Articles → themes, bull/bear, risks, catalysts, citations | NL request → choose tools → narrate results | Reconcile forecast + news/earnings/technicals → Integrated Analysis |
| Scope | Narrow, single-shot, no tools | Conversational, tool-calling loop | Single-shot over assembled signals |
| Module | `llm/news_summarizer.py` | `agent/` | `llm/synthesizer.py` |
| Numbers? | **invents none** | **reports tool numbers only** | **reports input numbers only** |

Role A invents no figures (anti-forecast guard). Roles B and C **may report** numbers that came from models/tools/news-facts but **never invent or revise** them — both enforced by the shared **numeric-grounding guard** (`llm.guards.NumberGrounding`). Role A is a tool Role B can call; Role C runs in the report pipeline to produce the Integrated Analysis section (the quantitative + qualitative reconciliation).

## 4. The `agent/` layer

The chat agent is a **router, not a calculator**. Its only job: **intent → tool calls → narration grounded in tool outputs.**

```
src/stock_agent/agent/
├── runtime.py    # tool-use loop (Anthropic SDK; prompt caching) — the LLM router/supervisor
├── router.py     # hybrid routing: deterministic fast-path + delegate to runtime (no new LLM call)
├── tools.py      # tool schemas → thin wrappers over pipelines/forecasting/backtesting
├── prompts/      # system prompt + numbers-vs-narrative rules
└── guards.py     # numeric-grounding check on agent output
```

### Exposed tools (thin wrappers; no logic duplication)

| Tool | Wraps | Returns | Status |
|---|---|---|---|
| `get_price_summary(ticker, days)` | `data.loader` | price stats + `data_warnings` | ✅ |
| `compute_indicators(ticker)` | `indicators.*` | indicator snapshot + `data_warnings` | ✅ |
| `get_news(ticker, days)` | `news.*` | deduped headlines | ✅ |
| `summarize_news(ticker, days)` | `llm.news_summarizer` (Role A) | themes + citations | ✅ |
| `get_news_sentiment(ticker, days, use_llm?)` | `features.news_features` | avg sentiment, %pos/neg, event flags (AV default; Claude opt-in) | ✅ |
| `get_earnings_context(ticker, horizon?)` | `data.earnings` | next/last earnings, days-to-next, in-horizon flag | ✅ |
| `run_forecast(ticker, horizon, model?)` | `pipelines.forecast` | bucket probs, E[r], VaR, CIs (any model) | ✅ |
| `run_backtest(ticker, horizon, model?)` | `pipelines.backtest` | OOS metric suite + calibration + trust label | ✅ |
| `get_calibration(ticker, horizon, model?)` | `pipelines.backtest` | reliability table, ECE/MCE, trust label, post-hoc recal | ✅ |
| `search_filings(ticker, question, top_k?)` | `research.synthesis` over `rag/` | cited answer from the SEC filings (citation + number guarded) | ✅ |
| `research_multistep(question)` | `research.agentic` (A4 ReAct loop) over `rag/` | cited multi-hop filing answer + step trace (reuses the P7 guards; ≤4 LLM calls) | ✅ |
| `research_summary(ticker, days?)` | `pipelines.research` (P8 memo) | integrated brief (filings + news + forecast), cited | ✅ |

Data tools surface `data_warnings` (stale/sparse) so the agent can caveat. Backtest/calibration let a user ask *"is your 30-day NVDA forecast well-calibrated?"* and be answered from `get_calibration`, not the model's own reasoning — bounded for chat cost (fast offline models only, horizon 5–60d, wall-clock timeout, per-session result cache; ML backtests stay a CLI op). Numbers in every tool result feed the grounding guard, so the agent may only state figures that came from a tool.

### Dependency rule

`agent/` depends on `pipelines/`, `forecasting/`, `backtesting/` — never the reverse. Tools are thin adapters; all logic lives in the core modules.

### Hybrid routing (deterministic fast-path + LLM routing)

`agent/router.py` is a thin front-door over the tools with two ways from a request to a capability:

- **LLM routing** (default) — `run_agent` (the tool-use loop above) *is* the supervisor: it selects and composes one or many tools and grounds the numbers. Handles compound requests ("pull news **and** forecast") in one turn.
- **Deterministic routing** — when the caller **names** the capability, dispatch straight to one tool with structured params, making **no routing LLM call** (faster/cheaper/unambiguous when the user already knows what they want; numeric routes need no LLM at all).

Two layers: a granular **route registry** (one route = one tool — also the future A6 RL action set) and a friendly **~5 domains** (`predictions`, `news`, `filings`, `technicals`, `brief`) that group sibling routes by a **variant**; the domains are a *complete, non-overlapping cover* of the routes (asserted in tests). Surfaces: CLI `chat --domain <area> [--variant …]` (plus `--tool <route>` for the exact route) and a Streamlit sidebar **Routing** selector. Grounding is unchanged — a deterministic turn just dispatches one already-guarded tool — and the deterministic result reuses the same chart + citation rendering as the LLM path.

### Numeric-grounding guard (the critical safeguard)

Chat-mode analogue of the anti-forecast guard. After the agent drafts a response:

1. Extract every numeric probability / return / VaR figure from the text.
2. Assert each appears in a tool result returned this turn.
3. Reject + regenerate (or strip) any fabricated figure.

System prompt rule: *"You may only state quantitative figures present in tool outputs. You may explain and contextualize them; you may not derive or estimate new ones."*

### Request walkthrough

> "Analyze NVDA over the last 3 months and forecast 15 and 30 days."

```
Agent parses → days=90, horizons=[15,30]
 ├─ get_price_series(NVDA, 90)        → PriceSeries
 ├─ compute_indicators(series)         → MA/RSI/MACD/vol/drawdown
 ├─ get_news(NVDA, 90) → summarize_news → themes + citations   (Role A)
 ├─ run_forecast(NVDA, 15)             → buckets, E[r], VaR, CI
 └─ run_forecast(NVDA, 30)             → buckets, E[r], VaR, CI
Agent narrates using ONLY the above → grounding guard → reply
```

`days=90` and `horizon=15/30` are arguments the agent extracts from natural language and passes to the **same** pipelines the CLI calls.

## 5. Repository structure

```
ai-stock-analysis/
├── pyproject.toml
├── .env.example                 # committed, empty values
├── README.md
├── Makefile                     # lint · test · typecheck · run
├── configs/
│   ├── default.yaml
│   ├── models.yaml              # model hyperparams / registry
│   └── providers.yaml           # provider priority, rate limits
├── src/stock_agent/
│   ├── __main__.py              # python -m stock_agent
│   ├── settings.py              # pydantic-settings (.env binding)
│   ├── logging_config.py        # structlog
│   ├── schemas/                 # market · news · forecast · report · earnings · synthesis · backtest (+ConformalReport) · documents · retrieval · research
│   ├── providers/               # base(Protocols: price/news/earnings) · registry · av · finnhub · yfinance · marketaux · sec_edgar (EDGAR official API) · _cache
│   ├── data/                    # loader · validation · earnings (context + cadence) · market_context (VIX)
│   ├── indicators/              # trend · momentum · volatility · returns
│   ├── features/                # price_features · news_features (display) · news_history (GDELT, leakage-safe) · assembler
│   ├── news/                    # fetch · dedup · rank · clean · aggregate · gdelt_ingest (BigQuery)
│   ├── documents/               # ticker_cik · download (+bulk_download, date-floor) · parsers (HTML→text + section detect) · manifest
│   ├── rag/                     # embeddings (Protocol: fastembed/openai/voyage, batched) · chunking · vector_store (Protocol: InMemory/Chroma, per-embedder namespaced) · retriever · pipeline (ingest/bulk_ingest) · eval (A/B)
│   ├── research/                # synthesis (single grounded call + citation guard) · memo · evidence · prompts
│   ├── llm/                     # client · prompts · news_summarizer (A) · synthesizer (C) · guards
│   ├── forecasting/             # base · buckets · historical · monte_carlo · ml · pooled · train_pooled ·
│   │                            #   ensemble · quantiles · conformal(+_calibrate · train_conformal) · large_move · verify · regime
│   ├── backtesting/             # splitter · runner · metrics · calibration
│   ├── reports/                 # builder · render_md
│   ├── agent/                   # runtime · tools · prompts · guards
│   ├── pipelines/               # analyze · forecast · backtest · research (SEC memo)
│   └── cli/                     # app.py (analyze · forecast · backtest · train · conformal-calibrate · verify-models · ingest-news · chat · documents download-sec/ingest · rag query/eval · research)
├── configs/  (default.yaml · models.yaml · providers.yaml · universe.txt · ticker_aliases.json)
├── ui/                          # chat_app.py (Streamlit frontend)
├── scripts/                     # one-off tools (cost est. · validate_news/ensemble/xgboost experiments)
├── tests/  (unit · integration · data · fixtures)
├── notebooks/                   # exploration only — no core logic
├── outputs/  (reports · experiments · models [+conformal.json] · news_sentiment — gitignored)
├── data/     (raw [SEC filings] · processed · vectorstore [Chroma] — gitignored; the RAG corpus)
└── docs/   (ARCHITECTURE · ROADMAP · TASKS · models_explanation · validations_results · NEWS_INGEST · app_enhancements · RAG_TODO · RAG_IMPLEMENTATION_PLAN · rag_concepts · rag_implementation_notes)
```

## 6. Module responsibilities

| Module | Responsibility | Key invariant |
|---|---|---|
| `settings` | Bind `.env` + YAML | Fail fast on missing required keys; no secrets in code |
| `schemas` | Typed domain contracts | Report schema has **no recommendation field** |
| `providers` | Fetch + **normalize** to domain objects; rate-limit; cache | Business logic never sees raw JSON |
| `providers.registry` | Priority + fallback chain per capability | Deterministic, config-driven |
| `data` | Orchestrate providers → clean `PriceSeries`; validate | Point-in-time correctness |
| `indicators` | Pure fns: prices → indicator series | No lookahead; stateless; vectorized |
| `features` | Point-in-time feature matrix (price + news) | Leakage prevention is the core concern |
| `news` | Fetch, dedup, clean, rank | Dedup before LLM (cost + quality) |
| `documents` | Download SEC filings (official EDGAR API), parse HTML→text, detect sections | Official API only (no scraping); raw never overwritten; idempotent |
| `rag` | Chunk, embed (once, at ingestion), store, retrieve (filter + top-k + dedup) | Embedder + store behind Protocols; **no LLM in retrieval** |
| `research` | One grounded synthesis call → cited memo / answer | Citation guard: every cite ∈ retrieved set; non-advisory |
| `llm` | Summarize news, extract signals, cite URLs | **No numbers**; schema-validated output |
| `forecasting` | Scenario probabilities, E[r], VaR, CIs | All probabilities model-derived; common interface |
| `backtesting` | Walk-forward OOS eval + calibration | Strict temporal separation |
| `reports` | Assemble + render typed report | Narrative ⊥ numbers; every claim traceable |
| `agent` | NL → tool calls → grounded narration | Router not calculator; numeric-grounding guard |
| `pipelines` | Compose modules into use-cases | Thin; no business logic |
| `cli` | Args → pipeline dispatch | Thin; validation in schemas |

## 7. Provider abstraction

Protocol-based (structural typing, no inheritance coupling):

```
PriceProvider:         get_daily_prices(ticker, start, end) -> PriceSeries
FundamentalsProvider:  get_fundamentals(ticker)             -> Fundamentals
NewsProvider:          get_company_news(ticker, start, end) -> NewsBundle
```

- **Normalization at the boundary** — concrete providers map their API shape into shared schemas.
- **Capability registry** — `providers.yaml` declares which providers implement which capability + priority; registry resolves a fallback chain (e.g. prices: `alpha_vantage → yfinance`).
- **Uniform error contract** — typed exceptions (`ProviderRateLimit`, `ProviderUnavailable`, `SymbolNotFound`); registry catches and falls through.
- **Caching + rate limiting** are uniform decorators, not per-provider code.
- **Adding a provider** = implement the Protocol + register in YAML. Zero business-logic changes.
- **Testability** — a `FakeProvider` returning fixtures runs the whole pipeline offline/deterministically.

## 8. Data flow

```
CLI / Chat Agent  →  pipelines.analyze
   ├─ data.loader (registry) ──────────► PriceSeries ─► indicators.* ─┐
   ├─ news.fetch→dedup→clean→rank ─► NewsBundle ─► llm.news_summarizer┤ (cites URLs)
   └─ providers.registry ───────────► Fundamentals ───────────────────┤
                                                                       ▼
                                                   features.assembler (point-in-time)
                                                                       ▼
                                                   forecasting.* (probs, E[r], VaR, CI)
                                                                       ▼
                                  reports.builder → ResearchReport → render_md → outputs/
```

`forecast` / `backtest` reuse the left/center branches and skip report narrative.

## 9. Modeling strategy

**Target:** forward return `r_{t→t+h} = P_{t+h}/P_t − 1`, bucketed into the six specified ranges; classifiers use one-vs-rest thresholds (`> +5/+10%`, `< −5/−10%`).

**Three tiers (strong baseline first), plus an ensemble over them:**

1. **Historical simulation (baseline)** — empirical distribution of past `h`-day returns → bucket frequencies. Reference for whether complex models add value.
2. **Volatility-based Monte Carlo** — simulate forward paths → bucket probs, E[r], VaR, percentile CIs. **Four variants:** GBM (parametric, constant vol), block-bootstrap (non-parametric, fat tails), **earnings-jump** (GBM + an empirical post-earnings jump bootstrapped from the stock's own ~4y of historical earnings moves when an announcement falls in the horizon; calibration fetch is point-in-time to `as_of`, falls back to GBM with a disclosing note otherwise), and **GARCH** (`monte_carlo_garch` — per-ticker GJR-GARCH(1,1)-t via the `arch` library: forecasts conditional volatility *forward* with mean-reversion + the equity leverage effect + Student-t fat tails; fits on daily returns ≤ `as_of`, bootstrap fallback on short history / non-convergence). **GARCH is the validated best at h30/h60** (beats both baselines on Brier *and* big-move AUC; ties at h20) — **promoted into the default offline comparison set and the chat agent** (the two promoted Monte-Carlo methods are bootstrap + GARCH; `gbm` dropped from the agent's vetted set). See validations_results.md.
3. **ML classifiers (pooled, price-only)** — **logistic + tuned LightGBM** (config #38; xgboost & random-forest were evaluated and dropped). One binary classifier per **horizon-scaled** return-threshold (h20 ±5/±10%, h30 ±10/±20%, h60 ±15/±30% — a fixed ±5/±10 band is degenerate at long horizons), combined into the six buckets with isotonicity enforced. **Post-hoc calibrated** — `CalibratedClassifierCV(cv=3, isotonic)` baked into the artifact (config-gated `settings.calibrate_ml`), validated calibrated-vs-raw OOS with AUC held flat. Two design decisions (see TASKS.md decision log):
   - **Pooled, not per-ticker.** Trained once offline across a ticker universe (`configs/universe.txt`), persisted as a `PooledModel` artifact (`forecasting/pooled.py`, `train_pooled.py`) under `outputs/models/`, and loaded at inference. Per-ticker overlapping windows give too-low effective sample size; pooling (~tens of thousands of rows) generalizes because the features are scale-free ratios. Missing artifact → graceful fallback to historical-sim.
   - **Price-only (Option A).** The model uses price/indicator features only. News sentiment is **never** a model input — we have no point-in-time historical news to train on, and a feature absent at training cannot be applied at inference. News sentiment is *display context* (`features/news_features.py`): default = free Alpha Vantage scores; Claude scoring is opt-in.

**Ensemble (`forecasting/ensemble.py`) — the interactive default.** A **linear probability pool** over the 5 validated members (historical + bootstrap + GARCH + logistic + lightgbm), equal-weighted: bucket masses are weighted-averaged; E[r] / P(up) / P(down) are exact weighted means (linear functionals); **VaR/CI are recomputed from the mixture CDF** (`forecasting/quantiles.py`), never by averaging the members' quantiles. ML members that lack an artifact self-report a historical fallback and are dropped. OOS-validated as a **robust no-regret default** rather than a Brier win: it ties (does not beat) the single best model on Brier but is **never the worst** and has the **best big-move discrimination at honest calibration** (pooled ECE ≈ 0.05) — the right choice when you can't know a-priori which single model wins for a given name/horizon. Skill-weighting (online stacking) was tested and **added nothing** → equal weights. **Promoted as the default on the interactive surfaces (chat agent + `forecast` CLI); the deterministic `analyze` report stays the transparent baseline + ML-overlay view.** See validations_results.md.

**Feature set (`features/price_features.py`, 24 scale-free features):** trailing returns (1/5/20/60d), `rsi14`, `macd_hist`, price-to-MA deviations (20/50, 50-to-200), volatility (20/60d + ratio), `atr_pct`, `drawdown`, `B_perc` (Bollinger %B), and `days_to_next_earnings` (leakage-safe cadence estimate from yfinance earnings dates — uses only past dates + the ~91-day cadence, so it is point-in-time valid; NaN when earnings data is unavailable), plus the **market-wide VIX** (`vix_level` = VIX/100, `vix_rel` = VIX vs its 20-day average) — a real-time, leakage-safe volatility-regime signal that mainly sharpens the big-move/vol prediction, not direction (validated neutral-to-positive; see validations_results.md). **Phase 1.6 expansion (+6, OHLCV-only):** `rvol_20` + `dollar_vol_z_20` (volume), `overnight_ret_20d` + `intraday_ret_20d` (session split), `realized_skew_60` + `semivol_ratio_60` (return-distribution shape) — promoted from the opt-in candidate-group staging set after a walk-forward logistic ablation (h20/30/60); `shape` was the robust winner. Rejected (hurt calibration): `high52w`, `relstr` — kept opt-in only. All ratios/bounded so cross-ticker pooling is valid. **Display vs feature:** the report/agent show the *real* upcoming earnings date (`get_earnings_context`); the model feature uses the *cadence estimate* (computed identically at train and inference). Known caveats / deferred: `macd_hist` is mildly price-scaled (consider `/close`); **calendar seasonality features deferred** (pooling does not raise effective N for shared-calendar effects). News sentiment as a model feature (Option B) awaits historical news. Validate any new feature via Phase 6 OOS backtesting before keeping it.

**Candidate feature groups (opt-in, ablation-gated).** New features are added behind an opt-in `feature_groups` argument rather than directly into the 18-feature baseline, so a candidate can be measured before it touches production. The mechanism (`features/price_features.py`):

- **`FEATURE_GROUPS`** maps a group name → its column list; `resolve_feature_cols(groups)` returns `PRICE_FEATURE_COLS` + the requested groups, in order. `build_price_feature_matrix(..., feature_groups=None)` defaults to **baseline only**, so the default matrix — and every committed artifact trained on it — is **byte-identical** until a group is deliberately promoted. Backward-compat is a tested invariant, not a convention.
- **Self-describing artifacts.** `PooledModel.feature_cols` records exactly the columns it was trained on. At inference (`forecasting/ml.MLForecaster`), `groups_for_cols(model.feature_cols)` recovers which groups the artifact uses and rebuilds the matching feature vector — so a baseline artifact stays baseline and a richer one auto-matches, with no extra metadata to keep in sync.
- **Auxiliary series threaded like VIX.** Groups needing data beyond the ticker's own OHLCV take an extra series fetched once over a span and reindexed/rolled per fold (point-in-time safe): `market` (SPY close, `data/market_context.fetch_market`) for the market-relative-strength group; `insider` (a per-`filing_date` Form 4 activity frame, `data/insider.py`) for the insider group. Each threads through `assembler → pooled trainer → backtest` exactly as `vix` does; absent the series, the group's columns are NaN (handled like any missing feature).
- **Tier 1 groups** (all from existing OHLCV ± SPY): `volume` (relative volume, dollar-volume z-score), `high52w` (52-week-high anchoring), `session` (overnight/intraday return split), `shape` (realized skew + downside/upside semivol ratio), `relstr` (return minus SPY return at 20/60d). **Tier 2:** `insider` (trailing-63d net open-market insider \$ ÷ dollar volume + buy/sell imbalance). All scale-free; all trailing-window PIT.
- **Form 4 kept out of the RAG corpus.** Insider data is structured ownership XML, not narrative text, so it has its own `schemas/insider.py` + `documents/form4.py` parser + `sec_edgar.list_form4_filings`/`download_form4` (XML disk-cached) flow — the RAG `DocumentType` enum (`10-K/10-Q/8-K`) is intentionally **not** widened.
- **Promotion is gated on measured lift.** `backtesting/ablation.py` + `scripts/ablate_feature_groups.py` run the walk-forward harness baseline-vs-baseline+group and report Brier/ECE/big-move-AUC/coverage deltas; the promote gate is conservative (Brier↓ **without** ECE↑). Only groups that clear it are folded into `PRICE_FEATURE_COLS`, after which {logistic, lightgbm} × {20,30,60} are retrained and the results recorded in validations_results.md. Until then the groups are inert experimental infra (same posture as the regime/LSTM/news-feature work).
- **Two promotion routes.** Groups that help the *whole* universe (volume/session/shape) are folded directly into `PRICE_FEATURE_COLS` (always-on, OHLCV-only). A group that helps only a *segment* uses the granular route: it stays out of the baseline but is enabled for **production training** via `settings.model_feature_groups` (CSV), so the trained artifact's `feature_cols` carry it and inference auto-adapts. `insider` (Form 4) took this route — it lifts Brier/calibration on **mid/small-caps** but is **nil on mega-caps** (validated, universe-dependent), so the production universe was broadened into that segment (`configs/universe.txt`, 159 tickers) and `insider` enabled by config. It needs `SEC_USER_AGENT` at train (CI secret) and inference time; absent it, insider columns are NaN (degraded, not broken). Bulk Form 4 fetch uses a pooled keep-alive client + retry (`data/insider.build_hardened_sec_provider`) to survive EDGAR fair-access throttling at universe scale.

**Leakage discipline (priority #1):**

- Feature at `t` uses only data available at `t` (indicator windows ending at `t`).
- Target window strictly after feature cutoff; enforce `feature_end < target_start`.
- Imputer fit on (pooled) train data only and **persisted** with the artifact — inference uses the same fill values (no per-row imputation leak).
- Tests assert no future timestamps in feature rows.

**Outputs:** bucket probabilities (sum to 1), expected return, upside/downside probability (partition at 0 → sum to 1), VaR (5%/1%), CIs, model identity + calibration status. Sparse-data tickers fall back to baseline with explicit low-confidence flags.

## 10. Backtesting & calibration

*Implemented in `backtesting/` (splitter · metrics · calibration · runner) + `pipelines/backtest.py` + `backtest` CLI.*

- **One comparison surface for every model** — any `ScenarioForecast` is reduced to per-threshold exceedance probabilities `P(r > θ_k) = Σ buckets above θ_k` (θ_k = the **horizon-scaled** bucket boundaries = the ML cut-points from `thresholds_for_horizon`), scored against the realized label `1[r > θ_k]`. So historical-sim, Monte-Carlo, and ML are evaluated on identical folds with the exact target the ML models train on.
- **Leakage discipline** — the runner forecasts from `bars[:t+1]` at each as-of (historical/MC point-in-time by construction); refittable models (pooled ML) are rebuilt per fold via `build_model(train_end_date)` on universe data ≤ cutoff.
- **Temporal splitting only** — walk-forward folds (expanding/rolling train → fixed OOS test, stepped). No random K-fold.
- **Embargo** between train end and test start equal to horizon `h` (prevents target overlap leakage).
- **Metrics** per fold + aggregated with dispersion: accuracy, precision, recall, ROC AUC, Brier, log loss.
- **Calibration first-class** — reliability diagrams + Expected Calibration Error measure trust; the served ML models are calibrated via `CalibratedClassifierCV(cv=k, isotonic)` baked into the artifact, validated calibrated-vs-raw OOS (ECE/Brier down, **AUC invariant**). The `backtest` harness's `calibrate=` flag drives that A/B.
- **Interval coverage is conformalized** — ECE calibrates *bucket probabilities*; **split conformal** (`forecasting/conformal.py`) calibrates the *prediction interval* so the stated CI has honest coverage. The backtest reports stated-vs-conformalized coverage (`ConformalReport`); a **pooled offline correction** `q` per (model, horizon) — calibrated as-of a cutoff, pooled across the universe (`train_conformal.py`), persisted `outputs/models/conformal.json` — is applied to served CIs/VaR at inference (config-gated `settings.conformal_intervals`). Distribution-free, finite-sample marginal coverage; leakage-safe (calibration window is the past).
- **Baselines as guardrails** — every ML model compared to historical-sim + MC on identical folds. Underperforming/uncalibrated models reported as such.
- **Reproducibility** — each run logs config, seeds, data window, provider versions, metrics to `outputs/experiments/<run_id>/`.

Explicit deliverable: a **trustworthiness measurement** of probabilities, surfaced to the agent via `get_calibration`.

## 11. Configuration, logging, testing

- **Config:** `.env` (secrets, gitignored) + `configs/*.yaml` (behavior) bound through `pydantic-settings`; env-specific via `ENV`.
- **Logging:** `structlog`, JSON in prod; API request logging; experiment logging to `outputs/experiments/`.
- **Testing:** unit (golden-file indicators, dedup, schemas), data validation (point-in-time/leakage), provider (normalization vs fixtures, fallback triggers, cache), integration (full pipeline via `FakeProvider`, LLM mocked), LLM/agent guards (schema conformance, numeric-grounding, citation validity), modeling (leakage, probs sum to 1, splitter non-overlap, calibration math), CLI. Deterministic: seed everything, mock all network + LLM.

## 12. Model lifecycle (train in CI, serve locally)

The ML artifacts are **trained in CI and served locally** — the repo never carries the ~100 MB of binaries (`outputs/models/` is gitignored).

- **Scheduled retrain** — `.github/workflows/retrain.yml` runs monthly (cron `0 6 1 * *`) and on manual `workflow_dispatch`: `pip install -e .` → `train --all` (logistic + tuned lightgbm × {20, 30, 60}, calibrated) → **`conformal-calibrate`** (pooled split-conformal interval-corrections → `conformal.json`, so served CIs/VaR have honest coverage and track the fresh models) → **promote gate** → publish. yfinance is keyless, so it needs no secrets and fetches the full universe from the runner.
- **Promote gate** — `forecasting/verify.py` (`verify-models` CLI), **network-free**: each pooled artifact loads, its `thresholds` match `thresholds_for_horizon(h)`, ≥ n-1 threshold-classifiers trained, dummy-feature `predict_exceedance` returns probs in [0, 1], **plus a data-quality floor** (≥ `verify_min_ticker_fraction` of the universe, ≥ `verify_min_rows` rows). It also **verifies `conformal.json` when present** (every required model×horizon has a finite `q` whose post-correction coverage reached ~target) — present in CI, so a broken conformal run fails the publish; absent locally, it just warns. A failure aborts the run so the previous release stays published. *Limitation:* still no OOS-Brier re-backtest (see ROADMAP item 3 "Deferred").
- **Distribution** — the workflow tars the artifacts **and `conformal.json`** into a rolling **`models-latest` GitHub Release** (plus a dated snapshot for rollback). Local side pulls with **`make pull-models`** (auth-free `curl`, public repo); the app loads from `outputs/models/`, falling back to historical-sim (and un-conformalized CIs) when an artifact is absent.
- **Train/serve pickle parity** — the serialization-sensitive deps are pinned to a single minor band (`scikit-learn>=1.9,<1.10`, `lightgbm>=4.6,<5`, `joblib>=1.5,<2`). Without this, a CI/local sklearn skew triggers `InconsistentVersionWarning` on load and risks silently wrong deserialization. Both sides install the pins via `pip install -e .`.
- **Schedule liveness** — GitHub auto-disables a scheduled workflow after 60 days of no repo activity; the retrain pushes a small **keepalive commit** on every run (before training, so a failed/slow run still resets the timer), keeping the ~30-day cadence self-sustaining without manual intervention.

## 13. SEC-grounded research layer (RAG)

A second, self-contained layer turns **SEC filings (10-K / 10-Q / 8-K)** into a grounded research
assistant. It is independent of the forecasting core but held to the **same invariants** — numbers
come from the models (RAG returns *qualitative evidence + citations only*), no recommendations, no
scraping — and obeys the same **downward-only** dependency rule. Detailed build steps + locked
decisions live in [RAG_TODO.md](RAG_TODO.md); per-phase mechanisms in
[rag_implementation_notes.md](rag_implementation_notes.md).

### Pipeline (dependencies point downward only)

```
providers/sec_edgar   EDGAR OFFICIAL API client (Protocol; throttle ≤10 rps; UA; DiskCache)   ← lowest
        ▼
documents/            download (+bulk, date-floor, idempotent) · parse HTML→text · detect Item sections · manifest
        ▼
rag/                  chunk (section-aware, pure) · embed ONCE (Embedder Protocol) · store (VectorStore Protocol) · retrieve (filter+top-k+dedup, NO LLM)
        ▼
research/             ONE grounded synthesis call (llm/ + guards) → GroundedAnswer / ResearchMemo
        ▼
research/agentic        A4 bounded ReAct loop (multi-hop) — reuses retrieval + the P7 guarded answer
        ▼
pipelines/research · cli (research · rag ask) · agent tools (search_filings · research_multistep · research_summary)
```

`rag/` depends on `documents/` + an `Embedder`; `research/` depends on `rag/` + `pipelines/`
(forecast/analyze for the integrated memo) + `llm/`. Never inverted.

### Embedding strategy (the key design decisions)

- **Embeddings are computed once, at ingestion** — never per query over the corpus. Both the
  embedder and the vector store sit behind **Protocols** (`Embedder`, `VectorStore`), so providers
  swap without touching chunking/retrieval/synthesis code.
- **Production embedder = Voyage `voyage-4`**, chosen by a **labeled retrieval A/B** (`rag/eval.py`)
  on a 25-question / 5-ticker set: voyage-4 beat local fastembed on ranking quality (MRR 0.72→0.89,
  precision@8 0.63→0.76, hit@8 tied); the finance-tuned `voyage-finance-2` *lost* and was dropped.
  Local **`fastembed`/BGE** (onnxruntime, no torch → dodges the macOS torch+lightgbm OpenMP segfault,
  $0, unlimited) remains a **complete on-disk fallback**.
- **Per-embedder collection namespacing.** Different embedders have different vector dimensions
  (BGE 384-d vs voyage-4 1024-d). `build_vector_store` derives the Chroma collection name from the
  embedder identity (`embedding_namespace`), so the local and voyage corpora live in **separate
  collections** — switching `EMBEDDING_PROVIDER` targets a fresh collection instead of corrupting one.
- **Cost controls.** A configurable `rag_max_embed_tokens` ceiling **refuses an ingest before it
  embeds** (no provider spend on an over-budget run). For large corpora, `embed_documents` **batches**
  requests under the provider's per-request caps, and `bulk_ingest` **isolates + retries per ticker**,
  so one transient network blip during a multi-hour embed never aborts the whole run (failed tickers
  are reported for an idempotent backfill).

### The single paid call + guards

The **only paid LLM call** in the whole flow is the final synthesis (`research/synthesis.py`):
download, parse, chunk, embed, and retrieval are 100% local. Two guards protect it (analogues of the
forecasting layer's anti-forecast / numeric-grounding guards):

- **Citation guard** — every cited marker (inline `[n]` and the `citations` list) must resolve to a
  source in the *retrieved* evidence set; a fabricated cite triggers one corrective retry, then raises.
- **Number grounding** — reuses `llm.guards.NumberGrounding`, seeded from the retrieved texts (and, in
  the integrated memo, the forecast + snapshot + news), so the synthesis may quote SEC figures but
  never invent them. Empty retrieval short-circuits to *"Insufficient evidence found."* — **no LLM call**.

The integrated memo (`research/memo.py`, `pipelines/research.py`) copies quant sections (technical
indicators, probability scenarios) **verbatim from the models** and lets the LLM write only the
narrative with cited SEC claims — **no recommendation field**, same as the analyze report.

### Front-ends

- **CLI:** `documents download-sec` → `documents ingest` → `rag query [--answer]` (single-shot) /
  `rag ask [--single]` (multi-hop ReAct) → `research`.
- **Chat agent:** three guarded tools — `search_filings(ticker, question)` (a specific filing
  question, cited), `research_multistep(question)` (multi-hop / comparative / change-over-time, the
  A4 ReAct loop), and `research_summary(ticker)` (the integrated brief). All make their own guarded
  synthesis call and return *validated, cited* output, so the agent's grounding guard stays intact
  (it never hands raw chunks to the model). Reachable only after a ticker's filings are ingested.

### Production state

Built incrementally P0–P9 (`RAG_TODO.md`), then an **advanced-RAG track A1–A4** ([ADVANCED_RAG_TODO.md](ADVANCED_RAG_TODO.md)):
A1 retrieval-eval harness, A2 reranking (kept available, OFF), **A3 hybrid dense⊕BM25 — promoted to the
default** on a measured eval win, A4 **agentic multi-hop** retrieval (`research_multistep`). The MVP
corpus = SEC filings only (transcripts / decks / GraphRAG (A5) / retrieval-RL (A6) are still future).
As shipped: ~3 years of 10-K/10-Q/8-K across the universe (≈93k chunks, with a backfilled BM25 index)
embedded with **voyage-4** in production (local BGE collection retained as fallback); the production
embedder is selected by `EMBEDDING_PROVIDER` in `.env`.
