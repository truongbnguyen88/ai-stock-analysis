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
├── runtime.py    # tool-use loop (Anthropic SDK; prompt caching)
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
| `run_backtest(ticker, horizon, model?)` | `backtesting.runner` | OOS metric suite | Phase 6.5 |
| `get_calibration(ticker, horizon, model?)` | `backtesting.calibration` | reliability curve, ECE, trust flag | Phase 6.5 |

Data tools surface `data_warnings` (stale/sparse) so the agent can caveat. Backtest/calibration (Phase 6.5) will let a user ask *"is your 30-day NVDA forecast well-calibrated?"* and be answered from `get_calibration`, not the model's own reasoning. Numbers in every tool result feed the grounding guard, so the agent may only state figures that came from a tool.

### Dependency rule

`agent/` depends on `pipelines/`, `forecasting/`, `backtesting/` — never the reverse. Tools are thin adapters; all logic lives in the core modules.

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
│   ├── schemas/                 # market · news · forecast · report · earnings · synthesis
│   ├── providers/               # base(Protocols: price/news/earnings) · registry · av · finnhub · yfinance · marketaux · _cache
│   ├── data/                    # loader · validation · earnings (context + cadence)
│   ├── indicators/              # trend · momentum · volatility · returns
│   ├── features/                # price_features · news_features (display) · assembler
│   ├── news/                    # fetch · dedup · rank · clean
│   ├── llm/                     # client · prompts · news_summarizer (A) · synthesizer (C) · guards
│   ├── forecasting/             # base · buckets · historical · monte_carlo · ml · pooled · train_pooled
│   ├── backtesting/             # splitter · runner · metrics · calibration
│   ├── reports/                 # builder · render_md
│   ├── agent/                   # runtime · tools · prompts · guards
│   ├── pipelines/               # analyze · forecast · backtest
│   └── cli/                     # app.py (analyze · forecast · train · chat)
├── configs/  (default.yaml · models.yaml · providers.yaml · universe.txt)
├── ui/                          # chat_app.py (Streamlit frontend)
├── scripts/                     # one-off tools (e.g. estimate_sentiment_cost.py)
├── tests/  (unit · integration · data · fixtures)
├── notebooks/                   # exploration only — no core logic
├── outputs/  (reports · experiments · models — gitignored)
└── docs/   (ARCHITECTURE.md · ROADMAP.md · TASKS.md)
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

**Three tiers (strong baseline first):**

1. **Historical simulation (baseline)** — empirical distribution of past `h`-day returns → bucket frequencies. Reference for whether complex models add value.
2. **Volatility-based Monte Carlo** — estimate drift/vol (EWMA/rolling → later GARCH); simulate paths → bucket probs, E[r], VaR, percentile CIs. Three variants: GBM (parametric), block-bootstrap (fat tails), and **earnings-jump** (GBM + an empirical post-earnings jump bootstrapped from the stock's own ~4y of historical earnings moves when an announcement falls in the horizon; calibration fetch is point-in-time to `as_of`, falls back to GBM with a disclosing note otherwise).
3. **ML classifiers (pooled, price-only)** — one binary classifier per return-threshold (`> +5/+10%`, `< −5/−10%`), combined into the six buckets with isotonicity enforced. **Must be calibrated** before reporting. Two design decisions (see TASKS.md decision log):
   - **Pooled, not per-ticker.** Trained once offline across a ticker universe (`configs/universe.txt`), persisted as a `PooledModel` artifact (`forecasting/pooled.py`, `train_pooled.py`) under `outputs/models/`, and loaded at inference. Per-ticker overlapping windows give too-low effective sample size; pooling (~tens of thousands of rows) generalizes because the features are scale-free ratios. Missing artifact → graceful fallback to historical-sim.
   - **Price-only (Option A).** The model uses price/indicator features only. News sentiment is **never** a model input — we have no point-in-time historical news to train on, and a feature absent at training cannot be applied at inference. News sentiment is *display context* (`features/news_features.py`): default = free Alpha Vantage scores; Claude scoring is opt-in.

**Feature set (`features/price_features.py`, 16 scale-free features):** trailing returns (1/5/20/60d), `rsi14`, `macd_hist`, price-to-MA deviations (20/50, 50-to-200), volatility (20/60d + ratio), `atr_pct`, `drawdown`, `B_perc` (Bollinger %B), and `days_to_next_earnings` (leakage-safe cadence estimate from yfinance earnings dates — uses only past dates + the ~91-day cadence, so it is point-in-time valid; NaN when earnings data is unavailable). All ratios/bounded so cross-ticker pooling is valid. **Display vs feature:** the report/agent show the *real* upcoming earnings date (`get_earnings_context`); the model feature uses the *cadence estimate* (computed identically at train and inference). Known caveats / deferred: `macd_hist` is mildly price-scaled (consider `/close`); **calendar seasonality features deferred** (pooling does not raise effective N for shared-calendar effects). News sentiment as a model feature (Option B) awaits historical news. Validate any new feature via Phase 6 OOS backtesting before keeping it.

**Leakage discipline (priority #1):**

- Feature at `t` uses only data available at `t` (indicator windows ending at `t`).
- Target window strictly after feature cutoff; enforce `feature_end < target_start`.
- Imputer fit on (pooled) train data only and **persisted** with the artifact — inference uses the same fill values (no per-row imputation leak).
- Tests assert no future timestamps in feature rows.

**Outputs:** bucket probabilities (sum to 1), expected return, upside/downside probability (partition at 0 → sum to 1), VaR (5%/1%), CIs, model identity + calibration status. Sparse-data tickers fall back to baseline with explicit low-confidence flags.

## 10. Backtesting & calibration

*Implemented in `backtesting/` (splitter · metrics · calibration · runner) + `pipelines/backtest.py` + `backtest` CLI.*

- **One comparison surface for every model** — any `ScenarioForecast` is reduced to per-threshold exceedance probabilities `P(r > θ_k) = Σ buckets above θ_k` (θ_k = the bucket boundaries = the ML `THRESHOLDS`), scored against the realized label `1[r > θ_k]`. So historical-sim, Monte-Carlo, and ML are evaluated on identical folds with the exact target the ML models train on.
- **Leakage discipline** — the runner forecasts from `bars[:t+1]` at each as-of (historical/MC point-in-time by construction); refittable models (pooled ML) are rebuilt per fold via `build_model(train_end_date)` on universe data ≤ cutoff.
- **Temporal splitting only** — walk-forward folds (expanding/rolling train → fixed OOS test, stepped). No random K-fold.
- **Embargo** between train end and test start equal to horizon `h` (prevents target overlap leakage).
- **Metrics** per fold + aggregated with dispersion: accuracy, precision, recall, ROC AUC, Brier, log loss.
- **Calibration first-class** — reliability diagrams, Expected Calibration Error, post-hoc isotonic/Platt fit on validation, re-evaluated OOS.
- **Baselines as guardrails** — every ML model compared to historical-sim + MC on identical folds. Underperforming/uncalibrated models reported as such.
- **Reproducibility** — each run logs config, seeds, data window, provider versions, metrics to `outputs/experiments/<run_id>/`.

Explicit deliverable: a **trustworthiness measurement** of probabilities, surfaced to the agent via `get_calibration`.

## 11. Configuration, logging, testing

- **Config:** `.env` (secrets, gitignored) + `configs/*.yaml` (behavior) bound through `pydantic-settings`; env-specific via `ENV`.
- **Logging:** `structlog`, JSON in prod; API request logging; experiment logging to `outputs/experiments/`.
- **Testing:** unit (golden-file indicators, dedup, schemas), data validation (point-in-time/leakage), provider (normalization vs fixtures, fallback triggers, cache), integration (full pipeline via `FakeProvider`, LLM mocked), LLM/agent guards (schema conformance, numeric-grounding, citation validity), modeling (leakage, probs sum to 1, splitter non-overlap, calibration math), CLI. Deterministic: seed everything, mock all network + LLM.
