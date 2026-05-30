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
- **Forecasting models share one interface** (`fit` / `predict_proba` / `metadata`) so baseline, Monte Carlo, and ML are swappable and directly comparable in backtests.
- **Two front-ends, one core.** The CLI and the chat agent are independent entry points over the *same* `pipelines/` and `forecasting/` logic.

## 3. Two LLM roles (keep separate)

| | Role A — News Summarizer | Role B — Orchestrating Agent |
|---|---|---|
| Job | Articles → themes, bull/bear, risks, catalysts, citations | NL request → choose tools → narrate results |
| Scope | Narrow, single-shot, no tools | Conversational, tool-calling loop |
| Module | `llm/news_summarizer.py` | `agent/` |
| May emit numbers? | **No** | **No** |

Role A is a tool that Role B can call. Neither produces quantitative figures.

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

| Tool | Wraps | Returns |
|---|---|---|
| `get_price_series(ticker, days)` | `data.loader` | `PriceSeries` |
| `compute_indicators(series)` | `indicators.*` | indicator set |
| `get_news(ticker, days)` | `news.*` | deduped articles |
| `summarize_news(articles)` | `llm.news_summarizer` (Role A) | themes + citations |
| `run_forecast(ticker, horizon, model?)` | `forecasting.*` | bucket probs, E[r], VaR, CIs |
| `run_backtest(ticker, horizon, model?)` | `backtesting.runner` | OOS metric suite |
| `get_calibration(ticker, horizon, model?)` | `backtesting.calibration` | reliability curve, ECE, trust flag |

Backtest and calibration are exposed as tools (per design decision) so a user can ask *"is your 30-day NVDA forecast well-calibrated?"* and the agent answers from `get_calibration`'s output, not from its own reasoning. These tools are heavier; the agent runtime should surface progress and may enforce timeouts / argument bounds.

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
│   ├── schemas/                 # market · news · forecast · report
│   ├── providers/               # base(Protocols) · registry · av · finnhub · yfinance · marketaux · _cache
│   ├── data/                    # loader · validation (point-in-time)
│   ├── indicators/              # trend · momentum · volatility · returns
│   ├── features/                # price_features · news_features (display) · assembler
│   ├── news/                    # fetch · dedup · rank · clean
│   ├── llm/                     # client · prompts · news_summarizer · guards
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
2. **Volatility-based Monte Carlo** — estimate drift/vol (EWMA/rolling → later GARCH); simulate paths (GBM + block-bootstrap for fat tails) → bucket probs, E[r], VaR, percentile CIs.
3. **ML classifiers (pooled, price-only)** — one binary classifier per return-threshold (`> +5/+10%`, `< −5/−10%`), combined into the six buckets with isotonicity enforced. **Must be calibrated** before reporting. Two design decisions (see TASKS.md decision log):
   - **Pooled, not per-ticker.** Trained once offline across a ticker universe (`configs/universe.txt`), persisted as a `PooledModel` artifact (`forecasting/pooled.py`, `train_pooled.py`) under `outputs/models/`, and loaded at inference. Per-ticker overlapping windows give too-low effective sample size; pooling (~tens of thousands of rows) generalizes because the features are scale-free ratios. Missing artifact → graceful fallback to historical-sim.
   - **Price-only (Option A).** The model uses price/indicator features only. News sentiment is **never** a model input — we have no point-in-time historical news to train on, and a feature absent at training cannot be applied at inference. News sentiment is *display context* (`features/news_features.py`): default = free Alpha Vantage scores; Claude scoring is opt-in.

**Feature set (`features/price_features.py`, 15 scale-free features):** trailing returns (1/5/20/60d), `rsi14`, `macd_hist`, price-to-MA deviations (20/50, 50-to-200), volatility (20/60d + ratio), `atr_pct`, `drawdown`, and `B_perc` (Bollinger %B). All ratios/bounded so cross-ticker pooling is valid. Known caveats / deferred: `macd_hist` is mildly price-scaled (consider `/close`); **calendar seasonality features deferred** — pooling does not raise effective sample size for market-wide calendar effects (all tickers share the calendar), so they overfit; the higher-value temporal feature is **earnings proximity** (needs an earnings-date provider with point-in-time history). News sentiment as a model feature (Option B) likewise awaits historical news. Validate any new feature via Phase 6 OOS backtesting before keeping it.

**Leakage discipline (priority #1):**

- Feature at `t` uses only data available at `t` (indicator windows ending at `t`).
- Target window strictly after feature cutoff; enforce `feature_end < target_start`.
- Imputer fit on (pooled) train data only and **persisted** with the artifact — inference uses the same fill values (no per-row imputation leak).
- Tests assert no future timestamps in feature rows.

**Outputs:** bucket probabilities (sum to 1), expected return, upside/downside probability (partition at 0 → sum to 1), VaR (5%/1%), CIs, model identity + calibration status. Sparse-data tickers fall back to baseline with explicit low-confidence flags.

## 10. Backtesting & calibration

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
