# Roadmap — AI Stock Research Assistant

Scope tiers and an ordered, incremental implementation plan. Each step is independently testable; each phase ends green-tested before the next. Risk-ordered: data correctness and leakage prevention land before modeling complexity; the agent layer lands only after the pipelines it wraps exist.

## Scope tiers

### MVP — deterministic core + LLM news + baseline forecast

- Providers: yfinance (price fallback), Alpha Vantage (prices/fundamentals/news), Finnhub (news); cache + rate limit.
- Indicators: MA20/50/200, RSI, MACD, daily/log returns, historical volatility, drawdowns, trend flags.
- News: fetch → dedup → clean → relevance rank → LLM summary (bull/bear/risk/catalyst) with **URL citations**, schema-validated, anti-forecast guard.
- Forecasting: **historical-simulation baseline only** (empirical forward-return distribution → buckets, E[r], simple VaR).
- Report: full markdown with all required sections incl. Uncertainty Notes + Source Citations.
- CLI: `analyze`, basic `forecast`.
- **Chat agent (analyze + baseline forecast tools)** over the same pipelines.
- Cross-cutting: config, logging, caching.

Out of MVP: ML models, Monte Carlo, walk-forward backtesting, calibration, ensembles.

### V1 — statistical/ML modeling + rigorous evaluation

- Monte Carlo forecaster (GBM + block bootstrap + earnings-jump variant).
- ML forecasters: logistic / XGBoost / LightGBM / random forest under common interface; threshold-bucket targets. **Pooled (cross-sectional) training** across a ticker universe → persisted `PooledModel` artifact loaded at inference (`train` CLI).
- **Price-only model (Option A):** news sentiment is display context (free AV scores by default; Claude opt-in), never a model input — no point-in-time historical news to train on. See TASKS.md decision log.
- Backtesting: rolling + walk-forward OOS; full metric suite (accuracy, precision, recall, ROC AUC, Brier, log loss).
- Calibration: reliability diagrams, isotonic/Platt, ECE, predicted-vs-realized.
- Forecast report section with CIs + VaR across 5/20/60-day horizons.
- `backtest` CLI + experiment logging.
- **Agent gains `run_backtest` + `get_calibration` tools** (e.g. "is the 30-day NVDA forecast well-calibrated?").

### Future

- Sequence/regime models (TFT/LSTM with temporal CV; HMM/changepoint).
- Probabilistic depth: quantile regression, conformal prediction CIs, GARCH, copula multi-horizon joints.
- **Temporal features:** ~~earnings proximity~~ **DONE** — `days_to_next_earnings` is now a model feature (leakage-safe cadence estimate; yfinance earnings dates) + `get_earnings_context` for display. Still validate its OOS lift via Phase 6. (Pure calendar seasonality remains deferred — pooling doesn't raise effective N for shared-calendar effects.)
- **News-as-model-feature (Option B):** only once point-in-time historical sentiment exists (buy data or log snapshots forward); then backtest whether it beats price-only.
- News depth: embedding dedup/relevance, entity linking, event-study impact.
- Cross-sectional/portfolio extension (still non-advisory).
- Serving: FastAPI report API, scheduled refresh, report diffing.
- MLOps: model registry, experiment tracking (MLflow/W&B), drift monitoring, automated calibration re-checks.
- LLM eval harness (faithfulness / citation-accuracy scoring) for both Role A and the agent.

## Step-by-step implementation

Bracketed = primary deliverable.

### Phase 0 — Scaffolding
1. `pyproject.toml`, package skeleton, `Makefile`, `.gitignore`, `.env.example`. [installable package]
2. `settings.py` (pydantic-settings) + `logging_config.py` (structlog). [config loads; fails fast on missing keys]
3. Core `schemas/` (market, news, forecast, report). [typed contracts + validators + tests]

### Phase 1 — Data layer
4. `providers/base.py` Protocols + `_cache.py` + `registry.py`. [interfaces + fallback + `FakeProvider`]
5. yfinance provider (no key → easiest first). [real prices end-to-end]
6. Alpha Vantage (prices/fundamentals/news) + Finnhub (news). [normalization + fixture tests]
7. `data/loader.py` + `data/validation.py`. [clean `PriceSeries` + point-in-time checks]

### Phase 2 — Indicators
8. `indicators/*` pure functions + golden-file tests. [all required indicators]

### Phase 3 — News + LLM (Role A)
9. `news/` fetch → dedup → clean → rank. [ranked, deduped articles]
10. `llm/client.py` (claude-api skill; prompt caching) + prompts + `news_summarizer.py` + `guards.py`. [schema-valid summary with citations; anti-forecast guard tested]

### Phase 4 — Baseline forecast + report (MVP analytical core)
11. `forecasting/base.py` + `buckets.py` + `historical.py`. [baseline probabilities + E[r] + VaR]
12. `reports/builder.py` + `render_md.py` + templates. [full markdown report]
13. `pipelines/analyze.py` + `cli/app.py` (`analyze`). [`python -m stock_agent analyze --ticker NVDA --days 30`]
14. Integration test: full analyze pipeline with fakes.

### Phase 4.5 — Chat agent (Role B), MVP scope ← **MVP milestone**
15. `agent/tools.py` — wrap `get_price_series`, `compute_indicators`, `get_news`, `summarize_news`, `run_forecast` over existing pipelines. [tool schemas]
16. `agent/runtime.py` (tool-use loop, prompt caching) + `agent/prompts/` (numbers-vs-narrative system prompt) + `agent/guards.py` (numeric-grounding guard). [working chat over deterministic + baseline forecast]
17. `cli/app.py` adds `chat` command + agent integration tests (LLM mocked, fabricated-number rejection tested). **← MVP milestone**

### Phase 5 — Statistical & ML models (V1) ✅
18. `forecasting/monte_carlo.py` (GBM + block bootstrap + earnings-jump) + `forecast` CLI wiring. [MC scenarios]
19. `features/price_features.py` + `assembler.py` (point-in-time, leakage-tested) — **price-only**; `news_features.py` builds display context only (AV sentiment default, Claude opt-in).
20. `forecasting/ml.py` + `pooled.py` + `train_pooled.py`: **pooled** classifiers persisted as an artifact, loaded at inference; `train` CLI. [price-only, calibratable]

### Phase 6 — Backtesting & calibration (V1) ✅
21. `backtesting/splitter.py` (walk-forward + embargo) + leakage tests.
22. `backtesting/runner.py` + `metrics.py`. [OOS metric suite]
23. `backtesting/calibration.py` (reliability, isotonic/Platt, ECE). [trustworthiness output]
24. `pipelines/backtest.py` + `backtest` CLI + experiment logging.

### Phase 6.5 — Agent gains evaluation tools ← **V1 milestone**
25. `agent/tools.py` adds `run_backtest` + `get_calibration` (with timeouts / argument bounds for heavier ops); extend system prompt so the agent reports calibration/trust honestly. [agent answers calibration/backtest questions, grounded] **← V1 milestone**

### Phase 7 — Hardening
26. Forecast report section with CIs / VaR / calibration status.
27. Coverage pass; finalize `docs/` (`ARCHITECTURE.md`, `MODELING.md`) + `README.md`.

## CLI / chat surface

```bash
# Deterministic CLI (scriptable, reproducible)
python -m stock_agent analyze  --ticker NVDA  --days 90
python -m stock_agent train    --model xgboost --horizon 20   # pooled ML (one-time)
python -m stock_agent forecast --ticker MSFT  --horizon 20  --model xgboost
python -m stock_agent backtest --ticker AAPL                  # Phase 6

# Conversational agent (same core)
python -m stock_agent chat
> Analyze NVDA over the last 3 months and forecast 15 and 30 days
> Is your 30-day NVDA forecast well-calibrated?

# Browser chat frontend (Streamlit)
make ui   # → http://localhost:8501
```

## Notes / open decisions

- **CLI framework:** Typer (Annotated style; a `@app.callback()` keeps subcommand names).
- **LLM integration:** Phases 3, 4.5, 6.5 use the `claude-api` skill with prompt caching (news context is large and reused across report sections and chat turns).
- **Heavy agent tools:** `run_backtest` / `get_calibration` can be slow; runtime should stream progress and bound arguments (max horizon, max window) to keep chat responsive.
- **UI (added, off-roadmap):** `ui/chat_app.py` — Streamlit chat over the agent; threads conversation history (stateful). Launch with `make ui`.
- **ML artifacts:** pooled models persist to `outputs/models/` (gitignored); `forecast --model <ml>` falls back to historical-sim until you `train`.
