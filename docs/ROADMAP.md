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

- Monte Carlo forecaster (GBM + block bootstrap + earnings-jump + **GARCH** conditional-volatility variant — the validated best at h30/h60, promoted into the default comparison set + the chat agent; see validations_results.md).
- ML forecasters: **logistic + tuned LightGBM** under a common interface; horizon-scaled threshold-bucket targets, **CalibratedClassifierCV** post-hoc calibration. **Pooled (cross-sectional) training** across a ticker universe → persisted `PooledModel` artifact loaded at inference (`train` CLI). *(xgboost + random_forest were evaluated and dropped — neither beat the toolkit at its cost; see validations_results.md.)*
- **Price-only model (Option A):** news sentiment is display context (free AV scores by default; Claude opt-in), never a model input — no point-in-time historical news to train on. See TASKS.md decision log.
- Backtesting: rolling + walk-forward OOS; full metric suite (accuracy, precision, recall, ROC AUC, Brier, log loss).
- Calibration: reliability diagrams, isotonic/Platt, ECE, predicted-vs-realized.
- Forecast report section with CIs + VaR across 5/20/60-day horizons.
- `backtest` CLI + experiment logging.
- **Agent gains `run_backtest` + `get_calibration` tools** (e.g. "is the 30-day NVDA forecast well-calibrated?").

### Future (after Phase 7)

- **Sequence / regime models (PARKED — revisit after Phase 7).** TFT/LSTM with temporal CV; HMM/changepoint. Explicitly deferred during the post-V1 ML track: every validation found **direction is ~efficient** and only **magnitude/volatility** is predictable, which logistic + tuned lightgbm already capture — so the bar for a heavier, fundamentally different (temporal-structure) paradigm is high. Worth a *bounded experiment* only after the toolkit is operationalized (calibration + scheduled retraining + Phase 7 report surfacing), and judged on the same held-out / calibration discipline.
- Probabilistic depth: quantile regression, conformal prediction CIs, copula multi-horizon joints. *(GARCH — DONE: shipped as the `monte_carlo_garch` variant.)*
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
18. `forecasting/monte_carlo.py` (GBM + block bootstrap + earnings-jump + **GARCH**) + `forecast` CLI wiring. [MC scenarios]
19. `features/price_features.py` + `assembler.py` (point-in-time, leakage-tested) — **price-only**; `news_features.py` builds display context only (AV sentiment default, Claude opt-in).
20. `forecasting/ml.py` + `pooled.py` + `train_pooled.py`: **pooled** classifiers persisted as an artifact, loaded at inference; `train` CLI. [price-only, calibratable]

### Phase 6 — Backtesting & calibration (V1) ✅
21. `backtesting/splitter.py` (walk-forward + embargo) + leakage tests.
22. `backtesting/runner.py` + `metrics.py`. [OOS metric suite]
23. `backtesting/calibration.py` (reliability, isotonic/Platt, ECE). [trustworthiness output]
24. `pipelines/backtest.py` + `backtest` CLI + experiment logging.

### Phase 6.5 — Agent gains evaluation tools ✅ ← **V1 milestone**
25. `agent/tools.py` adds `run_backtest` + `get_calibration` (with timeouts / argument bounds for heavier ops); extend system prompt so the agent reports calibration/trust honestly. [agent answers calibration/backtest questions, grounded] **← V1 milestone**

### Phase 7 — Hardening
26. **[DONE]** Forecast report surfaces CIs / VaR / **real calibration status** / **horizon-scaled big-move** / horizon-trust. The deterministic report now overlays the **calibrated ML forecast** (lightgbm) alongside the historical baseline per horizon (graceful skip where no artifact); `calibration_status` is sourced from the artifact (`PooledModel.is_calibrated`) instead of a hardcoded `"unknown"`; each block renders the horizon's inner-`k` big-move reading (`large_move_breakdown`) and a horizon-confidence caveat (h≤30 measurable / h60 low-confidence).
27. **[DONE]** Coverage pass (86% overall; added `load_universe` + verify/render-branch tests) and docs finalized — authored the top-level `README.md` and refreshed `docs/models_explanation.md` to the shipped toolkit (logistic + tuned lightgbm only, horizon-scaled buckets, calibration shipped; xgboost/RF/h5 removed). `models_explanation.md` is the modeling reference (no separate `MODELING.md`).

## Active ML work queue (post-V1 ML track)

Sequenced pipeline beyond the original phases — the ML-quality + MLOps track. Status as of **2026-06-03**. Empirical results land in [validations_results.md](validations_results.md).

1. **[DONE] Large-scale lightgbm tuning + horizon-scaled buckets.** Random search (~100 configs, ~13-D) with **ticker-level meta-validation against selection bias** (tune on a volatile basket → report once on a **disjoint held-out** basket {AVGO, MU, ARM, TSLA, VRT}). Winner **config #38** beat the incumbent on held-out h20 (+0.023 AUC — survived the winner's curse). Also shipped **horizon-scaled scenario buckets** (h20 ±5/±10, h30 ±10/±20, h60 ±15/±30; the inner boundary is the default big-move `k` = 5/10/15%), because a fixed ±5/±10 band is degenerate at long horizons. Retrained **logistic + lightgbm at {20, 30, 60}** via `train --all`; **dropped h5** (too short-term) and **dropped xgboost + random_forest** (neither promoted; RF's `class_weight="balanced"` was a latent calibration bug).
2. **[DONE] Post-hoc calibration — logistic + lightgbm.** Re-validation (volatile NVDA/SMCI/TSLA at each inner `k`) confirmed ML skill at **h20/h30** (AUC 0.59–0.67 > baselines) but **both models miscalibrated** (ECE 0.10–0.32, worsening with horizon); h60 unmeasurable (n≈19) → flagged low-confidence. Shipped calibration = **`CalibratedClassifierCV(cv=3, method="isotonic")`** baked into each pooled threshold-classifier (config-gated via `settings.calibrate_ml`, default on); the cross-threshold **monotone envelope** is enforced downstream in `ml._exceedance_to_buckets`. The **cv=3 A/B improved Brier in every (model, horizon) cell** and fixed the earlier *prefit*-isotonic regression that hurt lightgbm (20% data loss + isotonic overfitting small per-fold holdouts); `cv=k` uses all data via an averaged calibrator. The 6 served artifacts were retrained calibrated. Full tables → [validations_results.md](validations_results.md).
3. **[DONE] Scheduled monthly retraining (CI-train / local-serve).** `train --all` (`{logistic, lightgbm} × {20, 30, 60}`, calibrated) runs on a **GitHub Actions cron** (`.github/workflows/retrain.yml`, `0 6 1 * *`; chose GH Actions over `launchd` for zero local dependence). Pipeline: **fetch the universe from yfinance → retrain + refit calibrators → structural `verify-models` promote gate → publish a rolling `models-latest` GitHub Release + a dated snapshot for rollback.** Local pulls with **`make pull-models`** (auth-free `curl`, public repo); the app serves from `outputs/models/` (gitignored), so the repo never carries the ~100 MB of binaries. Hardening that made it reliable: **dependency pins** (`scikit-learn`/`lightgbm`/`joblib`) so CI-pickled artifacts deserialize cleanly on the local serve side, and a **keepalive commit** on each run to dodge GitHub's 60-day schedule auto-disable. Validated end-to-end (yfinance fetches the full 114-ticker universe from the runner; pulled artifacts are the cv=3 calibrated ensembles). **Deferred (next hardening):** the promote gate is *structural* only (artifact loads, thresholds match, valid probs) — it does **not** re-backtest or assert data-quality, so a degraded-data month could still publish; add an OOS-Brier / minimum-universe-size gate + versioned `as_of` metadata. Realizes the **MLOps** + **Serving (scheduled refresh)** Future items above.
4. **[DONE — ❌ neither promoted] Sequence / regime model exploration (Task 8).** Gaussian-HMM regime forecaster (*tied* the baselines) and a pooled LSTM (heavy 3–4 layer + LayerNorm, LR sweep, full universe, 114-ticker validation, four calibration methods incl. `CalibratedClassifierCV(cv=3)`) — the LSTM extracts a **real but redundant** vol signal and **loses Brier/ECE on ~75% of tickers**; no calibration helps. Conclusion: the binding constraint is **information, not model class or calibration**. Both kept as isolated experimental code (torch env-gated). Transformer/TFT ruled out by the same logic.
5. **[DONE — ✅ promoted] GARCH conditional-volatility forecaster (Task 9).** `monte_carlo_garch` (GJR-GARCH(1,1)-t via `arch`) — the **first new model to beat the baselines on the deployable proper score**, *because it adds a different mechanism* (forward vol mean-reversion + leverage + fat tails) rather than more capacity over the same inputs. Ties bootstrap at h20, **wins Brier at h30 (9/12) and h60 (10/12)**, highest big-move AUC at every horizon. Promoted into the default comparison set + the chat agent (the two promoted MC methods are bootstrap + GARCH; `gbm` dropped from the agent). Follow-up: surface GARCH as the preferred long-horizon baseline in the deterministic written report.

## CLI / chat surface

```bash
# Deterministic CLI (scriptable, reproducible)
python -m stock_agent analyze  --ticker NVDA  --days 90
python -m stock_agent train    --all                          # retrain logistic + lightgbm @ 20/30/60
python -m stock_agent forecast --ticker MSFT  --horizon 30  --model lightgbm
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
