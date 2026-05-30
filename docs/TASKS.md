# TASKS — Progress Tracker

Living checklist mirroring [ROADMAP.md](ROADMAP.md). Update on every step transition. Full detail lives in ROADMAP/ARCHITECTURE — keep this file terse.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (tests green)

---

## ▶ Current
- **Phase:** 6 — Backtesting & calibration
- **Next step:** Step 21 — `backtesting/splitter.py` (walk-forward + embargo) + leakage tests
- **Gate to advance:** `make check` green
- **Last updated:** 2026-05-30
- 🎉 **MVP milestone** (Phases 0–4.5): analyze CLI + chat agent live.
- ✅ **Phases 5 + 5.5 done** (162 pytest): pooled ML, earnings (context + feature), Role C synthesis layer. Phase 6 (backtest/calibration) is the next roadmap step — and what makes all forecasts *trustworthy* + lets synthesis/agent speak to calibration.

---

## Phase 0 — Scaffolding ✅
- [x] 1. `pyproject.toml`, package skeleton, `Makefile`, `.gitignore`, `.env.example`
- [x] 2. `settings.py` (pydantic-settings) + `logging_config.py` (structlog)
- [x] 3. Core `schemas/` (market, news, forecast, report)

## Phase 1 — Data layer ✅
- [x] 4. `providers/base.py` Protocols + `_cache.py` + `registry.py` (+ `FakeProvider`)
- [x] 5. yfinance provider
- [x] 6. Alpha Vantage + Finnhub + Marketaux providers
- [x] 7. `data/loader.py` + `data/validation.py`

## Phase 2 — Indicators ✅
- [x] 8. `indicators/*` + golden-file tests

## Phase 3 — News + LLM (Role A) ✅
- [x] 9. `news/` fetch → dedup → clean → rank
- [x] 10. `llm/` client + prompts + `news_summarizer.py` + `guards.py`

## Phase 4 — Baseline forecast + report ✅
- [x] 11. `forecasting/` base + buckets + historical
- [x] 12. `reports/` builder + render_md (rendered in code; no jinja2 dep)
- [x] 13. `pipelines/analyze.py` + `cli/app.py` (`analyze`)
- [x] 14. Integration test: analyze pipeline with fakes

## Phase 4.5 — Chat agent (Role B) ✅ ← MVP milestone
- [x] 15. `agent/tools.py` (5 tools: price/indicators/news/summary/forecast)
- [x] 16. `agent/runtime.py` + prompts + numeric-grounding guard
- [x] 17. `chat` CLI command + agent integration tests

## Phase 5 — Statistical & ML models ✅
- [x] 18. `forecasting/monte_carlo.py` (GBM + block bootstrap + earnings-jump) + `forecast` CLI
- [x] 19. `features/` price + news + assembler (point-in-time, leakage-tested)
- [x] 20. `forecasting/ml.py` (logistic, XGBoost, LightGBM, random forest)

## Phase 5.5 — Signal enrichment & synthesis ✅ (off original roadmap; details in decision log)
- [x] 20a. Pooled-training rework: cross-sectional `PooledModel` artifact + `train` CLI (replaces per-ticker fit); price-only (Option A); AV sentiment = display context (Claude opt-in)
- [x] 20b. Agent completeness: `get_news_sentiment` (de-orphaned AV sentiment), `data_warnings`, model selection in `run_forecast`, stateful multi-turn memory, guard false-positive fixes
- [x] 20c. Features: `B_perc` (Bollinger %B) + `days_to_next_earnings` (leakage-safe cadence estimate) → 16 features
- [x] 20d. Earnings: `EarningsProvider` (yfinance) + context in report + `get_earnings_context` agent tool
- [x] 20e. **Synthesis layer (Role C)**: `llm/synthesizer.py` → Integrated Analysis reconciling forecast + news + earnings + technicals (numeric-grounding guarded)

## Phase 6 — Backtesting & calibration
- [ ] 21. `backtesting/splitter.py` (walk-forward + embargo) + leakage tests
- [ ] 22. `backtesting/runner.py` + `metrics.py`
- [ ] 23. `backtesting/calibration.py`
- [ ] 24. `pipelines/backtest.py` + `backtest` CLI + experiment logging

## Phase 6.5 — Agent evaluation tools ← V1 milestone
- [ ] 25. `agent/tools.py` adds `run_backtest` + `get_calibration`

## Phase 7 — Hardening
- [ ] 26. Forecast report section (CIs / VaR / calibration status)
- [ ] 27. Coverage pass + finalize docs + README

---

## Decision log (append-only)
- 2026-05-28 — Future-tier work (Phase 8+) left unphased as backlog until V1 backtesting informs ordering.
- 2026-05-28 — Repo structure grows incrementally per phase; no upfront empty-tree scaffolding.
- 2026-05-29 — Phase 0 done; gate green (ruff + mypy strict + 12 pytest). Deps kept minimal (pydantic, pydantic-settings, structlog); heavier deps added per phase. API keys optional at load, validated at use via `Settings.require()`.
- 2026-05-29 — LLM default set to `claude-sonnet-4-6` (cost efficiency) instead of Opus.
- 2026-05-29 — Provider priority defaults reordered for free tiers: prices `yfinance,alpha_vantage`; news `finnhub,marketaux,alpha_vantage`. Rationale: Alpha Vantage free = 25 req/day (too low for primary); Finnhub free has no historical candles (news only); yfinance keyless for prices. Only a free Finnhub key needed to start Phase 1.
- 2026-05-29 — Dev Python pinned to 3.12 (`.python-version`); Python 3.14.0 framework build does not honor editable-install `.pth` files and lacks some wheels. `requires-python` floor stays >=3.11.
- 2026-05-29 — Some shells/sandboxes don't process `.pth` at all, so editable import is unreliable there. Added `pythonpath = ["src"]` to pytest so the suite imports from src directly. For ad-hoc runs in such shells use `PYTHONPATH=src python -m stock_agent`; a normally-activated venv works without it.
- 2026-05-29 — Local `.env` created with Finnhub + Marketaux keys (gitignored, verified untracked). Anthropic/Alpha Vantage left blank until needed.
- 2026-05-29 — Phase 1 done; gate green (ruff + mypy strict + 47 pytest). Live smoke verified: yfinance prices (NVDA, adj_close present), Finnhub 247 + Marketaux 3 merged = 250 news, Alpha Vantage skipped (no key). Deps added: httpx, yfinance, pandas.
- 2026-05-29 — Provider design: structural Protocols (base `Provider` + capability sub-Protocols); typed error contract (`ProviderError`/`Unavailable`/`RateLimit`/`SymbolNotFound`); disk TTL cache via `cached_model`; news = merge mode, prices = failover; HTTP via injectable `HttpJson` (MockTransport in tests). `FundamentalsProvider` Protocol defined; concrete impl deferred to Phase 4.
- 2026-05-29 — Alpha Vantage key added + live-tested. Fixed free-tier bug: `outputsize=full` is now premium → switched to `compact` (~100 trading days; fine since yfinance is primary price source). Live OK: AV prices (AAPL, 10 bars) and AV news (50 articles w/ sentiment). Note: AV free throttles to ~1 req/sec — registry merge fires providers sequentially so this is generally fine; heavy multi-call flows may need spacing.
- 2026-05-29 — Phase 2 done; gate green (ruff + mypy strict + 62 pytest). Indicators: pure pandas fns (returns/trend/momentum/volatility) + `wilder_smooth` (RSI/ATR) + frame adapter + typed `IndicatorSnapshot`. Conventions: Wilder RMA (SMA seed), MACD EMA adjust=False, vol = annualized log-return std ddof=1 (252d), drawdown = close/cummax-1, analysis close = adj_close|close. Golden values hand-derived (period=2 RSI/ATR). Live sanity on NVDA: MA20>MA50>MA200, vol 40%, MDD -20% — all consistent.
- 2026-05-29 — Phase 4.5 done (🎉 MVP milestone); gate green (ruff + mypy strict + 103 pytest). Chat agent (Role B): 5 thin tools over existing pipelines (`get_price_summary`/`compute_indicators`/`get_news`/`summarize_news`/`run_forecast`); `AnthropicToolClient` tool-use loop (system-block caching, injectable `ToolLLM`); numeric-grounding guard (only decimals/percents checked; numbers in tool text also grounded so news facts can be quoted; dropped 0-digit rounding which masked fabrications); 1 corrective retry then refuse. `chat` CLI (one-shot + REPL). Live NVDA "analyze + forecast 15/30d": agent auto-called all tools, grounded answer, 0 violations. Bumped agent answer budget to 4096 tokens (was truncating). Multi-turn memory deferred (each query independent for MVP).
- 2026-05-29 — Phase 4 done; gate green (ruff + mypy strict + 95 pytest). Forecasting: `ForecastModel` protocol, 6 return buckets ([lower,upper)), historical-sim baseline (empirical overlapping h-day returns → bucket probs, E[r], VaR95/99, 90% CI; flags low-confidence <30 samples). Reports: deterministic `build_report` (templated narrative from numbers + LLM news summary) + `render_markdown` (all required sections, disclaimer, no recommendation; rendered in code — chose not to add jinja2). Pipeline `run_analyze` degrades gracefully if LLM disabled/fails. CLI `analyze` via Typer (Annotated style; needs a `@app.callback()` so single command keeps its name). Deps: typer. Live: AAPL `--no-llm` (uptrend, RSI 78, 289 bars, scenarios/VaR/CI) + MSFT full LLM run (rich grounded news section, valid citations). Fixed: summarizer truncation (max_tokens 2048→4096) + pipeline now degrades on parse/validation errors too.
- Deferred to later: `forecast`/`backtest` CLI commands (Phases 5-6); fundamentals provider + report fundamentals section (when a provider is added).
- 2026-05-29 — Phase 3 done; gate green (ruff + mypy strict + 86 pytest). News pipeline (clean/canonical-url, dedup by canonical-url+title-Jaccard with field merge, recency+mention rank) + LLM Role A (AnthropicClient w/ system-block prompt caching, NewsSummary schema, anti-forecast + citation guards, summarizer w/ 1 corrective retry). Live NVDA end-to-end: 300→297 deduped→8 ranked→Sonnet summary; 0 forecast violations, 0 invalid citations. Fixed Sonnet-4.x bug: model rejects assistant-message prefill → removed prefill, rely on JSON instruction + lenient parse. Dep added: anthropic 0.105.
- 2026-05-29 — **DECISION: ML tier uses POOLED (cross-sectional) training, not per-ticker.** Train ONE classifier-per-threshold on stacked rows from a ~50–100 ticker universe (`configs/universe.txt`), then predict for any target ticker. Rationale: per-ticker overlapping windows give low effective sample size (≈n/h) → XGBoost overfits; pooling yields ~50k–100k rows and generalizes. Requires scale-free features (ratios/bounded indicators — already designed that way). Target ticker may be in the universe; backtest splits are TEMPORAL not by-ticker (never train on dates ≥ test date). News sentiment EXCLUDED from training (no point-in-time historical news) — inference-only context. Default lookback 3–5y/ticker; drop first ~200 bars (warmup) + last h bars (unlabeled).
  - **STATUS: not yet implemented.** Current `forecasting/ml.py` trains per-ticker at forecast time (`self._fit(series)`). Pooled training is a planned revision: add an offline training step (fetch+cache universe → build point-in-time features → fit → persist model artifact) and load the artifact at inference. Revisit when wiring backtesting (Phase 6) so walk-forward respects the pooled temporal split.
- 2026-05-29 — **DECISION: Option A — ML model is PRICE-ONLY; news sentiment is display context, never a model input.** You cannot feed a feature at inference that wasn't in training, and we have no point-in-time historical news to train on. So the calibrated probability comes 100% from price features (train = inference). News sentiment is shown *alongside* the forecast (report + agent), not mixed into the number. To make news a real model feature later (Option B) we'd need to buy historical news or log point-in-time snapshots going forward, then retrain and backtest whether it adds value.
- 2026-05-29 — **DECISION: news-sentiment cost tiering (display only, since news ≠ model feature).** Measured: Claude scoring all 296 articles = ~$0.44 AND truncates/resets (fragile). Tiering: (1) **AV pre-computed sentiment = free default numeric signal** (~17% coverage; show coverage caveat); (2) **Role A Claude summary (already built, ~$0.02, top ~15 articles) = the qualitative insight** — no separate full-scoring needed; (3) **full-300 Claude scoring = optional, on-demand, 24h-cached, index-based not URL-echo** (~$0.10/day with the fix). Default path uses tiers 1+2 only. Pooled rework will drop the news feature vector from the model and keep `news_features` for display/context via AV sentiment.
- 2026-05-29 — **Pooled rework IMPLEMENTED** (gate green: ruff + mypy strict + 136 pytest). `forecasting/pooled.py` (`PooledModel` artifact = classifiers + persisted imputer + metadata; joblib save/load; `train_pooled_from_series` pure/offline-testable) + `train_pooled.py` (universe → fetch deep history → stack → fit → persist) + `ml.py` reworked to LOAD the artifact at inference (fallback to historical_sim with note if absent) + CLI `train`. Model is PRICE-ONLY; `news_features` now defaults to AV sentiment with Claude opt-in (`use_llm_sentiment`). Correctness: persisted imputer (fixes per-row imputation leak), inf→NaN guard. Live: trained xgboost on 8-ticker mini universe → 11,952 rows / 5 thresholds; NVDA 20d forecast loaded the artifact. Fixed bug: upside/downside used `(b.lower or 0.0)` which miscounted open tails → P(up)+P(down)=121%; now None-guarded → sums to 100% (invariant test added). NOTE: ML forecasts need a trained artifact first — `python -m stock_agent train --model xgboost --horizon 20` over `configs/universe.txt` (61 tickers). `outputs/models/` is gitignored (artifacts not committed).
- 2026-05-29 — Post-MVP additions: (a) **Streamlit chat UI** `ui/chat_app.py` (`make ui` → localhost:8501); (b) **agent made stateful** — `run_agent(history=...)` threads prior turns; `AgentResult.messages` returned for the next turn; UI persists it (supersedes the "multi-turn memory deferred" note above); (c) guard false-positive fixes: anti-forecast no longer flags reported analyst "price target" (only LLM's own); numeric-grounding no longer flags `95%/99%` VaR labels (forecast tool now emits `var_confidence_levels_pct`).
- 2026-05-29 — Agent forecast tool now exposes **model selection** (closing a gap: it previously hardcoded historical_sim). `run_forecast` tool gained a `model` arg (enum from `MODEL_REGISTRY`) and routes via `pipelines.forecast.run_forecast`, so the chat agent can use Monte Carlo and pooled ML and compare models; ML falls back to the baseline with a note when no artifact is trained. System prompt updated. Live NVDA: historical_sim / monte_carlo_gbm distinct, xgboost falls back honestly.
- 2026-05-29 — Feature set: added **`B_perc`** (Bollinger %B — scale-free band position) → 15 price features. Evaluated a proposed indicator list (RSI/MACD/BB_High/BB_Low/SMA_20/EMA_9_of_MACD/MA_30): rejected most as raw price-levels that break pooling or already present in scale-free form; only %B added new signal. Noted `macd_hist` is mildly price-scaled (future fix: `/close`). NOTE: feature change makes existing pooled artifacts stale → retrain.
- 2026-05-29 — **DECISION: defer calendar/date seasonality features.** They're scale-free and leakage-safe, but pooling does NOT raise effective sample size for market-wide calendar effects (all tickers share the calendar → ~6 Novembers, cross-sectionally correlated → high overfit risk, weak at 20d horizon in liquid large-caps). The economically-motivated alternative is **earnings proximity** (`days_to_next_earnings`), deferred until an earnings-date provider with point-in-time history exists. Any new feature must prove out via Phase 6 OOS backtesting before keeping.
- 2026-05-29 — Integration audit of Phases 1–5 into the chat agent: closed two gaps. (1) `build_news_features` (AV sentiment + event flags) was **orphaned** — called by nothing in src; added a `get_news_sentiment` agent tool (AV default, Claude opt-in) so numeric sentiment is reachable. (2) `_load` dropped `LoadResult.issues`; `get_price_summary`/`compute_indicators` now include `data_warnings` (stale/sparse data) so the agent can caveat. Agent now exposes 6 tools. By-design (not a gap): no full-report tool — agent composes from granular tools.
- 2026-05-30 — **days_to_next_earnings built (both context + model feature).** Source: yfinance (`get_earnings_dates`, keyless, history to 2014; `.calendar` for upcoming) — chosen over Finnhub free (upcoming only) and AV (25/day). Part 1 (context): `EarningsProvider` protocol + yfinance impl + registry + `data/earnings.py` (`earnings_context` real next date) → report Earnings section + uncertainty note when earnings in horizon + `get_earnings_context` agent tool (7 tools). Part 2 (feature): 16th feature `days_to_next_earnings` = leakage-safe cadence estimate (only past dates + ~91d cadence; identical at train/inference) threaded through price_features/assembler/pooled/train_pooled (fetches earnings per ticker)/ml (registry fetches target's earnings); imputer `keep_empty_features=True`; ML models built per-call with registry (`MODEL_REGISTRY`→`MODEL_NAMES`+factory). **Display≠feature by design:** report shows real next date, model uses cadence estimate (NVDA live: feature 82 vs display 88). Retrained xgboost: 61 tickers / 91,073 rows / 16 features, 0 earnings-fetch failures. Gate green (159 pytest). Dep: lxml.
- 2026-05-30 — DECISION: defer ML+MC ensemble (`forecasting/ensemble.py`) until AFTER Phase 6. Rationale: principled ensemble weighting needs OOS skill (∝1/Brier) from backtesting, and only the backtest harness can validate whether an ensemble beats the best single model. Planned approach: linear probability pool over {historical, monte_carlo_bootstrap, xgboost}, skill-weighted, with derived stats (E[r], VaR, CI) RECOMPUTED from the blended bucket CDF (never average quantiles). Equal-weight version is trivial to add earlier for comparison but would be uncalibrated/unvalidated.
- 2026-05-30 — **Synthesis layer (Role C) built** — the integrated "numbers + narrative" read, the project's core goal made first-class (exec summary was templated). `llm/synthesizer.py` + `prompts/synthesis.py`: `Synthesis` schema (overview/alignments/tensions/confidence); reconciles forecast(s) + news summary + AV sentiment + earnings + technicals. Guard: reuses **NumberGrounding** (moved agent/guards→llm/guards; agent re-exports) — the LLM may interpret numbers but the grounding check rejects any decimal/percent not in the inputs (1 retry, else raises). Report gains an Integrated Analysis section (after exec summary); `run_analyze` computes AV sentiment + calls synthesize, degrades gracefully. Live NVDA: high-quality reconciliation — flagged institutional-selling-vs-high-P(up), bear-case-37%-vs-+13.4%-E[r], earnings-beyond-horizon→wider distribution; 0 grounding violations. Gate green (162 pytest). Decision: synthesis is orthogonal to Phase 6 but gets richer once `get_calibration` exists (can state how trustworthy the forecast is).
- 2026-05-30 — **Earnings-jump Monte Carlo (`monte_carlo_jump`) built.** GBM plus an empirical jump fired when the next earnings date lands in the horizon. The jump is bootstrapped from the stock's OWN historical post-earnings close-to-close log moves (`historical_earnings_moves` pure fn), capturing real, *skewed* event risk a plain price model can't see — NVDA live: 16 moves, σ=8.5%, positive-skewed → widens upper tail + worsens VaR (−14.5%→−15.4% at 65d). Key correctness choices: (a) **calibration needs a longer window than the forecast** — the ~420d forecast series holds only ~4–5 quarterly earnings (<`_MIN_JUMP_SAMPLES`=8), so the jump fetches ~4y of prices via the registry *just* for calibration, ending at `as_of` (point-in-time preserved); without this the jump silently never fired. (b) **horizon−1 day replacement** — simulate horizon−1 normal GBM days + 1 jump draw so the earnings day isn't double-counted as both a diffusion day and a jump day. (c) **moves used as-is (mean-inclusive)** — keeps the stock's real post-earnings skew rather than imposing a zero-mean shock; consistent with the tool's empirical-extrapolation philosophy and still non-advisory (every figure data-derived, no LLM number). Graceful GBM fallback + disclosing `note` when no earnings data / earnings beyond horizon / too few moves. Wired into `pipelines.forecast` (`MODEL_NAMES`+`_build_model`, registry-fed); `FakeProvider` gained `earnings_dates`. Gate green (156 pytest; 5 new: golden moves, 3 fallback notes, fattened-tails behavioral). NOTE: NVDA next earnings 2026-08-26 (~88 cal days) → fires only at horizon ≳62; shorter horizons correctly report "beyond horizon; GBM only".
