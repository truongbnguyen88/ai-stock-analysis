# TASKS — Progress Tracker

Living checklist mirroring [ROADMAP.md](ROADMAP.md). Update on every step transition. Full detail lives in ROADMAP/ARCHITECTURE — keep this file terse.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (tests green)

---

## ▶ Current
- **Phase:** 6 — Backtesting & calibration
- **Next step:** Step 21 — `backtesting/splitter.py` (walk-forward + embargo) + leakage tests
- **Gate to advance:** `make check` green
- **Last updated:** 2026-05-29
- 🎉 **MVP milestone reached** (Phases 0–4.5): analyze CLI + chat agent working live.

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
- [x] 18. `forecasting/monte_carlo.py` (GBM + block bootstrap) + `forecast` CLI
- [x] 19. `features/` price + news + assembler (point-in-time, leakage-tested)
- [x] 20. `forecasting/ml.py` (logistic, XGBoost, LightGBM, random forest)

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
