# TASKS — Progress Tracker

Living checklist mirroring [ROADMAP.md](ROADMAP.md). Update on every step transition. Full detail lives in ROADMAP/ARCHITECTURE — keep this file terse.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done (tests green)

---

## ▶ Current
- **Phase:** 2 — Indicators
- **Next step:** Step 8 — `indicators/*` pure functions (MA/RSI/MACD/returns/vol/drawdown) + golden-file tests
- **Gate to advance:** `make check` green
- **Last updated:** 2026-05-29

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

## Phase 2 — Indicators
- [ ] 8. `indicators/*` + golden-file tests

## Phase 3 — News + LLM (Role A)
- [ ] 9. `news/` fetch → dedup → clean → rank
- [ ] 10. `llm/` client + prompts + `news_summarizer.py` + `guards.py`

## Phase 4 — Baseline forecast + report
- [ ] 11. `forecasting/` base + buckets + historical
- [ ] 12. `reports/` builder + render_md + templates
- [ ] 13. `pipelines/analyze.py` + `cli/app.py` (`analyze`)
- [ ] 14. Integration test: analyze pipeline with fakes

## Phase 4.5 — Chat agent (Role B) ← MVP milestone
- [ ] 15. `agent/tools.py` (analyze + baseline-forecast tools)
- [ ] 16. `agent/runtime.py` + prompts + numeric-grounding guard
- [ ] 17. `chat` CLI command + agent integration tests

## Phase 5 — Statistical & ML models
- [ ] 18. `forecasting/monte_carlo.py` + `forecast` CLI
- [ ] 19. `features/` price + news + assembler (point-in-time)
- [ ] 20. `forecasting/ml.py` (logistic → tree ensembles)

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
