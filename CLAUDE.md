# CLAUDE.md — Project Working Agreement

Operational guide for executing the roadmap. Global standards live in `~/.claude/CLAUDE.md`; this file is **project-specific deltas only** — do not restate global rules.

## Orientation (read before non-trivial work)
- Architecture, layers, module responsibilities, agent design → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Scope tiers + ordered build plan → [docs/ROADMAP.md](docs/ROADMAP.md)
- Always know which **roadmap step** you are on; if unclear, ask before coding.

## Non-negotiable invariants
1. **Numbers vs. narrative.** LLM never produces probabilities, returns, VaR, or forecasts. All quantitative figures come from `indicators/`, `forecasting/`, or `backtesting/`. The LLM only summarizes, routes, and explains.
2. **Non-advisory by construction.** No buy/sell signals. The report schema has **no recommendation field** — do not add one.
3. **No web scraping.** Data only via the provider abstraction (official/free APIs).
4. **Secrets only in `.env`** (gitignored). Never hardcode keys; access via `settings.py`.
5. **Dependency direction is downward only** (see ARCHITECTURE §2). `agent/` and `cli/` → `pipelines/` → core modules → `providers/`. Never invert.
6. **Leakage prevention is a correctness requirement.** Features at time `t` use only data available at `t`; targets strictly after feature cutoff; scalers fit on train fold only. Treat leakage tests like any other failing test.

## Module boundaries (where code goes)
- Pure compute (no I/O, no state) → `indicators/`, `features/`.
- Anything touching an external API → behind a provider Protocol in `providers/`, normalized to `schemas/` at the boundary. Business logic never sees raw JSON.
- Forecast models implement the common interface in `forecasting/base.py` (`ForecastModel` Protocol: a `name` attr + `forecast(series, *, horizon_days, as_of) -> ScenarioForecast`).
- LLM calls → `llm/` (Role A summarizer) or `agent/` (Role B router) only. Both pass through their `guards.py`.
- `pipelines/` and `cli/` stay thin — orchestration only, no business logic.

## How to execute a roadmap step
1. State the step number + its bracketed deliverable.
2. Implement the smallest vertical slice that satisfies it.
3. Write tests in the same change (see Testing). A step is not done until its tests pass.
4. Run `make check` (lint + typecheck + test) before declaring done.
5. Do not start the next step until the current one is green.
- Respect phase gates: never wrap a pipeline in an `agent/` tool before that pipeline exists (agent phases 4.5 / 6.5 come after their dependencies).

## Code style (project-specific)
- **Comment for the next maintainer, not the parser.** Explain *why* and any non-obvious math/finance assumption (e.g. annualization factor, RSI smoothing choice, embargo length, bootstrap block size). Skip comments that merely restate the code.
- Every public function: typed signature + a concise docstring stating purpose, key assumptions, and units (e.g. "returns annualized volatility, 252-day basis").
- Each financial/statistical formula gets a one-line comment naming the convention and any source assumption.
- Prefer small pure functions over classes unless state/interface is needed (forecast models and providers are the main legitimate classes).
- `pathlib`, explicit typing, `structlog` (no `print`), config-driven — per global rules.

## Testing requirements (per module type)
- **Indicators / features:** golden-file tests with hand-verified expected values; edge cases (short series, NaNs, flat prices). Leakage assertions for features.
- **Providers:** normalization vs recorded fixtures; fallback chain triggers on simulated rate-limit/unavailable; cache hit/miss.
- **Forecasting:** probabilities sum to 1; baseline reproducible under fixed seed.
- **Backtesting:** splitter never overlaps train/test; embargo respected; calibration math vs known cases.
- **LLM / agent:** output schema conformance; anti-forecast guard rejects LLM-emitted numbers; agent numeric-grounding guard rejects fabricated figures; citations reference real fetched URLs.
- All tests deterministic: seed everything, mock all network + LLM (`FakeProvider`, canned LLM responses). No live API calls in tests.

## Cost-efficiency rules (generation + runtime)
**During development (token budget):**
- Read only the files a step touches; use search to locate, don't bulk-read the tree.
- Prefer incremental `Edit` over rewriting files. No drive-by refactors of unrelated code.
- Reuse existing utilities/schemas before writing new ones; check `schemas/` and `providers/` first.
- Keep responses focused on the current step; don't re-explain the architecture (it's in docs).

**At runtime (API cost):**
- Provider calls go through the cache layer (TTL, disk) — respect free-tier rate limits; never bypass cache in normal flow.
- LLM calls use **prompt caching** (use the `claude-api` skill); dedup + rank news *before* sending to the LLM so we summarize fewer tokens.
- Bound heavy agent tools (`run_backtest`, `get_calibration`) with argument limits + timeouts.
- Default to the configured model in `settings`; don't hardcode model IDs in logic.

## Commands
```bash
make check        # lint + typecheck + test (run before "done")
make test         # pytest
make lint         # ruff
make typecheck    # mypy
python -m stock_agent analyze  --ticker NVDA --days 90
python -m stock_agent forecast --ticker MSFT --horizon 20
python -m stock_agent chat
```
(If a target doesn't exist yet, it's a Phase 0 deliverable — create it.)

## When uncertain
State the assumption, pick the option consistent with the invariants above, and flag it — per global "When Uncertain" rules. Ambiguity that affects correctness or leakage: ask first.
