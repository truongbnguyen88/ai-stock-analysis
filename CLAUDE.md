# CLAUDE.md — Project Working Agreement

Operational guide for executing the roadmap. Global standards live in `~/.claude/CLAUDE.md`; this file is **project-specific deltas only** — do not restate global rules.

## Orientation (read before non-trivial work)
- Architecture, layers, module responsibilities, agent design → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Scope tiers + ordered build plan → [docs/ROADMAP.md](docs/ROADMAP.md)
- RAG layer (SEC-grounded research assistant): plan → [docs/RAG_IMPLEMENTATION_PLAN.md](docs/RAG_IMPLEMENTATION_PLAN.md); ordered build steps + locked decisions → [docs/RAG_TODO.md](docs/RAG_TODO.md); concepts/theory → [docs/rag_concepts.md](docs/rag_concepts.md); per-phase build journal → [docs/rag_implementation_notes.md](docs/rag_implementation_notes.md)
- Advanced RAG track (hybrid/rerank/eval/agentic/graph/RL) — **plan of record** → [docs/ADVANCED_RAG_TODO.md](docs/ADVANCED_RAG_TODO.md) (phases A1–A6); design rationale only (superseded on specifics) → [docs/RAG_IMPLEMENTATION_PLAN_ADV.md](docs/RAG_IMPLEMENTATION_PLAN_ADV.md)
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
- **RAG layer** (see RAG_TODO.md): `providers/sec_edgar.py` (EDGAR **official API**, not scraping — Protocol, throttled, cached) → `documents/` (download/parse/section-detect, normalized to `schemas/documents.py`) → `rag/` (chunk/embed/store/retrieve; embeddings + vector store each behind a Protocol — local `fastembed`/BGE default, Chroma default) → `research/` (single grounded synthesis call). `rag/` retrieval does **no LLM calls**; the only paid LLM call is the final memo synthesis. Embeddings are computed **once at ingestion**, never per query over the corpus.

## How to execute a roadmap step
1. State the step number + its bracketed deliverable.
2. Implement the smallest vertical slice that satisfies it.
3. Write tests in the same change (see Testing). A step is not done until its tests pass.
4. Run `make check` (lint + typecheck + test) before declaring done.
5. Do not start the next step until the current one is green.
- Respect phase gates: never wrap a pipeline in an `agent/` tool before that pipeline exists (agent phases 4.5 / 6.5 come after their dependencies).
- **RAG phases:** when a RAG_TODO phase (P-N) lands green, in the same change (a) mark it `[x]` in [docs/RAG_TODO.md](docs/RAG_TODO.md), and (b) **append a section to [docs/rag_implementation_notes.md](docs/rag_implementation_notes.md)** explaining that phase's mechanism (role, key files, how it works step-by-step, how the next phase uses it, key decisions). The notes doc is the learning archive — keep it complete and current.
- **Advanced RAG phases (A-N, [docs/ADVANCED_RAG_TODO.md](docs/ADVANCED_RAG_TODO.md)):** these are *learning-oriented* — the user is studying the advanced-RAG concepts (hybrid search, reranking, retrieval eval, agentic RAG, GraphRAG, retrieval+RL), so every phase carries a teaching obligation in addition to the P-N rules above. When an A-N phase lands green, in the same change:
  - (a) mark it `[x]` in [docs/ADVANCED_RAG_TODO.md](docs/ADVANCED_RAG_TODO.md), and (b) append the build-mechanism section to [docs/rag_implementation_notes.md](docs/rag_implementation_notes.md) (same rule as P-N).
  - (c) **In the chat response, clearly explain what was implemented and the foundational/mathematical concepts behind the feature** — definitions, the formula(s) with notation, why it works, assumptions/failure modes, and how it composes with the existing retrieval stack. High signal-to-noise; no filler.
  - (d) **Append a new concepts/theory section to [docs/rag_concepts.md](docs/rag_concepts.md)** capturing that same math/theory durably (not the build log — that's the notes doc). State each formula with its convention, define every symbol, give a worked micro-example where it clarifies, and cite the canonical source where relevant (e.g. RRF, nDCG, BM25, IPS/doubly-robust). Obey the repo's GitHub-MathJax escaping rules (see "Docs: math & Mermaid").
  - Division of labor: `rag_concepts.md` = the *why/math* (durable theory); `rag_implementation_notes.md` = the *how/where* (per-phase build journal). Keep both current.

## Code style (project-specific)
- **Comment for the next maintainer, not the parser.** Explain *why* and any non-obvious math/finance assumption (e.g. annualization factor, RSI smoothing choice, embargo length, bootstrap block size). Skip comments that merely restate the code.
- Every public function: typed signature + a concise docstring stating purpose, key assumptions, and units (e.g. "returns annualized volatility, 252-day basis").
- Each financial/statistical formula gets a one-line comment naming the convention and any source assumption.
- Prefer small pure functions over classes unless state/interface is needed (forecast models and providers are the main legitimate classes).
- `pathlib`, explicit typing, `structlog` (no `print`), config-driven — per global rules.

## Docs: math & Mermaid (rendered on GitHub)
Markdown docs render on the GitHub website via **MathJax** (`$…$` inline, `$$…$$` block). GitHub **unescapes backslash-escapes of markdown-significant punctuation — `#`, `_`, `*`, `` ` ``, `[`, `]` — even inside math**, so the bare char reaches MathJax and errors. Rules for any doc with equations:
- **Never write `\#`, `\_`, `\*` (or `` \` ``, `\[`, `\]`) inside `$…$`/`$$…$$`.** They throw *"macro parameter character #"* / *"'_' allowed only in math mode"* on GitHub even though they're valid LaTeX locally.
  - Count/cardinality → use an indicator sum `\frac{1}{N}\sum_t \mathbb{1}[\,\cdot\,]` or `\lvert\{\cdots\}\rvert`, **not** `\#`.
  - Literal star → `{*}` or `\ast`, **not** `\*`.
- **Code/config identifiers with underscores** (`as_of`, `days_to_next_earnings`, `hist_vol_20`) go in **markdown code spans** (backticks) *outside* math, or use a clean math symbol (`d_{\text{earn}}`, `t_0`). Never put a `_`-bearing name inside `\text{}`/`\texttt{}`.
- **Safe to keep:** `\%`, `\{`, `\}`, `\!`, and all backslash+letter macros (`\frac`, `\sigma`, `\le`, `\mathbb`, `\text{…}` with no underscore). Only the six markdown-significant escapes above are the hazard.
- **Mermaid:** use fenced ```` ```mermaid ```` blocks; quote any node label containing punctuation (`["P(r) ratio"]`); `<br/>` for line breaks; avoid raw `<`/`>` in labels (use words) — the `-->` arrows are fine.
- **Before committing a math/diagram doc**, grep the math spans for the hazards: no `\#`, `\_`, `\*` inside `$…$`, and no bare `_` inside `\text{}`/`\texttt{}`.

## Testing requirements (per module type)
- **Indicators / features:** golden-file tests with hand-verified expected values; edge cases (short series, NaNs, flat prices). Leakage assertions for features.
- **Providers:** normalization vs recorded fixtures; fallback chain triggers on simulated rate-limit/unavailable; cache hit/miss.
- **Forecasting:** probabilities sum to 1; baseline reproducible under fixed seed.
- **Backtesting:** splitter never overlaps train/test; embargo respected; calibration math vs known cases.
- **LLM / agent:** output schema conformance; anti-forecast guard rejects LLM-emitted numbers; agent numeric-grounding guard rejects fabricated figures; citations reference real fetched URLs.
- **RAG:** parsing/section-detection golden fixtures; chunking preserves section boundaries + carries metadata (no tiny/giant chunks); embeddings via a deterministic `FakeEmbedder` (**never download a model in CI**); vector store uses a temp dir; retrieval filter correctness + empty-retrieval → "Insufficient evidence found."; the synthesis **citation guard** rejects any cited source/chunk not in the retrieved set. SEC download tested against **recorded EDGAR fixtures** (no live calls).
- All tests deterministic: seed everything, mock all network + LLM (`FakeProvider`, `FakeEmbedder`, canned LLM responses). No live API calls in tests.

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
# RAG (see RAG_TODO.md — built incrementally P1→P8):
python -m stock_agent documents download-sec --ticker NVDA --forms 10-K 10-Q
python -m stock_agent documents ingest --ticker NVDA        # parse→chunk→embed→store (local, $0)
python -m stock_agent rag query --ticker NVDA --question "What AI growth drivers did management cite?"
python -m stock_agent research --ticker NVDA                # technicals + forecast + news + RAG → memo
```
(If a target doesn't exist yet, it's a Phase 0 deliverable — create it.)

## When uncertain
State the assumption, pick the option consistent with the invariants above, and flag it — per global "When Uncertain" rules. Ambiguity that affects correctness or leakage: ask first.
