# stock-agent — LLM-powered stock research assistant

A research/education tool that combines **statistical & ML forecasting** with an
**LLM that explains, never invents** numbers. It produces probabilistic scenario
forecasts, a direction-agnostic "big-move" reading, backtested calibration, and a
conversational agent over the same core.

> **Disclaimer:** For research and educational purposes only. **Not financial
> advice.** No buy/sell/hold recommendations. Every quantitative figure is produced
> by model code; the LLM only summarizes and explains.

---

## Core design invariants

1. **Numbers vs. narrative.** The LLM never produces probabilities, returns, VaR, or
   forecasts — those come from `indicators/`, `forecasting/`, `backtesting/`. A
   numeric-grounding guard rejects any model-emitted figure the LLM didn't get from a
   tool result.
2. **Non-advisory by construction.** The report schema has *no recommendation field*.
3. **No web scraping.** Data only via the provider abstraction (official/free APIs).
4. **Leakage prevention is a correctness requirement.** Features at time `t` use only
   data available at `t`; targets strictly after the feature cutoff; scalers/imputers
   fit on the train fold only.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## What it does

- **`analyze`** — prices + indicators + probabilistic scenarios (historical-sim
  baseline + calibrated ML overlay) + news summary → a structured research report
  with VaR, predictive intervals, calibration status, horizon-scaled big-move, and a
  horizon-trust caveat.
- **`forecast`** — a single ticker/horizon scenario forecast. Default `ensemble` — a
  calibrated probability pool of all the models (historical + Monte-Carlo + GARCH + ML);
  the robust no-regret default since you can't know a-priori which single model wins for
  a given name/horizon. Any individual model is selectable via `--model`.
- **Big-move signal** — `P(|r| > k)` with up/down tails at the horizon's inner `k`
  (5/10/15% for h20/h30/h60). This is ML's genuine niche: direction is ≈ efficient,
  magnitude/volatility is predictable. See [docs/models_explanation.md](docs/models_explanation.md).
- **`backtest`** — walk-forward, embargoed, leakage-safe evaluation (Brier / log-loss
  / AUC / ECE **+ prediction-interval coverage**) comparing every model to baselines on
  identical folds. CIs are **conformally calibrated** (`conformal-calibrate` → a pooled,
  distribution-free correction so a stated 90% interval actually covers ~90% OOS).
- **`chat`** — a conversational agent (Role C) that orchestrates the tools, grounds
  every number, and can export an executive summary to PDF/DOCX/Markdown. It can also
  answer **SEC-filing questions** (`search_filings`), run **multi-hop filing research**
  (`research_multistep` — a bounded ReAct loop for comparative / change-over-time /
  bridging questions one retrieval can't answer), and produce an **integrated brief**
  (`research_summary`) over the RAG layer below — all cited. **Hybrid routing:** by
  default the LLM picks the tool(s); pass `--domain <area>` to dispatch a capability
  **deterministically**, skipping the routing LLM call when you already know what you want.
  Domains: `predictions`, `news`, `filings`, `technicals`, `brief`, each with `--variant`
  (e.g. `chat --domain predictions --variant big-move --ticker NVDA`;
  `chat --domain filings --variant multi "compare NVDA and AMD risks"`). `--tool <route>`
  is the advanced exact-route escape hatch.
- **`research`** — a **SEC-grounded equity research memo** (RAG): fuses filings
  (10-K/10-Q/8-K) + news + the forecast into one cited, non-advisory brief. Retrieval is
  100% local; every filing claim carries a citation, and a citation + number guard rejects
  anything not in the retrieved evidence (same numbers-vs-narrative invariant).
- **`documents` / `rag`** — manage the SEC corpus: `documents download-sec` (official EDGAR
  API) → `documents ingest` (parse → chunk → embed → store) → `rag query` (grounded filing QA)
  or `rag ask` (multi-hop ReAct research; `--single` for one-shot).

## Quickstart

```bash
# 1. Install (editable, with dev tooling) — Python 3.12 recommended
make install            # = pip install -e ".[dev]"

# 2. Configure secrets (all optional; a missing key only errors when its
#    capability is invoked — yfinance needs none)
cp .env.example .env    # then fill in ANTHROPIC_API_KEY for LLM features

# 3. Get models (pick one)
make pull-models        # download the latest CI-trained artifacts + conformal.json (recommended)
#   …or train locally:
python -m stock_agent train --all            # logistic + tuned lightgbm × {20,30,60}, calibrated
python -m stock_agent conformal-calibrate    # (optional) honest-coverage CIs/VaR → conformal.json

# 4. Run
python -m stock_agent analyze  --ticker NVDA --days 90
python -m stock_agent forecast --ticker MSFT --horizon 30           # default model = ensemble
python -m stock_agent forecast --ticker MSFT --horizon 30 --model monte_carlo_garch  # or pick one
python -m stock_agent backtest --ticker AAPL
python -m stock_agent chat                 # conversational agent (CLI)
make ui                                     # Streamlit chat frontend

# 5. SEC-grounded research (RAG) — optional; needs the [rag] extra + SEC_USER_AGENT
python -m stock_agent documents download-sec --all --years 3   # official EDGAR API (free)
python -m stock_agent documents ingest --all                   # parse→chunk→embed→store
python -m stock_agent rag query --ticker NVDA --question "What AI growth drivers did management cite?" --answer
python -m stock_agent research --ticker NVDA                   # technicals + forecast + news + filings → cited memo
```

`analyze`/`chat` degrade gracefully without `ANTHROPIC_API_KEY` (numbers still
produced; narrative sections note the LLM was disabled). Pass `--no-llm` to skip it.

## Models: trained in CI, served locally

The ML artifacts (~100 MB) are **not** in the repo (`outputs/models/` is gitignored).
A monthly **GitHub Actions** job (`.github/workflows/retrain.yml`) retrains the
toolkit, runs a promote gate (`verify-models` — structural + a data-quality floor),
and publishes a rolling `models-latest` release. Pull it with `make pull-models`; the
app falls back to the historical-sim baseline when an artifact is absent. The ML toolkit
is **logistic** (stable names) + **tuned lightgbm** (volatile names' big-move tails),
at horizons **{20, 30, 60}**, isotonic-calibrated (`CalibratedClassifierCV(cv=3)`). The
interactive default is the **`ensemble`** — a calibrated pool of these plus the
Monte-Carlo / GARCH baselines (OOS-validated as the robust no-regret choice).

## SEC-grounded research (RAG layer)

A second, self-contained layer turns SEC filings into a **grounded research assistant** —
distinct from the forecasting core and held to the same invariants (numbers from models, no
recommendations, no scraping). Pipeline: `providers/sec_edgar` (official EDGAR API, throttled +
cached) → `documents/` (download · parse · section-detect) → `rag/` (chunk · embed · store ·
retrieve) → `research/` (one grounded synthesis call → cited memo).

- **Embeddings are computed once at ingestion**, never per query, each behind a Protocol.
  **Production = Voyage `voyage-4`** (chosen via a labeled retrieval A/B that beat local — MRR
  0.72→0.89; `voyage-finance-2` lost); **local `fastembed`/BGE** (onnxruntime, no torch, $0)
  remains a complete fallback. The vector store (Chroma) namespaces a **separate collection per
  embedder**, so switching providers never mixes vector dimensions.
- **Only one paid LLM call** in the whole flow — the final memo synthesis. Download, parse,
  chunk, embed, and retrieval are 100% local. A **citation guard** rejects any cited
  source/chunk not in the retrieved set; empty retrieval → *"Insufficient evidence found."*
- **Cost ceiling:** a configurable `rag_max_embed_tokens` refuses an ingest before it embeds;
  large corpus embeds batch + retry per ticker so a transient blip never aborts the run.

Build steps + locked decisions: [docs/RAG_TODO.md](docs/RAG_TODO.md); concepts:
[docs/rag_concepts.md](docs/rag_concepts.md); per-phase mechanism notes:
[docs/rag_implementation_notes.md](docs/rag_implementation_notes.md).

## Development

```bash
make check    # ruff + mypy --strict + pytest (the gate; must be green to advance)
make test     # pytest only
make format   # ruff format + safe autofix
```

All tests are deterministic (seeded; network + LLM mocked via `FakeProvider` and
canned responses) — no live API calls in tests.

## Documentation

| Doc | What |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, module boundaries, agent design, model lifecycle |
| [docs/models_explanation.md](docs/models_explanation.md) | Every forecasting model, math, assumptions, failure modes |
| [docs/validations_results.md](docs/validations_results.md) | Backtest / tuning / calibration results |
| [docs/RAG_TODO.md](docs/RAG_TODO.md) · [docs/rag_concepts.md](docs/rag_concepts.md) · [docs/rag_implementation_notes.md](docs/rag_implementation_notes.md) | SEC-grounded RAG layer: build plan, concepts, per-phase mechanism notes |
| [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/TASKS.md](docs/TASKS.md) | Build plan + progress log |

## Data & secrets

Free/official data providers only (yfinance keyless; Alpha Vantage / Finnhub /
Marketaux optional). Respect each provider's terms and rate limits. Secrets live only
in `.env` (gitignored). The RAG layer adds: **`SEC_USER_AGENT`** (a name + contact email,
required by EDGAR's fair-access policy for downloads) and an optional **`VOYAGE_API_KEY`**
(only when `EMBEDDING_PROVIDER=voyage`; the default local `fastembed` needs no key).
