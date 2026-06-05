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
  every number, and can export an executive summary to PDF/DOCX/Markdown.

## Quickstart

```bash
# 1. Install (editable, with dev tooling) — Python 3.12 recommended
make install            # = pip install -e ".[dev]"

# 2. Configure secrets (all optional; a missing key only errors when its
#    capability is invoked — yfinance needs none)
cp .env.example .env    # then fill in ANTHROPIC_API_KEY for LLM features

# 3. Get models (pick one)
make pull-models        # download the latest CI-trained artifacts (recommended)
#   …or train locally:
python -m stock_agent train --all   # logistic + tuned lightgbm × {20,30,60}, calibrated

# 4. Run
python -m stock_agent analyze  --ticker NVDA --days 90
python -m stock_agent forecast --ticker MSFT --horizon 30 --model lightgbm
python -m stock_agent backtest --ticker AAPL
python -m stock_agent chat                 # conversational agent (CLI)
make ui                                     # Streamlit chat frontend
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
| [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/TASKS.md](docs/TASKS.md) | Build plan + progress log |

## Data & secrets

Free/official data providers only (yfinance keyless; Alpha Vantage / Finnhub /
Marketaux optional). Respect each provider's terms and rate limits. Secrets live only
in `.env` (gitignored).
