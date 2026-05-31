# Validation Results

Durable record of **out-of-sample validation experiments**. The project's discipline:
a model or feature is not surfaced in the product until it beats the baselines on
*identical walk-forward folds*. Each such test — and the decision it drove — is recorded
here so the empirical basis for what we ship (and what we don't) is traceable.

- **Methodology:** `backtesting/` (`splitter` · `runner` · `metrics` · `calibration`); design in [ARCHITECTURE.md §10](ARCHITECTURE.md). Model mechanics in [models_explanation.md](models_explanation.md).
- **Raw artifacts:** each run writes config + per-model metrics to `outputs/experiments/<run_id>/` (gitignored). Reproduce commands are given per entry.

## Reading the metrics

Three complementary out-of-sample lenses — read them together, not in isolation:

| Metric | Direction | Measures |
|---|---|---|
| **Brier** | ↓ lower | Overall probabilistic accuracy = calibration **+** sharpness. The headline score. |
| **ECE** | ↓ lower | *Calibration only* — are the probabilities honest? `0` perfect; `<0.05` well, `0.05–0.10` moderate, `>0.10` poor. (**MCE** = worst single bin.) |
| **ROC AUC** | ↑ (0.5 = none) | *Discrimination* — can the model rank outcomes? `0.5` = no directional edge. |

The crucial interaction: a model can have **AUC > 0.5** (some real skill) yet still **lose on Brier** if it is *overconfident* (**high ECE**) — the sharpness it adds isn't earned. An unconditional baseline that honestly hedges near the base rate (AUC ≈ 0.5, low ECE) frequently wins overall. So "ML has a signal" and "ML is worth shipping" are different claims; Brier + ECE settle the second.

## Index

| Date | Experiment | Verdict | Decision |
|---|---|---|---|
| 2026-05-30 | Pooled xgboost vs baselines (h20, 3 tickers) | ❌ xgboost worse on Brier **and** ECE on all 3 | Keep baselines primary; **do not** expand ML training or surface ML |

---

## 2026-05-30 — Pooled xgboost vs baselines (h20)

**Question.** Does the pooled, price-only **xgboost** classifier beat the unconditional baselines (historical-sim, Monte-Carlo bootstrap) out-of-sample at a 20-day horizon? This is the gate for whether to train the full `{model type × horizon}` grid and surface ML as a forecast.

**Setup.**
- **Tickers:** NVDA (high-vol), MSFT (large-cap), KO (low-vol) — chosen to test generalization across volatility regimes.
- **Horizon:** 20 trading days. Walk-forward, expanding train, **embargo = stride = 20**, `min_train = 252`, `test_size = 6` → **11 folds, 62 OOS forecasts/ticker**, window 2021-06-21 → 2026-04-30.
- **Models on identical folds:** `historical_sim`, `monte_carlo_bootstrap` (5 000 paths), `xgboost`.
- **xgboost** = pooled over the 61-ticker universe, **retrained per fold** on data ≤ `train_end` only (leakage-safe walk-forward; training rows grew 14 152 → 87 352 across the expanding folds). `seed = 42`.

**Results** — Brier ↓, log loss ↓, ECE ↓ (lower is better); ROC AUC per return threshold (`0.5` = no skill, `n/a` = single-class).

**NVDA** (62 OOS / 11 folds)

| Model | Brier | log loss | ECE | AUC −10% | −5% | 0% | +5% | +10% |
|---|---|---|---|---|---|---|---|---|
| historical_sim | **0.2049** | 0.6022 | **0.042** | 0.40 | 0.34 | 0.39 | 0.31 | 0.39 |
| monte_carlo_bootstrap | 0.2050 | **0.6004** | 0.047 | 0.47 | 0.40 | 0.41 | 0.35 | 0.35 |
| ml_xgboost | 🔴 0.2492 | 🔴 0.7621 | 🔴 0.168 | 0.52 | 0.45 | 0.48 | 0.50 | 0.52 |

**MSFT** (62 OOS / 11 folds)

| Model | Brier | log loss | ECE | AUC −10% | −5% | 0% | +5% | +10% |
|---|---|---|---|---|---|---|---|---|
| historical_sim | **0.1562** | **0.4777** | 0.074 | 0.27 | 0.34 | 0.42 | 0.42 | 0.39 |
| monte_carlo_bootstrap | 0.1570 | 0.4777 | **0.059** | 0.51 | 0.37 | 0.43 | 0.48 | 0.33 |
| ml_xgboost | 🔴 0.1728 | 0.5338 | 0.077 | 0.52 | 0.37 | 0.45 | 0.55 | **0.61** |

**KO** (62 OOS / 11 folds)

| Model | Brier | log loss | ECE | AUC −10% | −5% | 0% | +5% | +10% |
|---|---|---|---|---|---|---|---|---|
| historical_sim | 0.0921 | 0.3005 | 0.046 | n/a | 0.44 | 0.42 | 0.32 | 0.44 |
| monte_carlo_bootstrap | **0.0916** | **0.2993** | **0.036** | n/a | 0.66 | 0.38 | 0.40 | 0.03 |
| ml_xgboost | 🔴 0.0978 | 0.3112 | 0.052 | n/a | 0.33 | 0.47 | **0.67** | **0.67** |

**Interpretation.**
- **The baselines win on the metrics that matter, on every ticker.** xgboost is worse on **Brier** (NVDA badly: 0.249 vs 0.205) and worse on **ECE** (NVDA "poorly calibrated" at 0.168 vs the baseline's well-calibrated 0.042).
- **There is a faint signal:** xgboost's **AUC on the upside thresholds is consistently > 0.5** (KO +5/+10% = 0.67, MSFT +10% = 0.61), while the unconditional baselines sit near 0.5. So xgboost has *a little* genuine resolution — it ranks upside moves slightly better than chance, most visibly for the lower-vol names.
- **But it pays for that sharpness with unearned overconfidence.** The resolution it gains is smaller than the calibration it loses → net-negative Brier. This is the textbook calibration-vs-resolution trade: the honest baseline (AUC ≈ 0.5, low ECE) beats the overconfident model (AUC ≈ 0.55, high ECE).
- **Likely causes:** 20-day equity *direction* is near-unpredictable from price features (efficient-ish market → tiny signal); a pooled model over 61 mixed names doesn't capture ticker-specific volatility as well as `historical_sim`, which uses each ticker's *own* history; and a 300-tree xgboost on a near-noise target overfits in-sample patterns → overconfident OOS. The raw classifier probabilities are also **not post-hoc calibrated** at inference.

**Decision.**
- **Keep `historical_sim` + `monte_carlo_*` as the primary forecasters** (well-calibrated, cheaper, no training).
- **Do not surface xgboost as the headline forecast** — it would present *less* trustworthy probabilities. The agent's existing baseline fallback is the correct default.
- **Do not expand training** to the full `{model × horizon}` grid — unjustified given the flagship model fails to beat a free baseline at the primary horizon.
- **Future experiment (deferred, not blocking Phase 7):** wrap the ML probabilities in **post-hoc isotonic calibration** (machinery already exists in `backtesting/calibration.py`). High ECE / overconfidence is exactly what isotonic fixes; *if* recalibration closes the Brier gap while preserving the upside-AUC edge, ML becomes worth revisiting. Until then ML stays experimental.

**Caveats.** 62 OOS forecasts × 3 tickers is a **directional read, not proof**. Stride = horizon keeps the OOS points quasi-independent, but the sample is small and three names is narrow. The verdict is acted on because the pattern is *consistent* — xgboost worse on both Brier and ECE across all three, with only a weak AUC consolation.

**Reproduce.**
```bash
# ML (per-fold pooled refit — the slow path) and baselines on identical folds:
python -m stock_agent backtest --ticker NVDA --model xgboost
python -m stock_agent backtest --ticker NVDA            # historical_sim + monte_carlo_*
# repeat for MSFT, KO. Artifacts land in outputs/experiments/<run_id>/.
```
