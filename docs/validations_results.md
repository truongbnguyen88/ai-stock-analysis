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
| 2026-05-30 | ML improvement campaign — model class / calibration / features (h20) | ⚠️ plain logistic ≫ xgboost (ties baselines on 2/3, real tail AUC) but still no clear win | **Adopt plain scaled logistic** as the ML model; baselines stay primary |

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

---

## 2026-05-30 — ML improvement campaign (model class · calibration · features), h20

**Question.** After the initial xgboost lost to the baselines, can the pooled ML model be *improved* to beat them? Explored — in order — model class, regularization, post-hoc calibration, and feature engineering, each validated on the same 3-ticker (NVDA/MSFT/KO) h20 walk-forward harness (62 OOS / 11 folds).

**The decisive diagnostic (Exp 1).** Reading the AUC *by threshold* exposed where the (little) signal lives:

> The **±5% / ±10% thresholds have real discrimination** (AUC up to 0.77), but the **0% threshold — pure direction — sits at ≈ 0.5 for every model.** So the models can predict *whether a big move happens* (volatility clusters → forecastable) but **not which way** (20-day direction ≈ efficient for liquid large-caps). All ML resolution here is *tail/volatility*, not directional.

**Experiments & findings.**

| Exp | What | Key result |
|---|---|---|
| 1 | Model-class sweep (logistic / lightgbm / xgboost) | lightgbm ≈ xgboost exactly → *not* booster-specific. `logistic` (balanced) had the strongest tail AUC (0.77) but catastrophic calibration (ECE up to 0.29). |
| 2 | Linear variants + regularized booster | **Dropping `class_weight="balanced"` is a big win** (NVDA Brier 0.340→0.231, ECE 0.291→0.119, tail AUC kept). Plain logistic **ties baselines on MSFT/KO** with *better* calibration (MSFT ECE 0.039). Regularizing xgboost helped calibration but it still found no tail signal (AUC ≈0.5). *Also fixed a latent bug: pooled logistic was unscaled.* |
| 3 | Post-hoc calibration of plain logistic (`CalibratedClassifierCV` isotonic / Platt) | Platt helped the overconfident high-vol case (NVDA ECE 0.119→0.093) and marginally beat the baseline on MSFT, but ~neutral overall / slightly worse on KO. Isotonic barely moved NVDA — its overconfidence is an **OOS regime shift**, which *training-data* CV calibration can't see (needs a temporal holdout). |
| 4 | Feature engineering (+`mom_12_1`, `volume_ratio`, `skew_60`, `high_252_ratio` → 20 features) | **Did not help; slightly hurt** (NVDA logistic 0.231→0.243 — overfit). The 0%-direction AUC stayed ≈0.5 → momentum added *no* directional signal at 20d. **Reverted.** |

**Outcome — what changed.**

- ✅ **ADOPTED: plain scaled logistic** as the pooled ML model (`StandardScaler` → `LogisticRegression`, **no** `class_weight="balanced"`). A real, validated upgrade over the original xgboost:

  | | avg Brier | avg ECE | tail resolution |
  |---|---|---|---|
  | original xgboost | 0.173 | 0.099 | AUC ≈ 0.5 (none) |
  | **plain logistic** | **0.161** | **0.076** | AUC 0.65–0.76 (real) |
  | baselines | 0.151 | 0.046 | AUC ≈ 0.5 |

  It **ties the baselines on MSFT/KO** and adds genuine tail/volatility discrimination they lack; only high-vol **NVDA** still lags (residual OOS overconfidence).

- ❌ **NOT adopted:** post-hoc calibration (marginal gain, added complexity), the 4 new features (no 20-day signal — reverted), the boosters (logistic dominates them here).

**Bottom line.** ML is now a competitive, well-calibrated model with real tail resolution — but it still does **not clearly beat the baselines at 20d**, because the only forecastable piece (tail magnitude / vol) is already captured by the vol-conditioned Monte-Carlo baseline, and 20-day **direction is ~efficient** (0% AUC ≈ 0.5 for everything tried). **Baselines remain primary.**

**Future leads (deferred, ranked).** (1) Different **horizon** — 5-day may hold more short-term-reversal signal; the whole campaign was h20-only. (2) **Vol-scaled targets** (thresholds in σ units) to de-muddle the cross-sectional pooling. (3) **Temporal-holdout** calibration for high-vol OOS shift. (4) **ML + baseline ensemble** (baseline calibration + ML tail resolution). Direction at 20d is likely a dead end regardless.

**Reproduce.** Variant experiments used a `_make_classifier` monkeypatch (see the campaign scripts); the adopted config is now the default `_make_classifier("logistic")`. Re-check with:
```bash
python -m stock_agent backtest --ticker NVDA --model logistic   # plain scaled logistic
python -m stock_agent backtest --ticker NVDA                    # baselines, same folds
```
