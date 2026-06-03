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
| 2026-05-30 | ML improvement campaign — model / calibration / features / **horizon sweep** (h5–60) | ⚠️ plain logistic ≫ xgboost & ties MC at short h, but **no horizon where it beats the baselines** | **Adopt plain scaled logistic**; baselines stay primary at all horizons |
| 2026-05-31 | **Big-move reframe** — P(\|r\|>k) magnitude target (h5/h20) | ✅ **First genuine ML win** — logistic beats baselines on big-move AUC everywhere + Brier at h5 (MSFT/KO) | Build a `prob_large_move` signal (logistic, calibrated); ML's real niche |
| 2026-05-31 | **VIX feature A/B** + universe → 110 tickers (big-move, h5) | ➖ neutral-to-positive (Brier flat, +AUC; KO AUC 0.84→0.93) | **Keep** VIX (sound, leakage-safe, optionality in high-vol regimes); expand universe |
| 2026-05-31 | **RF tuning + basket test** — tuned RandomForest vs logistic/lightgbm/baselines, 9-ticker vol spectrum (big-move, h20) | ✅ routing confirmed: `corr(vol, log−tree AUC gap) = -0.913`; RF only **ties** lightgbm | **Do NOT promote RF** (cost ≫ benefit); toolkit stays logistic + lightgbm |
| 2026-06-01 | **Large-scale lightgbm tuning** (~100 configs, ticker-level meta-validation) + **horizon-scaled buckets** | ✅ config #38 wins held-out h20 (+0.023 AUC); scaling balances the target for volatile names | Adopt #38; ship scaled buckets (h20 ±5/10, h30 ±10/20, h60 ±15/30); drop h5; drop xgboost/RF |
| 2026-06-01 | **Re-validation on scaled buckets** (volatile NVDA/SMCI/TSLA; h20/30/60 at inner k) | ✅ ML skill at h20/h30 (AUC 0.59–0.67 > baselines); ⚠️ both models **miscalibrated** (ECE 0.10–0.32); h60 unmeasurable (n=19) | **Calibrate logistic AND lightgbm** (Step 3); flag h60 big-move low-confidence |
| 2026-06-02 | **Calibration A/B** — prefit-isotonic vs `CalibratedClassifierCV(cv=3)` (volatile, h20/h30) | prefit helped logistic but **hurt lightgbm**; **cv=3 improves Brier in all cells** + fixes lightgbm ECE | Ship calibration (cv=3) for **both** models; default `calibrate_ml=True` |
| 2026-06-02 | **Task 8 spike — regime (Gaussian HMM) forecaster** vs unconditional baselines (volatile basket, h20; h60 sanity) | ❌ ties baselines (mean +0.0006 Brier, i.e. marginally worse) — vol-regime conditioning is **redundant** with bootstrap/VIX | **Do NOT promote**; keep `regime_hmm` experimental (CLI/backtest only); **reinforces parking TFT/LSTM** |
| 2026-06-02 | **Task 8 spike — pooled LSTM sequence forecaster** vs baselines (NVDA/TSLA, h20 single-split; config sweep) | ❌ **worse**, not just a tie — Brier 0.30/0.26 vs ~0.19, ECE 0.21–0.26 vs ~0.07; lighter fits only approach the baseline from below, never beat | **Do NOT promote**; `lstm_seq` + torch kept as an isolated optional extra; **confirms the ceiling is information, not model class** |
| 2026-06-03 | **Task 8 (cont.) — heavy LSTM + LR sweep + 114-ticker validation + calibration sweep** (3–4 layers, LayerNorm; CalibratedClassifierCV isotonic & sigmoid) | ❌ extracts a **real but redundant** vol signal (pooled big-move AUC +0.02–0.04 over historical) yet **loses Brier/ECE on ~75% of tickers**; **no** calibration method helps (val→test shift) | **Close LSTM/sequence track**; ceiling is *information*; next lever = GARCH (MC-tier upgrade) or Task 9 (news) |

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

## 2026-05-30 — ML improvement campaign (model class · calibration · features · horizon), h5–60

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
| 5 | **Horizon sweep** (h5/10/20/60, logistic vs baselines) | **No horizon where ML decisively wins.** Closest: a dead *tie* with the MC baseline at h5/h10 for stable names (MSFT/KO Brier Δ −0.0003 to −0.0009). 0%-direction AUC ≈0.5 at *every* horizon (0.17–0.26 at h60 → *anti*-predictive). ML's strong *tail* AUC (0.61–0.81 at h5) is **redundant with the vol-conditioned MC** → they tie. |

**Outcome — what changed.**

- ✅ **ADOPTED: plain scaled logistic** as the pooled ML model (`StandardScaler` → `LogisticRegression`, **no** `class_weight="balanced"`). A real, validated upgrade over the original xgboost:

  | | avg Brier | avg ECE | tail resolution |
  |---|---|---|---|
  | original xgboost | 0.173 | 0.099 | AUC ≈ 0.5 (none) |
  | **plain logistic** | **0.161** | **0.076** | AUC 0.65–0.76 (real) |
  | baselines | 0.151 | 0.046 | AUC ≈ 0.5 |

  It **ties the baselines on MSFT/KO** and adds genuine tail/volatility discrimination they lack; only high-vol **NVDA** still lags (residual OOS overconfidence).

- ❌ **NOT adopted:** post-hoc calibration (marginal gain, added complexity), the 4 new features (no 20-day signal — reverted), the boosters (logistic dominates them here).

**Bottom line.** ML is now a competitive, well-calibrated model with real tail resolution — but it does **not clearly beat the baselines at any tested horizon (5–60d)**. The horizon sweep made the reason precise: ML's only real signal is **conditional volatility** (tail AUC up to 0.81 at h5), but that information is **redundant with the vol-conditioned Monte-Carlo baseline**, so they *tie*; ML beats the *unconditional* historical-sim by a hair but cannot exceed vol-aware MC. And **direction is ≈ efficient at every horizon** (0%-threshold AUC ≈ 0.5, even *anti*-predictive at h60). **Baselines remain primary at all horizons.**

**Future leads (deferred, now lower priority).** Horizon is tested (above) → no win. Remaining and speculative: (1) **vol-scaled targets** (σ-unit thresholds) to clean the cross-sectional pooling; (2) a **different product framing** — ML's genuine strength is short-horizon *volatility / big-move* prediction (tail AUC 0.81 at h5), **not** direction, so a "probability of a large move" signal is where ML could actually add value over the baselines. The **ML+baseline ensemble is now low-EV** (ML's vol signal is redundant with MC). Equity *direction* is a dead end at every horizon tested.

**Reproduce.** Variant experiments used a `_make_classifier` monkeypatch (see the campaign scripts); the adopted config is now the default `_make_classifier("logistic")`. Re-check with:
```bash
python -m stock_agent backtest --ticker NVDA --model logistic   # plain scaled logistic
python -m stock_agent backtest --ticker NVDA                    # baselines, same folds
```

---

## 2026-05-31 — Big-move reframe: P(|r| > k) — the first genuine ML win

**Question.** ML can't beat the baselines on *direction* (efficient). But its real skill is *magnitude/volatility* (tail AUC up to 0.81). Does logistic predict a **large move regardless of direction** — `P(|r| > k)` — better than the baselines? `k` sized to horizon (~2–2.5σ): **5% at h5, 10% at h20**.

**Setup.** New `big_move` metric in the harness (`backtesting/runner.py`): scores `P(|r| > k) = P(< −k) + P(> +k)` (the two outer buckets) against realized `1[|r| > k]`. logistic vs baselines, identical folds, 3 tickers, h5 (250 OOS) + h20 (61 OOS).

**Results — big-move prediction** (Brier ↓, log loss ↓, AUC ↑).

**h5, |r| > 5%** (250 OOS):

| ticker | model | Brier | log loss | AUC |
|---|---|---|---|---|
| NVDA | historical_sim / MC | **0.334** | 0.95 | 0.46 / 0.48 |
| | ml_logistic | 0.347 | 1.14 | **0.599** |
| MSFT | historical_sim / MC | 0.1416 | 0.70 | 0.34 / 0.38 |
| | ml_logistic | **0.1367** | **0.560** | **0.661** |
| KO | historical_sim / MC | 0.0360 | 1.24 | 0.50 / 0.50 |
| | ml_logistic | **0.0348** | **0.155** | **0.878** |

**h20, |r| > 10%** (61 OOS — smaller, noisier): logistic AUC wins everywhere (0.59 / 0.66 / 0.82 vs baselines 0.55 / 0.40–0.60 / 0.13–0.53) but is **overconfident on Brier** (NVDA 0.281 vs 0.247) — too few positives (n+ = 2–28) and the high-vol OOS-overconfidence issue.

**Finding — YES, the reframe wins (the first time ML beats the baselines).**
- **AUC: logistic wins on *every* ticker/horizon** — it genuinely discriminates big-move periods, which the unconditional/recent-vol baselines cannot (their AUC hovers at 0.5, even *below*).
- **Brier + log loss: logistic wins at h5 for the stable names** (MSFT, KO). The KO log-loss collapse (**0.155 vs 1.24**) is the headline: the baselines are *terrible* at KO's rare 5% moves; logistic predicts them well.
- High-vol NVDA and h20 keep the AUC edge but go overconfident on Brier — fixable now because (unlike the directional task) there is **real resolution to calibrate**.

**Interpretation.** Unlike direction, **big-move/volatility is conditionally predictable from the features**, and the baselines are weak at it (unconditional history, or recent-vol that doesn't see the feature state) — so ML's signal is **non-redundant and additive** here. This is ML's genuine product niche.

**Decision.** Validated → **build a `prob_large_move` signal** powered by logistic, surfaced as a distinct *"probability of a large move (±k) over the horizon"* output (separate from the directional scenario forecast, which stays baseline-driven). Calibrate it (post-hoc) to fix high-vol overconfidence, and lead with short horizons where the win is strongest. Recorded metric: `BacktestResult.big_move` (configurable `big_move_k`).

**Reproduce.**
```bash
# big_move metric is computed by every backtest; sweep k via the pipeline:
#   run_backtest_pipeline(ticker, horizon, model_names=[...], big_move_k=0.05)
# then read result.big_move (a ThresholdMetrics for P(|r|>k)).
```

**Follow-up — model sweep + which model powers it.** Unlike *direction* (where trees were stuck at AUC ≈ 0.5), on **big-move the trees are competitive** — it has non-linear vol structure they capture. h5 |r|>5% AUC: xgboost/lightgbm 0.63–0.72 vs logistic 0.60–0.66. Key result: **regularizing lightgbm (shallow, `min_child_samples=200`, subsampling, L1+L2) fixes the high-vol overconfidence** — NVDA h20 log-loss **1.31 → 0.72**, Brier 0.281 → 0.259, AUC up. Verdict is **split**: **regularized lightgbm wins high-vol names (NVDA); logistic wins stable names (KO AUC 0.878 vs 0.807; MSFT h20 Brier 0.093 vs 0.102).** Neither dominates.

**Tuning sweep (2026-05-31, h5, 110-ticker universe + VIX).** Three reg configs (current / stronger / moderate-capacity). Winner **`lgbm_reg_moderate`** (depth 4 / 15 leaves, `min_child_samples=300`, subsample 0.8, λ=8, α=1) — the *more-capacity* config, confirming big-move has real signal that rewards some capacity, not maximal regularization. On NVDA big-move it beats logistic **and** the baseline (Brier **0.3286** vs 0.352 / 0.334, AUC 0.659); logistic still wins the stable names (MSFT/KO, better Brier + calibration). Gains over the first reg config are marginal (NVDA 0.332 → 0.3286) → **practical ceiling.** **Production `lightgbm` config swapped to this regularized version** (the default was the overfitting one). Toolkit stands: **logistic for stable names, regularized lightgbm for volatile ones**; logistic remains the simple `get_large_move` default.

**Built (2026-05-31).** `forecasting/large_move.py` (`large_move_breakdown` → `LargeMoveBreakdown`) + agent **`get_large_move`** tool (system prompt v3). Model-agnostic; **defaults to logistic**, with regularized lightgbm a documented high-vol swap. Live NVDA 20d k=10%: P(|r|>10%)=18% (up 11% / down 7%, lean up). Next enhancement: a precomputed per-ticker skill scorecard (inline AUC/calibration trust badge).

---

## 2026-05-31 — VIX macro feature + universe expansion

**Question.** Macro features are *market-wide* (one value per date) so they add ~nothing to *direction* (constant across tickers, pooling doesn't raise effective N — same issue as calendar seasonality). But **VIX *is* forward-looking volatility**, so it should help the *big-move / volatility* target. Does adding VIX (`vix_level` = VIX/100, `vix_rel` = VIX vs its 20-day average — both real-time, leakage-safe) improve big-move? (Employment/CPI deferred — publication lags need vintage alignment to avoid leakage.)

**Setup.** A/B on the **same expanded 110-ticker universe** (semis / AI-infra / AI-energy / AI-memory / ETFs added): logistic, **VIX-on vs VIX-off** (fetch monkeypatched to empty), big-move at h5 (|r|>5%), 3 tickers. VIX plumbed through training + inference + per-fold backtest, reindexed `≤` each point-in-time date.

**Results** (big-move, h5, VIX off → on):

| ticker | Brier | log loss | AUC |
|---|---|---|---|
| NVDA | 0.3518 → 0.3521 | 1.108 → 1.135 | 0.607 → 0.591 |
| MSFT | 0.1348 → 0.1348 | 0.520 → 0.527 | 0.657 → 0.669 |
| KO | 0.0342 → **0.0338** | 0.146 → **0.138** | 0.838 → **0.928** |
| **avg** | **0.1736 → 0.1736** | 0.591 → 0.600 | 0.701 → **0.729** |

**Finding — neutral-to-slightly-positive.** Brier exactly flat; **AUC up (+0.028 avg)**, driven by a real KO gain (0.84 → 0.93); log loss a hair worse (+0.008, all NVDA — within noise). Helps KO, neutral MSFT, slightly hurts NVDA → no consistent harm, a genuine discrimination gain. **Caveat:** the test window's recent years are *calm* (VIX ~15); VIX earns its keep in *high-vol* regimes (2020/2022), so its value here is likely **understated**.

**Decision.** **Keep VIX** (16 → 18 features). It's a sound, leakage-safe, economically-motivated feature that doesn't worsen results and adds high-vol-regime optionality. Also kept: the **110-ticker universe** and two robustness fixes the bigger universe surfaced — provider **OHLC sanitization** (clamp split-adjustment artifacts) and a **resilient universe fetch** (skip any bad/short-history ticker instead of aborting). **Deferred:** employment/CPI (vintage alignment); re-tuning regularized lightgbm *with* VIX on the bigger universe.

---

## 2026-05-31 — RandomForest: tuning sweep + basket test (big-move, h20)

**Question.** Two things the 3-ticker tests (NVDA/MSFT/KO) couldn't settle: (1) Does a *tuned* RandomForest beat the shipped trees enough to earn a place in the toolkit — or replace lightgbm? (2) Is the "trees on volatile, logistic on stable" split a **real** effect, or a base-rate artifact (stable names have few big moves → sparse positives starve the trees)?

**Setup.**
- **`class_weight="balanced"` was disqualified first** (basic RF probe): it wrecks calibration — NVDA big-move log-loss **3.93** vs ~1.0 for everything else, worst Brier on all 3 probe tickers. Same lesson as logistic; the production RF default (which still carried `balanced`) is a confirmed latent bug. All RF work below is **unbalanced**.
- **Tuning sweep** (h20, 8 configs × {NVDA, KO}): best all-round config **`d12_l50_sqrt`** (`n_estimators=300, max_depth=12, min_samples_leaf=50, max_features="sqrt"`). Depth 12–16 / leaf 50–100 is the sweet spot; heavy regularization (`leaf=200`, `max_depth=None`) and `n=500` add nothing.
- **Basket test** — that RF config vs logistic + regularized lightgbm + baselines on identical walk-forward folds. Big-move h20, |r|>5%, 11 folds, ~61 OOS/ticker. **Stable** {KO, PG, JNJ, WMT} vs **volatile** {NVDA, AMD, MRVL, SMCI, PLTR}. Per-ticker realized vol + big-move base rate recorded to test the base-rate confound.

**Results** — big-move AUC (↑; 0.5 = no skill). `gap = logistic − max(lightgbm, RF)`.

| ticker | vol | base % | logistic | lightgbm | RF `d12_l50` | gap |
|---|---|---|---|---|---|---|
| KO | low | 19.7 | **0.685** | 0.578 | 0.587 | +0.099 |
| PG | 0.17 | 24.6 | **0.642** | 0.564 | 0.588 | +0.054 |
| JNJ | 0.17 | 26.2 | **0.549** | 0.487 | 0.493 | +0.056 |
| WMT | 0.22 | 36.1 | **0.650** | 0.613 | 0.607 | +0.037 |
| NVDA | 0.47 | 65.6 | 0.636 | 0.650 | **0.675** | -0.039 |
| AMD | 0.55 | 80.3 | 0.536 | 0.519 | 0.527 | +0.009 |
| MRVL | 0.59 | 75.4 | 0.520 | 0.461 | **0.536** | -0.016 |
| SMCI | 0.97 | 73.8 | 0.715 | **0.781** | 0.703 | -0.065 |
| PLTR | 0.63 | 77.2 | 0.420 | 0.456 | 0.418 | -0.037 |

**Headline:** `corr(realized vol, logistic−best-tree AUC gap) = -0.913` — a strong negative correlation: trees gain over logistic precisely as volatility rises.

**Findings.**
- **Routing confirmed at n=9.** Logistic wins **all 4 stable** names (AUC 0.55–0.69 vs trees 0.49–0.61); trees win **4 of 5 volatile** names. The −0.913 correlation upgrades the prior n=3 anecdote to a cross-sectional result.
- **RF vs lightgbm = a wash.** RF wins more names (6/9 on AUC) and has lower Brier on most volatile names, **but** lightgbm wins the *most-volatile* name decisively (SMCI AUC 0.781 vs RF 0.703) — exactly the regime a tree model exists for. No uniform winner.
- **Base-rate confound is real but doesn't overturn the routing.** Vol and big-move base rate are correlated (stable 20–36% positives vs volatile 66–80%), so the mechanism is partly "enough positives to learn nonlinearity," not purely a linear-vs-nonlinear boundary. Either way the routing (logistic for stable/sparse, trees for volatile/dense) holds.
- **Weak absolute skill on several names.** AMD/MRVL/PLTR AUC ≈ 0.5 (PLTR < 0.5); the big-move edge is strong on NVDA/SMCI, marginal elsewhere. With ~61 OOS/ticker the per-ticker AUC CIs are wide (±~0.1) — trust the aggregate, not ±0.03 individual rankings.

**Decision.** **Do NOT promote RandomForest.** It only *ties* lightgbm overall while costing ~10–50× more at inference (300 deep trees over ~100k+ rows vs histogram boosting) and losing the most-volatile name. A heavy model that merely ties a cheap, validated one doesn't earn the swap. **Production toolkit stands: logistic (stable names) + regularized lightgbm (volatile names);** RF is documented as competitive-but-not-cost-justified. Sequence / deep models stay parked. **Next:** post-hoc calibration scoped to the *shipped* toolkit (logistic + lightgbm) — measure OOS ECE, calibrate where ECE > ~0.05 (lightgbm the prime candidate; logistic likely already well-calibrated).

---

## 2026-06-01 — Large-scale lightgbm tuning + horizon-scaled buckets

**Question.** Can a larger hyperparameter search beat the prior lightgbm config — *without* selection bias? And does a fixed ±5%/±10% big-move band stay informative across horizons?

**Tuning.** Random search, ~100 configs over a ~13-D space (learning rate, n_estimators, num_leaves, max_depth, min_child_samples, subsample, colsample, reg_lambda/alpha, **min_split_gain, min_child_weight, max_bin, path_smooth**). Objective = big-move ROC AUC (calibration-invariant; calibration handled separately). **Ticker-level meta-validation against selection bias:** stage 1 screen (NVDA/SMCI) → stage 2 validate (5-ticker volatile basket) → stage 3 **held-out** test on a DISJOINT basket {AVGO, MU, ARM, TSLA, VRT}. The funnel reordered (stage-1 winner #33 fell to rank 5 — winner's curse in action); the validation winner was **#38** (`n_estimators=400, num_leaves=47, max_depth=-1, learning_rate=0.024, min_child_samples=50, subsample=0.98, colsample=0.56, reg_lambda≈9, path_smooth=1, max_bin=127`).

**Held-out verdict.** #38 beats the incumbent on the **disjoint** basket: mean big-move AUC **0.583 vs 0.559 (+0.0234)** at h20 — the validation edge (+0.035) shrank but survived → a *real* (modest) improvement, not selection noise. h30 directional +0.026; h60 unmeasurable (few folds). **Adopted #38** for all horizons.

**Horizon-scaled buckets.** Base rates of `|r|>5%` over longer horizons are degenerate (~90% at h60 → the target is single-class). Scaled the bucket band with horizon — **h20 ±5/±10, h30 ±10/±20, h60 ±15/±30** — so each horizon's inner boundary (5/10/15%) is the natural "big move" threshold. `k` is a read-time parameter (`get_large_move`); the artifact predicts all boundaries.

**Decision.** Adopt #38; ship horizon-scaled buckets; retrain logistic + lightgbm at **{20, 30, 60}** (`train --all`); **drop h5** (too short-term for swing trading) and **drop xgboost + random_forest** (never promoted; RF's `class_weight="balanced"` was a latent calibration bug). Toolkit = logistic + tuned lightgbm only.

---

## 2026-06-01 — Re-validation on the horizon-scaled buckets (volatile names)

**Question.** The big-move *target* changed (h30 → P(|r|>10%), h60 → P(|r|>15%)), so prior numbers no longer describe these models. Did scaling balance the target? Do the models have skill at the new thresholds? And what's the calibration error (the Step-3 scope)?

**Base rates (direct from prices).** The scaled thresholds are right-sized **for volatile names** (the big-move regime): volatile {NVDA,AMD,SMCI} ≈ 55–75% across horizons (balanced/informative); stable {KO,PG,JNJ} fall to **1–9%** at h30/h60 — correct, *stable names don't make 15% moves in 60 days*. So big-move skill is only measurable on volatile names; that's the validation set.

**Setup.** Walk-forward backtest on **NVDA/SMCI/TSLA**, each horizon at its inner k (h20/5%, h30/10%, h60/15%), logistic + lightgbm + baselines. Means over the 3 names:

| horizon (k) | base % | hist AUC | mc AUC | **logistic AUC** | **lightgbm AUC** | **log ECE** | **lgb ECE** | n/ticker |
|---|---|---|---|---|---|---|---|---|
| h20 (5%) | 73 | 0.48 | 0.51 | **0.65** | **0.66** | 0.10 | 0.11 | 61 |
| h30 (10%) | 58 | 0.40 | 0.44 | **0.67** | **0.59** | 0.18 | 0.17 | 40 |
| h60 (15%) | 56 | 0.33 | 0.31 | 0.52 | 0.50 | 0.29 | 0.32 | **19** |

**Findings.**
- **Skill is real at h20 & h30.** Both ML models (0.59–0.67) clearly beat the baselines (~0.40–0.51). logistic ≈ lightgbm (logistic even edges at h30) → the "lightgbm for volatile" advantage is marginal, but no reason to change the toolkit.
- **h60 is unmeasurable.** Every model ≈ 0.5 with only **19 OOS predictions** (the non-overlapping-60-day-window limit). Can't validate h60 skill → serve it as **low-confidence**.
- **Both models are miscalibrated** (ECE 0.10 → 0.18 → 0.32, worsening with horizon), while baselines are well-calibrated (0.06–0.15). Tellingly, **ML Brier does not beat the baselines despite higher AUC** (h20: lgb 0.291 vs baseline 0.245) — real resolution wasted by overconfidence. (Earlier guess that logistic was "probably fine" was **wrong** — it's miscalibrated too.)

**Decision.** **Calibrate BOTH logistic and lightgbm** (Step 3: per-threshold isotonic + monotone-envelope + nested holdout). Priority h20 + h30 (skill + miscalibrated → calibration recovers Brier without touching AUC). h60: apply for honesty but flag the big-move signal as low-confidence. Baselines: already calibrated, left alone.

---

## 2026-06-02 — Calibration A/B: prefit-isotonic vs CalibratedClassifierCV(cv=3)

**Question.** Does post-hoc calibration of the pooled ML models actually cut OOS ECE/Brier without destroying discrimination? And which calibration strategy? Walk-forward backtest, `calibrate=False` vs `True`, on the volatile basket (NVDA/SMCI) at h20/h30 (where ML has skill), each horizon at its inner `k`.

**Round 1 — manual prefit isotonic (fit classifier on 80%, isotonic on the held-out 20%).** Mixed → **net negative**: it helped logistic but **hurt lightgbm** (means, raw→cal): lightgbm h20 ECE **0.117→0.176**, Brier 0.233→0.261; h30 ECE 0.162→0.239. Diagnosis: (1) the 80/20 split costs the booster 20% of its training data, and (2) isotonic (non-parametric, high-variance) **overfits the thin per-fold holdout**. Logistic's smooth outputs + lower variance tolerated it; lightgbm did not.

**Round 2 — `CalibratedClassifierCV(cv=3, method="isotonic")`.** Refits the base on k-1 folds, fits isotonic on the held fold, repeats, averages → uses **all** data (no holdout loss) and the **averaged** calibrator is robust on small folds. Means over the basket, raw→calibrated:

| model | h | ECE | AUC | Brier |
|---|---|---|---|---|
| logistic | 20 | 0.086 → 0.085 | 0.694 → 0.655 | 0.231 → **0.224** |
| lightgbm | 20 | 0.129 → **0.100** | 0.690 → 0.659 | 0.233 → **0.227** |
| logistic | 30 | 0.197 → **0.173** | 0.667 → 0.678 | 0.272 → **0.250** |
| lightgbm | 30 | 0.155 → 0.157 | 0.581 → 0.615 | 0.264 → **0.258** |

**Findings.**
- **Brier improves in all four cells** — the decisive proper-score result (Brier = calibration + resolution), so net forecast quality went up everywhere.
- **ECE improves where it was bad, holds where it was fine** — lightgbm h20 **0.129→0.100** (the fix; prefit had *worsened* this to 0.176), logistic h30 0.197→0.173; logistic h20 / lightgbm h30 already-fine → flat, no harm.
- **AUC is *not strictly* invariant** (dips ~0.03 at h20, rises at h30 → ≈flat on average). Expected: `cv=3` replaces the single classifier with an **ensemble of 3** (each on ⅔ the data), so base scores shift slightly. Within n=61 noise, and Brier (which subsumes resolution) improved regardless. Strict AUC-invariance only holds for single-classifier `cv="prefit"` — which is what hurt lightgbm.

**Decision.** **Ship calibration for both models via `CalibratedClassifierCV(cv=3, isotonic)`** (default `settings.calibrate_ml=True`); retrain the 6 served artifacts calibrated. The cross-threshold **monotone envelope** stays enforced downstream (`ml._exceedance_to_buckets`). `cv=3` chosen over 5 to bound the k× training cost. h60 still served **low-confidence** (unmeasurable, n≈19).

---

## 2026-06-02 — Task 8 spike: regime (Gaussian HMM) forecaster

**Hypothesis.** Direction is ≈ efficient but *volatility/magnitude* is predictable, and latent market regimes are volatility states. So conditioning the forward-return distribution on the **current regime** should sharpen the big-move tails the toolkit targets — possibly beating the unconditional baselines.

**Model.** `forecasting/regime.py` (`regime_hmm`): fit a `GaussianHMM(n_states=3, diag)` on the stock's daily `(log-return, |log-return|)` sequence using data ≤ `as_of`, take the current Viterbi state, and build the forecast from the **historical h-day forward returns whose start-day shared that regime** (reusing `historical.sample_to_forecast`). Leakage-safe (fit + conditioning strictly within `[0, as_of]`); deterministic (`random_state=42`); graceful fallback to unconditional history when the fit fails or the current regime has < 30 conditioned windows. Wired into the `forecast` / `backtest` CLI only.

**Setup.** Walk-forward, embargo = h, 11 folds, n=61 OOS at h20 (4 folds / n≈19 at h60). Compared `regime_hmm` vs `historical_sim` + `monte_carlo_bootstrap` on the volatile basket.

**Results (h20, mean Brier; `regime − historical`).**

| Ticker | regime | historical | bootstrap | regime − hist |
|---|---|---|---|---|
| NVDA | 0.2013 | **0.1981** | 0.1988 | +0.0032 |
| TSLA | 0.2360 | 0.2335 | **0.2334** | +0.0025 |
| SMCI | 0.2419 | 0.2410 | **0.2388** | +0.0009 |
| MU | **0.2130** | 0.2139 | 0.2127 | −0.0009 |
| AVGO | 0.1828 | 0.1828 | **0.1789** | +0.0001 |
| ARM | 0.2371 | 0.2369 | **0.2349** | +0.0002 |

Basket mean `regime − historical` = **+0.0006** (regime marginally *worse*); `monte_carlo_bootstrap` is generally best. ECE is comparable (regime sometimes slightly better, e.g. NVDA 0.040 vs 0.048). h60 sanity is mixed and unmeasurable (NVDA −0.0090 but ECE 0.19; TSLA +0.0056; n=19).

**Findings.**
- **Regime conditioning does not beat the baselines.** It ties them within noise at h20 and is unmeasurable at h60 — the same ceiling every prior experiment hit.
- **Why:** the signal it adds (recent volatility state) is **redundant** — the bootstrap baseline already resamples from recent realized returns, and the pooled ML already has `vix_*` + vol features. Conditioning on a latent vol-regime re-encodes information the comparison set already has, so no net resolution is gained. Same lesson as the VIX A/B and the "ML ties MC" horizon sweep.

**Decision.** **Do NOT promote** `regime_hmm`; keep it as an experimental CLI/backtest model (leakage-safe, tested) for future reference. This **reinforces parking TFT/LSTM**: a model that targets the *known-predictable* axis (volatility) directly still can't beat the baselines because the signal is redundant/efficient — so a heavier, higher-variance sequence net is even lower-EV at far greater cost. Revisit only with a genuinely new information source (e.g. Task 9 point-in-time news), not a new function class over the same price inputs.

**Reproduce.**
```python
from stock_agent.pipelines.backtest import run_backtest_pipeline
from stock_agent.settings import get_settings
run_backtest_pipeline("NVDA", 20,
    model_names=["regime_hmm", "historical_sim", "monte_carlo_bootstrap"],
    settings=get_settings(), test_size=6)
```

---

## 2026-06-02 — Task 8 spike: pooled LSTM sequence forecaster

**Hypothesis.** A sequence model might capture *temporal structure* (vol clustering, momentum/mean-reversion dynamics) that the tabular one-day feature snapshot flattens — adding skill beyond the toolkit. **LSTM chosen over a transformer**: sample-efficient on low *effective* N (overlapping windows), its autoregressive inductive bias fits vol-clustering, and it's deterministic + CPU-cheap; a transformer adds capacity, not information.

**Model.** `forecasting/sequence.py` (`lstm_seq`, optional `[sequence]` extra → torch): pooled LSTM over `L=60`-day sequences of the 18 scale-free daily features, softmax over the 6 horizon-scaled buckets → native `ScenarioForecast`. Leakage-safe (features ≤ t; labels use `P_{t+h}`; standardizer fit on train pool; model used only on as-of dates after its training cutoff). Deterministic (seeded, single-thread, `use_deterministic_algorithms`).

**Setup.** **Single temporal split** (a fast go/no-go before paying for full per-fold walk-forward): train on a 40-ticker universe sliced ≤ a cutoff date, then compare to the baselines on the **same** post-cutoff OOS folds (NVDA/TSLA, h20, n=28).

**Results (h20 mean Brier / ECE; baselines ~0.185 / 0.07).**

| Model | NVDA Brier | NVDA ECE | TSLA Brier | TSLA ECE |
|---|---|---|---|---|
| monte_carlo_bootstrap | **0.1841** | 0.077 | **0.2043** | 0.072 |
| historical_sim | 0.1850 | 0.070 | 0.2046 | 0.072 |
| lstm_seq | 0.3021 | 0.264 | 0.2627 | 0.209 |

**Config sweep (NVDA, to check tuning doesn't reverse it):** `e3·h24·L1` → 0.1926 / 0.115 · `e5·h32·L1` → 0.2046 / 0.145 · `e10·h32·L2` → 0.2856 / 0.232. More training/capacity **monotonically worsens** it (overfit); the best (heavily-underfit) config only approaches the baseline **from below** and stays worse-calibrated.

**Findings.**
- The LSTM is **decisively worse** than the baselines — overconfident and badly miscalibrated (ECE 3–4× the baselines) — the classic high-capacity / low-effective-N overfit on overlapping windows.
- It can be regularized *toward* the baseline but **never beats it**. With the regime model only *tying*, this nails the conclusion: **the binding constraint is information, not model class** — a higher-capacity function over the same price inputs cannot clear a signal ceiling.

**Decision.** **Do NOT promote** `lstm_seq`. Keep it as an isolated optional extra (torch is heavy and segfaults alongside lightgbm in one process on macOS, so its tests are env-gated — `RUN_SEQUENCE_TESTS=1`; the default gate + CI never load torch). **Do not pursue a transformer** — same inputs, more capacity, strictly higher cost/variance. The only lever with real upside is a **new information source** (Task 9 point-in-time news), not a new function class on price alone.

**Reproduce.** Install the extra (`pip install -e ".[sequence]"`), then the single-split eval in the session log (train `train_sequence_model` on a universe sliced ≤ a cutoff date; compare via `run_backtest` on post-cutoff folds with matched `min_train`).

---

## 2026-06-03 — Task 8 (cont.): heavy LSTM, LR sweep, 114-ticker validation, calibration sweep

The spike used 40 tickers, one config, no early stopping. To **refute** "it failed only because it was under-powered / under-calibrated," we gave the sequence model every advantage. Harness: `forecasting/sequence_tune.py` (early stopping on val Brier, temperature calibration, train-vs-val AUC diagnostic, stride subsampling). Protocol: full 114-ticker universe; **date split** train ≤ 2024-01-02 (60%) / val 60–80% / **held-out test > 2025-03-19**; per-threshold exceedance scoring; winners persisted to `outputs/experimental/` (gitignored).

**1. Heavy architecture search** (3–4 LSTM layers, hidden 128/256, LayerNorm, dropout): **flat** — val Brier 0.1707–0.1715, val AUC 0.62–0.64 across *all* configs; **train AUC ≈ val AUC** (generalizable, not overfit — capacity isn't the issue).

**2. Learning-rate sweep** (3e-5 → 1e-3): flat; best `lr=3e-4` (val Brier 0.1700, immaterial vs 1e-3's 0.1715). A flat response over both architecture and LR is the fingerprint of *no additional extractable signal*.

**3. Pooled held-out big-move AUC vs historical** (n=27k–32k):

| | LSTM | historical | edge |
|---|---|---|---|
| h20 | 0.665 | 0.644 | +0.021 |
| h30 | 0.698 | 0.674 | +0.024 |
| h60 | 0.720 | 0.683 | +0.037 |

A **small, real, generalizable** discrimination edge (growing with horizon) — so the heavy net *does* extract vol signal the light one (AUC≈0.5) missed.

**4. 114-ticker held-out Brier / ECE** (mean across tickers; the deployable verdict):

| Horizon | LSTM | historical | LSTM win-rate vs hist |
|---|---|---|---|
| h20 | 0.182 / 0.154 | **0.170** / 0.113 | **25%** (28/114) |
| h30 | 0.214 / 0.221 | **0.199** / 0.169 | **27%** (31/114) |
| h60 | 0.235 / 0.278 | 0.240 / 0.244 | 55%* (h60 unmeasurable) |

Worse Brier and ECE; loses to the trivial baseline on **~3 of 4 tickers** at the measurable horizons.

**5. Calibration sweep** (temperature · manual per-threshold isotonic · `CalibratedClassifierCV(cv=3)` isotonic & sigmoid). Raw pooled exceedance ECE was **already good** (0.085 / 0.075 / 0.058) — the model is *not* miscalibrated at the pooled level. **Every** val-fit calibrator made the **test** period *worse* (val→test distribution shift); AUC stayed invariant (confirming correct, resolution-preserving recalibration); the 114-ticker win-rate never approached 50% (18–40%). Sigmoid transferred marginally better than isotonic (parametric → shift-robust), as expected, but still net-negative.

**Why it loses (decomposed).** Brier = reliability − resolution + uncertainty. Calibration only shrinks *reliability*; it cannot add *resolution*. The LSTM's resolution (AUC ≈ 0.665) ≈ the baseline's (0.644) → **redundant**, so even a perfectly-calibrated LSTM converges to ≈ a *tie*, never a win. The residual per-ticker miscalibration (per-ticker ECE 0.15–0.29) is invisible to any *global* pooled calibrator.

**Decision.** **Close the sequence/regime track (Task 8).** Capacity (3–4 layers), learning rate, full universe, and four calibration methods incl. `CalibratedClassifierCV(cv=3)` were all ruled out across 114 tickers and three horizons. The binding constraint is **information, not model class or calibration**. Next levers: **GARCH** (a principled conditional-volatility upgrade to the Monte-Carlo/baseline tier — different mechanism, not just more capacity) and/or **Task 9** (point-in-time news = genuinely new information). `lstm_seq` + `sequence_tune.py` stay as an isolated, torch-gated experimental harness for the record.
