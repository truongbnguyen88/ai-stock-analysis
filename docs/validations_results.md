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
| 2026-06-03 | **Task 9 — GJR-GARCH(1,1)-t** (`monte_carlo_garch`) vs bootstrap/historical, 12-ticker basket, h20/30/60 | ✅ **first model to beat the baselines on Brier** — tie h20, wins h30 (9/12) + h60 (10/12); also **highest big-move AUC at every horizon** (+0.04–0.06 → win is *earned* resolution, not just calibration) | **Promote** to the default MC comparison set; modest *real* edge at long horizons (forward vol mean-reversion) |
| 2026-06-03 | **Task 10 — GDELT news features vs price-only** — single-split **then full walk-forward** (7 folds, 30 tickers, h20/30/60, both models, +retune control) | ❌ **no robust gain** — logistic nothing; lightgbm a faint negative-Brier lean (−0.001→−0.003, win 54–58%) but **within fold std** and the **retune control shows news ties retuned-price** (0.1886 vs 0.1887) → the gain is model-tuning, not news | **Do NOT promote**; keep ingest+features experimental (default-OFF); **information ceiling holds** (strongest test in repo) |
| 2026-06-04 | **XGBoost reconsidered** — heavy random-search tune (winner's-curse-guarded) + 3-way bake-off vs logistic/lightgbm (7-fold walk-forward, h20/30/60, with news), **raw then CCCV-calibrated** | **raw:** tuned-xgb looked best everywhere, lightgbm worst + poorly-calibrated (ECE 0.11–0.13). **calibrated (the fair test):** CCCV fixes lightgbm; all three converge & **logistic is nominally best at every h** (xgb last at h30/h60), all tied within fold std → xgb's raw edge was a **calibration artifact** | **Do NOT promote xgboost**; vindicates the original drop-xgboost call; logistic stays default |
| 2026-06-04 | **Ensemble** — 5-member linear pool (historical + bootstrap + GARCH + logistic + lightgbm), **equal vs online-stacked** weights, walk-forward via the real harness (4 tickers, h20/60, per-fold ML retrain) | does **not** beat the best single member on Brier (h20 0.1688 vs GARCH 0.1680; h60 0.1319 vs bootstrap 0.1288); **skill-weighting (stacking) adds nothing** (members' Briers cluster, too few folds). BUT best **big-move AUC** (h20 0.68 vs baselines ≤0.60) at **good calibration** (ECE 0.054 ≪ ML's 0.06–0.09) | **Promoted (user call) as the interactive default** (agent + `forecast` CLI) — not a Brier win, but never worst + best magnitude discrimination at honest calibration + one default. Equal weights (stacking dropped). |
| 2026-06-06 | **Conformal interval coverage** — split-conformal correction over the universe (pooled offline, per model × {20,30,60}), measured stated-vs-conformalized CI coverage | every model **under-covered**: a stated 90% CI really covered **82–86% (h20), 81–87% (h30), 76–81% (h60)** → conformal `q` (+0.04 to +0.16) fixed all to **90–91%** | **Ship conformal-calibrated CIs/VaR** at inference + in the monthly retrain. The intervals are now honest-coverage (marginal). |
| 2026-06-06 | **Sequence × news (Task 8 × 10)** — price-only vs price+news, **DEEP LSTM (3–4 layers, 128–256 hidden, LayerNorm), carefully tuned** (early-stop + temperature-calib + val-selected grid), 37 tickers, held-out TEST split, inner big-move band (h20 \|r\|>0.05, h60 \|r\|>0.15) | news adds **no skill** on the held-out test: big-move **ΔAUC −0.012 (h20) / −0.015 (h60)** (worse), Brier flat — and **robust across two hyperparameter grids** (dropout 0.2/lr 3e-4 and 0.3/lr 1e-3 agree). Deep nets **don't overfit** (train AUC ≈ val AUC). The one place news beat price was h60 *validation* Brier — it **reversed on held-out test** (spurious val signal). Price-only deep LSTM TEST big-move AUC **0.646 / 0.682**. | **Do not promote.** Robust to the obvious objections (depth, tuning, band, held-out) — the news-negative is an **INFORMATION ceiling, not a model-class limitation**: a deep, tuned temporal model given the raw sentiment sequence extracts no discrimination. **Caveat:** per-ticker + market sentiment only; **topic streams (July) untested.** |
| 2026-06-28 | **A5.3 GraphRAG promotion** — agentic+graph (alias-bridge OFF) vs the production multi-hop path agentic+hybrid+alias-bridge, on the SEC **bridging benchmark** (2×2 control×substrate, **12 Q × 2 seeds**, aspect-coverage) | ✅ **scoped win** — HARD `D≥B` held both seeds (1.00 / 0.92 ≥ 0.92), **D=1.00 on controls** (agentic+graph never regresses easy Qs); the strict `D−A≥+0.5`-vs-single-shot bar **missed** (+0.38) but the decision-relevant **D-vs-B** comparison passed. n=12/2-seed → *directional* | **Promote graph for the MULTI-HOP path only** (`graph_multistep_enabled` ON; routes `research_multistep`→`GraphRetriever`); **single-shot stays hybrid** (single-shot graph regressed controls); **alias-bridge kept ON** as universal fallback (non-graph tickers) |

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

---

## 2026-06-03 — Task 9: GJR-GARCH(1,1)-t conditional-volatility forecaster ✅

After three null results (regime tie, LSTM redundant), GARCH is the **first new model to beat the baselines on the deployable proper score** — exactly because it adds a *different mechanism*, not more capacity over the same inputs.

**Model.** `MonteCarlo(variant="garch")` (`monte_carlo_garch`): per-ticker **GJR-GARCH(1,1)** with **Student-t** innovations via `arch` — `o=1` captures the equity *leverage effect* (vol rises more after down-moves), Student-t the fat tails. Fits on daily log returns ≤ `as_of` (leakage-safe; *daily* obs → no overlapping-window / effective-N problem), simulates `n_paths` forward paths from the **current** conditional variance (which **mean-reverts** toward the long-run level), → terminal-return sample → `ScenarioForecast`. Deterministic (seeded `StudentsT`; arch's `forecast(random_state=)` is ignored, so the seed goes on the *distribution*). Degrades to the **block bootstrap** on short history (<250 returns) or any fit failure.

**Setup.** Walk-forward, embargo = h, 12-ticker basket spanning the vol spectrum (NVDA/TSLA/SMCI/MU/AVGO/AMD/AAPL/MSFT/KO/PG/JNJ/WMT). Compared to `monte_carlo_bootstrap` (the strongest baseline — it *also* conditions on recent vol) and `historical_sim`.

**Results (mean Brier / ECE across the basket; win-rate = GARCH beats bootstrap on Brier):**

| Horizon | GARCH | bootstrap | historical | win-rate |
|---|---|---|---|---|
| h20 (62 OOS) | 0.1705 / 0.050 | 0.1706 / 0.047 | 0.1713 / 0.051 | 5/12 — **tie** |
| h30 (41 OOS) | **0.1961** / 0.088 | 0.1973 / 0.095 | 0.1982 / 0.099 | **9/12** |
| h60 (20 OOS) | **0.2350** / 0.168 | 0.2398 / 0.154 | 0.2431 / 0.171 | **10/12** |

**Findings.**
- **Additive at h30/h60, redundant at h20** — exactly as the mechanism predicts. At short horizons the current vol ≈ the realized forward vol, so the bootstrap's recent-vol resampling already captures it (tie). At longer horizons the gap between *current* vol and the *long-run* level matters, and GARCH's mean-reverting variance forecast is genuinely more than "hold recent vol constant" → it wins on **9–10 of 12** tickers and beats **both** baselines on mean Brier.
- Gaps are small in absolute terms (0.001–0.005 Brier) but **consistent across a diverse basket** — signal, not noise. ECE is comparable (slightly better at h30, slightly worse at h60 — Brier, the proper score, is the arbiter).

**Discrimination (AUC) — the Brier win is *earned*, not just calibration.** Big-move `P(|r|>k)` AUC at the horizon-scaled `k`:

| Horizon | GARCH | bootstrap | historical | direction AUC (GARCH) | GARCH > boot |
|---|---|---|---|---|---|
| h20 (k=5%) | **0.545** | 0.497 | 0.502 | 0.493 | 10/12 |
| h30 (k=10%) | **0.518** | 0.476 | 0.468 | 0.431 | 7/12 |
| h60 (k=15%) | **0.477** | 0.418 | 0.412 | 0.371 | 7/10 |

- GARCH has the **highest big-move AUC at every horizon** (+0.04–0.06 over bootstrap, 7–10/12 win-rate) → its lower Brier is backed by **better resolution** (discrimination), unlike the LSTM whose AUC was redundant ~0.5. Calibration *and* sharpness both improved.
- Absolute discrimination is **modest** (0.52–0.55 at h20/h30) — vol clustering buys a little big-move predictability, not a lot. The h60 figures sit < 0.5 but that is **noise** (n=20 OOS/ticker); all models are < 0.5 there while the *ordering* (GARCH > bootstrap > historical) still holds.
- **Direction AUC ≈ 0.37–0.49 (≤ 0.5)** — direction stays efficient; GARCH correctly adds nothing on direction, only on magnitude.

**Decision.** **Promote** `monte_carlo_garch` into the default offline comparison set (`STATELESS_MODELS`) — it's a validated peer of the other MC variants and the strongest at h30/h60. Cheap, deterministic, interpretable, leakage-safe. `arch>=8` is a core dep (pandas-3.0 compatible; no OpenMP/lightgbm segfault). Surfacing it as the *preferred* long-horizon baseline in the written report is a follow-up.

**Reproduce.**
```python
from stock_agent.pipelines.backtest import run_backtest_pipeline
from stock_agent.settings import get_settings
run_backtest_pipeline("NVDA", 60,
    model_names=["monte_carlo_garch", "monte_carlo_bootstrap", "historical_sim"],
    settings=get_settings(), test_size=6)
```

---

## 2026-06-03 — Task 10: GDELT news features vs price-only (the information ceiling, tested directly)

**Question.** Every prior validation concluded the binding constraint is *information, not model class*. Task 10 is the one lever that adds genuinely **new information** — point-in-time news sentiment — which was previously *unavailable* (no historical news → "blocked on data"). We unblocked it with **GDELT 2.0 GKG via BigQuery**: free, multi-year, monitoring-timestamped (point-in-time), pre-scored lexicon tone. So this finally tests the ceiling hypothesis with real data instead of asserting it.

**Data.** Server-side-aggregated daily streams (`news/gdelt_ingest.py` → `outputs/news_sentiment/`), 2023-01→2026-06, 105/108 non-ETF tickers (3 `&`-name misses fixed for next pull). Features (`features/news_history.py`, leakage-safe, 1-day publication lag, scale-free): per-ticker `news_buzz` (count vs own trailing-60d mean), `news_tone`, `news_pos_frac`, `news_neg_frac`; market-wide (VIX-like) `pol_tone`, `epu_buzz`, `pres_tone`.

**Design.** Faithful A/B reusing the production blocks (`build_training_matrix` targets, `_make_classifier`, median imputer). 15-ticker pool (volatile semis + stable large-caps), h20, **identical rows** for both arms (only the 7 news columns differ), **temporal** split (train < 2025-09 ≤ test; 9,075 train / 2,535 test rows, 100% news coverage). Both **logistic** and **lightgbm**; directional thresholds + the **big-move** target `|r|>k` (ML's only historical niche). `scripts/validate_news_features.py`.

**Result — news adds no reliable edge (Δ = +news − price-only):**

| Model · target | mean ΔBrier | mean ΔAUC | Read |
|---|---|---|---|
| logistic · directional (5 thr) | **+0.0015** | **−0.018** | worse |
| logistic · big-move (k=5/10%) | +0.0015 / +0.0001 | −0.008 / −0.001 | worse / neutral |
| lightgbm · directional (5 thr) | −0.0010 | +0.002 | ~noise |
| lightgbm · big-move (k=5/10%) | +0.0027 / +0.0022 | −0.012 / −0.007 | worse |

- Deltas are **inconsistent in sign** across models/targets and **tiny** (all within ±0.003 Brier, ±0.02 AUC) — the signature of *no real signal*, not a near-miss.
- News does **not** even help the **big-move/volatility** target (its best shot) — it's neutral-to-worse there for both models. The per-ticker price/vol features + market VIX already encode the volatility regime; crude GDELT tone over noisy org→ticker mapping adds nothing incremental.
- A booster (lightgbm), the model most able to exploit weak interactions, finds **no extractable directional signal either** (directional Δ ≈ noise).

**Decision (single-split).** Pointed to "no gain" — but a single split is weak, so it was followed by a full walk-forward (below) before finalizing.

### Full walk-forward + retune control (the rigorous follow-up)

`scripts/validate_news_features_full.py`: **7 embargoed walk-forward folds** (2024-07→2026-04), **30 tickers**, horizons **20/30/60**, both models, directional + big-move targets, per-ticker win-rate, **`log1p`-normalized buzz**, and a **held-out lightgbm retune** with a `price-retuned` control. Mean Δ over folds (Δ<0 Brier / Δ>0 AUC = news helps):

| Horizon | logistic ΔBrier / ΔAUC / ticker-win | lightgbm ΔBrier / ΔAUC / ticker-win |
|---|---|---|
| 20 | +0.0009 / −0.005 / 46% | −0.0008 / +0.002 / 56% |
| 30 | −0.0002 / −0.006 / 46% | −0.0017 / +0.004 / 58% |
| 60 | −0.0018 / +0.004 / 55% | −0.0028 / +0.009 / 54% |

- **logistic: no gain at any horizon** (neutral-to-negative; AUC consistently ≤ 0).
- **lightgbm: a small negative-Brier lean that grows with horizon** (−0.0008 → −0.0028), helping 56–73% of cells — which the single split had missed. *But* it is **within the per-fold std** (0.002–0.011, ≥ the mean), win-rates are barely above 50%, and it does not appear for the linear model.

**The retune control is decisive** (h20 big-move, disjoint held-out test):

| Arm | Brier | AUC |
|---|---|---|
| retuned lightgbm **+ news** (25 feat) | 0.1886 | 0.694 |
| price-only, **production** config | 0.1950 | 0.684 |
| price-only, **retuned** config (the control) | **0.1887** | — |

Retuned-news (0.1886) **ties** retuned-price (0.1887) to within 0.0001 Brier. The real improvement over the production baseline (0.1950 → 0.1887) came from **retuning the model, not from the news columns**. (The script prints "BEATS" by comparing to the *better* price baseline, but the `price-retuned` control shows that label is noise.) **Conclusion: news adds no incremental information once the booster is properly tuned.**

**Decision.** **Do NOT promote** news into the model — price-only (Option A) stands. Ingest pipeline, store, loader, alias config kept as **validated-negative experimental infrastructure** (default-OFF; not wired into the production assembler), like the regime/LSTM experiments. News stays **display context**, never a model input. The precise claim (more careful than the single-split's "no gain"): **no robust, control-validated, economically-meaningful gain** — a faint lightgbm lean exists but is within noise, absent for logistic, and attributable to model flexibility rather than news (the retune control). Answers the question the project deferred "to last", with the strongest test in the repo (multi-fold walk-forward + held-out retune): the **information ceiling holds** — direction is efficient, magnitude is already captured by price/VIX/GARCH. Closes the post-V1 ML track.

**Caveats / what would change the verdict.** GDELT tone is a crude bag-of-words score and the org→ticker map is noisy; a future re-test could try the **Loughran-McDonald** finance lexicon (GCAM) instead of generic `V2Tone`, entity-disambiguated counts, or news-volume *surprise* — but the bar is high given how flat (and retune-explained) the response is, and feature-searching a near-null signal risks p-hacking.

**Reproduce.**
```bash
# 1. Pull the data (one-time, ~945 GiB scan, free tier — see docs/NEWS_INGEST.md)
python -m stock_agent ingest-news --start 2023-01-01 --end 2026-06-03 --project YOUR_GCP_PROJECT
# 2a. Quick single-split A/B (both models)
PYTHONPATH=src python scripts/validate_news_features.py
AB_MODEL=lightgbm PYTHONPATH=src python scripts/validate_news_features.py
# 2b. Full walk-forward + retune control (the rigorous run)
PYTHONPATH=src python scripts/validate_news_features_full.py
```

---

## 2026-06-04 — XGBoost reconsidered: heavy tuning + calibrated 3-way bake-off

**Question.** XGBoost was dropped early (it didn't beat the toolkit at its cost). Give it a second chance with the **same rigor lightgbm got** (config #38): heavy random-search tuning, then a rigorous walk-forward comparison vs logistic & lightgbm — run **with the news features**, and **calibrated the way production serves** (`CalibratedClassifierCV` isotonic).

**Setup** (`scripts/tune_validate_xgboost.py`). 30-ticker basket (volatile semis + mega-cap + stable large-caps); **18 price + 7 active news** features (the 10 topic columns are excluded — all-NaN until `topics.csv` is pulled); horizons 20/30/60; **7 embargoed walk-forward folds** (2024-07→2026-04, ~23k pooled rows); directional + big-move targets; metrics = Brier / AUC / **ECE** as mean ± std over folds, with a base-rate floor.
- **Tuning** (winner's-curse-guarded): random search of 60 configs on a **20-ticker** TUNE basket (train→val big-move Brier), then the winner is confirmed on a **disjoint 10-ticker held-out** basket. Winner: shallow + heavily regularized (`max_depth=3, lr=0.01, min_child_weight=50, colsample=0.48, reg_lambda≈4.7, reg_alpha≈1.8, gamma≈0.56, n=400`).

**The key methodological finding — calibration is the equalizer.**
- **Raw (no CCCV):** tuned-xgb looked **best** at every horizon; lightgbm was **worst** and the only **"poorly calibrated"** model (ECE 0.11–0.13). This made xgboost look like a real upgrade.
- **Calibrated (CCCV, all three — the production-faithful test):** CCCV fixes lightgbm's calibration, all three converge, and **logistic is the best Brier at every horizon**. XGBoost's raw edge was a **calibration artifact** — its heavy regularization self-calibrates, so it only led when the others were left un-calibrated. Equalize calibration and the advantage vanishes.

**Calibrated results** (30 tickers, 25 features, **CCCV cv=5**; cv=3 gave the same story):

| Horizon | base_rate Brier | **logistic** | lightgbm | xgboost | xgb ECE |
|---|---|---|---|---|---|
| h20 | 0.2016 | **0.1948** | 0.1993 | 0.1959 | 0.095 |
| h30 | 0.1584 | **0.1498** | 0.1546 | 0.1547 | 0.094 |
| h60 | 0.1543 | **0.1483** | 0.1516 | 0.1527 | **0.103 (poor)** |

- Per-fold Brier std ≈ ±0.011–0.029, so the model-class gaps are **within noise** — logistic is *nominally* best, all three statistically tied. AUC tells the same story (logistic ≈ trees, ~0.62–0.68, all ≫ 0.5).
- **XGBoost has no edge** and is the **only poorly-calibrated model at h60** (cv=5 did not rescue its long-horizon calibration). It can rank (AUC fine) but can't convert that to a better Brier.
- Direction stays efficient: base_rate wins the 0% target (and the far tails at h60) at every horizon.

**Decision. Do NOT promote XGBoost** — calibrated (as we serve), it does not beat logistic; it's heavier, more complex, and worse-calibrated at long horizons. This **vindicates the original drop-xgboost call**; logistic stays the default, lightgbm the volatile-name tree companion. The broader lesson recorded: **compare models on their served (calibrated) footing** — a raw comparison rewarded the model that happened to self-calibrate, not the most skillful one.

**Caveats.** 30-ticker basket (not the full 114-universe), single tuning seed, lightgbm left on config #38 (not re-tuned for this exact setup — user-accepted; the calibrated gap is small enough that re-tuning is unlikely to flip past a calibrated xgboost). News features are the 7 active ones (topic features pending the July pull).

**Reproduce.**
```bash
PYTHONPATH=src python scripts/tune_validate_xgboost.py   # tune + calibrated 3-way bake-off
```

---

## 2026-06-04 — Ensemble: 5-member linear probability pool (equal vs skill-weighted)

**Question.** GARCH's win showed different *mechanisms* are complementary — does a probability-pooled ensemble of the validated models beat the best single one? Ensembles usually win on Brier without new information, so this is a ceiling-proof lever.

**Model.** `forecasting/ensemble.py` — `EnsembleForecast`, a linear pool over 5 members: `historical_sim`, `monte_carlo_bootstrap`, `monte_carlo_garch`, pooled `logistic`, pooled `lightgbm`. Bucket probs linear-pooled; E[r]/upside/downside exact weighted means; **VaR/CI recomputed from the mixture CDF** (never averaging quantiles — `forecasting/quantiles.py`). ML members that lack a trained artifact self-report a historical fallback and are dropped. (Side improvement shipped with this: ML forecasters now emit bucket-derived VaR/CI, so they contribute quantile anchors and no longer flatten the mixture tail.)

**Setup** (`scripts/validate_ensemble.py`). Walk-forward via the **production harness primitives** (`walk_forward_splits` → `exceedance_probabilities` → `threshold_metrics`/`calibration_report`) with per-fold pooled ML retrain (shared cache). 4 tickers (NVDA/AMD/MSFT/KO), h20/h60, identical as-ofs for all models. Two weightings: **equal**, and **online convex stacking** — at each fold the simplex weights (ridge-pulled to uniform) that minimised the ensemble's exceedance Brier on ALL PRIOR folds only (leakage-safe; plain 1/Brier is too flat to matter since members' Briers cluster ~0.17).

**Result (mean over 4 tickers; Brier ↓, ECE ↓, bigAUC ↑):**

| h | best member | ens equal | ens skill | ens bigAUC | baselines bigAUC |
|---|---|---|---|---|---|
| 20 | GARCH **0.1680** | 0.1688 | 0.1689 | **0.682** (ECE 0.054) | 0.49–0.60 |
| 60 | bootstrap **0.1288** | 0.1319 | 0.1330 | — (n too small) | — |

- **Does NOT beat the best single member on Brier** at either horizon (ties GARCH at h20, slightly trails bootstrap at h60). The best single model is already near the price-only Brier floor; averaging correlated members can't beat it. Contrast GARCH, which won by adding a *new mechanism* — pooling existing ones adds no information.
- **Skill-weighting (stacking) adds nothing** — weights barely move from uniform (members' Briers cluster within ~0.01) and ~11 folds is too few for learned weights to generalize (h60 stacking is marginally *worse* than equal). **Dropped — equal weights shipped.**
- **Genuine strength:** highest **big-move AUC** (h20 0.68 vs baselines ≤0.60, near the ML members' 0.65–0.70) at **markedly better calibration** (ECE 0.054 vs ML 0.064–0.085). It's a *better-calibrated way to get the magnitude signal* than raw ML.

**Decision (user call). Promote the equal-weight 5-member ensemble as a robust all-rounder** — *not* on a Brier win (it has none), but because it is **never the worst**, has the **best magnitude discrimination at honest calibration**, and gives **one default forecast** with no per-horizon model-picking. Honest caveat recorded: on the headline Brier score it ties (not beats) the best single baseline; for the lowest Brier at a specific horizon, GARCH (h20) / bootstrap (h60) remain marginally ahead. **Status: available now (`forecast --model ensemble`); agent-vetting + docs wiring PENDING (next session).** Pre-promotion calibration check (verified, not assumed): the 2 ML members are CCCV cv=3 isotonic-calibrated (`is_calibrated=True` on served artifacts) + lightgbm tuned (#38); the 3 baselines aren't classifiers so CCCV is N/A but they're empirically well-calibrated (ECE 0.03–0.07); the *pooled* ensemble measured ECE 0.054 at h20. Open: promotion scope + whether to add a thin post-hoc isotonic so it reports `calibration_status="calibrated"`.

**Reproduce.**
```bash
PYTHONPATH=src python scripts/validate_ensemble.py   # equal + online-stacked, per-fold ML retrain
```

---

## 2026-06-06 — Conformal interval coverage: every model's CI was too narrow

**Question.** ECE says the bucket *probabilities* are honest — but is a stated **90% CI** actually a 90% interval out-of-sample? (For a prediction tool, an untrustworthy CI is worse than none.)

**Method.** Split-conformal (`forecasting/conformal.py`): the backtest harness now records each OOS `(ci_low, ci_high, realized)` and reports stated-vs-conformalized coverage (`ConformalReport`). The served correction is **pooled offline** (`train_conformal.py`, CLI `conformal-calibrate`): build each model as-of a cutoff, forecast the held-out window over a 24-ticker basket, pool the (CI, realized) pairs → one `q` per (model, horizon).

**Result — every model under-covered; conformal fixed all of them (full universe run):**

| Horizon | stated 90% CI actually covered | after conformal `q` |
|---|---|---|
| h20 | 82–86% | **90%** (q ≈ +0.04–0.07) |
| h30 | 81–87% | **90%** (q ≈ +0.04–0.07) |
| h60 | **76–81%** | **91%** (q ≈ +0.10–0.16) |

The intervals were systematically too narrow — worst at h60 (a "90%" interval really covering ~76%). The pooled `q` widened them to honest coverage everywhere, for every model (baselines + ML + ensemble).

**Decision.** Ship conformal-calibrated CIs/VaR — applied at inference (`settings.conformal_intervals`, default on) and **recomputed in the monthly retrain** so it tracks the served models (proven end-to-end on CI run 27050677469: train → conformal-calibrate → verify-both → publish `conformal.json` → `make pull-models`). Caveat: coverage is **marginal** (pooled across tickers), not conditional per ticker; `var_95` is corrected exactly (= `ci_low`), `var_99` shares the same `q` (approximate at the 1% tail).

**Reproduce.**
```bash
python -m stock_agent conformal-calibrate           # → outputs/models/conformal.json
python -m stock_agent backtest --ticker NVDA --horizon 60   # see interval coverage in the report
```

---

## 2026-06-06 — Sequence × news: the news-negative is an information ceiling, not model-class

**Question.** Two priors said news doesn't help (tabular ML) and the LSTM doesn't beat the toolkit (on price). The remaining cell: news sentiment is *temporal* (spikes, decay, momentum) — structure a one-day tabular snapshot flattens. Does a **sequence model**, handed the raw sentiment sequence, extract signal the snapshot ML couldn't?

**Method.** `scripts/validate_sequence_news.py` driving the careful tuning protocol in `sequence_tune.py`. For each arm (price-only / price+news): a **DEEP-LSTM grid** (3–4 layers, 128–256 hidden, LayerNorm, dropout, lookback 60/90), **early-stopped** on val Brier, **temperature-calibrated**, the best config selected by calibrated val Brier — then scored on a **held-out TEST split it never saw**. Fair same-window ablation: identical windows over full history (news NaN→0 before coverage, real after) and identical train/val/test dates (train ≤2025-01, val ≤2025-07, test >2025-07); arms differ **only** by the news channel. 37 tickers. Decision metric = **TEST** big-move AUC at the **inner band** (h20 |r|>0.05, h60 |r|>0.15 — where there's real skill *and* enough events) + TEST exceedance Brier.

> *(An earlier shallow, outer-band probe (1 layer, |r|>0.10/0.30) is superseded by this run — it over-stated the harm; the rigorous read is "no skill", below.)*

**Result — news adds no discrimination; robust across two hyperparameter regimes.**

dropout=0.2 / lr=3e-4 (the gentler grid), held-out TEST:

| arm | h20 TEST Brier | h20 big-move AUC | h60 TEST Brier | h60 big-move AUC |
|---|---|---|---|---|
| **price** | 0.1709 | **0.646** | 0.1237 | **0.682** |
| **price+news** | 0.1720 | 0.634 | 0.1240 | 0.666 |
| **Δ (news−price)** | +0.0011 | **−0.012** | +0.0002 | **−0.015** |

The decision metric — big-move **discrimination (AUC) — is *worse* with news at both horizons**, and Brier is flat-to-worse. The deep nets **do not overfit** (train AUC ≈ val AUC), so news isn't being "regularized away" — there's simply no signal to find. The price-only deep LSTM is itself respectable (TEST big-move AUC 0.646 / 0.682) — the model class is fine.

**Robustness + the held-out lesson.** An aggressive grid (dropout=0.3 / lr=1e-3) gave the same verdict (big-move ΔAUC −0.006 / −0.018), so the negative isn't an optimization artifact. And the one place news *looked* better — h60 **validation** Brier (0.1297 vs 0.1328 price) — **reversed on held-out TEST** (0.1240 vs 0.1237; AUC 0.666 vs 0.682): a spurious validation signal that didn't generalize, exactly what the disjoint test split exists to catch. News fits noise, not signal.

**Decision — do not promote.** The conclusion is now **robust to the obvious objections** (depth, careful tuning, correct inner band, held-out test, no overfit): a deep, tuned temporal model handed the raw sentiment sequence extracts no incremental discrimination. The news-negative is an **information ceiling, not a model-class limitation** — across all three cells: (1) sequence ⊀ tabular on price, (2) news ⊀ price-only on tabular, (3) news ⊀ price-only on the *deep, tuned* sequence model. **Caveat:** per-ticker + market sentiment only; the **topic streams (AI/tech/healthcare/energy) are not yet pulled** — the **July re-validation** (runnable through both the tabular harness and now this sequence harness) is the remaining open door.

**Reproduce.**
```bash
PYTHONPATH=src python scripts/validate_sequence_news.py
```

## 2026-06-12 — Feature expansion: Tier 1 candidate groups (volume / session / shape promoted)

**Question.** The production feature set was price/technical + VIX + earnings-cadence only. Do orthogonal, scale-free, point-in-time feature groups add out-of-sample value? P/E and other fundamentals were ruled out up front (quarterly step-functions, weak at 20–60d, point-in-time-reconstruction burden = leakage risk on free APIs). Five OHLCV(+SPY)-derived candidate groups were tested: `volume` (relative volume + dollar-volume z), `high52w` (nearness-to-52w-high), `session` (overnight/intraday return split), `shape` (realized skew + downside/upside semivol ratio), `relstr` (market-relative strength vs SPY).

**Setup** (`scripts/ablate_feature_groups.py` → `backtesting/ablation.py`). Walk-forward, per-fold pooled refit over the full 114-ticker universe; 8-ticker evaluation spread (NVDA MSFT AAPL JPM XOM JNJ TSLA WMT); horizons 20/30/60; `--test-size 24`. Each group compared **individually vs baseline** on identical folds. Δ shown as *improvement* (positive = group helped): ΔBrier, ΔECE (calibration), Δbig-move-AUC. Promote gate: Brier↓ without ECE↑.

**Model scope — logistic only.** The lightgbm walk-forward ablation is computationally infeasible: one cell ran **69 min without finishing even the baseline** (per-fold 400-tree refit over 114 tickers ≈ ~36h for the sweep). logistic did a full cell in ~6 min. So the *decision* is logistic-evidence-based; lightgbm is gated at retrain via the `verify-models` metric gate (granular per-artifact fallback available if it regresses).

| group | h20 ΔBrier/ΔECE/ΔAUC | h30 | h60 | verdict |
|---|---|---|---|---|
| **shape** | +.0004 / +.0043 / +.006 | +.0012 / +.0087 / +.002 | +.0049 / +.0083 / +.014 | **promote** — positive on all 3 metrics × all 3 horizons; Brier gain grows with horizon |
| **volume** | +.0002 / +.0027 / +.013 | −.0001 / +.0064 / +.001 | −.0007 / +.0050 / +.003 | **promote** — Brier ~flat (noise), ECE+AUC consistently +, never hurts |
| **session** | −.0001 / −.0044 / +.019 | +.0013 / +.0056 / −.001 | −.0009 / +.0107 / +.007 | **promote (marginal)** — Brier ~noise, net-positive ECE/AUC over h30/h60; weakest of the three |
| high52w | −.0002 / +.0001 / −.003 | −.0051 / −.0061 / −.004 | −.0101 / −.0405 / +.011 | **reject** — consistently hurts Brier + calibration (badly at h60) |
| relstr | −.0086 / −.0024 / −.017 | −.0078 / −.0171 / −.023 | −.0022 / −.0139 / +.002 | **reject** — hurts Brier + ECE + AUC at every horizon |

**Decision — promote `shape` + `volume` + `session`** into the baseline `PRICE_FEATURE_COLS` (18 → 24 features). All three are OHLCV-only, so they add **no new data dependency**. `shape` is the clear, robust winner (the tail-shape signal strengthens with horizon, as expected). `volume` is a clean low-risk add. `session` is marginal (Brier within noise) but net-positive on ECE/AUC and never harmful — included with that caveat; the retrain `verify-models` gate is the lgbm safety net. **Rejected `high52w` and `relstr`** — both degrade calibration; notably `relstr` (subtracting the market return) hurts the pooled logistic at every horizon. They remain opt-in (`FEATURE_GROUPS`) but are not in the baseline. Caveats: deltas are small and this is 8 evaluation tickers — `shape`'s cross-horizon, cross-metric *consistency* is what carries the decision, not any single cell. **Both models × {20,30,60} retrained on the 24-feature set; all calibrated; `verify-models` ✓; conformal `q` recomputed.**

**Not evaluated — `insider` (Tier 2, Form 4).** Its ablation was blocked by SEC EDGAR fair-access throttling at the 114×200-filing scale (handshake timeouts). A real bug was fixed en route: EDGAR lists the Form 4 `primaryDocument` as the XSL-rendered HTML path (`xslF…/form4.xml`), which fails XML parsing — `InsiderFilingRef.url` now de-renders to the raw ownership XML. Insider stays opt-in pending a bounded, resumable cache-warm + its own ablation.

**Reproduce.**
```bash
# Tier 1 ablation (logistic; ~6 min/cell):
for h in 20 30 60; do
  PYTHONPATH=src python scripts/ablate_feature_groups.py \
    --tickers NVDA MSFT AAPL JPM XOM JNJ TSLA WMT --horizon $h --model logistic \
    --groups volume high52w session shape relstr --test-size 24
done
# Retrain + recalibrate on the promoted 24-feature set:
PYTHONPATH=src python -m stock_agent train --all
PYTHONPATH=src python -m stock_agent conformal-calibrate
make verify-models
```

## 2026-06-12 — Insider (Form 4) feature: re-engineered, and the universe-dependence finding

**Question.** Does insider (Form 4) activity help the pooled classifiers? A first pass (v1) said no — but it was both bluntly encoded *and* tested only on the mega-cap production universe.

**v1 (rejected).** Features: `insider_net_63d` = trailing-63d net signed insider \$ ÷ dollar volume, and `insider_imb_63d` = buy/sell count imbalance. On the mega-cap universe, logistic × {20,30,60}: deltas ≤0.0003 Brier, inconsistent, **h60 exactly 0.000** — nil. Diagnosis: (1) **netting buys against sells** destroys the signal (informative buys swamped by routine sells); (2) **dollar-volume normalization** crushes a real buy to ~0 on mega-caps; (3) **sparsity** → near-zero variance; (4) no notion of *who*/*conviction*/*routine-ness*.

**Re-engineering (v2).** Parser extended to pull reporting-owner CIK, role flags (`isOfficer`/`officerTitle`/…), post-transaction holdings, and the Rule 10b5-1 flag (structured + footnote-text). New features separate the channels and weight by conviction:
- `insider_buy_conviction_63d` — Σ Δ-ownership of opportunistic **buys** (conviction, not dollars; self-normalizing; outlier-capped at 100%).
- `insider_senior_buy_63d` — count of opportunistic **CEO/CFO** buys (the highest-signal subset).
- `insider_sell_pressure_63d` — Σ |Δ-ownership| of opportunistic **sells**, kept **separate** (never netted).
All exclude Rule 10b5-1 (pre-scheduled, non-discretionary) trades; all `filing_date`-anchored (leakage-safe).

**Setup.** `scripts/ablate_feature_groups.py --groups insider` (now with `--universe`), logistic × {20,30,60}, `--test-size 24`, 8 eval tickers per universe. EDGAR Form 4 XML pre-warmed via `scripts/warm_insider_cache.py` (pooled keep-alive client + retry + 5 req/s — fixes the fair-access throttling that blocked the inline fetch; mid-cap 40/45 tickers, mega-cap 106/114, 0 download failures). Δ = improvement (>0 helped).

| universe | h20 ΔBrier/ΔECE/ΔAUC | h30 | h60 | gate |
|---|---|---|---|---|
| **mid-cap** (`configs/universe_insider.txt`) | +.0007 / +.0008 / −.008 | +.0001 / +.0053 / −.027 | +.0024 / +.0171 / −.012 | candidate ×3 |
| **mega-cap** (`configs/universe.txt`, control) | −.0001 / −.0001 / −.006 | +.0002 / −.0024 / +.009 | 0.000 / 0.000 / 0.000 | none |

**Finding — structural, not encoding.** Re-engineered insider **consistently improves Brier and calibration (ECE) at all three horizons on the mid-cap universe** (strongest at h60: ECE +0.017), clearing the promote gate every time. The *same features* on the mega-cap universe are **nil** (h60 again exactly 0.000). So insider signal is **universe-dependent**: it lives in small/mid-caps where insiders make opportunistic open-market *buys*; mega-cap Form 4s are dominated by routine scheduled *sells*, so there is no signal to extract regardless of encoding. Caveat: insider is directional — it modestly **lowers big-move (|return|>k) AUC** even where it helps Brier (a calibration-vs-tail-discrimination tradeoff).

**Decision.** **Keep `insider` opt-in; do NOT promote into the mega-cap production baseline** (nil there, confirmed twice). It is a **validated feature for a mid-cap-inclusive universe** (Brier + calibration gains), ready to enable in that regime. The parser enrichment, re-engineered features, and `warm_insider_cache.py` tooling are retained.

**Reproduce.**
```bash
PYTHONPATH=src python scripts/warm_insider_cache.py --universe configs/universe_insider.txt
for h in 20 30 60; do
  PYTHONPATH=src python scripts/ablate_feature_groups.py --tickers RF KEY ZION FITB AR SM OSK AGCO \
    --horizon $h --model logistic --groups insider --universe configs/universe_insider.txt --test-size 24
done
# mega-cap control: --tickers NVDA MSFT AAPL JPM XOM JNJ TSLA WMT --universe configs/universe.txt
```

---

## 2026-06-28 — A5.3: GraphRAG promotion for the multi-hop path (the §15.9 2×2)

**Question.** Does **agentic RAG over the GraphRetriever** match-or-beat the production multi-hop path
(**agentic + hybrid + the A4 alias-bridge**) on *bridging* questions — enough to route multi-hop
queries through the graph — or is GraphRAG an opt-in with no measured win?

**Design (control × substrate 2×2).** Unlike the backtests above this is a **retrieval** eval, scored
by **aspect coverage** (not Brier): each question has labeled answer-bearing spans per hop; coverage =
fraction of a question's aspects whose spans appear in the retrieved evidence union (`research/
multistep_eval.py`). The four cells:

| | hybrid (vector) | graph |
|---|---|---|
| **single-shot** | **A** | **C** |
| **agentic loop** | **B** (alias-bridge ON) | **D** (alias-bridge OFF) |

Two `rag eval-multistep` runs per seed yield all four cells (each run reports single-shot **and**
agentic-loop coverage for its retriever). Benchmark: `configs/rag_eval_multistep.json` — **12 Q**,
strata **HARD 0–5** (cross-filing bridges: NVDA→MU CAC/NAND, NVDA→TSM political-stability/site-
concentration, NVDA→AMD 7 nm, NVDA→INTC IDM 2.0), **MED 6–8** (TSM earthquake, MU capex, MU ASP),
**CTRL 9–11** (single-entity, non-bridging: NVDA data-center/competition, NVDA export-control/China,
MU DRAM/NAND). Every bridge aspect-2 span was corpus-verified **present in the bridged entity's own
filing and absent/rare in NVDA's** (e.g. "political stability" TSM:6 / NVDA:0; "7 nm" AMD:33 / NVDA:0;
"IDM 2.0" INTC:32 / NVDA:0; "critical information infrastructure" MU:50 / NVDA:0) so single-shot can't
cover it via the seed — the question genuinely tests the hop. **2 seeds** (loop LLM queries are
non-deterministic → results are directional, averaged).

**Per-question coverage (both seeds).** A = single-hybrid, B = agentic+hybrid+bridge, C = single-graph,
D = agentic+graph (bridge off):

| # | stratum | seed-1 A/B/C/D | seed-2 A/B/C/D | question (abbrev) |
|---|---|---|---|---|
| 0 | HARD | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | NVDA→memory supplier w/ CAC restriction |
| 1 | HARD | 0.00 / 1.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 / 1.00 | NVDA→memory supplier w/ NAND oversupply |
| 2 | HARD | 0.00 / 1.00 / 0.00 / 1.00 | 0.00 / 1.00 / 0.00 / 1.00 | NVDA→foundry w/ political-stability risk |
| 3 | HARD | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | NVDA→foundry site concentration |
| 4 | HARD | 1.00 / 0.50 / 0.50 / 1.00 | 1.00 / 0.50 / 0.50 / 0.50 | NVDA competitor on TSMC 7 nm |
| 5 | HARD | 0.50 / 1.00 / 0.50 / 1.00 | 0.50 / 1.00 / 0.50 / 1.00 | NVDA competitor IDM 2.0 foundry strategy |
| 6 | MED | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | NVDA→foundry earthquake risk |
| 7 | MED | 0.50 / 1.00 / 0.00 / 1.00 | 0.50 / 1.00 / 0.00 / 1.00 | NVDA→memory supplier capex intensity |
| 8 | MED | 0.50 / 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 / 0.50 | NVDA→memory supplier ASP decline |
| 9 | CTRL | 1.00 / 0.50 / 0.50 / 1.00 | 1.00 / 1.00 / 0.50 / 1.00 | NVDA data-center demand & competition |
| 10 | CTRL | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | NVDA export controls / China |
| 11 | CTRL | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | Micron DRAM/NAND product lines |

**Cell means (pooled over both seeds):**

| stratum | A | B | C | **D** | D−B | D−A |
|---|---|---|---|---|---|---|
| HARD | 0.58 | 0.92 | 0.67 | **0.96** | **+0.04** | +0.38 |
| MED  | 0.67 | 0.83 | 0.50 | **0.83** | 0.00 | +0.17 |
| CTRL | 1.00 | 0.92 | 0.83 | **1.00** | +0.08 | 0.00 |

**Finding.** The strict pre-registered rule failed *as written* — `D − A ≥ +0.5` on the bridging
subset missed (+0.38; single-shot hybrid started above the assumed 0.5), and **single-shot** graph (C)
regressed controls (0.83 < A 1.00, neighbor-noise on simple Qs). **But the decision-relevant comparison
is D vs B** — graph vs the production alias-bridge *on the multi-hop path*: **HARD `D ≥ B` held on both
seeds** (1.00, 0.92 ≥ 0.92) and **D = 1.00 on controls both seeds** (the seed-1 B=0.83 control dip was
loop noise; B=1.00 on seed-2). Read per-question, the signal is sharp on the true bridges Q1/Q2 (A=0 →
B=D=1.00: single-shot literally cannot reach the bridged entity's own disclosure; both the alias-bridge
*and* graph traversal recover it). So **agentic+graph matches-or-beats the alias-bridge everywhere and
is ~perfect on hard bridges with the bridge OFF** — graph *replaces* the brittle query-time alias scan
and adds hard-question lift. The one HARD wobble is Q4 (seed-2 D 0.50) — loop non-determinism, the
reason the HARD D−B margin is small (+0.04, one seed exact-tie). Q8 (ASP) is a stable 0.50 floor for
all cells (second aspect not reliably retrieved by any method).

**Decision — PROMOTE, scoped (tiered routing).**
- **EASY / single-topic → single-shot hybrid** (single-shot graph regressed controls; also $0).
- **MED/HARD multi-hop → agentic loop over `GraphRetriever`** (`settings.graph_multistep_enabled`,
  default ON; `ToolExecutor._get_multistep_retriever`). Graph-built tickers
  (`configs/graph_universe.txt`) bridge via stored edges; others degrade to the hybrid base.
- **A4 alias-bridge kept ON** as a universal fallback (revised from the plan's "gate off when graph
  active" — that was the experimental control). Non-graph tickers have no traversal, so the bridge is
  their only bridging mechanism; for graph tickers it is additive/redundant (dedup union), so live
  behaviour `≥` the measured D. Single-shot graph is **never** the default.

**Caveats.** n=12, 2 seeds, loop non-determinism → **directional, not definitive**; the promotion rests
on **D never falling below B**, not a large margin. Grow the benchmark (§A1) before any stronger claim.
A6 (retrieval RL) is the principled successor — *learn* the per-query routing instead of the hand-set
`graph_multistep_enabled` switch.

**Reproduce** (local, paid — only the agentic loop spends Sonnet; single-shot baseline is $0):
```bash
# Run 1 (cells A+B): hybrid, alias-bridge default ON
PYTHONPATH=src python -m stock_agent rag eval-multistep \
  -q configs/rag_eval_multistep.json --report outputs/rag_eval/a5_3_hybrid.json
# Run 2 (cells C+D): graph substrate, alias-bridge disabled via the existing setting
AGENTIC_BRIDGE_MAX_ENTITIES=0 PYTHONPATH=src python -m stock_agent rag eval-multistep \
  -q configs/rag_eval_multistep.json --graph --report outputs/rag_eval/a5_3_graph.json
# Repeat both for seed 2 (…_seed2.json); pooled 2×2 = scratchpad analyze_a5_3_pooled.py.
# Prereq: NVDA/MU/AMD/INTC 10-K + TSM 20-F ingested; graph built for configs/graph_universe.txt.
```

---

## 2026-07-05 — A6.1: retrieval contextual bandit + off-policy evaluation (verdict 2026-07-08: **REJECT** — keep default-OFF) ❌

**Question.** A5.3 showed the best retrieval config is **context-dependent** (HARD bridges → graph,
CTRL → hybrid/dense). Can a **learned contextual-bandit policy** realize that per-query lift — beating
the best *fixed* arm past a group-bootstrap CI on a group-wise held-out split — or is a rigorous
**negative** the outcome? Either way the logging + OPE + bandit infrastructure is the deliverable.

**Design (off-policy, not a deployment A/B).** Retrieval is a one-shot decision: context
$x = \mathrm{featurize}(\text{query})$ (label-free, 11-dim) → policy $\pi(a \mid x)$ picks 1 of **5
arms** (`dense, reranked, hybrid, hybrid+rerank, graph`) → reward $r = \mathrm{quality} - \lambda_c
c(a)$, quality = single-shot aspect **coverage** (the A6.0 metric, retrieval-only, **\$0**). The oracle
is deterministic + \$0, so we compute the **full-information reward matrix** $R[N,K]$ and (i) synthesize
a uniform-$\mu$ log (full support ⇒ OPE exact), (ii) fit LinUCB / ε-greedy offline on the **train**
fold, (iii) score every candidate on the **test** fold via **DR** (IPS/SNIPS/ESS reported), with the
$R$-derived **true value** as a ground-truth check. Group-wise split (`split_multihop`) + group-level
bootstrap throughout (§17.4, anti-pseudo-replication). Theory → [rag_concepts.md
§18](rag_concepts.md); mechanism → [rag_implementation_notes.md §A6.1](rag_implementation_notes.md).

**Pre-registered decision rule (fixed before seeing any numbers).** Promote (`adaptive_retrieval`
default→True) **iff**, on the group-wise held-out test fold,
$\hat{V}_{\mathrm{DR}}(\pi_{\text{bandit}}) - \hat{V}_{\mathrm{DR}}(\pi_{\text{fixed}}^\star) > 0$ **and**
the paired group-bootstrap 95% CI lower bound $> 0$ **and** no per-stratum regression (must not lose to
the best fixed arm on **CTRL**). Else keep default-OFF and **record the negative**. Report per-stratum
(HARD/MED/CTRL). Sensitivity-test $\lambda_c \in \{0, 0.05, 0.1\}$ via `--lambda-cost`. Likely outcome:
tuned hybrid(+graph) is strong → a **modest win or a rigorous negative**.

**Status — infra COMPLETE & GREEN; verdict EXECUTED 2026-07-08 → REJECT.** Slices A6.1a–f shipped,
`make check` green, default-OFF (read path byte-identical to A5.3). CI pins the mechanics against
hand-computed goldens and a synthetic reward matrix with a *known* contextual optimum (bandit promotes;
no-signal rejects). The local verdict run has now executed (\$0 retrieval, no LLM) — numbers below.

**Verdict (local run, 2026-07-08; `outputs/rag_eval/policy_eval_linucb_seed42.json`).** LinUCB α=1,
seed 42, `n_train=129 / n_test=83`, λ_c=0.05, 1000-resample **group** bootstrap over the 212-Q
multi-hop benchmark. DR is the pre-registered headline; `true_value` is the \$0-oracle full-information
value (ground-truth check).

| candidate | DR | DR 95% CI | IPS | SNIPS | true_value | ESS |
|---|---|---|---|---|---|---|
| fixed(dense) | 0.414 | [0.185, 0.616] | 0.392 | 0.406 | 0.355 | 16 |
| fixed(reranked) | 0.257 | [0.139, 0.373] | 0.347 | 0.360 | 0.328 | 16 |
| fixed(hybrid) | 0.287 | [−0.073, 0.546] | 0.445 | 0.308 | 0.387 | 24 |
| fixed(hybrid+rerank) | 0.309 | [0.184, 0.454] | 0.228 | 0.344 | 0.317 | 11 |
| fixed(graph) | 0.345 | [0.174, 0.469] | 0.347 | 0.360 | **0.437** | 16 |
| **linucb(α=1)** | **0.438** | [0.325, 0.533] | 0.442 | 0.458 | **0.441** | 16 |
| epsilon_greedy(0.1) | 0.433 | [0.242, 0.593] | 0.407 | 0.421 | 0.403 | 19 |

`best_fixed = fixed(dense)` (argmax DR). **Δ = DR(linucb) − DR(dense) = +0.0239**, paired group-bootstrap
95% CI **[−0.208, +0.273]**.

Per-stratum (bandit vs. fixed):

| stratum | n | DR bandit | DR fixed | Δ |
|---|---|---|---|---|
| HARD | 41 | 0.338 | 0.228 | **+0.110** |
| MED | 15 | 0.648 | 0.343 | **+0.305** |
| CTRL | 27 | 0.473 | 0.736 | **−0.263** |

**Decision: `promote = false` — keep `adaptive_retrieval=False`.** The pre-registered rule fails on **two
independent counts**: (1) the Δ 95% CI **includes 0** (Δ=+0.0239 ∈ [−0.208, +0.273]) — the effect is
statistically indistinguishable from zero; (2) a **CTRL regression** (−0.263) — the rule forbids losing
to the best fixed arm on control queries. Either alone blocks promotion; both fire. This is the
pre-registered **rigorous negative** — the logging + OPE + bandit infra is the shipped deliverable.

**Honest read (direction vs. certainty).** Point estimates favor the bandit: linucb is top on **DR
(0.438), SNIPS (0.458), and — tellingly — `true_value` (0.441, highest of all 7 candidates)**, and it
wins HARD (+0.11) and MED (+0.30) convincingly. On the \$0 oracle it *is* the best policy in the set.
But the test is underpowered and there's a real CTRL loss, so it cannot be certified:
- **Tiny effective sample.** Deterministic policies under a uniform log credit only rows where the logged
  arm matched the pick (`w ∈ {0,5}`), so **ESS ≈ 16 of 83**. A group bootstrap over ~16 effective points
  cannot resolve a +0.024 gap → CI width ~0.48. A *power* problem, not proof the bandit is bad.
- **DR misranks the fixed arms.** DR calls **dense** best-fixed (0.414), but the oracle `true_value` says
  **graph** is best-fixed (0.437 vs dense 0.355). The `best_fixed` anchor is itself noisy; anchored on the
  oracle, linucb (0.441) only *ties* graph (0.437) overall — and is still CTRL-negative.
- **The CTRL loss inverts the illustrative guess.** The example expected the bandit to *win* CTRL via
  cheap `hybrid`; in reality the best CTRL arm is **dense** (cost 0.0) and the bandit over-retrieves on
  control — λ_c=0.05 didn't steer it to dense. Concrete lever for a re-attempt (or A6.2): larger/swept
  λ_c, a variance-reduced logging design (stratified / propensity-blended μ to lift ESS), or richer
  "dense-is-sufficient" features.

**Sensitivity sweep (executed 2026-07-08; `outputs/rag_eval/policy_eval_linucb_seed42_lambda_sweep.json`).**
Tested the hypothesis "a larger cost penalty makes the unified bandit pick cheap `dense` on easy queries,
fixing the CTRL regression." Built the quality matrix **once** (λ_c=0) and re-scored `R(λ_c) = quality −
λ_c·cost` through the verbatim `evaluate_offline` — the λ_c=0.05 re-score reproduces the headline verdict
exactly (Δ=+0.0239, CI [−0.208,+0.273], per-stratum identical), validating build-once-recost ≡ fresh run.

| λ_c | Δ_DR | 95% CI | HARD Δ | MED Δ | CTRL Δ | promote |
|---|---|---|---|---|---|---|
| 0.05 | +0.0239 | [−0.208, +0.273] | +0.110 | +0.305 | −0.263 | ✗ |
| 0.10 | +0.0133 | [−0.216, +0.260] | +0.097 | +0.294 | −0.270 | ✗ |
| 0.20 | −0.0033 | [−0.209, +0.268] | +0.081 | +0.272 | −0.284 | ✗ |
| 0.30 | −0.0171 | [−0.216, +0.248] | +0.064 | +0.254 | −0.291 | ✗ |

**Hypothesis refuted, instructively.** (1) The gradient runs *backwards*: raising λ_c worsens CTRL
(−0.263→−0.291), shrinks the real HARD/MED wins, and turns Δ negative by λ_c=0.20 — no sweet spot; the
shipped λ_c=0.05 is already the most favorable value. (2) Mechanism: `dense` is the cheapest arm (cost 0),
so λ_c is invariant for it and only taxes the heavier arms the bandit *wins* with on HARD/MED. (3) **Sharp
finding — the CTRL loss is ~97% retrieval-quality, not cost:** extrapolating the near-linear CTRL trend to
λ_c=0 leaves ≈ −0.256, i.e. on CTRL the bandit routes easy queries to arms scoring ~0.47 coverage where
`dense` scores ~0.74 — a **featurization/routing** failure the label-free features can't prevent, not a
cost-accounting artifact. (4) CIs stay ~[−0.21,+0.26] across the sweep — the ESS≈16 power limit is a
property of the logging design, not the reward scale. **Conclusion:** cost tuning is *not* the lever; the
two real levers are (a) features that separate "dense-is-sufficient" easy queries and (b) a higher-ESS
logging design. Promotion scoped to MED/HARD would require a *gated* policy (force `dense` on easy by
construction), since at this feature resolution the bandit demonstrably cannot learn that routing itself.

**Reproduce** (local, **\$0** — retrieval-only, no LLM):
```bash
# Prereq: NVDA/MU/AMD/INTC 10-K + TSM 20-F ingested; graph built for configs/graph_universe.txt.
PYTHONPATH=src python -m stock_agent rag policy-eval \
  --queries configs/rag_eval_multistep_generated.json --policy linucb \
  --test-frac 0.3 --seed 42 --out outputs/rag_eval/policy_eval_linucb.json
# Sensitivity: rerun with --lambda-cost 0 and 0.1; compare with --policy epsilon_greedy.
```

## 2026-07-08 — A6.1 follow-up: gated router (deterministic gate → bandit on hard) ❌

**Question.** The A6.1 verdict rejected the *unified* bandit on two counts — the Δ CI included 0 **and**
it regressed CTRL (−0.263) — and diagnosed the CTRL loss as ~97% a **routing/featurization** failure
(the bandit sent easy queries to arms scoring ~0.47 coverage where `dense` scores ~0.74). It predicted:
*"promotion scoped to MED/HARD would require a gated policy (force `dense` on easy by construction)."*
This follow-up builds exactly that and asks **two** pre-registered questions: (1) does the **gated
router** `gated(dense | linucb)` beat the best fixed arm past the CI with no CTRL regression? (2) does
the bandit **earn** the hard branch — beat `fixed(graph)` (the A5.3 tiered default) on HARD+MED only,
rather than being *assumed* onto it? Theory → [rag_concepts.md §18.10](rag_concepts.md); mechanism →
[rag_implementation_notes.md §A6.1](rag_implementation_notes.md).

**Design.** A `GatedPolicy` composes two branches by a **deterministic, label-free** gate: `hard` iff
`is_bridging` = 1 (threshold `x[j] > 0`, standardization-invariant for a 0/1 feature), else `easy` →
`dense`. Because the gate is deterministic given `x`, $\pi_{\text{gated}}(a \mid x) =
\pi_{\text{active branch}}(a \mid x)$ **exactly**, so the identical DR / IPS / SNIPS + paired group
bootstrap harness (A6.1) applies with no re-derivation. Same reward matrix, same group-wise split
(`n_train=129 / n_test=83`), same LinUCB α=1, seed 42, λ_c=0.05 as the A6.1 run — so the two are
directly comparable. `evaluate_gated` scores 5 fixed + 2 learned + 2 gated candidates.

**Verdict (local run, 2026-07-08; `outputs/rag_eval/gated_eval_seed42.json`).**

| candidate | DR | DR 95% CI | true_value | ESS |
|---|---|---|---|---|
| **gated(dense \| linucb)** | **0.524** | [+0.327, +0.660] | 0.443 | 17 |
| gated(dense \| fixed graph) | 0.496 | [+0.320, +0.627] | **0.453** | 16 |
| linucb(α=1) | 0.438 | [+0.325, +0.533] | 0.441 | 16 |
| fixed(dense) *(best fixed)* | 0.414 | [+0.185, +0.616] | 0.355 | 16 |
| fixed(graph) | 0.345 | [+0.174, +0.469] | 0.437 | 16 |

**[1] Promote the gated router? — REJECT.** `best_fixed = fixed(dense)`. **Δ = DR(gated) − DR(dense) =
+0.1096**, paired 95% CI **[−0.0564, +0.2872]**, exact one-sided bootstrap **P(Δ>0) = 86.8%**
(868/1000 group-resamples positive). Per-stratum (gated − best fixed): HARD **+0.110**
(n=41), MED **+0.305** (n=15), **CTRL +0.000 (n=27)**. The rule fails on **one** count — the Δ CI
includes 0 — but **the CTRL regression is gone** (−0.263 → **exactly 0.000**: the gate routes CTRL to
`dense`, which *is* `best_fixed`, so the strata match by construction). This is a **strict improvement**
over the A6.1 unified bandit (failed 2 counts): the gate fixed precisely the failure A6.1 diagnosed,
and the pooled point estimate quadrupled (**+0.0239 → +0.1096**). It is now the single best policy by DR
(0.524). It still cannot be **certified**: at **ESS ≈ 17** the CI width (~0.34) cannot resolve a +0.11
gap — the same power limit as A6.1, a property of the uniform-log design, invariant to the gate.

**Follow-up (b), 2026-07-08 — exact bootstrap probability.** The pre-registered rule gates on
`CI_low > 0`, not on P(Δ>0); a natural question is *how close* the +0.1096 delta came. The exact
one-sided bootstrap probability the gated router beats best-fixed is **P(Δ>0) = 86.8%** (the fraction
of the 1000 paired group-resamples with Δ>0 — the same resamples the CI is read from). This is a touch
below the ~89% Gaussian approximation `Φ(0.1096 / (0.343/(2·1.96))) ≈ 0.896`, because the bootstrap
distribution is mildly left-skewed. Promotion under the pre-registered two-sided-95% CI is equivalent
to **P(Δ>0) ≥ 97.5%**; at 86.8% the router is *suggestive but uncertifiable* — it does **not** overturn
the REJECT, it quantifies the gap to certification (≈ 11 pts of bootstrap mass, i.e. more effective
samples, not a different point estimate). Recorded as a first-class harness field `delta_p_positive`
(printed by the CLI, stored in the verdict JSON) so the CI and its one-sided companion stay consistent.

**[2] Does the bandit earn the hard branch? — NO.** On the **HARD ∪ MED** rows only (n=56), `linucb`
vs `fixed(graph)`: **Δ = −0.0250**, CI **[−0.0945, +0.0458]**, exact **P(Δ>0) = 28.9%** (the bandit is
*more likely worse* than the fixed graph tier than better) → `linucb` DR ≤ `fixed(graph)`. The \$0
oracle agrees: `gated(dense | fixed graph)` `true_value` **0.4528 ≥** `gated(dense | linucb)` **0.4435**.
So even *within* the hard branch the learned policy does not beat the A5.3 fixed default — the bandit is
**superfluous**; the deterministic gate + `fixed(graph)` captures the available lift.

**Decision: `promote = false` — keep `adaptive_retrieval=False`; A5.3's tiered router is vindicated.**
The two verdicts compose cleanly: (1) the gate is the *right fix* for the A6.1 CTRL regression (removed
by construction, point estimate now clearly positive) but the residual HARD/MED lift is **uncertifiable
at this ESS**; (2) the *learned* hard branch adds nothing over `fixed(graph)`. The architecture the data
supports is therefore **deterministic gate → fixed graph on hard = exactly A5.3** — no learned policy is
justified. This is the pre-registered **rigorous negative**; the gate + two-verdict harness are the
shipped, green deliverable. The only lever left with upside is a **higher-ESS logging design**
(stratified / propensity-blended μ), not a different policy class — that would be an A6.2 concern.

**A6.1 (contextual bandit) is now CLOSED** across three tests — the unified bandit (2 fails), the λ_c
cost sweep (refuted), and the gated router (1 fail + exact P(Δ>0)=86.8% < 97.5%). All roads reduce to
*deterministic-gate → fixed-graph = A5.3*; a **learned contextual policy is not justified at this
logging design**. Next phase: **A6.2 (Full RL for Retrieval)** — the agentic loop as a finite-horizon
MDP (state = query + evidence-summary; action = {STOP} ∪ {(config, scope)}; PPO primary), which reuses
this reward oracle + featurizer + group split + OPE harness verbatim. A6.2's first job is the very
lever A6.1 could not turn: a **higher-ESS / propensity-blended logging design** so a learned policy is
even *testable* with power. See `docs/ADVANCED_RAG_TODO.md` §A6.2.

**Reproduce** (local, **\$0** — retrieval-only, no LLM):
```bash
# Same prereqs as the A6.1 run above (ingested corpus + graph for configs/graph_universe.txt).
EMBEDDING_PROVIDER=voyage PYTHONPATH=src python -m stock_agent rag gated-eval \
  --queries configs/rag_eval_multistep_generated.json \
  --easy-arm dense --gate-feature is_bridging --hard-fixed-arm graph \
  --test-frac 0.3 --seed 42 --out outputs/rag_eval/gated_eval_seed42.json
```

---

## 2026-07-13 — A6.2 (full RL): verdict **RETRACTED — the experiment was invalid**, and a reward bug found 🔴

**Headline.** The A6.2 REJECT is **withdrawn**. It is not evidence about RL. Two defects in the
*environment* meant the MDP could not express the correct action on 89% of the episodes it graded,
and the coverage reward paid out for evidence that answers nothing. Every coverage number below is
superseded pending a re-run on the fixed environment.

### What was run (all on the held-out fold; the numbers themselves are reproducible)
| run | result |
|---|---|
| `$0` sim eval, 63 episodes, 9 candidates | rl(reinforce) greedy **+0.3643** vs best baseline react(hybrid) **+0.3875** → Δ = **−0.023**, 95% group-CI [−0.108, +0.064], P(Δ>0)=0.26 → REJECT |
| paid sim-to-real, 24 HARD/MED, 49 calls | LLM-written queries beat the template: coverage 0.208 → 0.292 (**+0.083**); action sequences 96% unchanged |
| paid real-env head-to-head, 24 eps, 57 calls | with **both** policies on real queries: rl **+0.2765** vs react **+0.2590**, Δ = **+0.0175**, CI [−0.050, +0.133], P(Δ>0)=**0.55** — a coin flip |

The head-to-head *flips the sign* of the `$0` verdict, so the templated simulator was penalizing the
learned policy specifically (rl gains +0.083 from real query text; react gains **0.000**). But the
delta never clears the pre-registered gate (CI_low > 0), and the whole comparison rests on **9 bridge
groups** — the bootstrap unit — where the CI half-width is ±0.077. The experiment could only ever
have detected an effect ≥ ~0.08. It is underpowered by construction.

### 🔴 The two invalidating defects (diagnosed 2026-07-13, `$0`)

**1. The MDP cannot express the right action.** On 35 held-out HARD episodes:
- the target is never named in hop-1 evidence **62.9%** of the time — the env's hop-1 template
  searches the *whole question*, whose topic terms dominate, so it retrieves the seed's topic
  paragraphs instead of the paragraph that **names** the competitors;
- when the target *is* discovered, `disc0`/`disc1` expose only the **alphabetically first two** of a
  mean **9.7** candidates, so it is addressable **4/13** times.
- **Net: the correct bridge target is reachable by *any* policy in only 11.4% of HARD episodes.**
- And **no label-free ranking beats random** (alphabetical 42.9% top-2, mention-count 40%, chunk-score
  40%, random 40%) — you cannot know *which* competitor discloses topic X without reading their
  filings. The ranking signal does not exist at hop 1, so no policy and no heuristic can recover it.
- Working the ceiling: A1 ~40% + A2 ~11% ⇒ the action space caps HARD coverage at **~0.26**. React
  scored **0.271**. **The scripted baseline was already sitting on the ceiling of the MDP** — there
  was no headroom for any learner, and a null result was guaranteed regardless of RL's merit.

**2. `coverage()` was hackable — this invalidates numbers beyond A6.2.** The metric asked "does *any*
retrieved chunk contain the span?", never "does a chunk **from the right company**". A bridge
question's A2 aspect reads "*that competitor's own* {topic} disclosure", so the topic phrase turning
up in an unrelated company's 10-K is not evidence for it. Measured under a fan-out retrieval:
reward-scored coverage **77.1%** vs truly-evidenced **68.6%** → **8.6% spurious credit**, and **28.6%**
of episodes have some non-target company whose filings also carry the topic phrase. Any policy
rewarded for retrieving *broadly* was partly being paid for nothing. The narrow action space is the
only reason this never showed up. **Affects the A4/A5 multistep-eval coverage figures and the A6.1 +
A6.2 rewards.** Treat every published coverage number as an upper bound until re-run.

### Also established ($0, held-out HARD)
- **The corpus is not the bottleneck.** All 424 gold aspects are present in the ingested chunks of the
  company the aspect is bound to — a perfect retriever scores **1.000** (react scores 0.271).
- **The hop-1 miss is a query-formulation bug, not a retrieval bug.** Searching the seed's filings for
  the *relation* (`"competitors competition compete"`) instead of the whole question finds the naming
  chunk **100%** of the time (vs 48.6%).
- **Reranking makes hop-1 dramatically worse** (8.6% vs 48.6% with the template): the cross-encoder
  chases the topic and buries the competitor paragraph. Do **not** un-prune `hybrid+rerank` for the
  self hop.
- ~~**Fan-out works**: scoping hop 2 to *all* discovered candidates lifts A2 from 11.9% → **68.6%**.
  Ceiling on HARD rises **~0.26 → ~0.84**.~~ ⚠️ **RETRACTED (2026-07-13, same day).** This estimate
  measured whether the evidence was *retrieved into a branch* and never asked whether the chunk
  survived the capped union (`max_evidence=20`). It did not. See "E3/E6 measured results" below —
  the true seated rate is **21.4%**, and `sweep(hybrid)` beats `react(hybrid)` by only +0.014
  coverage while **losing on return**. A retrieval that never reaches the synthesis context did not
  happen.

### E3 / E6 — measured results (held-out fold, n=63; $0, no LLM)

**The hop-2 funnel** (42 bridge episodes with an A2 target):

| stage | pre-E3 | after E3 | after E3+E6 |
|---|---|---|---|
| 1. target IS a discovered candidate (**reachability**) | 11.4% | **78.6%** | 78.6% |
| 2. its branch **retrieved** the evidence | — | 33.3% | **50.0%** |
| 3. that chunk **seated** in the capped union | — | 21.4% | **21.4%** ← now binding |
| 4. A2 scored covered | ~0% | 21.4% | 21.4% |

**E3 did its job** — reachability 11.4% → 78.6%, exactly the defect that invalidated the verdict.
**But each fix reveals the next bottleneck**, and the sweep's headline gain is small:

| policy | HARD | MED | CTRL | ALL | arm cost | **return** |
|---|---|---|---|---|---|---|
| `react(hybrid)` | 0.486 | 0.429 | 0.571 | 0.508 | 0.225 | **+0.497** |
| `sweep(hybrid)` | 0.500 | 0.500 | 0.571 | 0.524 | 0.835 | **+0.482** |

Δ coverage = **+0.016** (6 episodes improved, 53 unchanged, 4 worse). Δ **return = −0.015**: the
sweep's 3.7× arm cost outweighs the coverage it buys. **The scripted sweep does not currently pay
for itself** — which is a genuine result, not a bug, and it is exactly the cost/quality margin a
learned policy would have to win on.

**Why seating is now the constraint.** The sweep seats ~1 chunk per branch (the union cap widens to
`len(union) + n_branches`; with 19 candidates that is already 25 chunks). But the target's
span-bearing chunk is only the *top* hit in its branch 27.3% of the time — rank histogram over the
33 reachable targets: `{rank0: 9, rank1: 1, rank2: 3, rank3: 4, rank4: 4}`. So **12 targets are
retrieved but never seated.**

~~Seating them needs `m ≥ 2` chunks per branch ⇒ cap = `e + m·N`, i.e. 6 + 2×19 = **44 chunks** of
synthesis context in the worst case. That is a real coverage-vs-context Pareto choice, not a free
fix.~~ ⚠️ **RETRACTED by E7 (below) — there was no Pareto choice.** This assumed the only way to seat
a deeper chunk is to seat *more* chunks per branch, i.e. it took the round-robin merge as given. Pool
the branches and rescore them with a **cross-encoder** instead and the target's chunk competes on
merit: `top_branches` beats this rule by **+0.083 coverage on a SMALLER context (13.5 vs 15.0
chunks)**. The depth-`m` ladder was the expensive way to buy what a posteriori branch selection gives
free. **Do not restore the 44-chunk framing.**

**Reranking the hop-2 branches — modest, not a silver bullet** (33 reachable targets, top_k=6):

| arm | evidence at rank 0 | in top-2 | found anywhere |
|---|---|---|---|
| `hybrid` | 27.3% | 30.3% | 63.6% |
| `hybrid+rerank` | **33.3%** | **42.4%** | 60.6% |
| `reranked` | 30.3% | 36.4% | 54.5% |

Rerank moves ~2 more targets to rank 0 and *lowers* recall-anywhere. Note this is the **opposite
sign** to hop 1, where reranking was catastrophic (8.6% vs 48.6%) — the two hops are different
tasks (hop 1 = find the chunk that *names* companies; hop 2 = find a *topic* inside one company),
so the arm that wins is hop-dependent. **That is itself an argument for a learned per-hop policy.**

Even with perfect seating, the retrieval ceiling is 63.6% of reachable ⇒ ~50% of bridge episodes:
the span simply is not in the target's top-6 for the topic query 36.4% of the time.

### E7 — seating: the cap was throwing away evidence the sweep had already found

**The rule was arithmetic, not a heuristic.** `breadth_first` merges branches round-robin by rank and
caps at `len(union) + N`. With a hop-1 union of 6 and N=19 that cap is *exactly* 25, and the
round-robin's first 19 entries are *exactly* the 19 rank-0 chunks. So the production seating rule
was, literally, **"be your branch's rank-0 hit, or you do not exist."** The target's span-bearing
chunk clears that bar only **27.3%** of the time — so **12 of the 21 episodes whose branch had
already retrieved the span were evicted by the cap**, and **18 of the 25 seated chunks were rank-0
noise from non-target companies**.

**The fix.** Pool every branch's chunks and rescore them with a **cross-encoder**. Its scores are
comparable *across* branches; RRF's are not (they are rank-derived, `1/(k+rank)`, so every branch's
rank-1 chunk carries an identical score — which is precisely why `breadth_first` had to exist).
Retrieval is held **byte-identical** in the table below; only the seating rule varies.

| seating rule | reranker | A2 cov | all cov | ctx chunks |
|---|---|---|---|---|
| `breadth_first` (**production**) | — | 21.4% | 0.500 | 15.0 |
| `pooled_rerank@20` | local MiniLM | 40.5% | 0.595 | 18.1 |
| `pooled_rerank@25` | local MiniLM | 42.9% | 0.607 | 21.7 |
| `pooled_rerank@25` | **Voyage rerank-2** | **47.6%** | 0.631 | 21.7 |
| `pooled_rerank@25` (fetch k=12) | **Voyage rerank-2** | **52.4%** | **0.655** | 23.5 |
| `top_branches` (b=3, m=3) | local MiniLM | 38.1% | 0.583 | **13.5** |
| `top_branches` (b=3, m=3) | Voyage | 40.5% | 0.595 | **13.5** |

**Coverage 0.500 → 0.607 (local) / 0.655 (Voyage) — roughly 7–10× the entire scripted sweep's
+0.016.** Seating *efficiency* (seated ÷ retrieved) goes **43% → 96%**.

**`I(Y;E₂) > 0` — the E3 impossibility result does not reach seating.** E3 established `I(Y;E₁) ≈ 0`:
hop-1 evidence carries no signal about *which* candidate discloses the topic, so candidates cannot be
ranked **a priori** and must be swept. That is silent about `I(Y;E₂)`. Once each candidate's filings
have actually been searched *for the topic*, the retrieved content is exactly the observation that
discriminates — the target discloses the topic, most of the other N−1 do not. Ranking candidates
**a posteriori** is legal, and it works: `top_branches` discards 16 of 19 branches and still beats
production by **+0.083 coverage on a SMALLER context (13.5 vs 15.0 chunks)** — a strict Pareto
improvement. **This kills the "seating them needs m≥2 ⇒ 44-chunk context" Pareto choice**: the
depth-`m` ladder was the expensive way to buy what a posteriori selection gives away free.

**The reranker's strength is load-bearing** (which is why the paid arm was run): Voyage beats local
MiniLM by 5–7 points at every setting, and `top_branches` under MiniLM *degrades* when fed a deeper
pool (k=12: 33.3% vs pooled's 45.2%) — a weak reranker picks the wrong branches. A local-only
experiment would have reported 42.9% and understated the fix. **Default stays LOCAL anyway**: the env
is the RL simulator (thousands of rollouts) and must remain `$0` and deterministic; Voyage is the
sim-to-real **arbitrator**, exactly like the LLM `QueryWriter`.

⚠️ **Upper bound, same caveat as E6:** the benchmark interpolates the gold A2 span verbatim as
`{topic}`, so the pooled query the cross-encoder scores against literally contains the gold text. The
*relative* ordering (pooled ≫ breadth-first) is robust — both rules see identical branches retrieved
with an identical query — but the magnitudes need the paid LLM-query-writer run to arbitrate.

### The funnel, after E7 (42 bridge episodes with an A2 target)

| stage | pre-E3 | after E3 | after E3+E6 | **after E7** |
|---|---|---|---|---|
| 1. target IS a discovered candidate (reachability) | 11.4% | 78.6% | 78.6% | 78.6% |
| 2. its branch **retrieved** the evidence | — | 33.3% | 50.0% | 50.0% (54.8% at k=12) ← **the wall** |
| 3. that chunk **seated** in the capped union | — | 21.4% | 21.4% | **42.9% / 52.4%** |
| 4. A2 scored covered | ~0% | 21.4% | 21.4% | **42.9% / 52.4%** |

**Seating is no longer the constraint — stage 2 is, again.** For 12 of the 33 reachable episodes the
span is simply not in the target's top-6 for the topic query, and **fetching deeper barely helps**
(k=6 → k=12 recovers only 2 more: 50.0% → 54.8%, and the two it finds sit at ranks 6 and 7). So the
residual loss is a **query-formulation** problem, not a depth problem and no longer a seating one.
That is the next real lever, and it is exactly what the paid LLM query-writer speaks to.

### Status
E1 (entity-bound coverage), E2 (relation-targeted hop-1 query), E3 (fan-out action + `sweep()`
baseline), E6 (topic-targeted hop-2 query) and **E7 (pooled-rerank seating)** are **landed and
green**. **E4 is closed as obsolete,
not built** — it existed so the policy could *rank* `disc0` vs `disc1`, and the information argument
that motivated E3 (`I(Y; E₁) ≈ 0`) says per-candidate features are uninformative *by construction*;
what the policy needs to *price* a sweep (`n_discovered_unretrieved`, `is_bridging`,
`budget_remaining`) is already in the 18-dim state.

**E5 (retrain + re-evaluate)** — now **CLOSED, REJECT** (2026-07-19; see the "VERDICT" block below).
It was run against **`sweep(hybrid)`**, not `react(hybrid)`: fan-out is trivially scriptable, so handing
the learner fan-out while the baseline stays on `disc0` would manufacture a win out of an action-space
asymmetry — the same error class as the original bug (see `rag_concepts.md` §20.5).

**The open question E5 answers is sharper than before.** The scripted sweep gets +0.016 coverage but
**−0.015 return** — it does not pay for itself. So the learnable margin is explicitly the *cost*
side: sweep only when it will pay (skip CTRL, skip episodes hop 1 already covered, skip candidate
sets too large to be worth `N × cost`), and possibly pick a **different arm per hop** (rerank helps
hop 2, is catastrophic on hop 1). If no policy beats the scripted sweeper on that margin, **RL
genuinely adds nothing here** — a legitimate finding, honestly obtained. No RL verdict is
supportable until that re-run.

## 2026-07-18 — A6.2g sim-to-real gap: the `$0` simulator flatters the policy by 0.18 coverage 🔬

**What was run (paid, $2.6).** The frozen REINFORCE policy (`outputs/experiments/e5/reinforce-s1/`,
E7-seated env, Voyage embeddings) rolled twice over the **same** 42 held-out HARD+MED episodes —
templated `$0` queries vs **real Sonnet-written** queries — changing only the query text (the
`rag rl-simreal` harness; theory in `rag_concepts.md` §20.8). This is the "measure the sim-to-real
gap" commitment in the A6.2 design (rollout realism), not the E5 retrain — it validates the frozen
policy's `$0` numbers against a realistic query-writer.

| metric | `$0` sim (templated) | real (LLM-written) | gap |
|---|---|---|---|
| coverage Φ(s_T) | 0.595 | 0.417 | **−0.179** (95% t-CI [−0.363, +0.006]; p≈0.06) |
| return | +0.544 | +0.389 | **−0.155** |
| same action sequence | — | — | 71.4% |
| synthesis refused (insufficient) | — | — | 33.3% |
| LLM calls (fan-out-aware) | — | 274 billed (473 ceiling) | — |

**The mechanism is the realization channel, not decisions.** 30/42 episodes took the **identical**
action sequence under real queries, yet **13 of those 30 still lost coverage** (mean −0.154). The
policy's "action" is an action-*type* (`hybrid@self`, `hybrid@fanout`, `STOP`); the query *text* is
what differs, and the templated text embeds the gold A2 span verbatim (the §20.7 / E6 upper-bound
caveat). So the −0.18 is a **direct estimate of how much that gold-text leak inflated the templated
coverage** — the caveat, quantified. Real coverage even *beats* sim on 7 episodes (impossible under a
"real union never reaches synthesis" wiring bug — the mixed {0.0, 0.5, 1.0} spread was the live check
that the pipeline was sound before the full spend).

**Why this matters for the promote question.** The RL-vs-`sweep(hybrid)` margin the templated eval was
built to detect is ~0.02–0.05. The sim-to-real bias (0.18) is **4–9× that margin**, so the `$0`
verdict is dominated by simulator bias, *independent of* the 18-group power limit (that limit sets the
CI width; this bias sets what the point estimate even measures — two separate obstructions, both must
clear). Practical consequence: **the templated eval cannot arbitrate promote on its own**; a real-query
eval is mandatory for any A6.2 promote claim, and the honest read of E5 stays "descriptive, not
promote-able" — now for *two* independent reasons.

**Caveats.** Direction is robust (sign split 7:18:17 pos:tie:neg; both strata agree, HARD −0.171 / MED
−0.214) but **not significant at n=42** — per-episode gaps are {−0.5, 0, +0.5}-valued (sd 0.59), so the
t-CI includes zero (one-sample t p≈0.06, Wilcoxon p≈0.08). The −0.18 is a point estimate of the bias,
not a resolved magnitude. Measured under **Voyage** (the higher-ceiling substrate, §20.7), so the sim side is the
0.595 Voyage number, not the 0.607 local one. The optional `sweep`-vs-RL real-query head-to-head
(`rag rl-h2h`, ~$3.2) — which would say whether the RL *advantage* survives real queries, not just
whether the policy degrades — is built and gated but **not yet run** (held).

## 2026-07-18 — A6.0 EXPANSION: grow the graph 20→48 seeds → benchmark 212→680 Q (power gate only) 📈

**What & why.** One-time expansion of the extraction graph and the multi-hop benchmark to attack the
**first** of E5's two promote obstructions — the group-count/CI-width limit (§17.5) — *not* the
sim-to-real bias (that stays untouched; see the 2026-07-18 A6.2g entry above). 28 already-ingested
semis/hyperscaler/software names were added to the graph via an **additions-only** universe file
(`configs/graph_universe_additions.txt`) so the original 20 seeds were never re-billed; then
`make rag-gen-multistep` regenerated over the merged 48-seed universe (`rng_seed=0`, `frac=0.3`
unchanged — only the universe varies).

**Cost.** Graph extraction is the only paid step: **83 LLM calls** (offline pre-count predicted exactly
83), **≈ \$10.80**, ceiling `GRAPH_MAX_EXTRACT_CALLS=100` never touched. Generation is \$0. Graph grew
634→1,832 nodes / 1,352→4,024 edges (+2,370). All 48 seeds have edges; original 20 unchanged.

**Before → after (v1 20-seed → v2 48-seed).**

| Metric | v1 (20 seeds) | v2 (48 seeds) | Δ |
|---|---|---|---|
| Full benchmark Q | 212 | **680** | 3.2× |
| Full HARD / MED / CTRL | 120 / 30 / 62 | 445 / 79 / 156 | — |
| Test Q (HARD / MED / CTRL) | 63 (35/7/21) | 199 (130/24/45) | — |
| **Test distinct HARD∪MED groups** (RL power denominator) | **13** | **38** | **2.9×** |
| Projected bootstrap CI half-width ($k/\sqrt{G}$) | ±0.077 | **±0.045** | 1.7× tighter |
| Supply: `distinct == emitted` (no cap hit) | ✓ | ✓ | — |

The realized ±0.045 **beats the pre-run projection** (~±0.056); the extra hubs (HPE, MSFT, ORCL, IBM,
TER — each 100+ edges) supplied far more bridge pairs than the conservative +10/+18-group estimate.

**Independent span audit (test HARD/MED, n=154, $0).** Target-aspect (bridge-answer) misses **0/154** —
the generator's probe is airtight. Seed-aspect misses **4/154 (2.6%), all one group** (`INTC|META`):
the `INTC competes_with META` edge is real (Intel's 10-K names "…Amazon, Google, Meta and Microsoft…")
but the gold spans are the formal aliases `meta platforms`/`facebook inc`, and Intel writes bare "Meta".
**Documented, not fixed** — the alias policy excludes bare "Meta" on purpose (it false-matches
`metal`/`metadata` across chip filings); the item stays valid (real edge + clean target hop). Known
limitation: 1/38 test HARD∪MED groups has a seed-hop surface-form gap.

**Verdict — this clears ONE of two gates.** The expansion tightens the CI for every *future* E5 run
(±0.077 → ±0.045) but does **nothing** for the sim-to-real bias (~0.18, 4–9× the ~0.02–0.05 effect),
which is orthogonal to $G$ (§17.5 two-obstruction table). So the standing read is unchanged: **E5 stays
descriptive, not promote-able**, now bottlenecked purely on validity (needs a real-query eval), no longer
on power. **All A6.1/A6.2 point estimates were computed on the v1 20-seed benchmark and remain valid in
git history; the v2 set is not backward-compatible with them** — a future E5 must be re-run on v2 to use
the tighter CI. **UPDATE 2026-07-19: E5 was re-run on v2 and REJECTED at `$0` on the *sim return* gate
(0/3 seeds; RL byte-identical to `sweep(hybrid)` on HARD) — see the "VERDICT" block below. That settles
the power axis empirically; the validity axis (real-query eval) was never reached because the sim margin
is ≈ 0, so there is nothing for a paid check to rescue. A6.2 is closed.**

### Frozen-policy inventory + next-session runbook (A6.2 close-out) — verified 2026-07-18

**On-disk state (verified this session; corrects the "E5 is open" framing above).** The E5 retrain was
already run on v1, and the checkpoints are **E7-seated** — `settings.rl_seating_rule="pooled_rerank"`
with provider `"local"` is the *default*, and the env resolves `seating_rule or settings.rl_seating_rule`,
so any run since E7 (2026-07-13) that did not override it trained under pooled-rerank/local:

| artifact | date | fold | `n_actions` | seating | key result |
|---|---|---|---|---|---|
| `outputs/experiments/e5/reinforce-s{0,1,2}/` | 07-17 | v1 (train n=149) | 9 | E7 local | **`rl_eval` REJECT**: `rl-reinforce(greedy)` Δ_return **−0.001** vs `sweep(hybrid)`, CI [−0.0045,+0.0026], 71% same action |
| `e5/reinforce-s1/rl_simreal.json` | 07-18 | v1 test | 9 | E7 Voyage | coverage −0.179 (sim 0.595→real 0.417) — the A6.2g gap above |
| `outputs/experiments/a62g_reinforce_s0/` | 07-13 | v1 | 7 | older | `rl_h2h.json` Δ=+0.0175 vs **`react:hybrid`** — the **SUPERSEDED invalid-env** h2h; not the sweep baseline |
| `e5_smoke*/` | — | — | — | — | smoke runs, ignore |

So on the **sim objective the verdict is effectively settled: RL reproduces `sweep(hybrid)` (Δ≈0)** with
a *tight* CI (this is not the power-limited coverage axis — it is the return-delta the promote gate
actually uses). What is genuinely **not** yet run: (i) a v2 re-confirmation on the 2.9×-larger fold, and
(ii) a real-query h2h **against `sweep:hybrid`** (only the old `react`-baseline/invalid-env h2h exists).

**Runbook.** Prefix: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m stock_agent`. CLI
defaults now auto-point at v2 (regenerated `train/test.json` + 48-seed `graph_universe.txt`); local
E7 reranker is cached, so training is `$0`, no model download.

1. **Re-confirm E5 on v2 (retrain + eval) — `$0`, minutes.**
   `for s in 0 1 2; do … rag rl-train --seed $s --out-dir outputs/experiments/e5_v2/reinforce-s$s; done`
   then `… rag rl-eval -p outputs/experiments/e5_v2/reinforce-s0/policy.json --seeds 0,1,2`.
   Read `promote`/`delta_return`/`delta_ci`. **If REJECT again (expected) ⇒ close A6.2 with no spend:**
   "RL not promote-able for multi-hop retrieval; deliverable = scripted `sweep(hybrid)` + E7 seating."
2. **Only if Step 1 shows a positive, resolvable sim margin — paid validity, ~`$3–6`.**
   `… rag rl-simreal -p <v2 policy> --n 42 --strata HARD,MED --yes`, then
   `… rag rl-h2h -p <v2 policy> --baseline sweep:hybrid --simreal <that rl_simreal.json> --n 42 --strata HARD,MED --yes`.
   ⚠️ `rl-h2h` defaults to the **wrong** `--baseline react:hybrid` — pass **`--baseline sweep:hybrid`**
   (§20.5: else the win is an action-space artifact). Promote only if RL beats `sweep:hybrid` on real
   queries with CI_low>0.

**Honest expectation:** v1 already shows RL ≈ sweep (Δ=−0.001) and the sim inflating coverage by 0.18,
so the most probable outcome is a **legitimate negative close-out at `$0`** (Step 1 confirms; Step 2 is
due-diligence only). Housekeeping alongside: re-baseline A4/A5/A6.1 under entity-bound coverage on v2.

**VERDICT 2026-07-19 — A6.2 CLOSED: E5-on-v2 REJECT (0/3 seeds), legitimate negative at `$0`.**
Runbook Step 1 (retrain) + Step 2 (`rag rl-eval` vs `sweep(hybrid)`, v2 held-out fold, `--seeds 0,1,2`)
both complete; Step 4 (paid real-query h2h) **not reached** (required a positive verdict — none). All
three v2 policies trained bit-identical to v1/seed 0 (`reinforce`/`pruned`/`n_actions=9`/`iterations=200`/
`gamma=1.0`/`lambda_cost=None`), only the fold differs (`n_train` 481 vs 149); final train `mean_return`
s0 0.5275, s1 0.5106, s2 0.5444.

**3-seed held-out gate — `rl-reinforce(greedy)` vs `sweep(hybrid)`** (v2, n_test=199, group-bootstrap CI):

| seed | Δ return | 95% group-boot CI | P(Δ>0) | HARD Δ (n=130) | MED Δ (n=24) | CTRL Δ (n=45) | promote |
|---|---|---|---|---|---|---|---|
| s0 | −0.0014 | [−0.0085, +0.0100] | 0.316 | −0.0009 | −0.0010 | −0.0030 | ❌ |
| s1 | +0.0037 | [−0.0030, +0.0158] | 0.682 | **+0.0000** | −0.0019 | +0.0173 | ❌ |
| s2 | −0.0012 | [−0.0019, −0.0003] | 0.006 | −0.0011 | −0.0017 | −0.0011 | ❌ |

- **0/3 promote; mean Δ ≈ +0.0004 return (≈ 0).** s0/s1 CIs straddle 0; **s2's CI is entirely negative**
  (a small, significant loss). This is now on the **power-cleared v2 fold** (CI half-width ~±0.005 on the
  return delta; the ±0.045 figure is the *coverage* axis) — tighter than v1's Δ=−0.001, and it agrees.
- **On HARD (the multi-hop target, n=130) the learned policy is byte-identical to `sweep(hybrid)`**
  across all seeds (Δ = −0.0009 / +0.0000 / −0.0011): **RL reproduces the sweep** — it recovers the
  sweep's stop policy and adds nothing on-target. s1's positive *point* estimate is an artifact of the
  **off-target CTRL stratum (+0.0173)**; HARD+MED are ≤ 0, so it is not a real edge.
- Sentinel(always-search) is beaten by the learned policy in every seed (return +0.2967 < ~0.421) — the
  policy *did* learn to gate/stop; it just learned the same thing the scripted sweep already encodes.
- Generalization gap ≈ 0 (s2 +0.0006; s0/s1 ≈ −0.003): no train→test transfer of any edge, consistent
  with "no promote-able edge exists."

**Close-out.** A6.2 deliverable for multi-hop retrieval = the **scripted `sweep(hybrid)` controller + E7
pooled-rerank seating**; a *learned* REINFORCE policy is **not promote-able** — it reproduces the sweep
(Δ≈0) on the sim objective, and the sim-to-real bias (~0.18 coverage inflation, §A6.2g) would only
subtract from a real-query claim. No spend incurred; the paid `rl-h2h --baseline sweep:hybrid` step was
gated on a positive sim margin and is not run. Enabling change: a determinism-preserving Voyage-embedder
resilience fix (`rag/embeddings.py`; client `max_retries=6`/`timeout=120s` + app-level retry over the
full transient set incl. `APIConnectionError`, which the SDK's own controller does not retry) so the
multi-hour live-embed retrains survive transient connection-aborts.
