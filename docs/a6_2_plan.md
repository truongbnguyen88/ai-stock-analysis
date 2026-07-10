# A6.2 — Full RL for Retrieval — Detailed Execution Plan

> **Status:** planning → executing (branch `feat/adv-rag-a6.2-rl`). This is the *detailed build plan*
> for [ADVANCED_RAG_TODO.md §A6.2](ADVANCED_RAG_TODO.md) (the plan of record) — the slice-by-slice
> execution notes, concrete module/interface designs, correctness invariants, and tests. Design brief
> (MDP / state / action / reward, worked trajectories) lives in
> [rl_rag_pre_questions.md](rl_rag_pre_questions.md) (Q2–Q5); durable RL theory lands in
> [rag_concepts.md §19](rag_concepts.md) on completion; the per-slice build journal in
> [rag_implementation_notes.md §A6.2](rag_implementation_notes.md).
>
> **Math convention in THIS doc:** ASCII / code-span formulas only (like Q6 of the pre-questions doc),
> to sidestep the GitHub-MathJax escaping hazards; the LaTeX theory is the concepts-doc's job.

---

## 0. What A6.2 is (one paragraph)

A6.2 replaces the **hand-written controllers** — the A4 ReAct loop's fixed LLM prompt
(`research/agentic.py`) and the A5 entity-bridge heuristic (`research/bridge.py`) — with **one learned
sequential policy** over a finite-horizon MDP. The corpus is fixed, so the only thing optimized is the
*sequence of retrieval decisions*: **which** retriever, **where** to point it, **when** to stop. A6.1
(contextual bandit) was the one-shot sub-problem and rejected (REJECT, 2026-07-08 — the tiered/gated
fixed router won); A6.2 is the genuine sequential-RL phase. A rigorous **negative** (a learned policy ≈
the A4 loop + fixed router) is an acceptable, valuable outcome — the env + RL harness are the durable
artifact and the core learning goal.

---

## 1. Reuse vs. new (A6.2 is mostly the MDP wrapper + learner + eval)

| Component | Status | Source of truth |
|---|---|---|
| Action executor (5 arms) | reuse | `rag/read_path.py`: `build_named_system`, `build_graph_system`, `LATTICE_SYSTEMS ∪ {graph}` |
| Reward oracle (`coverage`, `$0`, deterministic) | reuse | `research/multistep_eval.py`: `coverage`, `spans_present` |
| Static query features (label-free, 11-dim) | reuse | `rag/policy_features.py`: `featurize`, `FEATURE_NAMES` |
| Discovered-entity resolution | reuse | `research/bridge.py`: `mentioned_tickers`, `is_bridging` |
| Union dedup / anti-loop `(query,scope)` key | reuse | `research/agentic.py`: `_dedup_union` |
| Group-wise split / OPE / bootstrap CI | reuse | `research/multistep_gen.py` `split_multihop`; `rag/ope.py`; `rag/policy_eval.py` `_paired_delta` |
| Episode catalog (212 Q; strata; group_id) | reuse | `configs/rag_eval_multistep_generated.{train,test}.json` |
| Per-arm cost proxy + `λ_c` | reuse | `rag/reward.py`: `DEFAULT_ARM_COSTS`, `settings.reward_lambda_cost` |
| **Dynamic state featurizer** | **new** | `rag/rl/state.py` (A6.2a) |
| **Action space + query templates** | **new** | `rag/rl/action.py` (A6.2b) |
| **Environment (Gym-style, cached)** | **new** | `rag/rl/env.py` (A6.2c) |
| **REINFORCE (numpy) + BC** | **new** | `rag/rl/policy.py`, `rag/rl/reinforce.py` (A6.2d) |
| **PPO (torch, isolated `[rl]`)** | **new** | `rag/rl/ppo.py` (A6.2e) |
| **Train / eval CLI + verdict** | **new** | `rag/rl/train.py`, `rag/rl/rleval.py`, `rag rl-train` / `rag rl-eval` (A6.2f–g) |

---

## 2. The MDP, bound to code (finite horizon `T = agentic_max_steps = 3`)

- **State `s_t`** — pure numpy, **label-free** (leakage line, Q4.3). Static block = `featurize(query)`
  (11-dim, unchanged) **⊕** dynamic evidence-summary block:
  `[step_idx, budget_remaining=T−t, n_chunks, n_tickers_covered, n_sections_covered,
  n_discovered_unretrieved, last_new_chunks]`. `n_discovered_unretrieved` is the crux feature —
  "NVDA's filing named Micron, but I haven't pulled Micron's filing yet" — computed from
  `mentioned_tickers(union_text, alias_map) − searched`, exactly the bridge's own signal.
- **Action `a_t ∈ {STOP} ∪ {(config c, scope σ)}`**: `c ∈ {dense, reranked, hybrid, hybrid+rerank,
  graph}`; `σ ∈ {self-ticker, discovered-entity_j, none}`. One action jointly answers
  which / where / whether-to-stop. Strictly contains A6.1 (config-only) and A4/A5 (scope+stop).
- **Transition** — **deterministic**. Action → query string via a **template** (pure fn of
  `(σ, question)`, *no LLM*) → **real** `retrieve` against the **real** corpus → fold into the deduped
  union → recompute state. Determinism is what makes unlimited `$0` reproducible rollouts possible.
- **Reward** — potential-based shaping with `Φ(s) = coverage(union, aspects)`:
  ```
  r_t = γ·Φ(s_{t+1}) − Φ(s_t) − λ_c·cost(a_t)
  Σ_t γ^t·(γΦ(s_{t+1})−Φ(s_t)) = γ^T·Φ(s_T) − Φ(s_0)   (telescoping; Ng–Harada policy-invariant)
  ```
  Faithfulness = **hard floor to 0** on any guard failure — *inactive in the `$0` sim* (no synthesis
  runs), *active only in the paid held-out eval*. Sim-side proxy = the reward-hacking sentinel (a noisy
  high-recall arm, taxed by `λ_c`). `Φ` is the **only** place labels (aspects) enter, and it lives
  **inside the simulator** — never in the state (deployability).

---

## 3. Three architecture decisions that shape everything

### (a) Transition cache — the tractability lynchpin (NOT in the original spec; added here)
Each `step` runs a *real* embedder + Chroma + graph-BFS retrieval — cheap in dollars, **not** in
wall-clock, and PPO does thousands of rollouts. Transitions are deterministic and the action space is
finite, so the reachable `retrieve(arm, scope_ticker, question_template)` set **per episode is small and
enumerable**. The env **memoizes** each unique `(episode_id, arm, scope_ticker)` → chunk-list on first
hit; every later rollout/epoch is a table lookup. Turns "`$0` but slow" into "`$0` and fast" — what
actually makes on-policy PPO feasible on this hardware. Cache may be persisted per run for
reproducibility.

### (b) Action-space cardinality — the biggest modeling lever
Full flat space = `STOP + 5 configs × 3 scopes = 16` actions on ~149 train episodes × T=3 is very
sparse. **Decision (default): factored head** — separate config-head × scope-head × stop-bit sharing
the state encoder (shares statistics), with a **pruned flat space** `{hybrid, graph} × {self, disc#1,
disc#2} + STOP = 7` as the tractable default and the full-16 flat space as an ablation. Discovered-entity
slots use a **deterministic, label-free order**: lexicographic over
`sorted(mentioned_tickers(union) − searched)`, cap `J = 2` (= `agentic_bridge_max_entities`).

### (c) torch isolation — reuse the established pattern
pyproject already ships torch under `[sequence]` (the LSTM forecaster) with a mypy override + a
collect-ignore-without-torch pattern. A6.2 adds a clean **`[rl]` extra** (torch, no lightgbm),
**lazy-imported inside `rag/rl/ppo.py` only**, `KMP_DUPLICATE_LIB_OK=TRUE`, guarded by a **day-1 OpenMP
segfault smoke test** (import torch + lightgbm together). CI stays torch-free: numpy REINFORCE is the CI
default; PPO tests skip when torch is absent. **No gymnasium dep** — roll a minimal `reset/step`
interface (lighter, deterministic).

---

## 4. New modules & dependency direction (downward only)

```
rag/rl/__init__.py
rag/rl/state.py     # A6.2a  dynamic evidence-summary featurizer (numpy; extends policy_features)
rag/rl/action.py    # A6.2b  discrete action space + query templates + scope resolution (pure)
rag/rl/env.py       # A6.2c  Gym-style reset/step MDP + TransitionCache (reuses reward + read_path)
rag/rl/policy.py    # A6.2d  LinearSoftmaxPolicy + value baseline (numpy)
rag/rl/reinforce.py # A6.2d  rollout collection + REINFORCE + BC warm-start (numpy)
rag/rl/ppo.py       # A6.2e  torch MLP actor-critic, clipped surrogate, GAE (isolated [rl])
rag/rl/train.py     # A6.2f  training driver + checkpointing → outputs/experiments/<run_id>/
rag/rl/rleval.py    # A6.2g  held-out eval vs 4 baselines + sim-to-real gap
cli: rag rl-train / rag rl-eval
```
`rag/rl/*` depends on `rag/`, `research/`, `graph/`, `schemas/` — never inverted. The agent/CLI stay
thin adapters.

---

## 5. Ordered build slices (each `make check`-green, default-OFF, CI torch-free before the next)

### A6.2a — State featurizer (pure numpy) — **START HERE**
- `rag/rl/state.py`: `EvidenceSummary` dataclass (counts computed from a `RetrievedChunk` union) +
  `STATE_FEATURE_NAMES` (append-only) + `featurize_state(static: ContextVector, union, *, step_idx,
  max_steps, discovered_unretrieved) -> np.ndarray`. Reuses `policy_features.featurize` for the static
  block. **Label-free** (signature accepts NO aspects).
- Evidence summary fields: `n_chunks`, `n_tickers_covered` (distinct `chunk.ticker`),
  `n_sections_covered` (distinct non-None `chunk.section`), `last_new_chunks` (novelty of the last
  action), `n_discovered_unretrieved` (passed in from the env's discovered-set bookkeeping).
- **Tests (CI):** reproduce the Q4 worked `s_0 → s_1` deltas (`n_discovered_unretrieved 0→1`,
  `last_new_chunks 0→6`, `budget 3→2`); dimension pinned; leakage assertion (no aspect param).

### A6.2b — Action space + templates (pure)
- `rag/rl/action.py`: `Action` (STOP | `(config_idx, scope_idx)`); `build_action_space(*, configs,
  n_discovered_slots)` → ordered `list[Action]` (STOP first); `action_to_request(action, question,
  discovered) -> RetrievalRequest | None` where the request carries `(arm_name, query_str,
  scope_ticker)`; template (B) = `"{entity_name} {question}"` for a discovered scope, plain `question`
  for self/none. Invalid scope (discovered slot with no entity present) → returns `None` (masked / no-op).
- **Tests (CI):** enumeration stable + pinned; STOP carries no request; template goldens; out-of-range
  discovered slot masked.

### A6.2c — Environment + TransitionCache (deterministic, `$0`)
- `rag/rl/env.py`: `RagRetrievalEnv(episodes, *, settings, retriever_factory, alias_map, gamma,
  lambda_cost, arm_costs, max_steps, action_space)`; `reset(episode_idx | None) -> np.ndarray`;
  `step(action_idx) -> (state, reward, done, info)`. Holds the union, the issued `(query,scope)` set
  (reuse the A4 anti-loop key), the discovered set, `step_idx`. `retriever_factory(arm) ->
  RetrievalSystem` (default = `build_named_system` / `build_graph_system`, cached per arm + wrapped by
  the `TransitionCache`). Reward via `coverage` (the only labels touchpoint) + `DEFAULT_ARM_COSTS`.
- **`TransitionCache`**: `get(episode_id, arm, scope_ticker, query) -> list[RetrievedChunk]`, memoized.
- **Tests (CI, fake retriever + fake episodes):** reset/step determinism; horizon cap on never-STOP;
  STOP terminates; **telescoping identity** `Σ γ^t·shaping == γ^T·Φ(s_T)`; anti-loop dup handled; the
  full NVDA→TSM worked trajectory reproduced; **reward-hacking sentinel** (noisy high-recall fake arm
  scores low under `λ_c`); cache hit count == distinct requests.

### A6.2d — REINFORCE (numpy) + BC — the always-available CI learner
- `rag/rl/policy.py`: `LinearSoftmaxPolicy(d, n_actions, *, seed)` — logits `Wx`, softmax,
  `act(x)->(a, logp)`, `log_prob(a,x)`, analytic `grad_log_prob`; action masking (STOP always legal;
  masked actions get `−inf` logit). Optional linear value baseline `b(s)=v·s`.
- `rag/rl/reinforce.py`: `rollout(env, policy, *, seed) -> Trajectory`; `reinforce_update(policy,
  trajectories, *, gamma, lr, baseline)`; `behavior_clone(policy, expert_trajectories, *, lr, epochs)`
  (cross-entropy on (state → expert action)). Expert = the A4 loop replayed into the action space
  (self-ticker search = `(prod-arm, self)`; entity-bridge = `(arm, disc_j)`; terminal `STOP`).
- **Tests (CI):** finite-diff gradient check on `grad_log_prob`; **REINFORCE converges on a 2-arm /
  2-step toy** with a known optimum; BC reduces cross-entropy on expert trajectories; seeded determinism;
  masking respected.

### A6.2e — PPO (torch, `[rl]` extra) — the escalation rung
- Add `[rl]` extra to `pyproject.toml` (`torch>=2.2,<3`); mypy override; collect-ignore-without-torch.
- `rag/rl/ppo.py`: MLP actor-critic (torch.nn), `L^CLIP` clipped surrogate, GAE(λ), value head, entropy
  bonus; **lazy `import torch`** inside the module; `KMP_DUPLICATE_LIB_OK` set; **day-1 segfault smoke
  test** (import torch + lightgbm together). Same `Policy`-shaped `act/prob` surface as the numpy policy
  (so the eval harness is backend-agnostic).
- **Tests (local, torch present; skipped in CI):** PPO converges on the same 2-arm/2-step toy; ratio-clip
  correctness. Optional **GRPO** rung (group-relative advantage, no critic) for the tiny-data regime.

### A6.2f — Training CLI + experiment logging
- `rag rl-train --algo {bc,reinforce,ppo} --seed --episodes configs/…train.json --gamma --lambda-cost
  --action-space {pruned,full}`; freezes `{policy weights, STATE_FEATURE_NAMES order, action space,
  standardizer (mean,std)}` → `outputs/experiments/<run_id>/policy.json` (+ config + seed). Mirrors
  `backtesting/` discipline. Feature + action order pinned (reordering silently remaps weights).

### A6.2g — Held-out eval + verdict
- `rag rl-eval`: group-wise `split_multihop` held-out test; learned policy **vs 4 baselines** —
  (i) best fixed pipeline (A6.1f `FixedPolicy`), (ii) A6.1 LinUCB bandit (REJECT — reference),
  (iii) **A4 hand-prompted ReAct loop** (the strong baseline; B/D cells), (iv) random-action policy.
  Metrics: mean return, coverage, cost (steps + LLM calls), **train-vs-held-out gap** (overfit check),
  seed-averaged, reward-hacking sentinel; group-bootstrap CI via `bootstrap_delta_stats`.
- **Paid sim-to-real gap** run (frozen policy with *real* LLM-written queries, A5.3-style ~`$2–3`,
  single seed, ~12–24 held-out HARD/MED Q) reported as a measured number.
- **Verdict** → `validations_results.md`; docs per A-N rule (see §7).

---

## 6. Leakage, determinism & invariants (correctness gates)

- **State = label-free; reward = label-using, simulator-only** (Q4.3). A test asserts `featurize_state`
  never receives aspects; the frozen policy must run on a real query with no gold labels.
- **Group-wise split only** (D2) — `split_multihop`, never row-wise; no bridge pair straddles folds.
- **Grounding unchanged** — the terminal answer (only in the paid eval) still runs the citation +
  `NumberGrounding` guards; the hard floor makes "ungrounded ⇒ reward 0" the optimizer's incentive.
  Numbers stay model-only; no recommendation field; retrieval path stays `$0`/local.
- **Determinism** — no RNG in the env; seeded policies/rollouts; the `TransitionCache` makes rollouts
  reproducible byte-for-byte. Every experiment logs config + seed.

---

## 7. Risks & docs obligations

**Risks (ranked).** (1) **Tiny data** (~149 train episodes) → overfit; mitigated by the A6.0 benchmark
gate, factored action head, BC warm-start, regularization, train-vs-held-out gap reporting, and a
pre-registered rigorous-negative outcome. (2) **Strong baseline** (tuned hybrid/graph + A4 loop is
near-optimal) → beating it is not the success criterion; a measured comparison is. (3) **Sim-to-real
gap** → train `$0`, validate paid, report the gap (decision #3). (4) **Reward hacking** → composite
reward + hard faithfulness floor + sentinel arm. (5) **torch/OpenMP segfault** → isolated `[rl]` +
`KMP_DUPLICATE_LIB_OK` + day-1 smoke test; numpy REINFORCE is the CI floor. (6) **Retrieval wall-clock**
→ the `TransitionCache`.

**Docs obligations on landing (A-N rule).** Mark **A6.2 ✅** in `ADVANCED_RAG_TODO.md §A6.2` with the
verdict; mechanism → `rag_implementation_notes.md §A6.2`; **theory → `rag_concepts.md §19`** (MDP
formalization, policy-gradient theorem, PPO clipped surrogate, GAE, potential-based shaping invariance —
Ng–Harada; renumber References §19→§20), obeying the GitHub-MathJax escaping rules (run
`scripts/check-github-math.py`); chat explains the math per the teaching obligation.

---

## 8. Open decisions (recommendations; confirm at Go / revisit as evidence accrues)

1. **Action space** — factored head + pruned flat default `{hybrid,graph}×{self,disc#1,disc#2}+STOP=7`;
   full-16 flat as an ablation.
2. **Main learner backend** — ship numpy REINFORCE (CI) first as the real deliverable; PPO (torch `[rl]`)
   as the escalation rung; JAX only if torch isolation chafes.
3. **Paid held-out eval scope** — single seed, ~12–24 held-out HARD/MED Q (~`$2–3`), only to quantify
   the sim-to-real gap; alternatively skip and report the `$0`-sim verdict only.
4. **BC expert richness** — the A4 loop demonstrations are thin (it never varies the *config*), so BC is
   a warm-start / sanity baseline, not a strong teacher; acceptable, or start from REINFORCE-from-scratch.

---

## 9. Progress log (update as slices land)

- [x] **A6.2a — state featurizer** ✅ (2026-07-10) — `rag/rl/{__init__,state.py}` +
  `tests/unit/test_rl_state.py` (7 tests). Shipped: `STATE_FEATURE_NAMES` (=`FEATURE_NAMES` ⊕ 7-dim
  `EVIDENCE_FEATURE_NAMES`, **18-dim** total, append-only), `EvidenceSummary`,
  `discovered_unretrieved_entities` (reuses `research.bridge.mentioned_tickers` via lazy import),
  `summarize_evidence`, `featurize_state` (**label-free** — structural test asserts no aspect/label
  param). Tests reproduce the Q4 `s_0 → s_1` deltas (`n_discovered_unretrieved 0→1`, `budget 3→2`,
  `last_new_chunks 0→6`). ruff+mypy clean, 7 pass. Committed on `feat/adv-rag-a6.2-rl`.
- [ ] A6.2b — action space + templates
- [ ] A6.2c — env + TransitionCache
- [ ] A6.2d — REINFORCE (numpy) + BC
- [ ] A6.2e — PPO (torch, `[rl]`)
- [ ] A6.2f — train CLI + checkpointing
- [ ] A6.2g — held-out eval + verdict + docs

### Resume notes (for the next session)
- **On A6.2a:** state = static `featurize()` (11-dim, unchanged) ⊕ dynamic evidence block; the crux
  feature is `n_discovered_unretrieved` (env passes the *count*; env owns the `searched` set + the
  discovered *set* for action slots via `discovered_unretrieved_entities`).
- **Next = A6.2b** (`rag/rl/action.py`): discrete action space (`STOP` + `(config, scope)`), the
  deterministic query template (B) `"{entity} {question}"`, and label-free lexicographic
  discovered-entity ordering (cap `J=2`). Default action space = pruned `{hybrid,graph}×{self,disc#1,
  disc#2}+STOP=7` (§3b); full-16 as an ablation. Then A6.2c wires `state` + `action` into the env.
- **Invariants to keep:** state stays label-free; group-wise `split_multihop` only; `TransitionCache`
  in the env (§3a) so PPO rollouts don't re-hit the corpus; CI torch-free (numpy REINFORCE is the
  floor; PPO in the `[rl]` extra, lazy import, day-1 segfault smoke test).
