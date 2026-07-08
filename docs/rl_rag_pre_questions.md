# Retrieval + RL (A6) — Pre-Build Q&A

> **Purpose.** Design-clarification notes captured *before* building A6 (the retrieval-RL capstone).
> Three questions on (1) the ~100-question benchmark distribution, (2) the RAG-RL MDP, and (3) the
> templated training data. Companion to the plan of record [ADVANCED_RAG_TODO.md §A6](ADVANCED_RAG_TODO.md);
> durable theory lands later in [rag_concepts.md](rag_concepts.md), the build journal in
> [rag_implementation_notes.md](rag_implementation_notes.md). Grounded in the live code:
> `research/multistep_eval.py` (the `Aspect`/coverage oracle), `configs/rag_eval_multistep.json` (the
> current 12-Q set), `configs/graph_universe.txt` + `data/graph/*.db` (the A5 graph, 634 nodes / 1,352
> edges), `rag/read_path.py` (`LATTICE_SYSTEMS` + `build_graph_system` — the A6 action space).

---

## Q1 — Distribution of the ~100 corpus-verified questions

**Yes — explicitly stratified, and the stratification is load-bearing, not cosmetic: it is what makes
the bandit/RL problem well-posed.** Three orthogonal axes.

### Axis 1 — difficulty stratum (the A5.3 HARD/MED/CTRL, kept and grown)

| Stratum | What it is | Hops | Why it must be in the set | Target share |
|---|---|---|---|---|
| **CTRL** | single-entity, single-topic (e.g. *"What does Micron disclose about DRAM/NAND?"*) | 1 | The **control**: proves graph/agentic don't *regress* easy Qs, and supplies the cheap-arm side of the cost/quality tradeoff | ~20–25% |
| **MED** | one bridge, but the bridged fact is *co-disclosed* (the seed's own filing partly covers it) | ~1–2 | Ambiguous cases where single-shot sometimes suffices — the policy must learn *when* the extra hop pays | ~25–30% |
| **HARD** | genuine 2-hop bridge, answer span **present-in-target / absent-in-seed** | 2 (some 3) | The discriminating questions; the only ones where graph traversal earns its keep | ~45–50% |

### Axis 2 — relation type (mined from the A5 graph edges)

Coverage matches what the graph can actually answer:
- `depends_on` — memory (NVDA→MU) and foundry (NVDA→TSM) bridges — the bulk.
- `competes_with` — NVDA→AMD / INTC.
- `exposed_to` — company→`regulatory_topic` (export controls, CAC).
- `mentions_risk` — company→`risk` (earthquake, ASP decline, capex intensity).
- a handful of **2-hop** (NVDA→TSM→ASML) to give the horizon-3 policy real depth to exploit.

### Axis 3 — question type (the bandit context feature)

`question-type ∈ {risk, financial, business, overview, bridging}` — the categorical feature the
featurizer emits, so the benchmark must populate each value.

### Why the mix is load-bearing for A6 (the part the plan assumes but doesn't spell out)

- **An all-HARD benchmark has no decision to learn.** The optimal policy degenerates to "always
  graph+agentic," and that learned policy would then *over-spend* on easy production queries. The CTRL
  stratum is what creates a non-trivial **contextual** optimum (cheap arm here, expensive arm there).
  Adaptive retrieval only beats a fixed pipeline when some queries genuinely want a different arm.
- **PPO/LinUCB need state-distribution coverage.** The context vector $x$ (A6.1) and MDP state $s_t$
  (A6.2) must be populated across the whole spectrum or the policy won't generalize off the training
  queries. This is the literal reason **A6.0 (grow to ~100) is a gate for A6.2** — 12 points cannot
  cover a contextual policy's input space, and the honest result on 12 would be "insufficient data."

### Recommendation

A6.0 should add an explicit `stratum` (and `relation` / `type`) field to `MultiHopQuery` rather than
keep the fragile **index-range** convention A5.3 used (HARD = rows 0–5, MED 6–8, CTRL 9–11). RL needs
**stratified train/test splits** and **per-stratum reward attribution**, both of which want the label
on the row, not implied by position.

---

## Q2 — The MDP for RAG-RL, in detail

**Frame.** The A4 ReAct loop is *currently* driven by a fixed LLM prompt — the controller is
hand-written. A6.2 replaces that hand-written controller with a **learned policy** over a formal
finite-horizon MDP. The corpus is fixed, so the only thing being optimized is the *sequence of
retrieval decisions*.

### The five components

**Horizon** $T =$ `agentic_max_steps` (default 3 search steps + 1 terminal synthesis).

**State $s_t$** — a pure numpy function of the trajectory so far (the Markov sufficient statistic):
- query features (length, has-ticker, entity count, question-type, `is_bridging`) — *static across the episode*;
- step index $t$ and budget remaining ($T - t$);
- **evidence summary** (the dynamic part): union chunk count, distinct tickers covered, distinct
  sections covered, **marginal coverage of the last action**, and — critically — **the set of entities
  named in the union but not yet retrieved against**.

That last feature is the whole game: it is the encoded form of *"NVDA's filing just told me Micron
matters, but I haven't pulled Micron's filing yet."*

**Action $a_t \in \{\text{STOP}\} \cup \{(\text{config } c,\ \text{scope } \sigma)\}$:**
- $c \in \{\text{dense, reranked, hybrid, hybrid+rerank, graph}\}$ — *which retriever* (the A6.1 action space, = `LATTICE_SYSTEMS` ∪ {graph}).
- $\sigma \in \{\text{self-ticker},\ \text{discovered-entity}_j,\ \text{none}\}$ — *where to point it* (the bridge decision; in A4 this is the brittle alias-bridge heuristic, here it is **learned**).
- `STOP` — *when to stop* (in A4 the LLM's "reflective stop," here **learned**).

One action jointly answers **which / where / whether-to-stop**. This strictly contains both A6.1
(config only) and A4/A5 (scope + stop).

**Transition $P(s_{t+1}\mid s_t, a_t)$ — deterministic.** Given the action, the simulator builds a
query string from a template (Q3), runs the *real* retriever against the *real* corpus, folds the
returned chunks into the union, and recomputes the evidence summary. No stochasticity because (a) the
corpus is fixed and (b) the query is templated, not LLM-sampled. **This determinism is what makes
unlimited, $0, reproducible rollouts possible** — the key enabler for on-policy PPO.

**Reward.** Terminal

$$R = \text{coverage(final union)} - \lambda_c\cdot(\text{steps}+\text{LLM calls}) - \lambda_f\cdot(\text{guard failures})$$

with faithfulness (citation/number guard) as a **hard constraint** — any guard-failing trajectory is
floored to reward 0. Optional **potential-based shaping** $r_t = \gamma\Phi(s_{t+1}) - \Phi(s_t)$
with $\Phi(s) =$ current union coverage. Shaping telescopes to the same return (Ng–Harada → optimal
policy unchanged) but **densifies credit**: the agent is rewarded the step it adds coverage, not only
at the end — essential when the horizon is short and the terminal signal is a single scalar.

### Worked example — a full trajectory

**Question** (real config row): *"Among the memory suppliers NVIDIA names, which one discloses Chinese
cybersecurity-review restrictions in its own filings?"*
**Aspects** ($K=2$): **A1** = NVDA names supplier, spans `["Micron","SK Hynix"]`; **A2** = supplier's
own CAC restriction, span `["critical information infrastructure"]`.

| $t$ | State $s_t$ (salient parts) | Action $a_t$ | Transition result | $\Phi$ after | shaping $r_t$ |
|---|---|---|---|---|---|
| 0 | step 0, budget 3, union ∅, discovered={}, is_bridging=1, type=risk | `(hybrid, self-ticker=NVDA)` | retrieves NVDA chunk naming "Micron" → **A1 covered**; **MU now discovered** (named, not yet retrieved) | 0.5 | $0.5\gamma$ |
| 1 | step 1, budget 2, union={NVDA chunk}, **discovered={MU}**, last-marginal=+0.5 | `(graph, scope=discovered-entity MU)` ← *the learned bridge* | graph traverses NVDA→depends_on→MU, pulls MU's chunk with "critical information infrastructure" → **A2 covered** | 1.0 | $0.5\gamma$ |
| 2 | step 2, budget 1, union={NVDA,MU}, coverage=1.0, no undiscovered entities | `STOP` ← *the learned reflective stop* | terminal synthesis (guards run) | 1.0 | — |

Terminal $R = 1.0 - \lambda_c\cdot(2\ \text{steps} + \text{LLM calls}) - \lambda_f\cdot 0$.

**Why this is a genuine MDP and not a bandit.** The *correct* action at $t=1$ (bridge to MU) is only
knowable *because* $t=0$ discovered MU. The decision is conditioned on accumulated state. A6.1's bandit
picks one config from $x$ and is blind to "what have I gathered so far" — it structurally cannot
represent this. That gap is exactly what A6.2 exists to close.

**The reward landscape the policy learns from (by contrast):**
- STOP at $t=1$ → coverage 0.5 (misses A2) → low return. *Learns: don't stop while a named-but-unretrieved entity remains.*
- re-query NVDA self-ticker at $t=1$ → coverage still 0.5 but pays cost → strictly worse. *Learns: self-ticker re-retrieval has zero marginal coverage once the seed is exhausted.*
- the trajectory above → coverage 1.0 at minimal cost → the optimum.

---

## Q3 — The templated training data (and how it differs from the LLM loop)

"Templated" appears in **two distinct places** in A6; both must be understood because "templated data
for PPO" sits on top of both.

### (A) Template-generated *benchmark questions* (the A6.0 dataset)

How the ~100 labeled questions are produced cheaply and verifiably instead of hand-written:

1. **Enumerate graph edges** $(\text{seed}, \text{relation}, \text{target})$ from the A5 graph (already
   1,352 edges): e.g. `(NVDA, depends_on, TSM)`, `(NVDA, depends_on, MU)`, `(NVDA, competes_with, AMD)`.
2. **Enumerate candidate topics** = risk spans the *target* discloses (mined from the target's ingested
   chunks / its `mentions_risk` edges): for TSM → {`earthquake`, `political stability`, Hsinchu/Science
   Park site concentration, power/water constraints}.
3. **Slot into a relation-specific question template**, e.g. the foundry `depends_on` template:
   > `"Which company NVIDIA relies on to fabricate its chips warns about {topic} in its own SEC filings?"`

   Filling `{topic}=regional political stability` and `{topic}=earthquake risk` **reproduces the actual
   config rows 3 and 7** — confirming the existing 12-Q are already template-shaped.
4. **Auto-generate the aspect labels:** A1 spans = the edge's target surface names
   (`["Taiwan Semiconductor","TSMC"]`, taken straight from the graph); A2 spans = `[topic phrase]`.
5. **Auto-verify with the A5.3 span-isolation probe** (a corpus grep): keep the question **only if** the
   topic span is *present in the target's chunks* AND *absent from the seed's chunks*. If NVDA's own
   filing already contains the topic, the bridge is unnecessary and the question is discarded (it
   wouldn't test multi-hop). This is what makes the labels trustworthy without human review.
6. **Stratify:** no-bridge templates (single entity, single topic) → CTRL; 2-hop templates → HARD;
   co-disclosed → MED.

Net: $\text{edges} \times \text{topics} \times \text{templates}$, filtered by the probe, yields ~100
verified questions for ~$0 (grep + the already-built graph).

### (B) Template-generated *rollout queries* (the simulator transition — what PPO trains on)

In the **real** A4 loop the next query string is *generated by an LLM* ("decision call") — the
expensive, non-deterministic, non-reproducible step. For RL training we **replace the LLM-written query
with a template that is a pure function of $(\text{scope } \sigma, \text{config } c)$:**

```
action (σ=discovered-entity MU, c=graph)
   → template:  "{entity_name} {question_topic}"  →  "Micron critical information infrastructure"
   → run the REAL retriever (graph) against the REAL corpus, scope=MU
```

The corpus and retriever are real; **only the query string is templated**. Because the template is
deterministic, the transition $s_t \to s_{t+1}$ is deterministic → the simulator (`rag/rl/env.py`)
yields **unlimited, $0, reproducible rollouts**, which is precisely what on-policy PPO needs (it samples
many trajectories per update).

So **"PPO + templated data"** means: PPO rolls out trajectories in the simulator; at each step the
chosen action is converted to a query **via template (B)**; retrieval runs for real; reward comes from
the oracle (`coverage` over the **(A)**-generated aspect labels). **No LLM anywhere in the training loop.**

### The deliberately-accepted cost — the sim-to-real gap

Template (B) is a *proxy* for the LLM's real query-writing ability. A templated
`"Micron critical information infrastructure"` may retrieve slightly differently than the phrasing
Claude would emit live. A6 handles this honestly (decision #3): **train on templated $0 rollouts, then
validate the frozen policy held-out with a small real LLM-in-the-loop eval (A5.3-style, ~$2–3)**, and
**report the sim-to-real gap as a measured number** rather than pretending it is zero. That validation
run is the only paid part of A6.2.

### Tie-together

(A) supplies the reward labels, (B) supplies the transition function, the simulator composes them into
deterministic rollouts, PPO learns a **which/where/stop** policy over the MDP of Q2, and the held-out
LLM eval measures how much the template proxy cost you.

---

## Q4 — A concrete, literal walkthrough (edge → benchmark row → state transition)

The two follow-up confusions worth pinning down: (i) exactly how a benchmark row is *manufactured* from
a graph edge (Part A, Step 0→4), and (ii) what an *action* actually transitions between — because "go
from one data point to another with an action" mixes two different objects.

### A4.1 — Part A, literal: from one graph edge to one benchmark row

This is offline dataset construction — a pure function `(graph edge, topic, template) → labeled JSON
row`, kept only if a grep probe passes. **No LLM.**

**Step 0 — raw materials already on disk.**

```python
# (i) a graph edge, pulled from data/graph/voyage-voyage-4.db (the A5 graph, 1,352 edges)
edge = {
    "subject":  "NVDA",
    "relation": "depends_on",
    "object":   "TSM",
    "object_surface": ["Taiwan Semiconductor", "TSMC"],   # how NVDA's filing names it
    "provenance": ["NVDA_10K_2024::item1A::chunk_037"],   # the chunk the edge came from
}

# (ii) topics the TARGET (TSM) discloses — mined from TSM's own chunks / its mentions_risk edges
topics_TSM = [
    {"label": "geopolitical-stability", "spans": ["political stability"]},
    {"label": "earthquake",             "spans": ["earthquake"]},
    {"label": "site-concentration",     "spans": ["Science Park", "Hsinchu"]},
]

# (iii) a relation-specific question template (written once per relation family)
TEMPLATES["depends_on/foundry"] = (
    "Which company {seed_name} relies on to fabricate its chips "
    "warns about {topic_phrase} in its own SEC filings?"
)
```

**Step 1 — fill the template (pure substitution).** Take `edge` × `topics_TSM[0]`:

```python
question = TEMPLATES["depends_on/foundry"].format(
    seed_name="NVIDIA", topic_phrase="regional political stability")
# → "Which company NVIDIA relies on to fabricate its chips warns about
#    regional political stability in its own SEC filings?"
```

This is **literally row 3 of the committed `configs/rag_eval_multistep.json`** — confirming the existing
hand-written 12-Q are already in template shape.

**Step 2 — auto-generate the label (aspects) straight from the edge + topic.** Nobody writes the spans
by hand: aspect A1's spans come from the edge's `object_surface`, A2's from the topic.

```python
row = {
  "question": question,
  "aspects": [
    {"name": "NVDA names the foundry",                       "spans": edge["object_surface"]},  # ["Taiwan Semiconductor","TSMC"]
    {"name": "that foundry's own geopolitical-stability risk","spans": topics_TSM[0]["spans"]},  # ["political stability"]
  ],
  "stratum": "HARD", "relation": "depends_on",   # metadata A6.0 should add (vs A5.3's index ranges)
}
```

**Step 3 — the span-isolation probe (a grep) decides keep vs discard.** The bridged fact must be
**present in the target's chunks AND absent from the seed's chunks** — else the question doesn't require
the hop and is thrown out.

```python
def keep(row, seed="NVDA", target="TSM"):
    a2 = row["aspects"][1]["spans"]              # ["political stability"]
    return grep(a2, corpus[target]) and not grep(a2, corpus[seed])
```

- `"political stability"` → present in TSM's 20-F, absent from NVDA's 10-K → **keep** (genuine bridge). ✓
- topic `"export control"` → also present in NVDA's *own* filing → `keep == False` → **discard** (single-shot already covers it). ✗

**Step 4 — sweep the product.** Loop **(all edges) × (each target's topics) × (matching templates)**, run
`keep`, stratify the survivors:

| edge | template family | topic filled | probe | emitted? | stratum |
|---|---|---|---|---|---|
| NVDA→depends_on→TSM | foundry | political stability | ✓ | yes | HARD |
| NVDA→depends_on→TSM | foundry | earthquake | ✓ | yes | HARD |
| NVDA→depends_on→MU | memory | NAND oversupply | ✓ | yes | HARD |
| NVDA→competes_with→AMD | competitor | 7nm TSMC dependency | ✓ | yes | HARD |
| NVDA→depends_on→TSM | foundry | export control | ✗ (in NVDA too) | **discarded** | — |
| MU (no bridge) | single-entity | DRAM/NAND lines | n/a | yes | CTRL |

≈ 3–4 templates × ~16 graph tickers × a few verified topics each, minus probe rejections ≈ **~100 rows**,
for the cost of grep + the already-built graph (**~$0**). *This product of rows IS the "templated
benchmark data"* — `configs/rag_eval_multistep.json` grown from 12 to ~100.

### A4.2 — what an *action* actually transitions (the three "data points")

A benchmark row is a **fixed task (one episode)**. An action does **not** move you to another benchmark
row — it transitions the **state** (accumulated evidence) *within* the same episode. Three distinct
"data points" live at three levels:

| Level | The "data point" | How many | Does an action move between them? |
|---|---|---|---|
| **Episode / task** | the benchmark row `{question, aspects}` | ~100, **fixed** | **No.** Constant for the whole episode; you reach the next row only at `env.reset()`. |
| **MDP state** $s_t$ | a numeric evidence-so-far vector | many per episode | **Yes — this is what an action transitions.** $s_t \xrightarrow{a_t} s_{t+1}$ |
| **PPO training datum** | a transition tuple $(s_t, a_t, r_t, s_{t+1})$ | harvested from rollouts | it *is* the recorded action-transition |

**The episode (fixed):** the pasted row, e.g. the NVDA→TSM political-stability question with its two
aspects A1 (`["Taiwan Semiconductor","TSMC"]`) and A2 (`["political stability"]`). `env.reset(episode)`
builds the initial state $s_0$; the question is constant until the next `reset()`.

**The state vector $s_0$** (raw counts; static block from the question, dynamic block = evidence summary):

```
i  feature                     s_0
0  has_ticker                  1
1  n_query_entities            1     ← "NVIDIA"
2  is_bridging                 1
3  type=bridging               1
── evidence summary ───────────────
4  step_idx (t)                0
5  budget_remaining (T−t)      3
6  n_chunks (union size)       0
7  n_tickers_covered           0
8  n_sections_covered          0
9  n_discovered_unretrieved    0     ← entities NAMED in union but not yet retrieved
10 last_new_chunks (novelty)   0
```

**One action, the transition $s_0 \to s_1$.** Action space = enumerated `(config, scope)` + STOP:

```
a=0 STOP   a=1 (hybrid, self-ticker)   a=2 (graph, self-ticker)
a=3 (graph, discovered-entity#1)   a=4 (hybrid, discovered-entity#1) ...
```

Policy picks $a_0 = 1$ = `(hybrid, self-ticker=NVDA)`. Inside `env.step(1)`:

```python
query  = QUERY_TEMPLATES["self-ticker"].format(question_text)   # action → literal string, NO LLM
chunks = build_named_system("hybrid").retrieve(query, where={"ticker": "NVDA"})  # REAL retrieval
union  = dedup(union + chunks)        # NVDA chunk that names "Taiwan Semiconductor"
```

Re-featurize → $s_1$ (TSM is now named-but-unretrieved):

```
i  feature                     s_0 → s_1     Δ
4  step_idx                    0 → 1        +1
5  budget_remaining            3 → 2        −1
6  n_chunks                    0 → 6        +6
7  n_tickers_covered           0 → 1        +1   (NVDA)
8  n_sections_covered          0 → 1        +1   (Item 1A)
9  n_discovered_unretrieved    0 → 1        +1   ← TSM! the signal the policy must learn to act on
10 last_new_chunks             0 → 6        +6
(static features 0–3 unchanged)
```

**That** is "going from one data point to another with an action" — the data point is the *state vector*,
not a new benchmark question. Same question, more evidence.

**The reward on this transition** (potential shaping, $\gamma = 0.9$):

```python
Φ(s_0) = coverage(union_0, aspects) = 0/2 = 0.0    # uses the LABELS (aspects)
Φ(s_1) = coverage(union_1, aspects) = 1/2 = 0.5    # A1 covered (chunk contains "Taiwan Semiconductor")
r_0    = γ·Φ(s_1) − Φ(s_0) = 0.9·0.5 − 0 = 0.45
```

**The PPO training datum** (the literal "one data point to another with an action"):

```python
(s_0, a_0=1, logπ(a_0|s_0), r_0=0.45, s_1, done=False)
```

The episode continues — $a_1 = 3$ `(graph, discovered-entity#1=TSM)` → template emits
`"Taiwan Semiconductor political stability"` → graph pulls TSM's chunk → $s_2$ (`n_discovered_unretrieved
= 0`, coverage 2/2) → $a_2 =$ STOP. **One benchmark row yields three transition tuples:**

```
(s_0, a_0, r_0, s_1)   (s_1, a_1, r_1, s_2)   (s_2, STOP, r_term, terminal)
```

Then `env.reset()` loads the **next** of the ~100 rows — the only way you "change benchmark data points,"
and it is dataset iteration, not a policy action.

### A4.3 — two corrections worth nailing down

1. **Actions never generate or move between benchmark rows.** The ~100 rows are a fixed dataset (Part A).
   Within an episode the question is constant; the action transitions the **state** (accumulated
   evidence). Rows change only at episode boundaries via `reset()` — dataset iteration, not an action.
2. **Leakage line (operationally critical).** The **state** $s_t$ must be computable from evidence alone
   (`n_chunks`, discovered entities, novelty) with **no aspect labels**, because at deployment a real
   user query has no gold labels. The **reward** $\Phi = \text{coverage(union, aspects)}$ *does* use the
   labels — and that is fine **only because it lives inside the simulator** (training/eval). If any
   label-derived quantity leaked into the state, the frozen policy could not run on real queries.
   **State = label-free; reward = label-using, simulator-only.**

---

## Q5 — The reward function, term by term

Two equivalent views: the **episode return** (what we ultimately maximize) and the **dense per-step
reward** (what PPO/REINFORCE actually consume). Same objective, redistributed in time.

### Episode return (the objective)

$$
G(\tau)=
\begin{cases}
0 & \text{if any guard fails (faithfulness — hard constraint)}\\[4pt]
\underbrace{\text{quality(final union)}}_{\text{coverage or nDCG@}k}-\lambda_c\sum_t \text{cost}(a_t) & \text{otherwise}
\end{cases}
$$

### Dense per-step form (potential-based shaping, $\Phi=$ union coverage)

$$
r_t=\underbrace{\gamma\Phi(s_{t+1})-\Phi(s_t)}_{\text{coverage shaping}}-\lambda_c\text{cost}(a_t),\qquad \Phi(s_0)=0
$$

The two coincide because the shaping **telescopes**:
$\sum_t \gamma^t\big(\gamma\Phi(s_{t+1})-\Phi(s_t)\big)=\gamma^T\Phi(s_T)-\Phi(s_0)$ — the discounted
terminal coverage, spread across the steps that earned it. (Symbols: $\tau$ the trajectory; $\Phi(s)$
the union's aspect coverage at state $s$; $a_t$ the action; $\text{cost}(a_t)$ its steps + LLM calls;
$\gamma$ the discount; $\lambda_c$ the cost price; $k$ the cutoff for nDCG.)

### Term 1 — quality (coverage / nDCG@k): the **driver**

- **What.** `coverage` = fraction of the question's $K$ aspects whose span appears in the accumulated
  union (multi-hop); `nDCG@k` = graded ranking quality (single-shot). "Active where its labels exist"
  (decision #4): coverage for `rag_eval_multistep` rows, nDCG for `rag_eval_queries` rows.
- **Effect.** The *only positive* term — pulls the policy toward gathering relevant evidence (and, via
  the `n_discovered_unretrieved` feature, toward bridging).
- **Alone** (`λ_c=λ_f=0`): coverage is **monotone non-decreasing** in retrieval, so the optimum is
  "always retrieve to the budget, never stop" — no stop incentive, and a noisy high-recall arm looks
  free. That degeneracy is exactly why Terms 2–3 exist.

### Term 2 — cost penalty $-\lambda_c\text{cost}(a_t)$: the **stop / efficiency pressure**

- **What.** `cost = steps + LLM calls`. In the $0 simulator the per-step LLM cost is ~0 (templated
  queries) so cost ≈ step count + the one terminal synthesis; in the **real held-out eval** the decision
  calls also count.
- **Effect.** The **only reason to stop.** It turns the problem into a margin test — an action is worth
  taking iff its *marginal* coverage exceeds its marginal cost:

$$
r_t\approx \underbrace{\Delta\text{coverage}(a_t)}_{\text{newly-covered aspects}}-\lambda_c\text{cost}(a_t)\quad(\gamma\to1)\Rightarrow \text{STOP when expected }\Delta\text{coverage}<\lambda_c .
$$

- **Sensitivity.** `λ_c` too **large** → under-retrieval (stops before bridging, misses the A2-type
  aspect); too **small** → over-retrieval (burns the whole budget on every query, including CTRL).
  It sets the coverage/cost operating point and must be sensitivity-tested.

### Term 3 — faithfulness: the **hard constraint** (anti-reward-hacking)

- **What (resolved, decision #4).** Any trajectory whose terminal answer trips the citation or number
  guard has its **entire return floored to 0** — strictly stronger than the soft
  $-\lambda_f\cdot(\text{guard failures})$ penalty the A6.1 formula first sketched.
- **Effect.** Closes the reward-hacking loophole *by construction*: a policy cannot "cover" aspects with
  ungrounded text that merely name-drops the spans, because flooring to 0 makes any ungrounded
  trajectory worthless regardless of coverage. The guard already refuses such answers — this makes the
  incentive explicit. The **reward-hacking sentinel** (a deliberately noisy high-recall arm) is the
  test: it must score low, taxed by `λ_c` on its chunk dumping and capped by the guard on grounding.

### Term 4 — shaping + discount γ: **credit densification, policy-invariant**

- **What.** Potential-based shaping with $\Phi=$ coverage (Ng–Harada). It does **not change the optimal
  policy** (the telescoping identity gives the same return up to the constant $\Phi(s_0)=0$), but turns
  one terminal scalar into a per-step signal: each hop is credited the coverage *it* added.
- **Effect.** Essential for short-horizon credit assignment — without it REINFORCE/PPO must
  back-propagate a single end-of-episode reward through all $T$ steps (high variance). $\gamma<1$ mildly
  **front-loads** coverage (cover an aspect at $t{=}1$ over $t{=}2$) and bounds horizon value; for
  $T=3$ it sits ~0.95–1.0.

### The marginal-coverage reading (why the policy stops)

With $\gamma\to1$, $r_t \approx \Delta\text{coverage}(a_t)-\lambda_c\text{cost}(a_t)$. So the learned
behavior is: **keep retrieving while a hop's expected new-aspect coverage exceeds $\lambda_c$; otherwise
STOP.** On the worked NVDA→TSM episode: $a_0$ (self-ticker) buys +0.5 coverage ≫ cost → take it; $a_1$
(bridge to TSM) buys the final +0.5 → take it; at $s_2$ every aspect is covered so the next hop's
$\Delta\text{coverage}=0<\lambda_c$ → STOP is optimal.

### Hyperparameter cheat-sheet (effect of *increasing* each)

| Knob | ↑ does what | Risk if too high | Risk if too low |
|---|---|---|---|
| `w_q` (quality weight) | rewards evidence harder | over-retrieval / reward-hacking pressure | policy ignores coverage |
| `λ_c` (cost) | stops earlier, prefers cheaper arms | under-retrieval, misses bridges | budget blow-up on easy Qs |
| `λ_f` / hard floor | punishes ungrounded answers | (hard floor: none — it is the safety) | grounding becomes gameable |
| `γ` (discount) | values later coverage nearly as much | (≈1 fine for short $T$) | myopic, won't plan the 2nd hop |

**Invariant tie-in.** Term 3 is the RL-side restatement of the project's non-negotiable grounding
invariant — the citation + `NumberGrounding` guards already run on every synthesis call; the hard floor
just makes "ungrounded ⇒ no reward" the policy's incentive too, so the optimizer can never *learn* to
bypass them.

---

## Q6 — Contextual bandit (A6.1): end-to-end "training", serving, and challenges

> **Post-build addendum (2026-07-05).** Q1–Q5 above were captured *before* building the A6 RL
> capstone and largely describe the **A6.2** full-MDP/PPO design. This Q6 is scoped to the
> **A6.1 contextual bandit** that is now actually built and green (PR #60): how its offline "training"
> works with a concrete example, how it would be served in the chat agent, and the operational
> challenges. Math is kept in code spans / plain ASCII (no `$…$`) to match the Q4 walkthrough and avoid
> MathJax escaping hazards. Durable theory: [rag_concepts.md](rag_concepts.md) §18; build log:
> [rag_implementation_notes.md](rag_implementation_notes.md) §A6.1.

**Question.** *Can you give me a full picture of how the "training" for contextual bandit RL Retrieval
works? A concrete, end-to-end illustrating example would be very helpful. Then after training, if we
integrate contextual bandit RL into use in the chat agent UI, how would it work? Any challenges that I
should be aware of?*

### Part 1 — What "training" means here (and what it doesn't)

There is **no gradient descent, no online interaction, no episodes**. "Training" is a single **offline
batch pass** that solves a supervised-style least-squares problem (ridge normal equations) over a logged
dataset, then evaluates the result off-policy. The whole thing is `$0` and deterministic because the
reward oracle is retrieval-only (no LLM).

The bandit framing — each query is one independent round:
- **Context** `x` = an 11-dim label-free feature vector of the query.
- **Action** `a` = one of **5 retrieval arms** `(dense, reranked, hybrid, hybrid+rerank, graph)`.
- **Reward** `r(x,a)` = `single-shot coverage/nDCG(a on x) − λ_c·cost(a)`.
- **Goal** = learn `π(a|x)` maximizing `V(π) = E_x E_{a∼π}[r(x,a)]`, then *prove* it beats the fixed
  default before shipping.

Core difficulty OPE solves: at serve time we have **no labels**, so we can't just `argmax` the reward
row — we need a policy that generalizes from `x`. And we can only *log* one arm per query, so we must
estimate what a *different* policy would have scored (off-policy).

> **Bandit (A6.1) vs MDP (A6.2, Q2 above).** A6.1 picks one config from the query features and is blind
> to "what have I gathered so far" — it is a **one-shot** decision. The MDP of Q2 conditions each action
> on accumulated evidence state `s_t`. A6.1 is the strict sub-problem (config-only, single step).

### Part 2 — End-to-end, with a concrete example

Two real-shaped queries from the 212-Q benchmark:

| | q1 (HARD, bridging) | q2 (CTRL, simple) |
|---|---|---|
| text | "How does NVDA's data-center growth depend on its memory suppliers?" | "What was NVDA's revenue?" |
| seed | NVDA | NVDA |

**Step 0 — arms & costs (fixed).** `ARMS = (dense, reranked, hybrid, hybrid+rerank, graph)`, indices
0–4. Costs in that order `c = [0.0, 0.3, 0.1, 0.4, 0.3]`, `λ_c = 0.05`. This ordering is the
**load-bearing invariant**: reward-matrix columns, OPE `target_probs` columns, and policy action indices
all use it.

**Step 1 — full-information reward matrix `R[N,K]`** ([reward.py](../src/stock_agent/rag/reward.py)).
Every query × every arm: run `retrieve()` once, score coverage, subtract cost. The only step that
touches the corpus.

```
          dense  reranked hybrid  hyb+rr  graph
q1 (HARD) 0.400   0.485   0.595   0.580   0.885   ← graph wins (traversal reaches MU's own 10-K)
q2 (CTRL) 0.700   0.735   0.845   0.830   0.835   ← hybrid wins (graph adds nothing, costs more)
```
e.g. q1/graph = `0.90 − 0.05·0.3 = 0.885`; q2/hybrid = `0.85 − 0.05·0.1 = 0.845`. **The optimal arm
flips with context** — the entire premise. A fixed arm can't win both rows.

**Step 2 — featurize contexts `X[N,d]`** ([policy_features.py](../src/stock_agent/rag/policy_features.py)).
Label-free, 11 dims. Discriminating features:

```
        bias n_tok has_tk n_ent is_brdg qt_risk qt_fin qt_bus qt_ovr qt_brdg in_graph
q1        1    10     1      1      1       0      0      0      0      1        1
q2        1     4     1      1      0       0      1      0      0      0        1
```
`is_bridging` (1 vs 0) is the signal the policy keys on. The featurizer **never reads the gold
`stratum`/`qtype`** — every feature is deploy-time computable, so a policy trained on them is deployable
(leakage rule 1).

**Step 3 — group-wise split** ([split_dataset](../src/stock_agent/rag/policy_eval.py)). `split_multihop`
by `group_id` (bridge pair) → ~149 train / 63 test, **no bridge pair straddling the fold**. The z-score
standardizer is fit on **train only**; the constant `bias` column passes through unchanged.

**Step 4 — synthesize the logging dataset under `μ`**
([synthesize_log](../src/stock_agent/rag/policy_eval.py)). `μ = UniformPolicy` picks an arm uniformly
(`μ(a|x)=1/5`, propensity 0.2), revealing **only that arm's reward** — partial feedback, like a real log:

```
row   sampled arm   propensity   observed r
q1 →  hybrid(2)       0.2         0.595      (we do NOT get to see the 0.885 graph would give)
q2 →  graph(4)        0.2         0.835
```
Uniform `μ` has **full support** (every arm `μ>0`) — what makes the later OPE mathematically exact.

**Step 5 — fit the policy on the train log** ([LinUCB.fit](../src/stock_agent/rag/policy.py)). Per arm
`a`, accumulate over the rows where `μ` sampled it:
```
A_a = λI + Σ x xᵀ        b_a = Σ r x        θ_a = A_a⁻¹ b_a
```
Per-arm ridge regression of reward on context. One batch pass, then frozen.

**Step 6 — off-policy evaluate every candidate on the *test* log**
([ope.py](../src/stock_agent/rag/ope.py)). Fresh uniform `μ`-log on test (`seed+1`). For each of 7
candidates (5 fixed + linucb + ε-greedy) compute `π(a|x)` and:
- **IPS** `mean(w_i r_i)`, `w_i = π(a_i|x_i)/0.2`. Deterministic policy ⇒ `w ∈ {0, 5}` — credits only
  test rows where the logged arm matched the pick (~1/5 of rows) → unbiased, high variance.
- **DR** = ridge `q̂` baseline + IPS correction on the residual → same target, much lower variance.
  The headline.
- **true_value** `mean_i R[i, π(x_i)]` — genuine full-info value, available **only because the oracle is
  `$0`**. Ground-truth check that DR isn't lying (`|DR − true| < 0.15` asserted in tests).

```
candidate        DR      true_value
fixed(dense)    0.62      0.61
fixed(hybrid)   0.80      0.81      ← best on CTRL, mediocre on HARD bridges
fixed(graph)    0.855     0.86      ← best fixed overall (bridges dominate the set)
linucb          0.878     0.88      ← graph on HARD, hybrid on CTRL = best of both
```

**Step 7 — pre-registered verdict** ([evaluate_offline](../src/stock_agent/rag/policy_eval.py)).
```
best_fixed = fixed(graph)            # argmax DR among fixed
Δ = DR(linucb) − DR(fixed(graph)) = +0.023
group-bootstrap 95% CI on Δ = [+0.006, +0.041]   (resamples bridge groups, paired)
per-stratum Δ: HARD +0.01, MED +0.02, CTRL +0.05  (no CTRL regression)
⇒ promote = (Δ>0) ∧ (CI_low>0) ∧ (no CTRL loss) = TRUE
```
The CTRL win is the tell: on simple questions LinUCB picks cheap `hybrid` where fixed-`graph` overpays
for traversal that doesn't help — the cost term `λ_c` earns its keep.

> The numbers in Steps 1/6/7 are illustrative-but-plausible to make the mechanics concrete; the *real*
> numbers come from the local verdict run and land in [validations_results.md](validations_results.md).
> **Real verdict (2026-07-08): REJECT.** The illustrative Step 7 shows a *promote*; the actual run was a
> **rigorous negative** — DR(linucb) 0.438 vs best-fixed 0.414 (Δ=+0.024) but the group-bootstrap CI
> **[−0.208, +0.273] includes 0** (ESS≈16 ⇒ underpowered) **and** the bandit *regressed on CTRL*
> (−0.263, opposite of the illustrative CTRL win). Default-OFF stays. Instructive: the mechanics are
> exactly as diagrammed, but the *outcome* was decided by OPE variance and a control-stratum loss, not by
> the sign of Δ. See [validations_results.md](validations_results.md) for the full table.

**Zoom-in: LinUCB actually learning the flip (hand-traced).** Reduce to 2 features `x=[1, is_bridging]`
and 2 arms `{hybrid, graph}`, separated rewards (non-bridging: hybrid 0.80 / graph 0.60; bridging:
hybrid 0.40 / graph 0.90). With `λ=1` and a balanced uniform log (2 samples per arm per context):

```
design matrix Σ x xᵀ over 2×[1,0] + 2×[1,1] = [[4,2],[2,2]];  +λI = [[5,2],[2,3]], det 11
inverse (1/11)[[3,-2],[-2,5]]

b_graph  = 0.60·[1,0]·2 + 0.90·[1,1]·2 = [3.00, 1.80]
θ_graph  = (1/11)[[3,-2],[-2,5]]·[3.00,1.80] = (1/11)[5.40, 3.00] = [0.491, 0.273]
b_hybrid = 0.80·[1,0]·2 + 0.40·[1,1]·2 = [2.40, 0.80]
θ_hybrid = (1/11)[[3,-2],[-2,5]]·[2.40,0.80] = (1/11)[5.60,-0.80] = [0.509, -0.073]

predicted reward  θ·x:
  is_bridging=0 (CTRL):  hybrid 0.509 > graph 0.491  → picks HYBRID  ✓
  is_bridging=1 (HARD):  graph  0.764 > hybrid 0.436 → picks GRAPH   ✓
```
Ridge shrinks the magnitudes (0.49 vs true 0.60), but the **ordering across context is recovered** —
all `argmax` needs. The `+α·sqrt(xᵀ A⁻¹ x)` UCB bonus adds optimism for under-sampled arms; with
balanced counts it doesn't change this argmax, but on real data it stops a rarely-tried arm from being
prematurely written off.

### Part 3 — Serving it in the chat agent UI (after a promote)

**Current state.** `PolicyRetriever` ([policy_retriever.py](../src/stock_agent/rag/policy_retriever.py))
is fully built and satisfies the `RetrievalSystem` contract — but is **not auto-wired**.
`adaptive_retrieval` / `bandit_policy` are inert flags. Deliberate: **A6.1 training doesn't persist a
policy** (training *is* the verdict run), so there's nothing to load yet. The read path is byte-identical
to A5.3.

**To actually serve it, three additions:**

1. **Persist the promoted policy.** Serialize LinUCB's
   `{θ, A_inv per arm, α, arm_names, FEATURE_NAMES order, standardizer (mean,std)}` at the end of a
   promoting verdict run. Feature order and arm order must be pinned — reordering silently remaps every
   weight.
2. **Load + wire under the flag.** With `adaptive_retrieval=True, bandit_policy=linucb`,
   `build_retrieval_system` loads the saved policy, wraps it in `PolicyRetriever`, and injects it as the
   base retriever (the `_injected_base` hook from PR #43).
3. **Per chat turn** (what the user experiences):

```
user question ─▶ featurize(q, ticker=where.ticker, alias_map, graph_universe)   # cheap, pure
             ─▶ policy.act(x) → (arm, propensity)                               # e.g. bridging → graph
             ─▶ build/reuse that arm (embedder/graph DB cached per instance)
             ─▶ arm.retrieve(q, top_k, where)   → evidence
             ─▶ answer_question(evidence)        # unchanged synth + citation/number guards
             ─▶ (optional) log_retrieval(...)    # if retrieval_logging=True
```

The user sees **no new UI** — same answer schema, same guards, same non-advisory contract. Only
observable difference: *which* evidence got selected and slightly variable latency (graph turns are
heavier). The numbers-vs-narrative invariant is untouched: the bandit only *chooses a retrieval config*,
it emits no numbers.

**The virtuous loop (optional, toward A6.2).** With `retrieval_logging=True`, production decisions
`(context, action, propensity, chunks)` accumulate as real logs — the substrate for periodic
re-evaluation/retraining on true traffic rather than the synthetic benchmark.

### Part 4 — Challenges to be aware of (ranked)

1. **Off-policy → production distribution shift (biggest risk).** Trained on 212 semis/AI, bridging-heavy
   questions. Real chat traffic differs (unscoped queries, tickers with no graph, non-finance phrasings).
   LinUCB's linear `θ` can extrapolate badly on out-of-distribution `x`, and the 11-dim coarse features
   (keyword `qtype`, no embeddings) may be uninformative for many real queries → the policy collapses
   toward a near-constant arm. **Mitigation:** keep the fixed default as a floor; let the bandit override
   only where features are confident.
2. **Reward ≠ end objective.** Reward is *single-shot retrieval coverage* minus a *hand-set* cost proxy.
   It does **not** measure final-answer faithfulness (`λ_f` deferred — needs a synthesis call), real
   latency in ms, or user satisfaction. An arm that maximizes coverage need not maximize post-synthesis
   answer quality. The cost vector `[0,0.3,0.1,0.4,0.3]` is a guess, not measured latency — if wrong, the
   cost-adjusted optimum is wrong. **Calibrate costs to real ms before trusting CTRL-stratum wins.**
3. **OPE reliability on *production* logs.** The clean verdict math relies on a **uniform** `μ` with full
   support. Once you serve a deterministic LinUCB, its own log has `w ∈ {0,5}` and no exploration — any
   *future* re-eval from that log is high-variance (tiny Kish ESS → wide CIs) or biased (unsupported
   arms). To keep the loop honest, **log propensities and inject exploration** (ε-greedy or Thompson) so
   future OPE stays valid. Classic bandit feedback-loop confound: the policy shapes the data it later
   learns from.
4. **Frozen, not online.** Deployed LinUCB is static — won't adapt to new filings, tickers, or drift
   without a manual retrain. Real LinUCB updates `A_a,b_a` per round; we deliberately froze it (numpy,
   offline) for reproducibility. Auto-retraining reintroduces the loop risk in (3).
5. **Small, group-correlated benchmark.** 63 test queries, correlated within bridge pairs, `competes_with`
   skew (100/120 HARD). A `+0.023` promote can be within noise; the group bootstrap widens (correctly)
   but doesn't eliminate this. Treat the first promote as provisional; re-confirm on held-out tickers.
6. **Determinism vs exploration at serve time.** LinUCB's deterministic `argmax` is good for
   reproducibility and caching. Stochastic ε-greedy would inject randomness into which arm runs —
   undesirable for cache hit rates and reproducible answers. **Serve LinUCB (or ε-greedy's greedy
   branch), not stochastic ε-greedy.**
7. **Layering against existing routing.** The agent already routes multi-hop → graph (A5.3) and runs the
   alias bridge. `PolicyRetriever` is *another* controller choosing an arm. Decide the layering — bandit
   governs single-shot default only, or also overrides multistep routing? Otherwise two systems fight
   over the graph decision.
8. **Cold-start arm build cost.** `featurize` is cheap, but the *first* time each arm is selected,
   `PolicyRetriever` builds it (embedder / graph DB load). Per-instance caching amortizes it, but the
   first graph turn in a session pays the load.

**Highest-value follow-up if promoted:** wire the `λ_f` faithfulness term into the reward (challenge 2)
so the bandit optimizes answer quality, not just retrieval coverage — but that turns training from `$0`
into a metered run, which is exactly the A6.2 boundary.
