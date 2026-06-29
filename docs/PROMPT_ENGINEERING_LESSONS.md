# Prompt Engineering Lessons — ai-stock-analysis

A teaching archive of **every system prompt we ship**, why each one is written the way it is, and
the reusable techniques behind them. This doc is for *learning*: it pairs each real prompt with the
design reasoning, so you can write prompts of the same quality yourself.

- **Scope:** the LLM-facing prompts in `llm/`, `agent/`, `research/`, `graph/`, and `news/`.
- **Companion docs:** RAG theory → [rag_concepts.md](rag_concepts.md); per-phase build journal →
  [rag_implementation_notes.md](rag_implementation_notes.md); architecture → [ARCHITECTURE.md](ARCHITECTURE.md).
- **The two invariants that shape *every* prompt here** (from [CLAUDE.md](../CLAUDE.md)):
  1. **Numbers vs. narrative** — the LLM never *produces* a probability, return, VaR, or forecast.
     Those come from `indicators/` / `forecasting/` / `backtesting/`. The LLM summarizes, routes,
     and explains. Every prompt below re-states this as a hard rule, and a *guard* enforces it at the
     output boundary (prompt sets the contract; code rejects violations).
  2. **Non-advisory by construction** — no buy/sell/hold, no price targets of our own. There is no
     recommendation field in any schema.

If you remember one thing: **a prompt is a contract, and the contract is only trustworthy if a guard
or a schema enforces it.** We never rely on the prompt alone for correctness.

---

## 0. The inventory — what tools exist and which prompt drives each

| # | Tool (your name for it) | Role | Version tag | Source |
|---|---|---|---|---|
| 1 | News analysis for a ticker | Role A — summarizer | `news_summary.v3` | [llm/prompts/news_summary.py](../src/stock_agent/llm/prompts/news_summary.py) |
| 2 | News + model synthesis | Role C — reconciler | `synthesis.v1` | [llm/prompts/synthesis.py](../src/stock_agent/llm/prompts/synthesis.py) |
| 3 | LLM as orchestrator (chat) | Role B — router | `agent.v16` | [agent/prompts/agent.py](../src/stock_agent/agent/prompts/agent.py) |
| 4 | SEC filings Q&A (single-hop) | grounded QA | `research.v1` | [research/prompts.py](../src/stock_agent/research/prompts.py) |
| 5 | Integrated research memo | grounded synthesis | `memo.v1` | [research/prompts.py](../src/stock_agent/research/prompts.py) |
| 6 | **Agentic RAG** — the controller | ReAct decision | `react.v1` | [research/prompts.py](../src/stock_agent/research/prompts.py) |
| 7 | Knowledge-graph extraction | structured extraction | `graph_extract.v1` | [graph/prompts.py](../src/stock_agent/graph/prompts.py) |
| 8 | Topic keyword expansion | query rewriting | (inline `_SYSTEM`) | [news/topic_expand.py](../src/stock_agent/news/topic_expand.py) |

> **Mapping to your mental model.** Your "agentic RAG" is actually *two* prompts working together:
> the **ReAct controller** (#6) decides *what to retrieve next*, and the **QA prompt** (#4) or
> **memo prompt** (#5) writes the *final cited answer* from the gathered evidence. Splitting "decide"
> from "answer" is a core agentic-RAG pattern — see §6.

Every one of these has a **`VERSION` constant** next to the `SYSTEM` string. Bump it on any material
change. This makes prompt changes greppable, lets you A/B, and ties a logged output back to the exact
prompt that produced it (reproducibility — a project invariant).

---

## 1. The anatomy of a good system prompt (the template behind all 8)

Read any prompt in this repo and you will see the same skeleton. Internalize this and you can write
the rest yourself.

```
[1] PERSONA + JOB        "You are a <role>. Your ONLY job is <one task>."
[2] HARD RULES / GUARDRAILS   what you must NOT do (the negative space), stated as enforceable bullets
[3] WHAT IS ALLOWED      the positive carve-outs that prevent over-refusal
[4] OUTPUT CONTRACT      exact shape: "Return ONLY a single JSON object {…}"
[5] (user message)       the per-request, volatile data — kept OUT of the system text
```

Why this order and these pieces:

- **[1] Persona + a single job.** Narrow the model. "financial news analyst… your ONLY job is
  qualitative synthesis" closes off the whole space of things it *could* do (advise, forecast,
  chit-chat) before it starts. A diffuse persona ("helpful assistant") invites scope creep.
- **[2] Hard rules as the *negative space*.** Most of our prompt budget goes to *what not to do*,
  because our failure modes are specific and costly: inventing a probability, fabricating a URL,
  giving advice. Each rule is phrased so a human (and a guard) can check it. Vague rules ("be
  accurate") are unenforceable; precise rules ("MUST NOT invent your own probabilities, odds,
  likelihoods, or numeric forecasts") are.
- **[3] Explicit allowances.** Negative rules over-trigger. If you only say "no numbers," the model
  refuses to repeat "revenue rose 20%" *that the article stated*. So we carve out the allowed case:
  "You MAY report concrete facts and figures that appear in the articles." **Naming the boundary on
  both sides** (forbidden vs. allowed, with examples) is what makes the model behave precisely instead
  of timidly.
- **[4] A rigid output contract.** Every prompt that feeds downstream code says **"Return ONLY a
  single JSON object"** and prints the exact shape. Downstream we `json.loads` it into a Pydantic
  model. The schema in the prompt *mirrors* the Pydantic schema in code — they must agree.
- **[5] Static system vs. volatile user.** The `SYSTEM` string is **static** (so it caches — see §9).
  The per-ticker / per-question data goes in the **user** message via a `build_user(...)` function.
  Never interpolate the ticker, the date, or the articles into the system text — that breaks prompt
  caching and bloats the cached prefix.

**How I would write one from scratch:** start by writing the *output contract* (what does the caller
need back?), then the *one-sentence job*, then enumerate the *failure modes* you fear and turn each
into a hard rule, then add the *allowances* that keep the rules from over-firing. Persona last — it's
the cheapest part.

---

## 2. Tool #1 — News analysis (Role A summarizer, `news_summary.v3`)

**Job:** turn a ranked list of news articles for one ticker into a structured qualitative synthesis
(overview, themes, bullish/bearish/risks/catalysts), each point cited to article URLs.

### The system prompt (verbatim, abridged for the rules)

```
You are a financial news analyst assistant. Your ONLY job is qualitative synthesis of the provided
news articles for a single stock ticker.

STRICT RULES:
- Summarize and interpret … Do NOT give investment advice …
- You MUST NOT invent your own probabilities, odds, likelihoods, or numeric forecasts …
- You MAY report concrete facts and figures that appear in the articles (e.g. "revenue rose 20%" …)
- Do NOT set your own price targets …
- Use ONLY the article URLs provided below as sources. Never invent URLs.
- Base everything on the supplied articles; if evidence is thin, say so briefly …

OUTPUT: Return ONLY a single JSON object … { "overview", "key_themes", "bullish":[{point,sources}],
"bearish":[…], "risks":[…], "catalysts":[…] }
Each "sources" entry must be one of the provided article URLs. Use [] when no specific article
supports a point.
```

### Why each clause is there

- **"ONLY job is qualitative synthesis"** — encodes the numbers-vs-narrative invariant *as the
  persona*, not just as a rule.
- **The forbidden-vs-allowed pair on numbers** — the single most important design move in the whole
  repo. "MUST NOT invent probabilities/forecasts" + "MAY report figures that appear in the articles."
  This is what lets the summary be *useful* (it can say "analyst raised PT to $300") without it being
  *dangerous* (it can't say "I expect 30% upside"). The matching guard
  (`find_forecast_violations`) only flags *forward-looking* number language, so a reported fact like
  "shares fell 5%" passes — the prompt and the guard are deliberately aligned.
- **"Use ONLY the article URLs … Never invent URLs"** — citation grounding. Hallucinated URLs are the
  classic RAG failure. The guard strips any cited URL not in the provided set; the prompt sets the
  expectation so the model rarely produces one in the first place.
- **"if evidence is thin, say so"** — an explicit escape hatch. Without it, a thin news day produces
  confident-sounding filler. With it, the model is *permitted* to be uncertain, which is the honest
  output.
- **JSON shape with `sources` per point** — forces *point-level* attribution, not a bibliography
  dump. Each claim carries its evidence.

### The reflection (self-review) pass — a technique worth stealing

`news_summary.v3` runs the model **twice** with the *same* `SYSTEM` (so caching still hits):

1. **Draft** — `build_user(ticker, articles)` → first JSON.
2. **Reflect** — `build_reflection(ticker, articles, draft_json)` re-sends the articles **+ the draft
   + a critique rubric** and asks for an improved JSON of identical shape.

The rubric (`_REFLECTION_RUBRIC`) checks completeness, balance, evidence, specificity, and re-states
the hard rules. Crucially it says: *"If the draft is already strong, return it lightly refined — do
NOT pad, invent, or speculate to make it longer."*

**Why this works and why the caveat matters.** A second pass reliably improves *recall* (catches a
missed risk/catalyst) and *grounding* (drops uncited points). But a naive "improve this" instruction
makes models *inflate* — adding plausible-sounding but unsupported points to look thorough. The
anti-padding clause is the guardrail against reflection's own failure mode. **Lesson:** when you ask a
model to critique itself, you must also tell it that "no change" is an acceptable, often correct,
outcome.

---

## 3. Tool #2 — News + model synthesis (Role C, `synthesis.v1`)

**Job:** reconcile the **quantitative** forecast figures (already computed by our models) with the
**qualitative** signals (news, sentiment, earnings timing, technicals) into a deeper read.

### Key clauses and the reasoning

```
You are an integrative equity research analyst. You are given QUANTITATIVE forecast figures … and
QUALITATIVE signals … Reconcile them …

- You may reference and interpret the numbers provided, but you must NOT invent, revise, or estimate
  any new number … Every figure you state must appear verbatim in the inputs. You may say a forecast
  "looks optimistic given X" but never produce a different number.
- The single most valuable output is identifying where … signals AGREE and where they DISAGREE.
  Lead with the disagreements.
- If a scheduled event (e.g. earnings) falls inside the forecast horizon, note that the price-only
  model cannot see it …
```

- **"verbatim in the inputs"** — a *stronger* version of the numbers rule than Role A's. Role C is
  handed real model numbers and its temptation is to *transform* them (average two models, annualize,
  re-round). The rule "every figure must appear verbatim, you may interpret but not produce a
  different number" is enforced by a **numeric-grounding guard** that checks each number in the output
  traces to an input. Note the carve-out: it *may* say a number "looks optimistic" — qualitative
  interpretation of a real number adds value without minting a new one.
- **"Lead with the disagreements"** — this is *task design inside the prompt*. The whole reason to run
  this call is the tension between a price-only model and the news. Telling the model what is *most
  valuable* (not just what's allowed) steers it toward the high-signal output instead of a bland
  recap.
- **The earnings-horizon clause** — injects *domain knowledge the model wouldn't reliably supply*: a
  price-only forecaster is blind to scheduled catalysts, so its distribution is too narrow near
  earnings. Encoding this makes the model surface a real, specific caveat. **Lesson:** put the
  domain's known failure modes *into the prompt* so the model flags them for you.

Output is again a strict JSON object: `overview / alignments / tensions / confidence`.

---

## 4. Tool #3 — LLM as orchestrator (Role B chat agent, `agent.v16`)

This is the biggest prompt (it has been revised 16 times — the version tag tells you so). It is the
**router**: it plans which tools to call, calls them, and explains results. It computes nothing
itself.

### Its structure (note the SECTION headers — a technique in itself)

```
=== HARD CONSTRAINTS (never violate) ===     1) router not calculator 2) no forward numbers
                                             3) non-advisory 4) cite real sources only
=== NUMBER DISCIPLINE (what keeps answers from being rejected) ===
=== ORCHESTRATION (multi-tool, compound questions) ===   plan → parallel calls → routing PATTERNS
=== EXECUTIVE SUMMARY (combining forecast + news) ===
=== DEGRADATION & HONESTY ===
=== TOOLS ===                                one line per tool, when-to-use baked in
```

### The big lessons from `agent.v16`

1. **"ROUTER, NOT CALCULATOR."** The agent's defining rule: *every* quantitative figure it states must
   come **verbatim from a tool result**; if it needs a number it doesn't have, it must call the tool
   that produces it. This is the orchestrator translation of numbers-vs-narrative. A **grounding
   guard** rejects any figure in the answer that doesn't trace to a tool result — and the prompt
   tells the model *that the guard exists and how to stay verifiable* ("quote each figure at the SAME
   precision… re-rounding a real number can make it look unverified and get the whole answer
   blocked"). Telling the model about the enforcement mechanism measurably improves compliance.

2. **No arithmetic on tool numbers.** "no averaging models, no summing buckets, no annualizing… To
   get a derived quantity, call the tool that computes it." This closes the subtle leak where a router
   quietly *becomes* a calculator. The honest path is always "route to the tool," never "do the math
   in your head."

3. **Routing patterns = a decision table in prose.** The `ORCHESTRATION` section maps question shapes
   to tools: "forecast" → `model='ensemble'`; "is it reliable" → `get_calibration`/`run_backtest`;
   "chance of a big move" → `get_large_move`; "how does <news> affect <stock>" →
   `conditional_outlook` (with the driver-proxy mapping spelled out); single-hop filing →
   `search_filings`; multi-hop filing → `research_multistep`; full picture → `research_summary`. This
   is how you make an LLM route *reliably*: don't hope it infers the right tool — **give it the
   mapping explicitly, including the tie-breakers** (single-hop vs multi-hop, when to use the
   expensive path).

4. **Parallelism instruction.** "Issue independent tool calls TOGETHER in one step… they run in
   parallel." A pure capability hint that improves latency — worth stating because models default to
   serial calls.

5. **Tool descriptions carry *when to use*, not just *what*.** Each `=== TOOLS ===` line is
   prescriptive: "`get_large_move` … Use for 'chance of a big move / spike / crash / how volatile'."
   On recent models, trigger conditions in a tool's description give measurable lift in *should-call*
   rate. The tool's name and schema tell the model *what it does*; the description tells it *when to
   reach for it*.

6. **Degradation & honesty.** Explicit instructions for the unhappy paths: tool errors → say so and
   continue; ML model fell back to a baseline → tell the user; ambiguous request → pick a sensible
   default, *state the assumption*, proceed. This is where a chat agent earns trust — the prompt
   refuses to let it paper over failures.

**How I would write an orchestrator prompt:** lead with the hard constraints (the things that get an
answer rejected), then a decision table from question-shape → tool, then degradation rules, then a
compact tool catalog with when-to-use phrasing. Keep the *static* tool catalog in the system prompt
(it caches); pass the user's actual question in the user turn.

> **Caching note for big agent prompts:** `agent.v16` is large enough to clear the cacheable-prefix
> minimum, so the whole static system block caches across turns. The tool list must be **stable and
> deterministically ordered** — tools render before `system` in the cache prefix, so reordering or
> conditionally including a tool invalidates the entire cache. Keep the tool set fixed per session.

---

## 5. Tools #4 & #5 — Grounded SEC answers (`research.v1`) and the memo (`memo.v1`)

These are pure **RAG synthesis** prompts: answer *only* from retrieved source excerpts, with inline
`[n]` citations.

### `research.v1` — single-question QA

```
You are a SEC-filings research assistant. Answer the user's QUESTION using ONLY the numbered SOURCES
below — excerpts retrieved from the company's own SEC filings.

- Ground every statement in the sources. Cite … inline with [n] … Cite only sources that are
  provided; never refer to a source number that is not in the list.
- Use ONLY the sources — no outside or prior knowledge. If the sources do not contain enough
  information … set "insufficient_evidence" to true and make the answer exactly
  "Insufficient evidence found."
- State figures only as they appear in the sources. Do NOT invent, estimate, project, or compute …

OUTPUT: { "answer" (with inline [n]), "citations":[n,…], "insufficient_evidence": bool }
```

Design points worth absorbing:

- **Numbered sources + inline `[n]`.** The `build_user` function renders each retrieved chunk as
  `[i] <citation_label>\n<text>`. The model cites those exact numbers. Downstream a **citation guard**
  rejects any `[n]` not in the retrieved set — same defense-in-depth as the URL guard in Role A.
- **The exact insufficient-evidence sentinel.** "make the answer exactly `Insufficient evidence
  found.`" A *fixed literal* string is something code can detect deterministically. This is how you
  let a RAG system *refuse to answer* cleanly instead of hallucinating from prior knowledge — and the
  empty-retrieval case is a tested requirement.
- **"no outside or prior knowledge."** The defining constraint of grounded QA. The model knows plenty
  about NVDA from pre-training; the whole point is to *suppress* that and answer only from the filing
  text, so the answer is auditable and current.

### `memo.v1` — the integrated research memo (`research_summary`)

Same grounding discipline, but it fuses **three** input types: (a) quantitative signals (the only
source of numbers), (b) news themes, (c) numbered SEC sources. Extra clauses:

- **"The quantitative signals are the ONLY source of numbers"** — Role C's verbatim rule, applied to
  a richer input. Management commentary / drivers / risks must be grounded in SEC sources and cited
  `[n]`; numbers come only from the signals block.
- **"Lead the Bullish/Bearish evidence with where … signals AGREE or DISAGREE."** Same high-value
  framing as `synthesis.v1` — the cross-source tension is the product.
- Output is a structured memo object (`executive_summary`, `management_commentary`,
  `business_drivers`, `risk_factors`, `bullish_evidence`, `bearish_evidence`, `uncertainty_notes`,
  `citations`) — and, per the non-advisory invariant, **no recommendation field**.

**Lesson on multi-source prompts:** when several input kinds feed one call, *label each kind in the
user message* and *assign each its allowed role in the system prompt* ("numbers only from signals;
qualitative claims cited to SEC sources"). Ambiguity about which input may supply what is where
multi-source synthesis goes wrong.

---

## 6. Tool #6 — The agentic-RAG controller (`react.v1`)

This is the heart of "agentic RAG." It implements a bounded **ReAct** (reason → act → observe) loop.
The key architectural decision: **the controller does not write the answer.** It only decides the
*next retrieval step*.

```
You are a retrieval controller answering a multi-hop question … You do NOT write the final answer —
you only decide the NEXT retrieval step in a ReAct loop. Each turn you see the QUESTION, the STEPS
taken so far, and a compact summary of the EVIDENCE gathered.

- Emit exactly ONE decision as JSON. NO prose, NO markdown, NO answer text, NO citations, NO numbers …
- action "search": provide a focused "query" and, when targeted, a "ticker". Make each query depend
  on what the evidence so far still lacks … Never repeat a query you already issued.
- action "stop": the gathered evidence is sufficient … Stop as soon as that is true …
- For a multi-entity question (e.g. "compare A and B"), retrieve for each entity … before stopping.

OUTPUT: { "thought", "action": "search"|"stop", "query": …|null, "ticker": …|null }
```

Why this is the right shape for agentic RAG:

- **Separation of "decide" from "answer."** The controller (`react.v1`) plans retrieval; the terminal
  call (`research.v1`/`memo.v1`) writes the cited answer from *all* gathered evidence. Mixing them
  makes a model both wander and hallucinate. Splitting them means each prompt has one job.
- **"Emit exactly ONE decision as JSON … NO answer text, NO numbers."** The controller is forbidden
  from doing the answerer's job. This keeps the loop's state machine clean: each turn returns a
  parseable action the orchestration code dispatches.
- **State is fed back compactly.** `build_react_user` shows the steps-so-far and a *short snippet per
  chunk* (240 chars) — not full text. The decision call is cheap and reasons over a *summary* of
  evidence; the full chunk text is only ever sent to the (single) terminal synthesis call. **This is
  a cost-architecture decision encoded in the prompt + the user-builder together.**
- **Anti-loop instructions.** "Make each query depend on what the evidence still lacks… Never repeat
  a query… Stop as soon as sufficient." These prevent the two ReAct pathologies: spinning on the same
  query, and padding with needless extra hops. The loop is also *bounded* in code (max steps / max LLM
  calls) — prompt + code together.
- **Multi-entity rule.** "retrieve for each entity across separate steps before stopping" — the
  comparison case ("compare NVDA and AMD risk factors") is exactly what single-shot retrieval can't
  do, so the controller is told to fan out per entity.

**Lesson:** an agentic loop needs (1) a controller prompt that outputs *only an action*, (2) compact
observation feedback, (3) explicit progress/anti-repeat/stop criteria, and (4) hard bounds in code.
The prompt makes the policy; the code makes it safe.

---

## 7. Tool #7 — Knowledge-graph extraction (`graph_extract.v1`)

**Job:** read numbered excerpts from *one* company's 10-K and emit typed `(subject, relation, object)`
triples, each tagged with the source chunk and a confidence.

```
You extract a small knowledge graph from one company's SEC filing. … The SUBJECT of every triple is
the filing's own company — you only name the OBJECT and the relation.

ALLOWED RELATIONS (use these exact strings, nothing else):
- "depends_on" … object = that company's name.
- "competes_with" … object = that company's name.
- "mentions_risk" … object = a SHORT risk phrase (3-6 words) …
- "exposed_to" … object = a short topic phrase …

- Extract a triple ONLY if the cited chunk literally states it. Do NOT infer or guess … If unsure, omit.
- "chunk" MUST be the number of the single excerpt that states the relationship.
- "confidence" in [0,1] …
OUTPUT: { "triples":[{subject, relation, object, chunk, confidence}] }  (or {"triples":[]})
```

Extraction-prompt techniques on display:

- **Closed label set, exact strings.** "use these exact strings, nothing else." Free-form relation
  names are unusable downstream; a fixed enum makes the output a clean typed graph. (In code you'd
  back this with strict structured output / a schema enum; the prompt enforces it for models that
  don't.)
- **Fix what you can, ask only for what you can't.** "The SUBJECT is always the filing's company —
  you only name the OBJECT and the relation." Reducing the model's degrees of freedom raises accuracy:
  it can't get the subject wrong because it never supplies it.
- **Provenance is mandatory.** "`chunk` MUST be the number of the single excerpt that states it."
  Every edge is citeable back to a source span — the same grounding discipline as RAG, applied to
  extraction. This `chunk` number is later mapped to a real `chunk_id`.
- **"ONLY if literally stated … If unsure, omit."** Precision over recall. A spurious edge pollutes
  the graph more than a missing one costs; the prompt biases hard toward omission.
- **Graceful empty case.** "If the excerpts state no qualifying relationship, return `{"triples": []}`."
  Always give the model a clean way to say "nothing here."

---

## 8. Tool #8 — Topic keyword expansion (the small one)

The smallest prompt in the repo, and a good lesson in *not over-engineering*:

```
You expand a news-search topic into precise search keywords.
Return ONLY a JSON object: {"keywords": ["...", "..."]}.
Rules: 3-6 short keyword phrases a news search engine would match … include close synonyms and the
key named entities; English; no explanations.
```

Why it's deliberately tiny:

- It produces **search terms, not numbers or narrative** — so it doesn't touch the numbers-vs-narrative
  invariant and needs no guard.
- It is **best-effort with a safe fallback**: any failure returns just the original phrase, so search
  precision never *depends* on the LLM. The prompt can be minimal because the *system* around it is
  robust.

**Lesson:** match prompt effort to stakes. A low-risk, fallback-protected helper gets four lines; a
load-bearing orchestrator gets 200. Don't pay for guardrails a task doesn't need.

---

## 9. Cross-cutting techniques (the reusable toolkit)

These recur across the eight prompts. Steal them.

### 9.1 Static system / volatile user split (and prompt caching)
Every prompt keeps `SYSTEM` **static** and builds the per-request payload in a `build_user(...)`
function. The client caches the system block (`cache_control: {"type": "ephemeral"}` on the system
text). Because caching is a **prefix match**, the rule is: stable content first (the frozen system
prompt), volatile content (ticker, date, articles, question) after — in the user turn. Interpolating
the ticker or `datetime.now()` into the system text would silently invalidate the cache on every
call. See [llm/client.py](../src/stock_agent/llm/client.py:57).

### 9.2 Determinism: `temperature=0`
`complete_json` sets `temperature=0.0` for reproducibility — we want the same articles to yield the
same synthesis. (See the gotcha in §10 about newer models that reject sampling params.)

### 9.3 JSON-only output contract, mirrored by a Pydantic model
"Return ONLY a single JSON object" + the exact shape in the prompt, parsed leniently in code into a
schema (`llm/guards.py:NewsSummary`, etc.). The prompt's shape and the code's schema **must agree** —
they are two halves of one contract.

### 9.4 Forbidden-vs-allowed pairing
Never state only the prohibition. Pair "MUST NOT invent forward-looking numbers" with "MAY report
figures stated in the source." Naming both sides (with examples) is what produces *precise* behavior
instead of timid over-refusal.

### 9.5 Grounding + a guard (defense in depth)
The prompt says "cite only provided URLs / source numbers"; a **guard** then strips/rejects any
citation not in the retrieved set, and a **numeric-grounding guard** rejects fabricated figures. The
prompt reduces violations; the guard guarantees they never reach the user. Telling the model *that the
guard exists* (as `agent.v16` does) further improves compliance.

### 9.6 Self-review / reflection — with an anti-padding clause
A second pass over (sources + draft + rubric) improves recall and grounding, but you must explicitly
permit "no change" and forbid padding, or reflection inflates the output with unsupported points.

### 9.7 Decision-only outputs for control loops
Agentic loops use a controller prompt that emits *only an action* (`react.v1`), fed compact
observations, with explicit progress/anti-repeat/stop rules — and hard bounds in code.

### 9.8 Versioned prompts
Every `SYSTEM` has a `VERSION` (or `*_VERSION`) sibling constant. Bump on material change. This buys
reproducibility, A/B testing, and traceability from a logged output back to its prompt.

### 9.9 Encode domain knowledge and known failure modes
Put the things the model won't reliably know into the prompt: "a price-only model can't see scheduled
earnings," "the large-move total is more reliable than the up/down split," driver-proxy mappings
(oil→USO/XLE). This is where domain expertise becomes prompt text.

### 9.10 Reduce the model's degrees of freedom
Fix everything you can (the triple's subject is always the filing company; relations are a closed
enum; the insufficient-evidence string is a fixed literal). The fewer free choices, the fewer ways to
be wrong.

---

## 10. Gotchas & honest caveats

- **`temperature=0` and newer models.** [llm/client.py](../src/stock_agent/llm/client.py:64) passes
  `temperature=0.0`. On **Claude Opus 4.7 / 4.8 and Fable 5**, sampling parameters
  (`temperature`/`top_p`/`top_k`) are *removed* and return a **400**. This call works on Sonnet 4.6
  (which still accepts `temperature`) but would break if `settings.llm_model` were pointed at an
  Opus-4.7+/Fable model. If you migrate the configured model upward, drop the `temperature` kwarg.
  (Determinism is then steered via prompting / low effort, not a temperature knob.)
- **Assistant prefill for JSON is *not* used here, on purpose.** A common trick to force JSON is to
  prefill the assistant turn with `{`. The client comment notes "Sonnet 4.x rejects assistant-message
  prefill," and indeed **last-assistant-turn prefills 400 on the whole 4.6/4.7/4.8 family and Fable
  5.** So instead of prefill we rely on (a) the system prompt mandating a bare JSON object and (b)
  lenient parsing that tolerates stray prose. If you want a *hard* JSON guarantee on a current model,
  use **structured outputs** (`output_config.format` with a JSON schema), not prefill.
- **The prompt is necessary but not sufficient.** Re-read §9.5: correctness comes from the
  prompt *and* the guard *and* the schema. A prompt change that weakens a rule without a corresponding
  guard is a silent regression. Bump the `VERSION` and check the matching guard/test when you touch
  any rule.

---

## 11. A checklist for writing the next prompt in this repo

1. **Write the output contract first.** What exact JSON does the caller parse? Add/locate the Pydantic
   model; make the prompt's shape mirror it.
2. **State the one job in one sentence.** "You are a <role>. Your ONLY job is <task>."
3. **Enumerate failure modes → hard rules.** For each thing that would be wrong/dangerous, write an
   enforceable bullet. Lean on the two invariants (no invented numbers; non-advisory).
4. **Add the allowances** that stop the rules from over-firing (forbidden-vs-allowed, with examples).
5. **Ground every claim** to provided sources (URLs / numbered chunks / tool results) and give a clean
   "insufficient evidence" / empty path.
6. **Decide static vs. volatile.** Static → `SYSTEM` (caches). Volatile → `build_user(...)`. Never put
   ticker/date/data in the system text.
7. **Pair the prompt with a guard or schema** that enforces the load-bearing rules at the output
   boundary. Tell the model the guard exists if compliance matters.
8. **Add a `VERSION` constant** and bump it; for RAG/advanced-RAG phases, follow the doc-update rules
   in [CLAUDE.md](../CLAUDE.md).
9. **Test deterministically** — `FakeEmbedder` / canned LLM responses, no live calls; add a test that
   would *fail* on the failure mode you wrote the rule for.
```
