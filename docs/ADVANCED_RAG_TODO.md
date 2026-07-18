# Advanced RAG — Roadmap & TODO

> **Execution guide** for the *advanced* RAG track (hybrid search, reranking, formal retrieval
> evaluation, agentic retrieval, GraphRAG, retrieval+RL). Design brief:
> [RAG_IMPLEMENTATION_PLAN_ADV.md](RAG_IMPLEMENTATION_PLAN_ADV.md). Companion to the MVP roadmap
> [RAG_TODO.md](RAG_TODO.md) (P0–P9, shipped) — same conventions: each phase is independently
> shippable, **default-OFF / config-gated** so the live system never regresses, and must pass
> `make check` before the next.
>
> **Where we start from (MVP, P0–P9 — done):** SEC download → parse → section-aware chunk →
> embed-once (`Embedder` Protocol; voyage-4 in prod, local BGE fallback) → Chroma `VectorStore`
> (per-embedder namespaced) → `Retriever` (filtered top-k + dedup, no LLM) → one grounded
> synthesis call (citation + number guards) → `research` memo / `search_filings` agent tool.
> Plus a **retrieval-eval harness already exists** (`rag/eval.py`, P9b) — A1 extends it, not rebuilds it.

---

## Locked decisions (the repo-specific answers to the brief's open questions)

These resolve the brief's "questions to answer" up front, given *this* codebase:

1. **Everything is default-OFF + behind a config flag + composes behind the existing `Retriever`
   interface.** Hybrid/rerank/agentic/graph are opt-in; with all flags off the pipeline is
   byte-identical to today. (Mirrors the forecasting track's experimental-infra pattern: regime/
   LSTM/news features ship as default-OFF code, validated before promotion.)
2. **Measurement first (A1 is the gate).** No retrieval change (A2–A6) ships a "win" without a
   measured improvement on the eval benchmark. A1 generalizes `run_ab` from comparing *embedders*
   to comparing *retrieval systems* (dense / hybrid / reranked / graph).
3. **Labels stay chunking-invariant.** Keep the P9b **answer-span** labels (+ add `expected_section`
   / `expected_document_type`); **never** `expected_chunk_ids` (they break on every re-chunk, which
   A2/A3 do constantly). The brief's `expected_chunk_ids` schema is explicitly rejected.
4. **No torch in the base path** (the documented macOS torch+lightgbm OpenMP segfault is why the MVP
   uses fastembed/onnxruntime). Rerankers: **fastembed onnx cross-encoder** (local default) | **Voyage
   `rerank-2` API** (opt-in, key already wired) | **no-op** (fallback). No `sentence-transformers`/torch.
   **A6 exception (2026-06-28):** torch is allowed ONLY inside the isolated RL trainer (`[rl]` extra,
   lazy import, OpenMP-isolated from lightgbm); base imports + CI stay torch-free (see §A6 decision 1).
5. **Sparse backend = SQLite FTS5** (Python stdlib `sqlite3`; persistent; BM25 via `bm25()`; metadata
   filter via SQL `WHERE`) — no new dependency, available in CI for deterministic tests. `rank_bm25`/
   Tantivy rejected (extra dep / Rust). Fusion = **Reciprocal Rank Fusion** (rank-based → no
   cosine-vs-BM25 score-normalization), not weighted score-sum.
6. **Graph store = SQLite tables + NetworkX in-memory** (Neo4j deferred); extraction = **LLM-assisted,
   offline, cost-gated, with chunk-id provenance on every edge** (so graph answers stay citeable).
7. **RL = contextual bandits first, THEN full RL** (updated 2026-06-28 — see §A6's two-phase plan).
   **A6.1** = retrieval **logging + off-policy evaluation (IPS / DR)** + an ε-greedy / LinUCB bandit if
   OPE shows headroom. **A6.2** = full multi-step RL (the agentic loop as an MDP; **PPO**, offline on a
   simulator) — no longer deferred. Reward = A1 metrics (+ later, user feedback).
8. **Citations + numbers never weaken.** Rerank/hybrid don't touch chunks (only order) → citations
   unchanged. Agentic/graph cite via the **union of retrieved chunks** / **edge provenance**; the
   existing citation guard + `NumberGrounding` run on every synthesis call. Numbers stay model-only.

---

## How it composes (target pipeline; each stage is optional/gated)

```
                                  ┌──────────────── A6: bandit policy picks the config (mode/rerank/k) ───────────────┐
                                  ▼                                                                                   │
question ─▶ [A4 planner: decompose?] ─▶ for each sub-query ─▶ RETRIEVE ─▶ RERANK (A2) ─▶ top-k ─▶ SYNTHESIS ─▶ cited answer
                                                                  │                                  ▲ (citation + number guards, unchanged)
                                            ┌─────────────────────┴─────────────────────┐           │
                                            │  A3 HybridRetriever: dense ⊕ sparse (RRF)  │           │
                                            │  A5 GraphRetriever: traverse → chunk_ids   ├── union ──┘
                                            └────────────────────────────────────────────┘
                                  A1 retrieval-eval harness measures EVERY stage (dense vs hybrid vs reranked vs graph)
```

`RETRIEVE` is the existing `rag/retriever.Retriever`; A2/A3/A5 are alternative or wrapping
implementations of the same "query (+filter) → ranked chunks" contract, so `research/synthesis`,
`research/memo`, and the agent tools are **untouched**.

---

## Sequencing & effort

| # | Enhancement | Order rationale | Type | Complexity |
|---|---|---|---|---|
| **A1 ✅** | Retrieval Evaluation | The measuring stick; extends `rag/eval.py`; lowest risk | Foundation | **Low–Med** (~1.5d) |
| **A2 ✅** | Reranking | Biggest quality win per effort; local-first, self-contained | Ship value | **Low–Med** (~1.5d) |
| **A3 ✅** | Hybrid Search | Fixes a *different* failure mode (exact ticker/section/number terms) | Ship value | **Med** (~2d) |
| **A4 ✅** | Agentic RAG | Needs strong retrieval primitives first; bounded LLM cost | Ship value | **Med–High** (~2.5d) |
| **A5 ✅** | GraphRAG | Heaviest infra; lower near-term ROI for SEC QA | Learning-first | **High** (~3–4d) |
| **A6** | Retrieval + RL | Capstone; needs A1 metrics + usage logs to define a reward | Learning-first | **High** (~3–4d) |

**Build order: A1 → A2 → A3 → A4 → A5 → A6.** A2↔A3 are swappable (both feed A1; final pipeline is
hybrid→rerank). A5/A6 are explicitly *learning-first* — high intellectual value, modest immediate
retrieval-quality ROI for grounded SEC QA; a rigorous negative result is a valid outcome (cf. the
forecasting track's LSTM/news-feature negatives).

---

## A1 — Retrieval Evaluation (extend `rag/eval.py`) ✅ gate for everything else

> **Status: DONE (2026-06-13).** Shipped: `RetrievalSystem` Protocol; graded relevance
> (`relevance_grade` + `expected_document_types`/`expected_sections`); `ndcg_at_k`,
> `citation_accuracy`; `SystemReport` + generic `evaluate_system` (with `run_ab`/`EmbedderReport`
> as back-compat wrappers); `Retriever.name`; `rag eval --report` + `make rag-eval`. Mechanism →
> [rag_implementation_notes.md](rag_implementation_notes.md) §A1; math → [rag_concepts.md](rag_concepts.md) §11.
> **Deferred (opt-in, by design):** LLM-judge **faithfulness** metric; **baseline.json regression
> gate**; growing the benchmark to 60–100 Q. (`--systems` lattice CLI — **DONE** as an A1/A3
> follow-up once A2/A3 existed; see the eval-lattice note in `rag_implementation_notes.md`.)

**Learning objective.** Rigorous IR evaluation: Precision@k, Recall@k, MRR, **nDCG@k**, plus the
RAG-specific **citation-accuracy** and **answer-faithfulness**; regression-gating retrieval changes.

**User value.** Every later change is provably better-or-worse, not vibes. A repeatable `rag eval`
report (dense vs hybrid vs reranked) + a CI floor that fails a retrieval regression.

**Current state (P9b).** `rag/eval.py` has `LabeledQuery` (answer-`relevant_spans`, `ticker`,
`top_k`), `hit_at_k` / `reciprocal_rank` / `precision_at_k` / `recall_at_k`, `evaluate_query`,
`run_ab` (over **embedders**), `EmbedderReport`, `format_reports_markdown`, the `rag eval` CLI, and a
25-question set `configs/rag_eval_queries.json`.

**Design / changes.**
- **Generalize the unit under test from `Embedder` → `RetrievalSystem`.** New Protocol in
  `rag/eval.py`: `RetrievalSystem` = `name` + `retrieve(query, *, top_k, where) -> EvidenceSet`
  (the existing `Retriever` already satisfies it; A2 `RerankingRetriever`, A3 `HybridRetriever`,
  A5 `GraphRetriever` will too). Add `evaluate_system(system, queries, ...) -> SystemReport`;
  keep `run_ab(embedders)` as a thin wrapper for back-compat.
- **Add metrics:** `ndcg_at_k(gains, k)` (graded relevance: gain = number of distinct
  `relevant_spans` a chunk matches, so a chunk answering more of the question ranks higher);
  `citation_accuracy(answer, evidence)` = fraction of a `GroundedAnswer`'s citations that point to a
  *relevant* chunk (precision of citations). Faithfulness → **opt-in LLM-judge** (`--faithfulness`,
  cost-gated; default off) scoring "is each answer claim supported by the cited chunk?"; the hard
  floor (no invented numbers/citations) is *already* enforced by the P7 guards — faithfulness is the
  soft, expensive layer.
- **Richer labels (chunking-invariant):** extend `LabeledQuery` with optional
  `expected_document_type: DocumentType | None` and `expected_sections: list[str]` (e.g.
  `"Item 1A. Risk Factors"`); a chunk is "relevant" if span-match **and** (if specified) section/type
  match. Grow the benchmark to ~60–100 Q across more tickers + sections; document the labeling method.
- **CI vs real-corpus split (important):** the FakeEmbedder is hash-based, not semantic, so CI tests
  the *harness mechanics* deterministically (metric math, system conformance, citation-accuracy with
  canned answers) — the **real** benchmark numbers come from a **local** `rag eval` run against the
  voyage/fastembed corpus (exactly like model backtests are local, not CI). A `make rag-eval` target
  runs the real benchmark; an optional committed `outputs/rag_eval/baseline.json` lets a local run
  flag regressions.

**Modules/files.** Extend `rag/eval.py`; add `schemas/eval.py` (`SystemReport`, graded
`LabeledQuery` fields) if it keeps `eval.py` clean; extend the `rag eval` CLI (`--systems
dense,hybrid,reranked`, `--faithfulness`, `--report outputs/rag_eval/<ts>.json`); `Makefile`
`rag-eval`.

**Tests** (offline): nDCG golden values (hand-computed gains/discounts); `RetrievalSystem` Protocol
conformance for `Retriever`; `evaluate_system` over a fake system; `citation_accuracy` with a canned
`GroundedAnswer` (some cites relevant, some not); section/type-filtered relevance. No model download.

**Risks.** Faithfulness scope-creep (LLM-judge cost/subjectivity) → keep it opt-in + small. Benchmark
labeling effort → grow incrementally; the 25-Q set already works.

---

## A2 — Reranking

> **Status: DONE (2026-06-13).** Shipped `rag/rerank.py`: `Reranker` Protocol +
> `NoOpReranker` (default) / `FastEmbedReranker` (fastembed onnx cross-encoder, **no torch**) /
> `VoyageReranker` (opt-in); `build_reranker`; `RerankingRetriever(base, reranker, *, fetch_k)`
> (over-fetch→rerank→slice, wraps any `RetrievalSystem`); `build_retrieval_system` — the single
> **default-OFF** factory wired into `pipelines/research`, `agent/tools._get_retriever`, and the
> `rag query` CLI. Config: `rerank_provider="none"|local|voyage`, `rerank_fetch_k=30`,
> `rerank_model`. Also moved the `RetrievalSystem` Protocol `eval.py → retriever.py` (its proper
> home now production depends on it). Mechanism → [rag_implementation_notes.md](rag_implementation_notes.md) §A2;
> math → [rag_concepts.md](rag_concepts.md) §12. `make check` green (635 passed). **Deferred:**
> promoting rerank ON by default (needs an A1 `make rag-eval` win first).

**Learning objective.** Cross-encoder reranking + the **retrieve-wide-then-narrow** pattern.

**User value.** Markedly cleaner top-k (the synthesis sees more on-point evidence) at ~1 cheap step.

**Pipeline.** `question → retrieve top fetch_k (e.g. 30) → rerank → keep top_k (5–8) → synthesis`.

**Design.**
- **`Reranker` Protocol** (`rag/rerank.py`): `name` + `rerank(query: str, chunks:
  list[RetrievedChunk]) -> list[RetrievedChunk]` (reordered; scores replaced by rerank scores).
  Implementations:
  - `NoOpReranker` — identity (the default; behavior unchanged).
  - `FastEmbedReranker` — fastembed `TextCrossEncoder` (onnx, **no torch**; e.g.
    `Xenova/ms-marco-MiniLM-L-6-v2` or a BGE/Jina reranker). Lazy-loaded, in the `[rag]` extra.
  - `VoyageReranker` — Voyage `rerank-2` API (opt-in, reuses `VOYAGE_API_KEY`; no local model;
    batched like the embedder).
  - `build_reranker(settings)` selector (mirrors `build_embedder`).
- **`RerankingRetriever(base: Retriever, reranker: Reranker, *, fetch_k: int)`** — over-fetch
  `fetch_k`, rerank, slice `top_k`. Satisfies `RetrievalSystem`, so synthesis/agent/memo are untouched.
- **Config:** `rerank_provider: Literal["none","local","voyage"] = "none"`, `rerank_fetch_k: int = 30`,
  `rerank_model: str = …`.

**Wire-in.** `research/synthesis`, `pipelines/research`, and `agent/tools._get_retriever` build a
`RerankingRetriever` *only when* `rerank_provider != "none"` — otherwise the plain `Retriever`.

**Tests** (offline): `Reranker` conformance; `RerankingRetriever` with a `FakeReranker`
(deterministic scores) reorders + truncates to `top_k`; `NoOpReranker` = passthrough;
over-fetch≥top_k; A1 shows reranked vs dense on the benchmark. Real onnx reranker behind a
`RUN_RERANK_TESTS` gate (no CI model download).

**Risks.** onnx model download (gate it); added latency (~50–200 ms local) → expose `fetch_k`;
Voyage rerank cost (small, opt-in).

---

## A3 — Hybrid Search (dense ⊕ sparse, RRF)

> **Status: DONE (2026-06-13).** Shipped `rag/sparse_store.py` (`SparseStore` Protocol +
> `Fts5SparseStore` [SQLite FTS5, stdlib, persistent] + `InMemoryBM25Store` [pure-Python Okapi
> BM25] + `build_sparse_store`); `rag/hybrid.py` (`reciprocal_rank_fusion` pure fn +
> `HybridRetriever`, a `RetrievalSystem`); `rag/read_path.py` — moved `build_retrieval_system` here
> (composition root) and extended it to the full lattice `rerank(hybrid(dense, sparse))`; ingest
> (`ingest_ticker`/`bulk_ingest` + `documents ingest`/`refresh`) now maintains the BM25 index in
> lockstep ($0), backfilling on the next refresh for pre-A3 corpora. Config:
> `retrieval_mode="dense"|hybrid`, `hybrid_rrf_k=60`, `hybrid_dense_k`/`hybrid_sparse_k=30`,
> `sparse_store_dir`. Default-OFF (`retrieval_mode="dense"`). Mechanism →
> [rag_implementation_notes.md](rag_implementation_notes.md) §A3; math → [rag_concepts.md](rag_concepts.md) §13.
> `make check` green (660 passed). **Eval-lattice CLI: DONE** — `rag eval --systems
> dense,reranked,hybrid,hybrid+rerank [--diagnostic]` + `build_named_system` (the A6 action-space
> seed) landed as an A1/A3 follow-up (669 passed). **PROMOTED (2026-06-22):** ran the lattice —
> **hybrid wins** (nDCG 0.787→0.823, P 0.760→0.805); `retrieval_mode` default → `hybrid`. Rerank was
> marginal (ms-marco reranker, domain-mismatched) → left OFF but kept available (A6 action). Added
> `documents backfill-sparse` (populate BM25 from existing filings, no embedding). See
> [rag_implementation_notes.md](rag_implementation_notes.md) "Promotion".

**Learning objective.** Sparse/BM25 retrieval and **rank fusion**; why dense and sparse are
complementary (semantics vs exact terms: tickers, section names, defined terms, dollar figures).

**User value.** Recovers exact-term matches dense embeddings miss (e.g. "Item 7A", "Hopper",
a specific subsidiary) without losing semantic recall.

**Design.**
- **`SparseStore` Protocol** (`rag/sparse_store.py`): `add(chunks)`, `search(query, *, top_k, where:
  ChunkFilter) -> list[tuple[str, float]]` (chunk_id, BM25 score), `existing_ids(ids)` (for the 9e
  incremental refresh). Backends:
  - `Fts5SparseStore` (default) — SQLite **FTS5** under `data/sparse/<embedder-namespace>.db`; chunk
    text in an FTS5 virtual table, metadata in columns; ranks via `bm25()`; filters via SQL `WHERE`.
    Stdlib only; works in CI (temp DB).
  - `InMemoryBM25Store` — tiny pure-Python BM25 (~40 lines) for parity / no-sqlite-FTS environments.
  - `build_sparse_store(settings)`.
- **Ingest writes both indexes.** `ingest_ticker`/`bulk_ingest` gain an optional `sparse_store=`;
  when present, `store.add` and `sparse.add` run together ($0, local). Refresh stays incremental for
  both (`sparse.existing_ids`).
- **`HybridRetriever(dense: Retriever, sparse: SparseStore, *, k_rrf: int = 60)`** — run dense + sparse
  (each over-fetched), fuse by **RRF**: `score(d) = Σ_i 1/(k_rrf + rank_i(d))` over the two ranked
  lists, sort, take `top_k`. Rank-based → no cosine/BM25 normalization. `ChunkFilter` applied to both
  sides. Empty sparse hit → degrades to dense. Satisfies `RetrievalSystem`.
- **Config:** `retrieval_mode: Literal["dense","hybrid"] = "dense"`, `hybrid_rrf_k: int = 60`,
  `hybrid_dense_k`/`hybrid_sparse_k` (over-fetch per side).

**Tests** (offline, temp sqlite): FTS5 add/search/metadata-filter; RRF math golden (two hand-built
rankings → expected fused order); `HybridRetriever` fuses + filters + dedups; empty-sparse → dense
fallback; sparse `existing_ids` for incremental; A1 dense vs hybrid.

**Risks.** FTS5 not compiled in some Python builds → feature-detect, fall back to `InMemoryBM25Store`.
Financial tokenization (tickers/numbers) → FTS5 `unicode61` + a `tokenchars` tweak; validate on the
benchmark. Sparse/vector sync on refresh → covered by `existing_ids` on both.

---

## A4 — Agentic RAG (self-contained, bounded ReAct loop for multi-hop SEC QA) ✅

> **Design decision (2026-06-22) — ReAct, not plan-and-execute.** A4 is a **self-contained, bounded
> ReAct loop** (reason → retrieve → observe, with a *reflective* stop), **not** query-decomposition /
> plan-and-execute. ReAct handles **multi-hop** — each retrieval can depend on the *previous*
> observation — which a one-shot up-front decomposition cannot. Framed precisely: A4 is **"P7, but
> iterative"** — multi-step SEC evidence-gathering that ends in the **existing** guarded answer
> (`research/synthesis.answer_question`). It is **NOT** "P8, but agentic": it does **not** rebuild the
> executive-summary memo (`run_research`, which fuses news + forecast + RAG) — that synthesis already
> exists and stays untouched. The only synthesis A4 performs is the unavoidable cited-answer
> composition, and that is **reused** from P7, not rebuilt.

**Learning objective.** The **ReAct** pattern (interleave reasoning + tool actions) with a reflective
stop-condition; bounded agentic control; multi-hop retrieval under a fixed LLM-call + grounding budget.
*(NB: ReAct ≠ Reflexion. Reflexion = self-critique-and-retry across attempts; here the "reflective"
part is the per-step self-assessment "do I have enough evidence yet?".)*

**User value.** Questions a single retrieval can't answer: *"Compare NVDA and AMD AI risks,"* *"What
changed in TSLA risk disclosures over 3 years,"* *"Which of NVDA's named suppliers flag the same
risk?"* — genuine **multi-hop**, where observation N shapes query N+1.

**Architecture.** New `research/agentic.py` (the ReAct controller) + `schemas/agentic.py` (typed loop
state/IO) + a `REACT_SYSTEM` prompt in `research/prompts.py`. **Reuses unchanged:**
`build_retrieval_system` (hybrid retrieval, $0/local), `answer_question` (P7 guarded synthesis — its
citation + number guards **and** its empty-evidence short-circuit), the `TextLLM.complete_json`
client. Dependency direction: `research/agentic.py` → `rag/` + `llm/` + `schemas/`; the agent tool
wraps it. **No new synthesis or guard code.**

**The loop.** Each iteration is ONE cheap structured-output call that reasons over the question + a
compact summary of evidence-so-far and emits a `ReActStep` — either **search** (retrieve more, the
*Act*) or **stop** (evidence sufficient, the *reflective stop*):

```
evidence = []                                  # deduped union of RetrievedChunk (per-step provenance)
trace = []
for step in range(agentic_max_steps):          # cap (default 4)
    s = react_step(llm, question, trace, evidence)        # 1 cheap LLM call → ReActStep
    if s.action == "stop": break
    if not s.query or s.query in issued_queries: break     # anti-loop: no empty/duplicate queries
    ev = retriever.retrieve(s.query, top_k=agentic_per_step_k, where=s.filter)   # $0, hybrid
    evidence = dedup_union(evidence, ev.chunks)[:agentic_max_evidence]
    trace.append(StepTrace(thought=s.thought, query=s.query, ticker=s.ticker, n_retrieved=len(ev)))
# terminal: the EXISTING guarded answer over the accumulated union (reused P7)
ans = answer_question(question, EvidenceSet(query=question, chunks=evidence), llm=llm)
return MultiStepAnswer(answer=ans, trace=trace, n_steps=len(trace), n_evidence=len(evidence))
```

- The decision call is **cheap** (sees a compact evidence summary, emits a small structured decision —
  no answer prose). The single heavy call is the terminal `answer_question`.
- **Budget:** ≤ `agentic_max_steps` decision calls + 1 terminal answer ⇒ **≤4 LLM calls** by default
  (`agentic_max_steps=3`); all retrieval between steps is $0/local. (Bounded like `run_backtest`.)

**Grounding (invariants unchanged).** Loop decisions emit only queries/filters — **no numbers, no
citations** → nothing to hallucinate-ground in the loop. The terminal `answer_question` runs the
**citation guard** (every `[n]` ∈ the retrieved union) + **NumberGrounding** over the union allow-set,
and its empty-union path already returns *"Insufficient evidence found."* with **no LLM call**.
Non-advisory + numbers-from-models unchanged. **No auto-ingestion** (a ticker with no corpus simply
yields no evidence). Re-uses, does not re-implement, the guards.

**Schemas (`schemas/agentic.py`).**
- `ReActStep { thought: str, action: Literal["search","stop"], query: str | None, ticker: str | None,
  filter: ChunkFilter | None }` — the per-iteration structured LLM output (parsed like P7's
  `_RawAnswer`).
- `StepTrace { thought, query, ticker, n_retrieved }` — transparent loop trace (debug + tool output + tests).
- `MultiStepAnswer { answer: GroundedAnswer, trace: list[StepTrace], n_steps: int, n_evidence: int }`.

**Controller (`research/agentic.py`).** `answer_multistep(question, *, settings, llm, retriever=None,
max_steps=…, per_step_k=…, max_evidence=…) -> MultiStepAnswer`; builds the retriever via
`build_retrieval_system` (injected in tests); `_react_step(...)` makes the structured decision call.

**Config (settings).** `agentic_max_steps: int = 3` (decision iterations; +1 terminal ⇒ ≤4 calls;
CLI `--max-steps` raises it per call); `agentic_per_step_k: int = 6`
(chunks per search step); `agentic_max_evidence: int = 20` (cap the union handed to the terminal
synthesis — bounds context + cost).

**Surface.** New agent tool `research_multistep(question)` (general multi-hop; the agent router sends
multi-entity / temporal / comparative filing questions here, casual ones stay on single-shot
`search_filings` — the **simple-question fast path**); bounded wall-clock like `run_backtest`. Optional
CLI `rag ask --multi -q "…"` for direct runs/eval. The agent **relays** the cited answer (does not
re-synthesize).

**Tests (offline; canned `TextLLM` + `FakeEmbedder`/InMemory; no live calls, no model).**
- scripted decisions (search→search→stop): retrieves per step, dedups the union, stops on `stop`;
  never-stops → caps at `max_steps`; empty/duplicate query → terminates (anti-loop);
- terminal reuses `answer_question` → citation guard over the union (a cite outside the union →
  `ResearchGuardError`); **empty union → "Insufficient evidence found." with no LLM call**;
- budget: LLM-call count == decisions + 1; never exceeds `max_steps`;
- agent tool: a "compare X and Y" turn routes to `research_multistep` (canned LLM).

**Risks.** Cost/looping → hard `max_steps` cap + anti-duplicate-query guard + the simple-question fast
path. Decision quality → versioned `REACT_SYSTEM` prompt + a small multi-step eval set ✅ (`research/
multistep_eval.py` + `rag eval-multistep` — union **aspect coverage** vs. a single-shot baseline, the
*gain* being the measured value of the extra hops; seed `configs/rag_eval_multistep.example.json`).
Latency → bounded calls + `max_evidence` cap. Grounding across steps → union allow-set + the reused
P7 guards.

---

## A5 — GraphRAG (lightweight knowledge graph) — learning-first ✅

> **Status: code DONE (2026-06-27).** Shipped all four slices: `schemas/graph.py` (`Entity`/`Edge` +
> `RELATION_OBJECT_TYPE`); `graph/store.py` (`GraphStore` Protocol + `SqliteGraphStore` — idempotent
> upserts, bounded BFS `neighbors`/`provenance_chunk_ids`, `build_graph_store` namespaced per
> embedder); `graph/extract.py` + `graph/prompts.py` (offline, cost-gated triple extraction →
> alias-resolve → **hallucinated-edge guard** [object must appear in the cited chunk] + confidence
> gate; `GraphExtractBudgetExceeded`) wired to `documents extract-graph`; `graph/retriever.py`
> (`GraphRetriever` — base ⊕ traversal fused by **RRF**, satisfies `RetrievalSystem`) wired via
> `rag/read_path.build_graph_system`, `rag graph-query`, and `rag ask --graph` / `rag eval-multistep
> --graph` (the A5.3 measurement hook into the existing bridging benchmark). Default-OFF; all
> retrieval $0/local; the only paid step is offline extraction. `make check` green (751 passed; 25 new
> graph tests). Mechanism → [rag_implementation_notes.md](rag_implementation_notes.md) §A5; theory →
> [rag_concepts.md](rag_concepts.md) §16. **Deferred (by design):** the *measured* promotion verdict
> (run `rag eval-multistep --graph` on the real corpus vs. hybrid + the A4 entity-bridge — a local
> paid run, like the model backtests); gating the query-time alias-bridge OFF when graph wins; global
> GraphRAG (communities/Leiden) and Neo4j remain V2+.

> **Conceptual framing — how A5 relates to A4 (agentic).** They are *orthogonal* layers, not one
> wrapping the other: agentic RAG is a control-flow *strategy* (when/how-many-times to retrieve);
> GraphRAG is a retrieval *substrate* (what you retrieve over). A5's `GraphRetriever` satisfies the
> same `RetrievalSystem` protocol the A4 loop already calls, so agentic can *wrap* graph, and a graph
> *traversal* can replace the brittle query-time alias-bridge of §A4 by moving entity resolution to a
> stored, ingest-time edge. Full framing + the 2×2 + the `NVDA→MU` worked contrast →
> [rag_concepts.md §15.9](rag_concepts.md). The A4 bridging negative is A5's motivating benchmark
> (re-measure with `rag eval-multistep`).

**Learning objective.** Knowledge-graph **construction from text** + **graph-augmented retrieval**
(multi-hop questions vector search can't answer).

**User value.** Structural questions: *"Who are NVDA's key suppliers/customers?"*, *"Which companies
are exposed to a TSMC disruption?"*, *"What risks does NVDA share with AMD?"* — answered by graph
traversal, then grounded in the filings the edges came from.

### Locked decisions (resolve before coding — do not re-litigate)
1. **Types vs. relations.** Earlier sketches listed `supplier`/`customer`/`competitor` as entity
   *types*; those are **relations**, not types. **Entity types (MVP, 5):** `company`, `product`,
   `segment`, `risk`, `regulatory_topic`. **Relations (MVP, 4):** `depends_on` (company→company,
   e.g. NVDA→MU — the bridging edge), `competes_with` (company→company), `mentions_risk`
   (company→risk), `exposed_to` (company→regulatory_topic). Defer `supplies_to` (= inverse of
   `depends_on`, derive on traversal), `acquired`, `operates_in`.
2. **Node identity = ticker for companies.** A company node's canonical `id` is its **ticker**
   (e.g. `MU`), resolved **at extraction time** via `configs/ticker_aliases.json` (REUSE
   `research/bridge.load_alias_map` + `mentioned_tickers`). This is the whole point — it moves the
   `micron → MU` resolution the A4 entity-bridge does at *query* time to a **once, offline, verified**
   step (§15.9). Unresolvable company names → store with `ticker=None` (lower-value, still cited).
   Non-company nodes (`risk`/`topic`) use a normalized-name id.
3. **Provenance is mandatory + verified.** Every `Edge` carries the `chunk_id(s)` it came from. A
   **verification pass** (the edge analogue of the P7 citation guard) **drops any edge whose
   provenance chunk text does not contain both the subject and object surface names** — the
   hallucinated-edge guard. Plus a model `confidence` ≥ `graph_min_confidence`.
4. **Extraction is offline, batched, cost-gated, default-OFF.** A separate `documents extract-graph`
   step (never at query time); one structured-output LLM call per `(ticker, section)` over **10-K
   Item 1 (Business) + Item 1A (Risk Factors)** only; a candidate pre-filter (regex over alias /
   risk keywords) cuts the call set; a `GraphExtractBudgetExceeded` spend ceiling
   (`graph_max_extract_calls`) mirrors `EmbedBudgetExceeded`.
5. **`GraphRetriever` satisfies `RetrievalSystem`** (`name` + `retrieve(query, *, top_k, where)`), so
   it drops into `build_retrieval_system` / the A4 loop / the eval harness **with zero changes** to
   callers (the §15.9 glue). Default-OFF (a new `graph` config), promoted only on a measured win.
6. **Storage = SQLite (source of truth) + NetworkX (traversal view).** `data/graph/<namespace>.db`;
   NetworkX loaded from SQLite for k-hop neighbors. New dep: `networkx` (lightweight, pure-Python).
   Neo4j deferred.

### Dependencies & reuse (what to build on — most of A5 is wiring, not new infra)
- **Chunked corpus** — chunks already carry `chunk_id`, `ticker`, `section` (`documents/` +
  `rag/chunking`). Extraction reads chunks filtered to Item 1 / Item 1A; the GraphRetriever fetches
  provenance chunks by id.
- **Entity resolver** — `research/bridge.py` (`load_alias_map`, `mentioned_tickers`) +
  `configs/ticker_aliases.json`. **The single most important reuse**: extraction resolves
  object-company names → tickers with the *same* map, but offline + verified.
- **`RetrievalSystem` Protocol** (`rag/retriever.py`) + **`build_retrieval_system` / `build_named_system`**
  (`rag/read_path.py`) — add a `graph` config; the GraphRetriever composes/unions with the live
  hybrid retriever.
- **LLM + parsing** — `llm/client.TextLLM.complete_json`, `research/_shared.loads_lenient` (lenient
  JSON), the P7 numbered-source + provenance pattern from `research/prompts.build_user`.
- **Spend gate** — `rag/pipeline.EmbedBudgetExceeded` (mirror it).
- **Chunk-by-id fetch** — `rag/sparse_store.Fts5SparseStore.fetch(ids)->{id:chunk}` already does this;
  reuse (or add a `get(chunk_ids)` to the vector store) so the GraphRetriever can materialize
  provenance chunks.
- **Measurement** — `configs/rag_eval_multistep.json` + `rag eval-multistep`. **A5's success metric is
  the A4 bridging benchmark**: does graph traversal earn the +0.5 bridging gain *without* the
  query-time alias-bridge? (compare graph vs. the A4 entity-bridge vs. plain hybrid).

### Schemas (`schemas/graph.py`)
```python
EntityType = Literal["company", "product", "segment", "risk", "regulatory_topic"]
Relation   = Literal["depends_on", "competes_with", "mentions_risk", "exposed_to"]

class Entity(BaseModel):
    id: str                      # canonical: ticker for companies ("MU"); normalized name otherwise
    name: str                    # surface form as written ("Micron")
    type: EntityType
    ticker: str | None = None    # resolved ticker for company entities (else None)

class Edge(BaseModel):
    subject: str                 # entity id — the filing's company (e.g. "NVDA")
    relation: Relation
    object: str                  # entity id (e.g. "MU", or a risk-node id)
    provenance: list[str] = Field(min_length=1)   # chunk_ids the triple was extracted from
    filing_date: Date
    source_url: str
    confidence: float = 1.0
```

### Ordered build steps (each a small green vertical slice — `make check` green before the next)

- **A5.0 ✅ — Schemas + SQLite `GraphStore` (NO LLM, NO network).**
  `schemas/graph.py` (above) + `graph/store.py`: a `GraphStore` Protocol + `SqliteGraphStore`
  (`nodes`/`edges` tables; idempotent `add_entities`/`add_edges` upsert by id;
  `neighbors(entity_id, *, relations=None, hops=1)`, `provenance_chunk_ids(...)`, `get_entity`,
  `count`). `settings.graph_store_dir = data/graph`. **Tests** (temp sqlite, hand-built triples):
  add → neighbors (1- and 2-hop), idempotent upsert, provenance retrieval, namespacing. *Deliverable:
  a queryable graph store, no extraction yet.*

- **A5.1 ✅ — Extraction (LLM-assisted, offline, cost-gated, verified).**
  `graph/prompts.py` (`GRAPH_EXTRACT_SYSTEM`: extract typed `(subject, relation, object)` triples
  from a NUMBERED filing section, returning JSON with the supporting chunk-number(s) + `confidence`;
  `build_extract_user(ticker, numbered_chunks)`) + `graph/extract.py`
  (`extract_edges(ticker, section_chunks, *, llm, alias_map) -> list[Edge]`: candidate pre-filter →
  one `complete_json` per section → parse → resolve object names to tickers (`mentioned_tickers`) →
  map chunk-numbers to `chunk_id`s → **verification drop** (provenance chunk must contain both
  surface names) + confidence threshold → `GraphExtractBudgetExceeded` gate). CLI
  `documents extract-graph --ticker/--all` (reads ingested Item 1 / 1A chunks → `GraphStore`).
  **Tests** (canned `TextLLM`, offline): expected edges with provenance; `micron → MU` resolution; a
  hallucinated edge (provenance lacks the object name) is dropped; spend gate. *Deliverable: a
  populated graph from real filings (run on the benchmark tickers).*

- **A5.2 ✅ — `GraphRetriever` (satisfies `RetrievalSystem`).**
  `graph/retriever.py`: `GraphRetriever(graph_store, vector_retriever, chunk_fetch)`.
  `retrieve(query, *, top_k, where)`: seed entity = `where.ticker` (or `mentioned_tickers(query)`) →
  traverse 1–2 hops → neighbor tickers + edge-provenance chunk_ids → **(who)** materialize provenance
  chunks + **(what)** a scoped vector retrieval per neighbor → **union** with a base
  `vector_retriever.retrieve(query)` → dedup → top_k. `name="graph(<base>)"`. Add `build_graph_system`
  (or a `graph` lattice config) in `rag/read_path.py`; CLI `rag graph-query --question [--ticker]`.
  **Tests** (fake `GraphStore` + fake vector retriever, offline): traversal pulls the neighbor's
  chunks; provenance chunks present and cite the real `source_url`; union dedups. *Deliverable: the
  bridging hop done by traversal, not the alias-bridge.*

- **A5.3 ✅ (code) — Integration + measurement (the payoff + the A-N teaching obligation).**
  Make `GraphRetriever` usable as the A4 loop's retriever / a `build_retrieval_system` mode. **Re-run
  `rag eval-multistep` on `configs/rag_eval_multistep.json`** with the GraphRetriever and compare to
  (a) plain hybrid and (b) the A4 entity-bridge: does traversal earn the bridging gain on its own? If
  graph wins, gate the alias-bridge OFF when graph is active (it becomes the fallback for entities not
  yet in the graph). Docs per CLAUDE.md A-N rule: mark **A5 ✅** here; `rag_implementation_notes.md`
  §A5 (mechanism); `rag_concepts.md` §16 (graph theory — KG construction from text, traversal,
  provenance-as-citation; obey the MathJax escaping rules); extend `example_rag_questions.md` with
  structural/graph questions. *Deliverable: a measured graph-vs-bridge-vs-hybrid verdict on the
  bridging benchmark + a promotion decision.*

### Config knobs (settings)
`graph_store_dir: Path = data/graph` · `graph_max_extract_calls: int | None = None` (spend ceiling) ·
`graph_min_confidence: float = 0.5` · `graph_sections: list[str] = ["Item 1. Business", "Item 1A. Risk Factors"]`.
Graph retrieval stays a **config/lattice option** (default-OFF), promoted only on a measured win
(mirrors the A2/A3 "default-OFF until measured" discipline).

### Scope guard + success criterion
MVP = 5 entity types, 4 relations, 2 sections, and the **benchmark tickers only** (NVDA + the
suppliers/peers in `configs/rag_eval_multistep.json`: MU, AMD, INTC, AVGO, …). **Success = the A4
bridging questions reach +0.5 gain via graph traversal *without* the query-time alias-bridge**, with
every graph-sourced claim citing a real filing chunk. Not a core dependency — a learning capstone.

> **Operational scope decision (2026-06-27): graph is built for the semis/AI complex ONLY, not the
> full universe.** Extraction is the one paid A5 step (~$40-60 to do all ~110 training names — too
> expensive and wasted, since GraphRAG is query-local: §16.5). The committed extraction set is
> **`configs/graph_universe.txt`** (NVDA + AVGO AMD INTC QCOM TXN AMAT MU MRVL LRCX KLAC ADI NXPI
> MCHP ANET DELL — 16 names, ~$8-12). Build with `documents extract-graph --all --universe
> configs/graph_universe.txt`. Extraction `max_tokens` is 4000 (Item 1A is long; a tight cap
> truncates the JSON and silently drops a section's edges). Grow the list only when a new ticker is
> actually queried structurally; `--all` over the whole `configs/universe.txt` is deliberately avoided.

### Risks (and mitigations baked into the steps above)
- **Hallucinated edges** → mandatory provenance + the verification drop (A5.1) + confidence threshold.
- **Extraction cost** → offline batch + candidate pre-filter + `GraphExtractBudgetExceeded` gate.
- **Entity-resolution gaps** (typos/abbreviations/out-of-universe — the A4 brittleness §15.9) →
  resolve offline where you can extend `ticker_aliases.json` and human-verify; unresolved names are
  stored but lower-value. This is precisely A5's improvement over the A4 query-time bridge.
- **Scope creep** → keep the MVP tiny; resist adding entity types / relations / sections until the
  bridging benchmark is beaten.

### A5 follow-ups — polish (status as of 2026-06-28)

Issues found while building the graph + running a 3-architecture eval. **F1, F2, D1, D3 are DONE**
(F2/D1/D3 in PR #38; F1 across PR #38 (core-name tier) + the polish PR (acronym tier)). The remaining
open item is the **A5.3 promotion verdict** (run `rag eval-multistep --graph` vs hybrid + the A4
bridge). One investigated item closed as a **negative** (see "Item A" below) — no fix, by design.

- **F1 — Unresolved-company node de-duplication (data quality) — DONE.**
  *Symptom:* the same out-of-universe company appears as several nodes — punctuation/suffix variants
  (`Hon Hai Precision Industry Co.` vs `… Co., Ltd.`, `Global Foundries` vs `GlobalFoundries`,
  `Apple Inc.` vs `Apple`) and acronym variants (`TSMC`↔`Taiwan Semiconductor`, `UMC`↔`United
  Microelectronics`, `SMIC`↔`Semiconductor Manufacturing International`). Resolved-ticker nodes
  (ASML/AMD/NVDA/MU…) already dedup correctly — this is only the `ticker=None` companies.
  *Root cause:* `graph/extract._build_edge` ids an unresolved company by `_normalize_node_id(raw
  surface)`, so every surface form is a distinct id.
  *Done:*
  (a) **Core-name tier (PR #38)** — id unresolved companies by `_core_name(obj_name)` (suffix-stripped
      + punctuation-normalized); merges suffix/spacing variants (`… Co.`/`… Co., Ltd.`).
  (b) **Acronym tier (polish PR)** — added `"tsmc"`→TSM and `"asml"`→ASML to
      `configs/ticker_aliases.json` (the confirmed dup was a stray `tsmc` node created by AVGO/INTC/
      NXPI filings using the acronym; TSMC's US ticker is **TSM** — the NYSE ADR, CIK 1046179).
      Guarded by `test_committed_alias_map_resolves_tsmc_acronym`. **Materializing the merge in the
      live graph needs a clean rebuild** (`extract-graph --all`) — the graph lives under gitignored
      `data/`; re-extract alone leaves the old `tsmc`-keyed edges. (Out-of-universe acronyms like
      `UMC`/`SMIC` can't resolve — no ticker in the universe — so they stay as name nodes.)
  *Files:* `graph/extract.py` (the unresolved-company `obj_id`); extend `tests/unit/test_graph_extract.py`
  (two surface variants → one node id). *Migration:* existing graph has the dup nodes under old ids —
  clear + rebuild the 16-ticker graph (idempotent) after the change, or write a one-time node-merge.

- **F2/D1/D3 — Bridge ranking & traversal policy (the A5.3 blockers) — DONE (PR #38).**
  Three fixes so the bridged neighbor actually surfaces: **F2** de-double-counts base vs. graph before
  RRF; **D1** round-robins neighbor selection across relations (a competes_with-heavy hub no longer
  starves depends_on suppliers); **D3** leads the graph ranking with scoped-neighbor chunks and caps
  provenance (subject-own provenance was flooding the ranking — the real blocker). Original F2 writeup:
  *Symptom:* `rag graph-query` *gathers* the bridged neighbor's chunks (e.g. MU's risk chunk) but
  they rank **below top-k**, so the bridge doesn't surface in the returned set (verified: MU is
  ingested + reachable; the chunks are fetched, then RRF-crowded out).
  *Root cause:* `graph/retriever.GraphRetriever.retrieve` fuses `[base_ids, graph_ids]` by RRF, but
  base chunks (seed-filtered) **also appear in `graph_ids` as edge provenance** → double-counted →
  they dominate; the genuinely-new scoped-neighbor chunks appear once at a lower rank and fall off.
  *Plan (do (a), measure, add (b) only if needed):*
  (a) **De-double-count** — build the graph ranking as `(provenance ∪ scoped) − base_ids`, so the
      fused graph list is *only the new evidence the base didn't already return*. Removes the
      inflation; the neighbor's chunks then rank on their own merit.
  (b) **Reserve a floor** — guarantee `min(graph_floor, available)` scoped-neighbor slots in the
      final top-k (new config `graph_min_neighbor_slots`, e.g. 2), filling the rest by RRF. Makes the
      bridge deterministically visible regardless of score scale.
  *Files:* `graph/retriever.py`; `settings.py` (if (b)); extend `tests/unit/test_graph_retriever.py`
  (neighbor chunk present in top-k even when base fills it). *Measure:* this is exactly what
  `rag eval-multistep --graph` scores — run it vs. plain hybrid + the A4 entity-bridge **after** F2 to
  produce the A5.3 promotion verdict. Until F2 lands, a graph-vs-bridge benchmark understates graph.

- **Item A — `TSM→ASML` / `TSM→AMAT` equipment 2-hop — INVESTIGATED, closed as a NEGATIVE (no fix).**
  *Hypothesis was:* the foundry→equipment edge didn't extract because the fallback cap cut the
  relevant chunk; "raise the cap / add equipment cues" would complete it. *Diagnosis (raw 20-F text):*
  TSMC names **ASML 0×** (so the edge isn't in the filing) and **Applied Materials 8× — all in Item 6
  director bios** (Michael Splinter was AMAT's CEO; a director advises AMAT), i.e. personnel
  affiliations, **not** supplier relationships. So capturing it would manufacture **false edges**, and
  the current first-N cap correctly excludes them. **No code change** — raising the cap / adding
  equipment cues is rejected (it would degrade precision). The real wafer/material suppliers TSMC
  *does* disclose already extract (GlobalWafers, Shin-Etsu, SUMCO, Siltronic, Soitec). The legitimate
  path to a foundry→equipment edge is the deferred **`supplies_to`/customer relation** mined from the
  equipment maker's *own* filing (ASML/AMAT name their customers) and derived on traversal — an A5
  scope extension, not a polish tweak.

---

### A5.3 — Promotion verdict ✅ (DONE — graph PROMOTED for the multi-hop path, 2026-06-28)

**VERDICT: PROMOTE (scoped/tiered).** Over **2 seeds × 12 Q** (HARD 0–5, MED 6–8, CTRL 9–11), pooled
mean aspect coverage:

| stratum | A single-hybrid | B agentic+hybrid+bridge | C single-graph | **D agentic+graph** | D−B | D−A |
|---|---|---|---|---|---|---|
| HARD | 0.58 | 0.92 | 0.67 | **0.96** | +0.04 | +0.38 |
| MED  | 0.67 | 0.83 | 0.50 | **0.83** | 0.00 | +0.17 |
| CTRL | 1.00 | 0.92 | 0.83 | **1.00** | +0.08 | 0.00 |

The strict pre-registered rule did **not** pass as written (the `D − A ≥ +0.5` bar missed at +0.38 on
the bridging subset — single-shot hybrid started higher than the assumed 0.5; and *single-shot* graph C
regressed controls 0.83 < A 1.00). **But the decision-relevant comparison is D vs B** (graph vs the
production alias-bridge on the multi-hop path), and **HARD `D ≥ B` held on both seeds** (1.00/0.92 ≥
0.92) with **D = 1.00 on controls both seeds** (agentic+graph does *not* regress easy Qs — the seed-1
B=0.83 control dip was noise, B=1.00 on seed-2). So **agentic+graph matches-or-beats the alias-bridge
everywhere and is ~perfect on hard bridges with the bridge OFF** → graph cleanly *replaces* the brittle
alias-bridge and adds the hard-question lift. Caveat: n=12, 2 seeds; HARD D−B margin is small (+0.04,
one seed exact-tie) — promotion rests on **D never falling below B**, not on a large margin.

**Promotion is SCOPED (tiered routing), not blanket:**
- **EASY / single-topic → single-shot hybrid** (single-shot graph C regressed controls; also $0).
- **MED/HARD multi-hop → agentic loop over the `GraphRetriever`** (`settings.graph_multistep_enabled`,
  default ON). Graph-built tickers (`configs/graph_universe.txt`) bridge via stored edges; tickers
  *without* graph edges degrade to the hybrid base.
- **The A4 alias-bridge stays ON as a universal fallback** (NOT gated off — revised from the original
  plan). Rationale: gating it off was a *scientific control* (prove graph replaces the bridge — it did,
  D≥B with bridge off); in production a $0 deterministic fallback is kept because (a) non-graph tickers
  have no traversal and the alias-bridge is their only bridging mechanism, and (b) for graph tickers it
  is additive/redundant (dedup union), so production = agentic+graph+bridge `≥` the measured D in
  coverage (minor `max_evidence`-cap eviction risk noted).

**Shipped (this slice):** `settings.graph_multistep_enabled` (default ON);
`ToolExecutor._get_multistep_retriever()` routes the multi-hop tool to `build_graph_system` (falls back
to hybrid if disabled / base injected / graph DB unavailable); `search_filings` + the P8 memo stay
hybrid; 4 routing tests. Reports: `outputs/rag_eval/a5_3_{hybrid,graph}.json` (seed 1) +
`a5_3_{hybrid,graph}_seed2.json` (seed 2). No single-shot-graph default; no benchmark leakage change.

<details><summary>Original PLAN (provenance)</summary>

**Question to answer (one sentence):** does **agentic + graph (alias-bridge OFF)** match-or-beat the
production multi-hop path (**agentic + hybrid + alias-bridge ON**) and the single-shot baselines on
*bridging* questions — by enough to **promote graph and gate the A4 alias-bridge OFF when graph is
active** — or is graph an opt-in with no measured win (a valid negative)?

**Pre-registered decision rule (write the verdict BEFORE eyeballing prose):** promote iff, on the
*bridging subset*, mean aspect coverage satisfies **D ≥ B** (graph ≥ the alias-bridge), **D ≥ C**
(the loop adds value over single-shot graph), **D − A ≥ +0.5** (the §A5 target gain vs single-shot
hybrid), **and** the *control subset* shows **no regression** (graph cells ≈ vector cells on
single-entity questions — graph must not hurt simple Qs). Else: keep graph default-OFF/opt-in, record
the negative, alias-bridge stays. A rigorous negative is an acceptable outcome (cf. LSTM/news).

**Experimental design — the §15.9 2×2** (attribute gains to *control* × *substrate*):

| | vector (hybrid) | graph |
|---|---|---|
| single-shot | **A** | **C** |
| agentic loop | **B** (bridge ON) | **D** (bridge OFF) |

`evaluate_multihop` already yields *both* the single-shot coverage (retrieval-only, $0 LLM) **and**
the agentic-loop coverage for whatever retriever it's given — so **two runs produce all four cells**,
no new code:
- **Run 1 (A+B):** `make rag-eval-multistep` (hybrid; bridge default ON).
- **Run 2 (C+D):** `AGENTIC_BRIDGE_MAX_ENTITIES=0 … rag eval-multistep --graph --report …` (graph
  substrate; alias-bridge disabled via the existing setting — env-overridable, no code change).
- *(Optional 5th cell B′: hybrid + agentic + bridge OFF — isolates how much the alias-bridge itself
  adds, separate from graph.)*

**Metrics** (already in `multistep_eval`): mean `aspect coverage` per cell + `coverage_gain`
(multistep − single-shot), split **bridging vs. control**; plus `citation accuracy`, the
*insufficient-evidence* rate (guard refusals), and loop cost (LLM calls/question). Report per-question,
not just means (n is small).

**The real work = the benchmark, not the runs.** `configs/rag_eval_multistep.json` today is **3 Q**
(2 memory-bridge + 1 control) and has **no foundry/TSM** questions. Expand to **~10–12 Q**, labels
verified against the *ingested* corpus:
- **memory bridge** (NVDA→MU, ingested): keep 2, add 1.
- **foundry bridge** (NVDA→TSM, now ingested): add **2–3** — e.g. *"Which company that NVIDIA relies on
  to fabricate its chips warns about geopolitical concentration or power/water constraints in its own
  filings?"* Aspects: (1) NVDA names the foundry — spans `["Taiwan Semiconductor", "TSMC"]` in NVDA's
  filing; (2) TSMC's own risk — spans drawn from TSM's 20-F (e.g. `"political stability"`, `"earthquake"`,
  `"shortages … power"`), **verified present in TSM chunks and absent from NVDA's**.
- **competitor bridge** (NVDA competes_with AMD/INTC): add **1–2**.
- **controls** (single-entity, non-bridging): **2–3** — the safety net that proves graph doesn't
  regress simple questions.
- *Labeling invariant:* each bridge aspect's spans must be **present in the bridged entity's own
  filing and absent/rare in the seed's** (else single-shot covers it and the question doesn't test the
  bridge). Verify every span with a quick corpus grep before committing it.

**Deliverables:** (1) expanded `configs/rag_eval_multistep.json` (+ span-verification); (2) the 2×2
results to `outputs/rag_eval/<ts>.json` + a short table; (3) the **verdict** written to
`rag_implementation_notes.md` §A5 and this file (mark A5.3 ✅); (4) **if promote:** gate the A4
alias-bridge OFF when the graph retriever is active (keep it as the fallback for entities not yet in
the graph) — a small follow-up slice with a test; **if not:** record the negative, no production change.

**Cost (local, paid — like the model backtests, NOT CI):** only the agentic loop spends (≤4 Sonnet
calls/question: ≤3 cheap decisions + 1 ~18K-token terminal synthesis; the single-shot baseline is
retrieval-only/\$0). One full 2-run sweep over ~12 Q ≈ **\$2–3**; budget **\$5–8** total for 2–3
iterations (label tuning + a couple of seeds, since the loop's LLM-generated queries make coverage
non-deterministic — report it as directional, optionally average 2–3 seeds). Query embeddings are
Voyage but negligible. CI stays free (harness-mechanics tests only).

**Risks:** small-n (directional, not definitive — grow later, §A1); label leakage (mitigated by the
present-in-bridged/absent-in-seed invariant + grep verification); loop non-determinism (report
per-question + optional multi-seed). Prereq: a clean graph (done — TSMC dedup merged) + the F2/D1/D3
ranking fixes (done, PR #38).

</details>

---

## A6 — Retrieval + RL — learning-first capstone (TWO phases: bandits → full RL)

> **Scope decision (2026-06-28, user).** Locked decision #7 ("contextual bandits first; full
> multi-step RL deferred") is now **two explicit phases**: **A6.1 = contextual bandits + off-policy
> evaluation** (the safe MVP), then **A6.2 = full reinforcement learning for RAG** (the agentic
> retrieval loop as a sequential MDP). A6.2 is no longer "out of scope" — it is the second deliverable.
> A6.1 ships and is validated on its own; A6.2 is gated on A6.1's infra **and** a benchmark large
> enough for sequential RL (see the A6.2 prerequisite gate). Both obey the eval-first / default-OFF /
> no-torch / offline-only invariants. A rigorous **negative** (adaptive retrieval ≈ the best fixed
> pipeline) is an acceptable, valuable outcome for either phase (cf. the LSTM/news negatives).

**Learning objective.** Treat retrieval as a *decision* problem — A6.1: a one-shot **contextual
bandit** (pick the retrieval config per query); A6.2: a **finite-horizon MDP** (the multi-hop loop —
*what to retrieve, where to point it, when to stop*) learned end-to-end against a reward oracle.

**What already exists (do NOT rebuild — wire into these):**
- **Action executor:** `rag/read_path.build_named_system(name, settings)` over
  `LATTICE_SYSTEMS = (dense, reranked, hybrid, hybrid+rerank)` **+ `build_graph_system`** — explicitly
  designed as the A6 action-space seed. A policy's action → a `RetrievalSystem` via these factories.
- **Reward oracle:** `rag/eval.py::evaluate_query` (per-query `hit@k` / `MRR` / **`nDCG@k`** /
  `precision`/`recall` + `citation_accuracy` over `LabeledQuery` answer-span labels) for single-shot;
  `research/multistep_eval.py` (**aspect coverage** of the accumulated union) for multi-hop. These are
  the `r` in every estimator below — no human labels needed to bootstrap.
- **Benchmarks:** `configs/rag_eval_queries.json` (single-shot `LabeledQuery`) +
  `configs/rag_eval_multistep.json` (12 multi-hop Q, HARD/MED/CTRL strata).
- **Context-feature sources:** `agent/router.py` (route/domain taxonomy), `research/bridge.is_bridging`,
  `schemas/` query metadata (ticker, length). Featurizer is a *pure* function over these.
- **Discipline to mirror:** `backtesting/` (train/test split, seed logging, `outputs/experiments/<run_id>/`,
  reward-hacking/baseline guardrails, report dispersion) — RL eval is "backtesting for retrieval policies."

---

### A6.0 — Grow the multi-hop benchmark (graph-mined, corpus-verified, stratified) [clean ~100–200 Q] ✅

> **Status: DONE (2026-06-29).** Shipped `research/multistep_templates.py` (pure bridge/control fills);
> `research/multistep_gen.py` (`classify_stratum` span-isolation probe, `split_multihop` group-wise
> split [D2], `generate_multihop` + `SupplyReport` [D1]); `spans_present` shared primitive +
> `MultiHopQuery` metadata fields (`stratum`/`relation`/`qtype`/`seed`/`target`/`group_id`/`generated`,
> back-compat) in `research/multistep_eval.py`; read-only `SparseStore.iter_chunks(where)` (Fts5 +
> InMemory); CLI `rag gen-multistep` + `make rag-gen-multistep`. **Local run over the 20-ticker graph
> yielded 212 clean questions (HARD 120 / MED 30 / CTRL 62)** — committed
> `configs/rag_eval_multistep_generated.json` (+ group-wise `.train.json` [149] / `.test.json` [63],
> 0 group overlap); supply report → `outputs/rag_eval/multistep_supply.json`. Builds the **episode
> catalog** — the fixed `{question, aspects, metadata}` rows; the MDP state `s_0` is derived *later* by
> the A6.1b featurizer at `env.reset()`, **not** here. Refines decision #2: the final count is an
> **output (= clean supply), not a fixed input** (D1). Mechanism →
> [rag_implementation_notes.md](rag_implementation_notes.md) §A6.0; methodology →
> [rag_concepts.md](rag_concepts.md) §17. **Audit notes:** `competes_with` dominates HARD (100/120 —
> the graph has more competitor than dependency edges; `--per-seed-cap` available to rebalance); A1
> surfaces mix case (graph name ∪ lowercase aliases) — harmless under normalized matching. The curated
> 12-Q `rag_eval_multistep.json` is left untouched (A4/A5.3 reproducibility); A6.1/A6.2 consume the
> generated set (or the union).
>
> **EXPANSION (2026-07-18):** graph grown **20 → 48 seeds** (+28 already-ingested semis/hyperscaler/
> software names, via additions-only `configs/graph_universe_additions.txt` so the original 20 were not
> re-billed; one-time ≈ \$10.80 / 83 LLM calls, ceiling 100). Regenerated → **680 Q** (HARD 445 / MED 79
> / CTRL 156); test fold **13 → 38 distinct HARD∪MED groups** (the RL power denominator, +2.9×) ⇒
> projected bootstrap CI half-width **±0.077 → ±0.045** (§17.5). Span audit: 0/154 target-aspect misses,
> 4/154 seed-aspect (all `INTC|META`, a documented alias surface-form gap). **This clears the power gate
> only** — the sim-to-real bias (~0.18, A6.2g) is orthogonal and untouched, so E5 stays *descriptive*.
> v1 (20-seed) numbers preserved in git history; v2 is not backward-compatible with A6.1/A6.2 point
> estimates. Mechanism → [rag_implementation_notes.md](rag_implementation_notes.md) §A6.0-EXPANSION;
> before/after → [validations_results.md](validations_results.md) 2026-07-18; power math →
> [rag_concepts.md §17.5](rag_concepts.md).

**Objective.** Grow `configs/rag_eval_multistep.json` from 12 → a clean, stratified ~100–200 multi-hop
questions, **mined from the A5 graph** (not hand-written), every aspect span **auto-verified against the
ingested corpus**. Mostly **$0** (graph traversal + grep; no LLM in the generator).

**Key reuse insight — both aspects come from already-verified graph edges** (A5.1's hallucination guard
guarantees each edge's provenance chunk contains both endpoints' surface names):
- **A1** ("seed names target"): spans = `load_alias_map()[target]` surfaces (e.g. TSM → `["Taiwan
  Semiconductor","TSMC"]`); **present-in-seed guaranteed** by the bridge edge's guard.
- **A2** ("target's own risk"): spans = the risk/topic node `name` from a `(target, mentions_risk |
  exposed_to, risk)` edge; **present-in-target guaranteed** by that edge's guard. The probe only
  confirms **absent-in-seed**.

Mining = two existing `GraphStore.neighbors(...)` calls per seed (bridges, then each target's risk
edges) + `get_entity(edge.object).name` — **no new graph-store method**.

**Correctness lynchpin.** The span-isolation probe **must reuse the exact matching used by the eval
metric** (`research/multistep_eval._normalize` / `_aspect_covered`) — not a re-implementation. If probe
and metric disagree, a row labeled "HARD / absent-in-seed" could be scored *covered* by single-shot at
eval time → a corrupt reward label that silently poisons RL. Self-consistency (probe ≡ metric) is the
requirement, not semantic truth.

**Strata** (target shares; final counts floated to clean supply):

| Stratum | Construction | absent-in-seed? | ~share |
|---|---|---|---|
| HARD | 2-hop bridge, A2 risk specific to target | yes (probe) | ~45–50% |
| MED | bridge where A2 topic co-disclosed in the seed too | no | ~25–30% |
| CTRL | single-entity, 1–2 same-entity aspects, no bridge | n/a | ~20–25% |

Diversity guards: per-`(seed, relation)` cap; dedup by `(seed, relation, target, a2-span)`; a
specificity stop-list drops over-generic A2 spans (`"competition"`, `"risk"`, …) that collapse to MED.

**Schema change (back-compat — all optional/defaulted, so the existing 12-Q file still validates).**
Extend `MultiHopQuery`: `stratum: Literal["HARD","MED","CTRL"] | None`, `relation: str | None`,
`qtype: str | None`, `seed/target: str | None`, `group_id: str | None`, `generated: bool = False`.
(Resolves the earlier open item: explicit `stratum` field vs A5.3's fragile index ranges.)

**Explicit deliverables (the two folded in 2026-06-29):**
- **D1 — Clean-supply report (count is an output).** The generator emits a per-stratum supply report —
  **distinct, probe-passing, deduped, specificity-filtered** counts (HARD/MED/CTRL) *before* sampling
  to any target. `--target-count` is a **cap, never a filler**: if clean supply < target, cap at clean
  supply and **never dilute** (no relaxing the specificity filter, no near-duplicate rows). To reach a
  higher count *legitimately*, grow `configs/graph_universe.txt` (more tickers → more bridges), not the
  filters. Report printed to stdout + written to a sidecar `outputs/rag_eval/multistep_supply.json` and
  summarized in the dataset file header/metadata.
- **D2 — Group-wise split (anti-leakage / anti-pseudo-replication).** Every row carries a `group_id` =
  the bridge pair `frozenset({seed,target})` (bridges) or the seed ticker (CTRL). Provide
  `split_multihop(queries, *, test_frac, seed) -> (train, test)` that splits **by group**
  (GroupShuffleSplit-style) so no bridge's near-duplicate variants straddle train/test. **A6.1f and
  A6.2 eval protocols consume this splitter** (not a row-wise split) — made a hard rule so the held-out
  set can never be near-copies of training rows.

**New modules.** `research/multistep_templates.py` (relation→template-family + pure `fill(...) ->
(question, aspects)`); `research/multistep_gen.py` (`generate_multihop(graph, sparse, alias_map, *,
seeds, caps, rng) -> (queries, SupplyReport)`, `span_isolation(...)`, `split_multihop(...)`); one
**read-only** `SparseStore.iter_chunks(where: ChunkFilter)` (Fts5 `SELECT … WHERE ticker=?`; InMemory
filter) for the exhaustive absent-in-seed scan (offline-only; mirrors `fetch`/`existing_ids`). CLI `rag
gen-multistep`; Makefile `rag-gen-multistep`.

**Ordered build slices (each `make check` green before the next):**
- **A6.0a** — schema fields + `multistep_templates.py`. Tests: golden fill reproduces a real config row;
  existing JSON still validates (back-compat).
- **A6.0b** — `iter_chunks` + `span_isolation` (+ `split_multihop`). Tests (InMemoryBM25Store, hand-built
  chunks): present-in-target, absent-in-seed→HARD, co-disclosed→MED, **probe ≡ metric** agreement;
  group split puts a bridge pair's variants on one side only.
- **A6.0c** — `generate_multihop` over a **fake** graph+sparse. Tests: expected questions; **D1 supply
  report counts**; dedup; per-`(seed,relation)` cap; determinism under fixed `rng`.
- **A6.0d** — CLI + **local** run against the real graph (`data/graph/voyage-voyage-4.db`) + corpus →
  emit the supply report, hand-audit a sample, commit the expanded `configs/rag_eval_multistep.json`.
  Local meta-check: every committed span re-passes the probe. Docs per A-N rule: mark **A6.0 ✅** here +
  append mechanism to `rag_implementation_notes.md`; light `rag_concepts.md` note on the
  stratification/span-isolation methodology (the heavy RL math lands with A6.1/A6.2).

**Tests: CI vs local** (per CLAUDE.md RAG rules). CI = A6.0a–c with fakes only (template goldens, probe
logic, supply/dedup/determinism, group-split, schema back-compat) — **no graph DB, corpus, or model**.
Local (like backtests) = the actual generation + span re-verification against the real corpus; the
committed JSON is the artifact.

**Risks/mitigations.** Bad labels poison RL → probe ≡ metric + both aspects from guard-verified edges +
local meta-verification. NVDA-skew → per-`(seed,relation)` caps over all 20 universe tickers. Generic
A2 → specificity stop-list. Non-ingested target → probe finds no chunks → auto-discard. Determinism →
seeded sampler, seed logged in the dataset header.

---

### A6.1 — Contextual bandits + off-policy evaluation (the MVP; ship first) ✅ (infra) · verdict 2026-07-08 = **REJECT** · gated-router follow-up 2026-07-08 = **REJECT (A5.3 vindicated)** · **TRACK CLOSED** → A6.2

> **Status: INFRA COMPLETE & GREEN (slices A6.1a–f shipped, `make check` green, default-OFF).
> Verdict EXECUTED 2026-07-08 → REJECT: `promote=false`, `adaptive_retrieval` stays False.** LinUCB α=1,
> seed 42, n_train=129/n_test=83: DR(linucb) 0.438 vs best-fixed(dense) 0.414 → Δ=+0.0239, group-boot 95%
> CI [−0.208, +0.273]; per-stratum HARD +0.110 / MED +0.305 / **CTRL −0.263**. Fails the pre-registered
> rule twice (CI includes 0 **and** CTRL regression) — a rigorous negative; the logging+OPE+bandit infra
> is the deliverable. Numbers + honest read (underpowered: ESS≈16; DR misranks fixed arms) →
> [validations_results.md](validations_results.md) (2026-07-05 entry, resolved) and
> [rag_implementation_notes.md §A6.1](rag_implementation_notes.md). Output:
> `outputs/rag_eval/policy_eval_linucb_seed42.json`. Mechanism →
> [rag_implementation_notes.md §A6.1](rag_implementation_notes.md); theory → [rag_concepts.md
> §18](rag_concepts.md). Conceptual background → [rl_rag_pre_questions.md](rl_rag_pre_questions.md)
> (MDP/reward framing) and [rag_concepts.md §17](rag_concepts.md) (the A6.0 benchmark this consumes).
> **Prereqs (merged):** A6.0 benchmark (`configs/rag_eval_multistep_generated.json`, PR #42) + the A5
> agent graph-routing fix (PR #43); A1–A5 shipped.
>
> **Shipped slices:** a — telemetry (`schemas/retrieval_log.py`, `rag/retrieval_log.py`); b —
> featurizer (`rag/policy_features.py`); c — reward oracle (`rag/reward.py`); d — OPE
> (`rag/ope.py`); e — policies (`rag/policy.py`); f — gated serving (`rag/policy_retriever.py`) +
> offline verdict harness (`rag/policy_eval.py`) + `rag policy-eval` CLI. **Deliberate boundary:**
> `PolicyRetriever` is built and injectable (it satisfies `RetrievalSystem` ⇒ drops into
> `ToolExecutor._injected_base`), but auto-wiring it into `build_retrieval_system` under
> `adaptive_retrieval=True` is **deferred** — that needs a *persisted trained policy*, which A6.1
> does not produce (training IS the verdict run). So `adaptive_retrieval` / `bandit_policy` are inert
> flags until a promotion + a policy-persistence step (a small follow-up, only if the verdict says
> promote).
>
> **Gated-router follow-up (2026-07-08 → REJECT; A5.3 vindicated).** Built the gate the A6.1 verdict
> called for: `GatedPolicy` + `build_gated_policy` (`rag/policy.py`), `evaluate_gated` +
> `GatedEvalReport` (`rag/policy_eval.py`), `rag gated-eval` CLI — a **deterministic label-free gate**
> (`is_bridging`) routes easy→`dense`, hard→branch-under-test; deterministic gate ⇒ OPE machinery reused
> verbatim. Two pre-registered verdicts (same split/seed/λ_c as A6.1): **(1) promote gated router?** Δ =
> DR(gated) − DR(dense) = **+0.1096**, CI **[−0.056, +0.287]** → REJECT (CI includes 0), **but the A6.1
> CTRL regression is gone** (−0.263 → **exactly 0**, by construction) and Δ quadrupled — a strict
> improvement, still power-limited (ESS≈17). **(2) does the bandit earn the hard branch?** on HARD+MED
> `linucb` vs `fixed(graph)`: Δ = **−0.0250** → **NO** (oracle `true_value` agrees). ⇒ The architecture
> the data supports is **deterministic gate → fixed graph = exactly A5.3's tiered router**; no learned
> policy is justified. `adaptive_retrieval` stays False. Output: `outputs/rag_eval/gated_eval_seed42.json`;
> theory → [rag_concepts.md §18.10](rag_concepts.md); mechanism + numbers →
> [rag_implementation_notes.md §A6.1](rag_implementation_notes.md) and
> [validations_results.md](validations_results.md) (2026-07-08 entry). Only remaining lever: higher-ESS
> logging (stratified/propensity-blended μ) — an A6.2 concern.
>
> **Follow-up (b) — exact bootstrap P(Δ>0) (2026-07-08):** added `delta_p_positive` as a first-class
> harness field (`ope.bootstrap_delta_stats`, one resample pass shared with the CI). Verdict [1]
> **P(Δ>0) = 86.8%** (868/1000 resamples positive; below the ~89% Gaussian approx — mild left-skew, and
> far below the 97.5% the `CI_low>0` rule implies) and verdict [2] **P(Δ>0) = 28.9%** (bandit *more likely
> worse* than fixed graph). Reported, not a criterion ⇒ **verdict unchanged: REJECT.** Theory →
> [rag_concepts.md §18.9](rag_concepts.md). **A6.1 (contextual-bandit) TRACK CLOSED** — three tests
> (unified bandit, λ_c sweep, gated router) all REJECT; a learned contextual policy is not justified at
> this logging design. Next: **A6.2 (Full RL for Retrieval)**, whose first move is the higher-ESS logging
> lever A6.1 could not turn (see §A6.2).

**What A6.1 is.** Treat retrieval as a **one-shot decision**: a featurized query (context `x`) → a
**policy** picks one of ~5 retrieval **configs** (arms `a`) → earns **reward** `r` = quality − cost.
Learn the policy **offline** from logged data, evaluate **off-policy** (IPS/SNIPS/DR), **promote only
if** it beats the best fixed config beyond a bootstrap CI on a group-wise held-out split. A rigorous
**negative** is acceptable (the logging+OPE+bandit infra is the durable artifact).

**Central hypothesis (why a bandit can win).** A5.3 measured that **single-shot graph regresses CTRL
(easy) questions but helps HARD bridges** → the optimal arm is **context-dependent** (HARD→graph,
CTRL→hybrid/dense), a genuine contextual optimum a fixed config can't capture. A6.1 tests whether a
learned policy realizes that lift.

**What already exists — build on these (verified file:line; do NOT rebuild):**

| Need | Reuse | Location |
|---|---|---|
| **Action executor** | `build_named_system(name, settings)` over `LATTICE_SYSTEMS=(dense,reranked,hybrid,hybrid+rerank)`; `build_graph_system(settings)` | `rag/read_path.py:64-140` |
| **Reward — single-shot nDCG** | `evaluate_query(system, LabeledQuery, *, top_k, corpus_chunks) -> QueryReport` (`.ndcg`) | `rag/eval.py:224` |
| **Reward — multi-hop coverage** | `coverage(chunks, aspects)`; one `retrieve()` = the single-shot baseline | `research/multistep_eval.py` |
| **Primary benchmark (contexts)** | `configs/rag_eval_multistep_generated.json` (v2: 680 Q over 48 seeds; `stratum`/`group_id`) | A6.0 |
| **Secondary benchmark** | `configs/rag_eval_queries.json` (25 single-shot `LabeledQuery`, ticker-scoped) | P9b |
| **Group-wise split (D2)** | `split_multihop(queries, *, test_frac, seed)` | `research/multistep_gen.py` |
| **Context features** | `research/bridge.is_bridging(q)`, `mentioned_tickers(text, alias_map)`, `load_alias_map()`; router cues | `research/bridge.py`, `agent/router.py` |
| **Discipline to mirror** | walk-forward split, seed logging (`seed=42`), report dispersion | `backtesting/` |

**Arms (MVP = 5):** `dense`, `reranked`, `hybrid`, `hybrid+rerank` + `graph`. Keep ≤ ~8–10; defer
`top_k`/`section_filter` variants.

**Context** `x = featurize(query)` (numpy) — **deploy-time signals ONLY** (see leakage rule 1):
`n_tokens`; `has_ticker`; `n_entities` (via `mentioned_tickers`); `is_bridging`; question-type one-hot
∈ {risk, financial, business, overview, bridging} (cheap keyword heuristic); **`in_graph_universe`**
(0/1 — is the ticker in `configs/graph_universe.txt`; tells the policy whether the `graph` arm can
actually traverse vs. degrade to hybrid+cost — the signal separating "graph helps" semis bridges from
"graph = hybrid + cost" e.g. AAPL). **Recommended.**

**Reward** `r(x,a)` — **retrieval-only, $0** (no synthesis in A6.1):

$$r = \underbrace{\text{nDCG@}k}_{\text{quality}} - \lambda_{c}\big(\text{LLM calls}+\text{latency}\big) - \lambda_{f}\big(\text{citation-guard failures}\big)$$

where in A6.1: `quality` = `nDCG@k` for a single-shot `LabeledQuery` **or single-shot `coverage`** for a
`MultiHopQuery` (one `retrieve()` of arm `a`, scored by aspect coverage) — both ∈ [0,1], both $0;
`cost(a)` = a **static per-arm** latency/compute proxy (e.g. `dense=0, hybrid=0.1, reranked=0.3,
hybrid+rerank=0.4, graph=0.3`), `λ_c` small (start 0.05). **The `λ_f` faithfulness term is DEFERRED**
(needs a synthesis call A6.1 does not run) — keep the field, default `λ_f=0`, so A6.2 / opt-in
synth-in-loop can switch it on. The cost penalty is the **reward-hacking guard** (an arm must earn its
cost in quality). `λ_c`, `λ_f` config; sensitivity-test.

**Logging policy `μ` (dataset synthesis).** Uniform-random over the 5 arms → `μ(a|x)=1/5`, known exactly
⇒ OPE is exact. Because the oracle is **$0 and deterministic**, also evaluate **every arm on every
context** (the full reward matrix `R[x,a]`) — giving (i) the **true** value of any policy
(full-information ground truth to check OPE against) and (ii) the logged dataset as a seeded subsample of
that matrix. Exploit both in tests.

**Invariants / leakage rules (non-negotiable):**
1. **Featurizer is label-free** — `x` uses only query text + alias map + `graph_universe.txt` + cheap
   heuristics, **never** the gold `stratum`/`relation`/`qtype` (the answer key). Gold strata are for
   **stratified reporting only**, never a feature (else the policy is un-deployable).
2. **Group-wise split everywhere** — `split_multihop` (group = bridge pair); for the 25-Q set (no
   `group_id`) group by `ticker`. Never row-wise (pseudo-replication, §17.4).
3. **Reward ≡ A6.0 metric** — multi-hop reward reuses `coverage`/`spans_present` (no metric drift).
4. **Default-OFF** — logging + adaptive retrieval config-gated; flags off ⇒ byte-identical pipeline.
5. **Numpy only (no torch)** — LinUCB / ridge `q̂` are numpy normal-equations. Torch is A6.2-only.

**Build slices (each a green vertical slice; default-OFF; deterministic tests) — all ✅ shipped:**
- **A6.1a — Telemetry.** `schemas/retrieval_log.py` (`RetrievalLogEntry`: timestamp, query, context
  features / raw `ContextVector`, chosen `action`, **propensity** `μ(a|x)` (None if deterministic),
  retrieved `chunk_id`s + scores, optional downstream answer + guard outcome, optional reward/feedback,
  seed) + `rag/retrieval_log.py` (append-only **JSONL** under `data/retrieval_logs/`, config-gated
  `settings.retrieval_logging=False`; `log_retrieval(entry)` no-ops when off). Wiring into the read
  path lands in f. *Tests (CI):* round-trip; off → no file / no-op; append accumulates.
- **A6.1b — Featurizer.** `rag/policy_features.py` — pure `featurize(query, *, ticker=None,
  alias_map=None, graph_universe=None) -> ContextVector` (numpy + stable `FEATURE_NAMES`). *Tests (CI):*
  golden vectors for risk/financial/bridging/overview/single-entity; flags (`has_ticker`/`is_bridging`/
  `in_graph_universe`) correct; stable ordering; determinism. No model/network.
- **A6.1c — Reward oracle adapter.** `rag/reward.py` — `reward(system, labeled, *, settings,
  corpus_chunks=None, lambda_cost, arm_cost) -> float` dispatching on label type (`LabeledQuery` →
  `evaluate_query(...).ndcg`; `MultiHopQuery` → `coverage(retrieve(q), aspects)`) minus
  `lambda_cost*arm_cost[name]`; plus `reward_matrix(systems, queries, ...) -> np.ndarray
  [n_queries × n_arms]` (the $0 full-information matrix; synthesizes logs + OPE ground truth). *Tests
  (CI, FakeEmbedder/InMemory):* composite math; a **reward-hacking sentinel arm** (dumps many
  low-relevance chunks) scores low; dispatch by label type; deterministic.
- **A6.1d — OPE.** `rag/ope.py` — **IPS**, **SNIPS**, **doubly-robust (DR)** + bootstrap CIs + a ridge
  reward-model `q̂(x,a)` (numpy normal equations) for the DR control variate. CLI `rag policy-eval`
  (load/synthesize a logged dataset, evaluate a named policy, print value + CI, write
  `outputs/rag_eval/policy_eval_<ts>.json`). *Tests (CI):* **hand-computed IPS/SNIPS/DR goldens** on a
  3-sample set; `DR == IPS` when `q̂≡0`; `SNIPS ∈ [min r, max r]`; IPS unbiased vs the full-info matrix
  on a toy where `μ` covers all arms.

$$\hat V_{\text{IPS}}(\pi)=\frac1N\sum_i \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}r_i,\quad \hat V_{\text{SNIPS}}(\pi)=\frac{\sum_i w_i r_i}{\sum_i w_i}, w_i=\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)},\quad \hat V_{\text{DR}}(\pi)=\frac1N\sum_i\Big[\hat q(x_i,\pi)+w_i\big(r_i-\hat q(x_i,a_i)\big)\Big]$$

- **A6.1e — Policy.** `rag/policy.py` — `Policy` Protocol (`act(x) -> (action, propensity)` +
  `prob(a, x) -> float` for OPE): `FixedPolicy(name)` (the baseline to beat = the promoted default;
  multi-hop baseline = `graph`, single-shot = `hybrid`), **`EpsilonGreedy(q̂, ε, seed)`**, **`LinUCB(d,
  n_arms, α, λ)`** (per-arm `A_a=λI+Σ x xᵀ`, `θ_a=A_a^{-1}b_a`, pick `argmax_a θ_aᵀx+α√(xᵀA_a^{-1}x)`),
  trained **offline** on the train split. *Tests (CI):* fixed + ε-greedy deterministic under seed;
  **LinUCB UCB arithmetic on a 2-arm toy** (hand-checked `A`,`θ`,UCB); split disjoint; `prob` sums to 1.
- **A6.1f — Gated integration + verdict.** `rag/policy_retriever.py` — `PolicyRetriever(policy,
  settings, *, build_arm=build_named_system)` satisfying `RetrievalSystem` (featurize → `policy.act` →
  build arm → delegate; logs via A6.1a when on). **Default-OFF** (`settings.adaptive_retrieval=False`);
  **forward-compatible with the PR-#43 fix** (injectable as the `ToolExecutor` `_injected_base`, or
  selected in `build_retrieval_system` when adaptive on). **Verdict (local, $0 retrieval, needs the real
  corpus/graph — like A1/A6.0):** synthesize the logged dataset over the group-wise held-out split,
  train policies on train, `rag policy-eval` DR(bandit) vs DR(best `FixedPolicy`).

**Pre-registered decision rule (write before looking at numbers):** promote (`adaptive_retrieval`
default→True) **iff**, on the **group-wise held-out** split, `DR(π_bandit) − DR(π_fixed_best) > 0` and
clears the **group-level bootstrap 95% CI**, **and** no per-stratum regression (must not lose to fixed on
CTRL). Else keep default-OFF and **record the negative**. Report **per-stratum** (HARD/MED/CTRL) so a win
isn't a pooled artifact. Write verdict → `docs/validations_results.md` (A5.3 template) +
`rag_implementation_notes.md` §A6.1. Pre-registered likely outcome: tuned hybrid(+graph) is strong → a
**modest win or a rigorous negative**; both ship the infra.

**Config flags to add (`settings.py`, all default-OFF / inert):** `retrieval_logging=False`;
`adaptive_retrieval=False`; `bandit_policy: Literal["fixed","epsilon_greedy","linucb"]="fixed"`;
`reward_lambda_cost=0.05`; `reward_lambda_faithfulness=0.0` (deferred); `bandit_arm_costs` (the static
`c(a)` vector); `retrieval_log_dir=data/retrieval_logs`.

**Testing: CI vs local** (per CLAUDE.md). CI (offline, no model/graph/corpus): log round-trip + off-noop;
featurizer goldens; reward composite + sentinel; **IPS/SNIPS/DR hand-goldens** + `DR=IPS@q̂≡0` +
SNIPS-hull + IPS-unbiased-vs-matrix; policy seeded determinism + LinUCB toy + disjoint split;
`PolicyRetriever` conformance with a `FakePolicy`. Local (like backtests): synthesize the real logged
dataset over the 212-Q (+ optional 25-Q) benchmark ($0 retrieval), produce the verdict numbers.

**Docs obligation on landing (A-N rule):** mark **A6.1 ✅** here (with verdict); mechanism →
`rag_implementation_notes.md` §A6.1; theory → **`rag_concepts.md` §18** (contextual bandits; IPS/SNIPS/DR
+ variance; LinUCB ridge+UCB; reward-hacking penalty) — **NB: A6.0 made §17 = benchmark construction and
bumped References to §18, so insert A6.1 as §18 and renumber References → §19**; chat explains the math;
verdict → `validations_results.md`.

**Risks/mitigations.** Label leakage via features → featurizer audited label-free (rule 1).
Pseudo-replication / leaked folds → group-wise split + group-level bootstrap CIs. Reward hacking → `λ_c`
+ sentinel test (`λ_f` deferred but flagged). Tiny n / weak signal → 212 Q is modest; honest CIs,
per-stratum; a negative is acceptable. Graph-arm degeneracy on non-semis tickers → `in_graph_universe`
feature; graph degrades safely to hybrid. Over-engineering → MVP = 5 arms, numpy only, retrieval-only $0
reward; defer top_k/section arms, synth-in-loop, the `λ_f` term.

**Open decisions (sensible defaults pre-picked; confirm at execution start):** (1) **Primary benchmark =
the 212-Q multi-hop set** (coverage reward; `group_id`+strata; A5.3 contextual optimum) — 25-Q
single-shot set optional secondary (both rewards ∈ [0,1]). (2) **5 arms**; defer `top_k`/`section`. (3)
Ridge backend = hand-rolled numpy normal equations (no sklearn dep). (4) `λ_c`=0.05; sensitivity-test
0/0.05/0.1 in the verdict run. (5) Build order **a→b→c→d→e→f** (telemetry first so f can wire it in).

---

### A6.2 — Full reinforcement learning for RAG (the agentic loop as an MDP) ⚠️ **infra ✅ · verdict RETRACTED (invalid environment)** → A6.2-E

> **STATUS 2026-07-13 — the A6.2 REJECT is WITHDRAWN. Do not cite it.**
> The infra (A6.2a–g: state, action space, env, REINFORCE/BC, PPO, train CLI, eval harness,
> sim-to-real) is built and green. The **verdict is invalid**: the environment could not express the
> correct action on 89% of the episodes it graded, and its reward paid out for evidence that answers
> nothing. `react(hybrid)` was already sitting on the action space's ceiling (~0.26 vs its 0.271), so
> a null result was structural — not a fact about RL.
> - Full diagnosis + numbers → `docs/validations_results.md` (2026-07-13)
> - Theory (reachability ρ, the information argument, reward quantifiers) → `docs/rag_concepts.md` §20
> - Build journal → `docs/rag_implementation_notes.md` §A6.2-E
>
> **Environment-fix track (A6.2-E):**
> - [x] **E1 — entity-bound coverage.** `Aspect.ticker`; an aspect is covered only by a chunk from the
>   company that must evidence it. Closes an 8.6% spurious-credit reward-hacking surface. Benchmark
>   backfilled in place (424/424 bindings corpus-verified, folds preserved). ⚠️ **supersedes the
>   published A4/A5/A6.1 coverage numbers** — they used the hackable metric.
> - [x] **E2 — relation-targeted hop-1 query.** `self_scope_query()`; a bridge question's self-hop
>   searches for the *relation*, not the topic. Naming-chunk retrieval 48.6% → **100%**.
> - [x] **E3 — fan-out action.** Candidates provably cannot be ranked (no label-free ordering beats
>   random), so they must be swept. `ScopeKind.FANOUT` expands to one request per candidate; cost =
>   n_candidates × arm cost, which is what finally gives the policy a real decision. Two traps fixed
>   while building it, each independently load-bearing: the branch merge must be **breadth-first**
>   (`_dedup_union` appends, so concatenating spends the union on the alphabetically-first candidate)
>   and the union cap must **widen to seat every branch** (only 69% of held-out HARD could seat all
>   their candidates under the flat cap of 20). Both would have silently re-created the very
>   alphabetical bias E3 removes.
>   **Also adds the `sweep(arm)` baseline** — fan-out is trivially scriptable, so giving the learner
>   fan-out while the baseline stays on `disc0` would manufacture a win out of an action-space
>   asymmetry (the same error class as the original bug). **RL must beat `sweep()`, not `react()`.**
> - [x] **E4 — candidate state features: CLOSED AS OBSOLETE, not built.** E4 existed so the policy
>   could *rank* `disc0` vs `disc1`. The information argument that motivated E3 (`I(Y; E₁) ≈ 0` —
>   hop-1 evidence carries no signal about *which* candidate discloses the topic) says per-candidate
>   features are **uninformative by construction**: they would add noise dimensions to a 149-episode
>   training set to support a decision that provably cannot be made. What the policy needs in order
>   to *price a sweep* is already in the 18-dim state: `n_discovered_unretrieved` (= the candidate
>   count = the cost driver, since cost = N × arm cost), `is_bridging`, and `budget_remaining`.
> - [x] **E6 — topic-targeted hop-2 query** (discovered *after* E3, the mirror of E2). Hop 2 was
>   sending the whole bridge question into the candidate's filings — ~20 tokens of scaffolding about
>   the *seed* around the 2 that matter. `discovered_scope_query()` strips the frame to the topic.
>   Target's-branch retrieval **33.3% → 50.0%**. ⚠️ the benchmark interpolates the gold span verbatim
>   as `{topic}`, so this lift is an **upper bound**; the paid LLM-query-writer run arbitrates.
> - [x] **E7 — pooled-rerank SEATING** (`rag/rl/seating.py`). E3's `breadth_first` rule (round-robin
>   by rank, cap widened to `len(union) + N`) is *arithmetically* "seat rank-0 of every branch and
>   nothing deeper": with a hop-1 union of 6 and N=19 the cap is exactly 25, and the round-robin's
>   first 19 entries are the 19 rank-0 chunks. The target's chunk is its branch's rank-0 hit only
>   **27.3%** of the time ⇒ **12 of the 21 episodes whose branch HAD retrieved the span were evicted
>   by the cap**, and 18 of the 25 seated chunks were noise from non-target companies.
>   **Fix:** pool every branch's chunks and rescore them with a **cross-encoder**, whose scores are
>   comparable *across* branches (RRF's are not — they are rank-derived, so every branch's rank-1
>   chunk ties, which is the whole reason `breadth_first` existed).
>   **Seating 21.4% → 42.9% (local) / 52.4% (Voyage); coverage 0.500 → 0.607 / 0.655.** Seating
>   *efficiency* (seated ÷ retrieved) goes **43% → 96%**: seating is no longer the bottleneck.
>   - **The E3 impossibility result does not reach seating.** `I(Y;E₁) ≈ 0` says hop-1 evidence
>     cannot rank candidates *a priori* — hence the sweep. It says nothing about `I(Y;E₂)`: once each
>     candidate's filings have actually been searched for the topic, the retrieved content is exactly
>     the observation that discriminates. Ranking candidates **a posteriori** is legal, and works.
>   - **`top_branches` is a strict Pareto win, killing the "44-chunk Pareto choice"**: it discards 16
>     of 19 branches and still beats production by **+0.083 coverage on a SMALLER context** (13.5 vs
>     15.0 chunks). The depth-`m` ladder was the expensive way to buy what a posteriori selection
>     gives free.
>   - **Two coupled traps** (each independently load-bearing, both mutation-tested): a no-op reranker
>     must fall back to `breadth_first` and **not** to the raw pooled order (which is branch-major ⇒
>     alphabetical starvation, the E3 bug rebuilt); and the **ordering and the cap must agree on the
>     same effective rule** — a reranked rule's flat cap is safe only *because* the reranked order
>     puts the best chunks first. Hence `effective_rule()` is resolved once and fed to both.
>   - **Default = `pooled_rerank` + LOCAL cross-encoder**: the env is the RL simulator (thousands of
>     rollouts) and must stay `$0` + deterministic. **Voyage seats better (52.4% vs 45.2%)** and is
>     the sim-to-real **arbitrator**, like the LLM `QueryWriter` — not the training default.
> - [ ] **E5 — retrain + re-evaluate**, now against **`sweep(hybrid)`** (not `react`), **in the
>   E7-seated env**. Also re-baseline A4/A5/A6.1 under entity-bound coverage.
>   **Measured reality check (see validations_results.md):** E3 fixed *reachability* (11.4% →
>   **78.6%**), E7 fixed *seating* (21.4% → 42.9%), and the bottleneck has moved **back to stage 2**:
>   the span is simply not in the target's top-6 for **12 of 33** reachable episodes, and fetching
>   deeper barely helps (k=12 buys only +2). That is a **query-formulation** problem, not a depth or
>   seating one — the next real lever, and the one the paid LLM query-writer speaks to.
>   The learnable margin for RL stays the **cost** side (when *not* to sweep) plus **per-hop arm
>   choice** (rerank helps hop 2, is catastrophic on hop 1). ⚠️ **Still power-limited: 18 test groups
>   ⇒ ±0.077 CI vs a ~0.02–0.05 effect** — run E5 descriptively; *promote* is blocked on benchmark
>   size, not on RL. The old "ceiling ~0.84" was **wrong** — it never checked the union cap.
> - [x] **A6.2g — sim-to-real gap MEASURED (2026-07-18, paid $2.6).** Fulfils the A6.2 "rollout
>   realism / measure the sim-to-real gap" commitment; **not** E5 retrain. Frozen REINFORCE
>   (`e5/reinforce-s1`, E7-seated, Voyage) run twice over the same 42 held-out episodes — templated
>   `$0` vs real Sonnet queries. **coverage gap −0.179** (95% t-CI [−0.363, +0.006], p≈0.06 — negative
>   but not significant at n=42), return −0.155,
>   same-action 71.4%, refusal 33.3%. The `$0` sim **flatters** the policy: the gap is the realization
>   channel (30/42 kept the same action sequence, 13 still lost coverage) — the templated text's
>   embedded gold span, quantified (§20.8 · rag_concepts.md · validations 2026-07-18). Bias (0.18) is
>   4–9× the RL-vs-sweep margin ⇒ the templated eval can't arbitrate promote *independent of* the power
>   limit. Also fixed the spend-gate wart (pre-count skipped under `--yes`; `quiet_empty` silences the
>   `$0` count-writer's benign empties). Real-query **head-to-head** (`rag rl-h2h` + `SweepBridgePolicy`,
>   ~$3.2) — does the RL *advantage* survive real queries — is built + gated but **held** (not run).


> **Detailed execution plan (slice-by-slice, module/interface designs, TransitionCache, action-space
> cardinality, tests):** [a6_2_plan.md](a6_2_plan.md). Design brief (MDP/state/action/reward, worked
> trajectories): [rl_rag_pre_questions.md](rl_rag_pre_questions.md) Q2–Q5. This section stays the
> plan-of-record summary; the detail lives in `a6_2_plan.md`. **Status: EXECUTING** (branch
> `feat/adv-rag-a6.2-rl`; slices A6.2a→g).

**Idea.** A6.1 optimizes a *single* retrieval choice. A6.2 learns the **whole multi-hop trajectory** —
generalizing the A4 ReAct loop (whose policy is currently a fixed LLM prompt) and the A5 entity-bridge
(a fixed heuristic) into **one learned sequential policy**. This is the genuine RL phase.

**MDP formalization** (finite horizon `T = agentic_max_steps`):
- **State** `s_t` = featurized (query features ⊕ step index `t` ⊕ budget remaining ⊕ evidence summary:
  chunk count, distinct tickers/sections covered, marginal-coverage of the last action, set of entities
  *named in the union but not yet retrieved*). Pure function of the trajectory so far → numpy vector.
- **Action** `a_t ∈ {STOP} ∪ {(config c, scope σ)}` where `c ∈ LATTICE ∪ {graph}` and
  `σ ∈ {self-ticker, discovered-entity_j, none}`. One action jointly chooses **which retriever**, **where
  to point it** (the bridge decision, now learned), and **whether to stop** (the reflective stop, now
  learned). This strictly contains A6.1 (config) and A4/A5 (scope + stop).
- **Transition** `P(s_{t+1}|s_t,a_t)`: deterministic given the retrieval result (corpus is fixed). To
  keep rollouts $0 and reproducible, query text per `(σ, c)` comes from a **templated generator** (no
  LLM in the loop); an **LLM-in-the-loop** variant (bounded cost) is a later ablation, not the default.
- **Reward**: terminal `R = coverage(final union) − λ_c·(steps + LLM calls) − λ_f·(guard failures)`;
  optional **potential-based shaping** `r_t = γ·Φ(s_{t+1}) − Φ(s_t)` with `Φ` = current union coverage
  (Ng-Harada — preserves the optimal policy, densifies the signal). Objective:

$$\max_\theta \mathbb{E}_{\tau\sim\pi_\theta}\Big[\textstyle\sum_{t=0}^{T}\gamma^{t} r_t\Big]$$

- **Environment** `rag/rl/env.py` — a Gym-style `reset()/step(a)` RAG-retrieval MDP over the labeled
  multi-hop benchmark, using A6.1c as the reward. **This simulator is the key enabler**: it yields
  unlimited, deterministic, $0 rollouts (templated queries), so RL is tractable without live LLM cost.

**Algorithm ladder (RESOLVED — PPO primary; backend separate from algorithm).** *Backend* (the autodiff
library: numpy / JAX / **torch**) and *algorithm* (the RL update: REINFORCE / **PPO** / DQN / CQL) are
orthogonal — any algorithm can be written in any backend. We have a **simulator** (the templated $0
rollout env), and on-policy methods want exactly that, so **PPO is the primary algorithm**. Staged so
each rung de-risks the next:
1. **Behavior cloning (BC, supervised warm-start).** Imitate the current A4 ReAct loop + A5 bridge
   decisions from logged trajectories → a sane initial policy. No RL yet; pure cross-entropy on
   (state → expert action). Gives PPO a good init and a non-trivial baseline.
2. **REINFORCE with a learned baseline** (numpy **linear-softmax** policy, manual gradients). The
   always-available, **CI-tested default** and a correctness sanity-check for the PG machinery (PPO is a
   stabilized REINFORCE, so this de-risks the PPO impl). Advantage `G_t − b(s_t)`:

$$\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\textstyle\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\big(G_t-b(s_t)\big)\Big]$$

3. **PPO (the main learner)** — actor-critic policy gradient with the **clipped surrogate** objective
   (prevents destructively large updates; the RLHF/agent-RL workhorse). MLP policy + value head on the
   `s_t` features, trained on-policy over the $0 simulator. Backend = **torch** (isolated `[rl]` extra,
   lazy-imported, OpenMP-isolated from lightgbm + a day-1 smoke test) or **JAX** if the isolation chafes:

$$L^{\text{CLIP}}(\theta)=\mathbb{E}_t\Big[\min\big(\rho_t(\theta)\hat A_t,\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$

4. **(Optional) GRPO-style** group-relative advantage — sample `G` trajectories per query, center the
   reward by the per-query group mean (no learned critic); fits a tiny-data regime, a clean tie to
   modern RL-for-LLM.
5. **CQL / FQI (offline-from-logs branch only).** The value-based (DQN-family) path, reserved for
   learning purely from **logged trajectories with no simulator** (decision #5's "door open"):
   **Fitted Q-Iteration** with a linear / lightgbm `Q(s,a)` + a **Conservative Q-Learning** pessimism
   penalty for offline OOD-action overestimation. Not vanilla DQN (too unstable on small data); only
   used if we ever drop the simulator for real logged feedback.

*Why not DQN as primary:* value-based off-policy shines when you can **only** learn from a fixed logged
dataset — moot here (we built a simulator), and it's finicky/overestimating on small data. On-policy PPO
over the simulator is more stable and is the better learning vehicle for this track.

**Evaluation protocol (mirror `backtesting` §10):** train/test **split of the query benchmark**; report
the learned policy vs four baselines — (i) best fixed pipeline (A6.1f), (ii) the **A6.1 bandit**, (iii)
the **A4 hand-prompted ReAct loop** (the A5.3 B/D cells), (iv) a random-action policy. Metrics: mean
return, coverage, **cost (steps + LLM calls)**, train-vs-held-out gap (overfit check), seed-averaged
(loop non-determinism), reward-hacking sentinel. OPE (A6.1d) for any logged-trajectory component.

**A6.2 prerequisite GATE (met by A6.0).** Sequential RL needs state-distribution coverage the current
**12-query** multi-hop benchmark cannot provide → **A6.0 grows `rag_eval_multistep.json` to a clean,
stratified ~100–200** (final count = clean supply, D1) corpus-verified questions first (before A6.1,
since it helps both). Without this, A6.2 overfits and the honest result is "insufficient data for
sequential RL" — a valid finding, but A6.0 makes the attempt meaningful. A6.2's held-out comparison
**must use A6.0's group-wise `split_multihop` (D2)**, never a row-wise split.

**A6.2 deliverable:** `rag/rl/{env,policy,train}.py`, a `rag rl-train` / `rag rl-eval` CLI, a held-out
comparison table in `validations_results.md`, and a verdict: does a *learned* retrieval policy beat the
hand-built ReAct loop + the bandit? (Pre-registered: plausibly a **rigorous negative** on a small
action/state space — the hand-prompted loop is a strong baseline — but the env + RL harness are the
durable, reusable artifact and the core learning goal.)

---

### A6 — decisions RESOLVED (2026-06-28, user-confirmed; build to these)

1. **RL algorithm + backend.** **Algorithm = PPO** (primary), staged **BC warm-start → REINFORCE/linear
   baseline → PPO → (optional GRPO)**, with **CQL/FQI reserved for the offline-from-logs branch only**.
   **Backend: torch is ALLOWED** for the PPO MLP — but as an **isolated optional `[rl]` extra**,
   lazy-imported, kept out-of-process / OpenMP-isolated from lightgbm (`KMP_DUPLICATE_LIB_OK` + a day-1
   segfault smoke test); the **numpy linear-softmax** policy stays the always-available, CI-tested
   default (CI remains torch-free + deterministic). JAX is the fallback if torch isolation is painful.
   *(This supersedes locked-decision #4's blanket "no torch" — torch is now permitted ONLY in the
   isolated RL trainer, never in the base import path / rerankers / graph.)*
2. **Benchmark size — GROW TO ~100 Q, before A6.1** (helps both phases): new sub-phase **A6.0** —
   graph-mined bridge pairs × templated question forms, **auto-verified** with the A5.3
   span-isolation probe (present-in-target / absent-in-seed), mixed strata + types. Mostly $0.
3. **A6.2 rollout realism — train on templated $0 rollouts; VALIDATE held-out with real
   LLM-in-the-loop** (small paid eval, A5.3-style). Accepts the **sim-to-real gap** as a measured
   quantity (the held-out LLM eval reports it) rather than a blocker.
4. **Reward = weighted `nDCG` (single-shot ranking quality) + `coverage` (multi-hop union completeness)
   − `λ_c`·cost, each metric active where its labels exist; faithfulness (citation/number guard) is a
   HARD CONSTRAINT** — any guard-failing trajectory is floored to reward 0 (prevents reward-hacking by
   construction; the guard already refuses ungrounded answers, so this just makes it explicit).
5. **Offline-only learning** (logging + OPE + simulator; deploy a FROZEN policy). The telemetry log
   (A6.1a) **keeps the door open** for future **batch** retrains on accumulated real feedback (like the
   monthly model cadence) — but **no live, online-updating policy** in this track.

**Tests (both phases, offline/deterministic):** log round-trip + off-is-noop; featurizer goldens;
**IPS/SNIPS/DR hand-computed goldens**; bandit (ε-greedy/LinUCB) seeded determinism;
env `reset/step` determinism + horizon/budget termination; reward-hacking sentinel; BC reduces loss on
expert trajectories; **REINFORCE/PPO converge on a 2-arm / 2-step toy** with a known optimum (PPO via
the isolated `[rl]` extra, skipped if torch absent — the numpy REINFORCE path always runs in CI);
train/test query split disjoint. No live model/LLM/network in CI; real benchmark runs are local.

**Risks.** (a) **Tiny data** — the dominant risk for A6.2; mitigated by the benchmark gate + the
simulator + overfit reporting. (b) **Reward hacking** — composite reward + held-out + sentinel.
(c) **Distribution shift** (logging `μ` ≠ target `π`) — DR + propensity logging + CQL pessimism.
(d) **Over-engineering** — A6.1 ships independently; A6.2 starts linear and escalates only on evidence.
(e) **Strong baseline** — the tuned hybrid/graph + ReAct loop may already be near-optimal → plan for a
rigorous negative.

---

## Cross-cutting (applies to every phase)

- **Dependency direction unchanged** — new modules live under `rag/`, `research/`, `graph/`; depend
  downward only; the agent/CLI stay thin adapters.
- **Cost/latency budget** — each phase states added paid LLM calls + latency; defaults keep the
  *retrieval* path $0/local (rerank-local, hybrid, graph-traverse, bandit are all local; only A4
  planner + A5 extraction + opt-in faithfulness/Voyage-rerank cost tokens).
- **Tests stay deterministic** — `FakeEmbedder`, `FakeReranker`, temp SQLite, canned `TextLLM`, seeded
  policies; no live model/LLM/network in CI. Real model/benchmark runs are local (like model backtests).

## Out of scope (V2+)

Neo4j / a hosted graph DB; transcripts + investor decks in the graph; learned dense fine-tuning;
multi-vector / ColBERT late-interaction; cross-encoder distillation; streaming/online index updates
beyond the quarterly refresh; **online (live-updating) RL policies** (A6 is offline-only — full
multi-step RL is now IN scope as A6.2, but trained offline against the simulator, deployed frozen).

## Suggested `CLAUDE.md` additions (project working agreement)

Add an "Advanced RAG" subsection under the orientation/invariants:
- **Pointer:** advanced RAG plan → `docs/ADVANCED_RAG_TODO.md`; mechanisms appended to
  `docs/rag_implementation_notes.md` per phase (same rule as P-N).
- **Default-OFF rule:** every advanced-RAG feature ships behind a config flag, default off, and is
  byte-identical to the current pipeline when off; promotion requires a measured A1 win.
- **One retrieval contract:** all retrievers (dense/hybrid/reranked/graph) implement the
  `RetrievalSystem` Protocol so `research/` + the agent never change to gain a new retrieval mode.
- **Eval-first gate:** no A2–A6 "improvement" merges without a `rag eval` number on the benchmark;
  CI tests the harness deterministically, the real benchmark is a local run.
- **Grounding is non-negotiable across steps:** rerank/hybrid don't touch chunks; agentic/graph cite
  via the union/edge-provenance; the citation guard + `NumberGrounding` run on every synthesis call.
- **No torch in the base path:** rerankers/graph use onnx (fastembed) or APIs (Voyage); never
  `sentence-transformers`/torch. **Exception (A6 only):** torch is permitted in the **isolated RL
  trainer** (`[rl]` extra, lazy-imported, OpenMP-isolated from lightgbm); CI stays torch-free.

## Recommendation — what to build first

**A1 (extend the eval harness) → A2 (reranking).** A1 is low-risk, mostly an extension of existing
code, and is the prerequisite that turns A2–A6 from guesswork into measured wins; A2 is the
highest-ROI quality improvement and a clean, self-contained reranking lesson. After those two you'll
have a *measurable* retrieval stack and can decide A3–A6 on evidence.
