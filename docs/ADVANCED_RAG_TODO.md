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
4. **No torch — ever** (the documented macOS torch+lightgbm OpenMP segfault is why the MVP uses
   fastembed/onnxruntime). Rerankers: **fastembed onnx cross-encoder** (local default) | **Voyage
   `rerank-2` API** (opt-in, key already wired) | **no-op** (fallback). No `sentence-transformers`/torch.
5. **Sparse backend = SQLite FTS5** (Python stdlib `sqlite3`; persistent; BM25 via `bm25()`; metadata
   filter via SQL `WHERE`) — no new dependency, available in CI for deterministic tests. `rank_bm25`/
   Tantivy rejected (extra dep / Rust). Fusion = **Reciprocal Rank Fusion** (rank-based → no
   cosine-vs-BM25 score-normalization), not weighted score-sum.
6. **Graph store = SQLite tables + NetworkX in-memory** (Neo4j deferred); extraction = **LLM-assisted,
   offline, cost-gated, with chunk-id provenance on every edge** (so graph answers stay citeable).
7. **RL = contextual bandits first, not full RL.** MVP = retrieval **logging + off-policy evaluation
   (IPS / doubly-robust)**; an ε-greedy / LinUCB policy only if OPE shows headroom. Reward = the A1
   metrics (+ later, user feedback). Full multi-step RL (agentic retrieval as an MDP) is deferred.
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

### A5 follow-ups — polish (PLANNED, deferred 2026-06-27; do before the A5.3 promotion verdict)

Two known issues surfaced while building the 16-ticker graph. Both are **deferred, not abandoned** —
fix #2 in particular gates the A5.3 "does graph beat the A4 bridge" measurement, so do these before
running the promotion benchmark. Neither is implemented yet.

- **F1 — Unresolved-company node de-duplication (data quality, Low effort).**
  *Symptom:* the same out-of-universe company appears as several nodes — punctuation/suffix variants
  (`Hon Hai Precision Industry Co.` vs `… Co., Ltd.`, `Global Foundries` vs `GlobalFoundries`,
  `Apple Inc.` vs `Apple`) and acronym variants (`TSMC`↔`Taiwan Semiconductor`, `UMC`↔`United
  Microelectronics`, `SMIC`↔`Semiconductor Manufacturing International`). Resolved-ticker nodes
  (ASML/AMD/NVDA/MU…) already dedup correctly — this is only the `ticker=None` companies.
  *Root cause:* `graph/extract._build_edge` ids an unresolved company by `_normalize_node_id(raw
  surface)`, so every surface form is a distinct id.
  *Plan:*
  (a) **Cheap tier** — id unresolved companies by `_core_name(obj_name)` (suffix-stripped +
      punctuation-normalized) instead of the raw slug. Merges the suffix/spacing variants. Low risk
      (watch for rare core-name collisions between genuinely different firms).
  (b) **Acronym tier (optional)** — add the common foreign suppliers/foundries to
      `configs/ticker_aliases.json` (or a sidecar alias map) so `TSMC`/`Taiwan Semiconductor` resolve
      to one canonical id; defer unless the duplicates actually hurt traversal.
  *Files:* `graph/extract.py` (the unresolved-company `obj_id`); extend `tests/unit/test_graph_extract.py`
  (two surface variants → one node id). *Migration:* existing graph has the dup nodes under old ids —
  clear + rebuild the 16-ticker graph (idempotent) after the change, or write a one-time node-merge.

- **F2 — Bridge ranking: guarantee neighbor evidence in top-k (the A5.3 blocker, Med effort).**
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

---

## A6 — Retrieval + RL (contextual bandits → off-policy eval) — learning-first, capstone

**Learning objective.** Frame retrieval as a **sequential/decision problem**; start with **contextual
bandits** and **off-policy evaluation** (the safe, offline way) — connecting RAG to your RL /
decision-systems focus.

**User value.** A policy that *adapts retrieval per query* (a risk question wants section-filtered
hybrid; a broad "overview" wants dense + high-k) instead of one fixed config — *if* it beats the
tuned A2+A3 pipeline. Plus a reusable retrieval-telemetry + OPE harness.

**Design (non-overengineered, staged).**
- **Frame as contextual bandits, not full RL** (single-step decision; no long horizon):
  - **Context (state):** query features — length, has-ticker, question-type (risk/financial/business
    via a cheap keyword/heuristic classifier), entity count.
  - **Action:** a discrete retrieval config — `{mode: dense|hybrid, rerank: on|off, top_k, fetch_k,
    section_filter: on|off}` (a small enumerated set, e.g. 6–10 arms).
  - **Reward:** the A1 metrics on labeled queries (nDCG / citation-accuracy − latency penalty);
    later, **implicit user feedback** (citation clicked / answer accepted / question rephrased) +
    explicit thumbs. Multi-objective → watch **reward hacking** (a high-recall/noisy arm games nDCG
    but hurts faithfulness — penalize).
- **MVP = telemetry + off-policy evaluation, NOT online learning:**
  1. **Retrieval log** (`rag/retrieval_log.py`, `schemas/retrieval_log.py`): per retrieval — context
     features, chosen action, propensity (if randomized), retrieved chunk_ids + scores, downstream
     answer + guard outcomes, optional feedback → JSONL under `data/retrieval_logs/` (config-gated
     `retrieval_logging`, default off).
  2. **Off-policy evaluation** (`rag/ope.py`): IPS + doubly-robust estimators on logged data —
     "what reward *would* policy π have gotten?" without deploying it. CLI `rag policy-eval`.
  3. **Policy** (`rag/policy.py`): a fixed baseline policy + an ε-greedy / LinUCB contextual bandit,
     trained offline against the eval-harness reward oracle; promoted only if OPE shows headroom over
     the best fixed pipeline.
- **Honest framing:** a well-tuned hybrid+rerank pipeline is a strong baseline; "does adaptive
  retrieval help?" is a real research question with a possible **rigorous negative** (cf. the LSTM/
  news-feature negatives). The logging + OPE infra is valuable regardless.

**Tests** (offline): log round-trip; action-space enumeration; **IPS/DR golden values** (hand-computed
on a tiny logged dataset); fixed + ε-greedy policies deterministic under seed; reward computed from
the eval harness; reward-hacking sentinel (a degenerate arm scores low on the composite reward).

**Risks.** Reward design / reward hacking → multi-objective + held-out check. Sparse feedback →
bootstrap reward from the A1 benchmark first. Over-engineering → bandits + OPE only; defer deep/
multi-step RL. Distribution shift (logged policy ≠ eval policy) → DR estimator + propensity logging.

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
multi-vector / ColBERT late-interaction; full multi-step deep RL; cross-encoder distillation;
streaming/online index updates beyond the quarterly refresh.

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
- **No torch:** rerankers/graph use onnx (fastembed) or APIs (Voyage); never `sentence-transformers`/torch.

## Recommendation — what to build first

**A1 (extend the eval harness) → A2 (reranking).** A1 is low-risk, mostly an extension of existing
code, and is the prerequisite that turns A2–A6 from guesswork into measured wins; A2 is the
highest-ROI quality improvement and a clean, self-contained reranking lesson. After those two you'll
have a *measurable* retrieval stack and can decide A3–A6 on evidence.
