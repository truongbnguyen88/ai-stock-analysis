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
| **A1** | Retrieval Evaluation | The measuring stick; extends `rag/eval.py`; lowest risk | Foundation | **Low–Med** (~1.5d) |
| **A2** | Reranking | Biggest quality win per effort; local-first, self-contained | Ship value | **Low–Med** (~1.5d) |
| **A3** | Hybrid Search | Fixes a *different* failure mode (exact ticker/section/number terms) | Ship value | **Med** (~2d) |
| **A4** | Agentic RAG | Needs strong retrieval primitives first; bounded LLM cost | Ship value | **Med–High** (~2.5d) |
| **A5** | GraphRAG | Heaviest infra; lower near-term ROI for SEC QA | Learning-first | **High** (~3–4d) |
| **A6** | Retrieval + RL | Capstone; needs A1 metrics + usage logs to define a reward | Learning-first | **High** (~3–4d) |

**Build order: A1 → A2 → A3 → A4 → A5 → A6.** A2↔A3 are swappable (both feed A1; final pipeline is
hybrid→rerank). A5/A6 are explicitly *learning-first* — high intellectual value, modest immediate
retrieval-quality ROI for grounded SEC QA; a rigorous negative result is a valid outcome (cf. the
forecasting track's LSTM/news-feature negatives).

---

## A1 — Retrieval Evaluation (extend `rag/eval.py`) ✅ gate for everything else

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

## A4 — Agentic RAG (bounded multi-step retrieval)

**Learning objective.** Query **decomposition**, multi-retrieval **planning**, evidence comparison —
while keeping LLM calls and grounding under control.

**User value.** Answers questions a single retrieval can't: *"Compare NVDA and AMD AI risks,"*
*"What changed in TSLA risk disclosures over 3 years,"* *"Why does the model forecast upside while
filings show risk?"* (this last one bridges RAG ⊕ the forecast pipeline).

**Design.**
- **Planner** (`research/planner.py`): one structured-output LLM call decomposes the question into a
  bounded plan — `RetrievalPlan { sub_queries: list[SubQuery], op: Literal["compare","trend",
  "synthesize"] }`, `SubQuery { ticker, question, filter: ChunkFilter | None }`. The planner emits
  **no numbers/citations** (only sub-queries) → nothing to ground there.
- **Executor**: run each `SubQuery` through the configured retrieval system (dense/hybrid/reranked +
  `ChunkFilter` for ticker/date/section scoping), collect `EvidenceSet`s, then **one** final
  synthesis call over the **union** with a compare/trend prompt → `MultiStepAnswer` (cited).
- **Bounds (cost-controlled, like `run_backtest`):** `agentic_max_subqueries: int = 4`,
  `agentic_max_llm_calls` (planner + synthesis = 2 typical); no auto-ingestion (constraint); a
  ticker not ingested → the existing "insufficient evidence" path. Falls back to single-shot
  `search_filings` for simple questions (a cheap is-this-complex heuristic / the planner returning 1
  sub-query).
- **Guards:** citation guard + `NumberGrounding` run on the final synthesis with the **union**
  evidence as the allow-set; non-advisory + router-not-calculator unchanged.
- **Surface:** a new agent tool `research_compare(tickers, question)` and/or `rag ask --multi`; the
  agent prompt routes multi-entity / temporal-comparison questions to it.

**Tests** (canned `TextLLM`): planner decomposes a comparison question → expected `SubQuery` list;
executor runs N fake retrievals; final synthesis citation guard over the union (a cite outside the
union → retry→raise); bound enforcement (>max → truncate/refuse); a "compare X and Y" agent turn
routes to `research_compare`. No live calls.

**Risks.** LLM-cost blowup → hard sub-query cap + the simple-question fast path. Decomposition quality
→ versioned planner prompt + eval on a small multi-step benchmark. Latency → bounded. Guard coverage
across steps → union allow-set + tests.

---

## A5 — GraphRAG (lightweight knowledge graph) — learning-first

**Learning objective.** Knowledge-graph **construction from text** + **graph-augmented retrieval**
(multi-hop questions vector search can't answer).

**User value.** Structural questions: *"Who are NVDA's key suppliers/customers?"*, *"Which companies
are exposed to a TSMC disruption?"*, *"What risks does NVDA share with AMD?"* — answered by graph
traversal, then grounded in the filings the edges came from.

**Design (deliberately minimal MVP).**
- **Schema** (`schemas/graph.py`): `Entity { name, type ∈ {company, product, segment, competitor,
  customer, supplier, risk, regulatory_topic} }`, `Edge { subject, relation ∈ {competes_with,
  supplies_to, depends_on, exposed_to, acquired, operates_in, mentions_risk}, object, provenance:
  list[str] (chunk_ids), filing_date }`. **Provenance is mandatory** — every edge carries the
  chunk_id(s) it was extracted from, so graph answers cite real filings.
- **Extraction** (`graph/extract.py`): **LLM-assisted, offline, batched, cost-gated** — one
  structured-output call per (ticker, section) over **10-K Item 1 (Business) + Item 1A (Risk
  Factors)** only for the MVP → triples with provenance. Rules/regex pre-filter to cut cost
  (candidate entity spans). A `documents extract-graph --ticker/--all` step (like ingestion); reuses
  the `EmbedBudgetExceeded`-style spend ceiling (`graph_max_extract_calls`).
- **Storage** (`graph/store.py`): **SQLite** `nodes`/`edges` tables (persistent, queryable, no dep) +
  **NetworkX** loaded from SQLite for traversal/neighbors/multi-hop. Neo4j deferred.
- **Retrieval** (`graph/retriever.py`): an entity/relation question → resolve entities → traverse
  (1–2 hops) → collect edge-provenance `chunk_id`s → fetch those chunks → **union with vector
  retrieval** → rerank/synthesis. A `GraphRetriever` that satisfies `RetrievalSystem` for the union
  case. CLI `rag graph-query`.
- **Scope guard:** MVP = ~5 entity types, ~4 relations, 2 sections, a handful of tickers. Success =
  correctly answers N hand-written multi-hop questions that dense retrieval demonstrably fails.

**Tests** (canned `TextLLM`): extraction → expected triples **with provenance**; SQLite graph
add/query/neighbors; `GraphRetriever` traverse → chunk_ids; graph-sourced chunks cite correctly
(provenance → real `source_url`); offline.

**Risks.** Extraction accuracy / hallucinated edges → provenance + a verification pass (drop edges
whose provenance chunk doesn't contain the entities) + confidence threshold. Cost → offline batch +
spend gate. Scope creep → keep the MVP tiny; this is a learning project, not a core dependency.

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
