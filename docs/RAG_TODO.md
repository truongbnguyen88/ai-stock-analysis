# RAG Implementation — Roadmap & TODO

> **Execution guide** for the RAG layer (SEC-grounded equity research assistant).
> Full context + rationale: [RAG_IMPLEMENTATION_PLAN.md](RAG_IMPLEMENTATION_PLAN.md).
> Each phase is independently shippable and must pass `make check` (ruff + mypy +
> pytest) before the next. Pick up by saying e.g. *"implement RAG P3"*.
>
> **Decisions locked (2026-06-09):**
> 1. **Embeddings = local `fastembed` + `bge-small-en-v1.5`** (onnxruntime, NO torch —
>    avoids the documented macOS torch+lightgbm OpenMP segfault, see
>    `tests/conftest.py`). OpenAI `text-embedding-3-small` wired behind the same
>    `Embedder` Protocol, **off by default** (one config flag to A/B). A
>    `sentence-transformers` local backend (torch; MPS accel / full precision / any HF
>    model) is the planned future alternative — same Protocol, swap-only, no retrieval
>    code change; deferred to avoid torch in the MVP.
> 2. **Vector DB = ChromaDB** (built-in persistence + metadata filtering Phase 6
>    needs), behind a `VectorStore` Protocol so it can be swapped (LanceDB/FAISS later).
> 3. **MVP corpus = SEC filings only** (10-K / 10-Q / 8-K). Transcripts, investor
>    decks, hybrid search, reranking → V1.
> 4. **One paid LLM call** in the whole flow: the final research synthesis (Claude).
>    Ingestion + embedding + retrieval are 100% local / $0.

---

## Architecture (fits the existing layers; dependencies point downward only)

```
providers/sec_edgar.py     EDGAR OFFICIAL API client (Protocol; throttle ≤10 req/s; UA; cache)   ← lowest
schemas/documents.py       Document, DocumentChunk, DocumentMetadata
schemas/retrieval.py       RetrievedChunk(score, metadata), EvidenceSet
documents/                 ticker_cik · download · parsers (HTML→text + section detect) · manifest
rag/embeddings.py          Embedder Protocol: FastEmbedEmbedder (default) | OpenAIEmbedder
rag/chunking.py            section-aware chunker (PURE)
rag/vector_store.py        VectorStore Protocol over ChromaDB
rag/retriever.py           filters (ticker/type/date/source) + top-k + dedup (NO LLM)
rag/pipeline.py            ingest(once): parse→chunk→embed→store ; query: embed→retrieve
rag/prompts.py             versioned grounded-QA / synthesis system prompts
research/synthesis.py      SINGLE grounded synthesis call (llm/ + guards)
research/memo.py           assemble the research memo (parallels reports/builder.py)
research/evidence.py       format retrieved chunks → citations
cli:                       `documents download-sec` · `documents ingest` · `rag query` · `research`
```

**Dependency direction:** `providers/sec_edgar` → `documents/` → `rag/` → `research/` → `cli/`.
`rag/` depends on `documents/` + an `Embedder`; `research/` depends on `rag/` + `pipelines/`
(forecast/analyze) + `llm/`. Never inverted.

---

## Invariant reconciliations (these are correctness requirements, not preferences)

- **No web scraping (#3).** SEC access is via the **official EDGAR API only**
  (`data.sec.gov/submissions`, `company_tickers.json`, `www.sec.gov/Archives/edgar/...`),
  with the SEC-required `User-Agent` (name + contact email), throttled, behind a
  `providers/` Protocol, normalized to `schemas/`. This is an official API, not scraping.
- **Numbers vs narrative (#1).** RAG returns **qualitative evidence + citations only**.
  The memo's probability/scenario numbers come from `run_forecast` / `indicators/`,
  **never** the LLM or a retrieved chunk. The synthesis call is checked by the existing
  `NumberGrounding` guard **plus a new citation guard**: every cited `chunk_id`/source in
  the answer must exist in the retrieved set (analogous to the URL guard in `llm/guards.py`).
  "Insufficient evidence found." when retrieval is empty — no fabrication.
- **Non-advisory (#2).** Memo sections = Bullish/Bearish **Evidence**, Uncertainty Notes,
  Citations. **No recommendation field.**
- **Secrets in `.env` (#4).** New optional keys: `OPENAI_API_KEY` (only if OpenAI embed),
  `SEC_USER_AGENT` (required by SEC for downloads). Access via `settings.py`.
- **Leakage (#6).** MVP memo doesn't feed models, so leakage is N/A *now* — but every chunk
  carries `filing_date` / `as_of` metadata so a future point-in-time use stays correct.
- **Reproducibility.** Tests inject a deterministic `FakeEmbedder` + recorded EDGAR
  fixtures; **no model download, no network in CI**. Vector store uses a temp dir per test.

---

## Config additions (`settings.py`) + `.gitignore`

- `openai_api_key: str | None = None`
- `sec_user_agent: str | None = None`  (required for live SEC downloads; clear error if absent)
- `embedding_provider: Literal["local", "openai"] = "local"`
- `embedding_model: str = "BAAI/bge-small-en-v1.5"`
- `rag_chunk_tokens: int = 900`, `rag_chunk_overlap: float = 0.15`, `rag_top_k: int = 8`
- `documents_dir: Path = data/raw`, `processed_dir: Path = data/processed`,
  `vector_store_dir: Path = data/vectorstore`
- **`.gitignore`: add `data/`** (raw filings + vectorstore are large/binary — never commit).

Dependencies follow the repo's **add-deps-when-first-needed** convention (not all in P0):
`fastembed` lands in **P4**, `chromadb` in **P5**, `openai` (optional embedder) in P4. HTML→text
reuses the present `lxml`. Heavy model loads are **lazy-imported** so import stays cheap + CI fast.

---

## Phases (each: state the step, smallest vertical slice, tests in the same change, `make check` green)

### P0 — Scaffolding & config ✅
- [x] `settings` fields above, `.gitignore data/`, empty `documents/`, `rag/`, `research/`
      packages, and `schemas/documents.py` + `schemas/retrieval.py` (models only). Deps deferred
      to the phase that needs them. Test: schema conformance + settings defaults. *(PR #3)*

### P1 — SEC document download ✅
- [x] `providers/sec_edgar.py` — EDGAR client (CIK via `company_tickers.json`, submissions
      index, filing download); throttle (≤10 rps) + `DiskCache`; UA from settings. `HttpJson`
      gained headers + `get_text`. `documents/ticker_cik.py` (ticker normalization + universe),
      `documents/download.py` (raw never overwritten, idempotent), `documents/manifest.py`.
      `schemas/documents.py` += `FilingRef`. CLI `documents download-sec`
      (`--ticker` / `--all` / `--forms` / `--limit`). Tests (+23): CIK + filing parsers, fetch
      via mocked EDGAR responses, UA header, download idempotency / raw-preservation / manifest,
      CLI arg handling — all offline. *(feat/rag-p1-sec-download)*

### P2 — Parsing ✅
- [x] `documents/parsers.py` — `html_to_text` (lxml; drops script/style/head, decodes
      entities + nbsp, block-aware newlines; fail-soft tag-strip + plain-text passthrough) +
      `detect_sections` (EDGAR Item anchors: 10-K/10-Q `1A`/`7`, 8-K decimal `2.02`; preamble;
      single-section fallback) + `parse_metadata` (sidecar → `DocumentMetadata`, extras ignored)
      + `load_filing` → `ParsedFiling` (metadata + text + `Section`s; `to_document()`). `Section`
      is the P3 chunking unit. Tests (+11): noise-strip/entities/block-breaks/malformed,
      Risk-Factors + MD&A + 8-K sections, metadata, end-to-end load. *(known: TOC repeats Item
      headers → short dup sections, deferred to P3 dedup.)* `lxml.*` added to mypy overrides.

### P3 — Chunking ✅
- [x] `rag/chunking.py` — PURE section-aware chunker: `chunk_sections` / `chunk_filing`.
      Sliding word-window per section (token budget → words via a 0.75 proxy, no tokenizer
      dep), `target_words` cap + exact `overlap_words`, **never crosses a section boundary**;
      `DocumentChunk.from_metadata` copies metadata onto every chunk with a document-global
      `chunk_index`. Folds in the **TOC dedup** P2 deferred (drops sub-`min_chunk_words`
      sections). Tests (+12): size bounds, exact overlap, complete/ordered coverage, boundary
      preservation, dedup, metadata/`chunk_id` integrity, determinism, overlap=0, edges.

### P4 — Embeddings ✅
- [x] `rag/embeddings.py` — `Embedder` Protocol (`name`, `dim`, `embed_documents`,
      `embed_query`); `FastEmbedEmbedder` (default, lazy onnxruntime/BGE) + `OpenAIEmbedder`
      (opt-in) + `VoyageEmbedder` (opt-in, default `voyage-4` + `input_type`
      asymmetry) + `FakeEmbedder` (deterministic unit-vector test double, reused by P5/P6) +
      `build_embedder(settings)`. Deps as **extras** (`[rag]` fastembed, `[openai]`, `[voyage]`)
      — NOT core, so CI never installs them or downloads a model; `fastembed.*`/`openai.*`/
      `voyageai.*` in mypy overrides. Tests (+10, +1 gated `RUN_EMBED_TESTS`): determinism,
      unit-norm, Protocol conformance, selector routing, OpenAI + Voyage via injected fake
      clients (input_type asserted), key-required.

### P5 — Vector store ✅
- [x] `rag/vector_store.py` — `VectorStore` Protocol (`add`/`query`/`count`) + `ChunkFilter`
      (schemas; ticker/type/section/date, AND-combined) + **`InMemoryVectorStore`** (pure-Python
      cosine + filter, CI-tested, chromadb-free fallback) + **`ChromaVectorStore`** (persistent,
      lazy chromadb, **cosine space** → `score = 1 − distance` = cosine similarity) +
      `build_vector_store(settings)`. Store takes **pre-computed vectors** (embedder runs in P6),
      upserts by `chunk_id` (idempotent); `filing_date` stored as ISO + `YYYYMMDD` int for range
      filters; `section` omitted when None (Chroma rejects None). `chromadb` joins the `[rag]`
      extra (lazy → CI never imports it; `chromadb.*` in mypy overrides). Tests (CI, InMemory):
      count/upsert, top-k order, each filter (ticker/type/section/date-range), empty, Protocol,
      score==cosine; gated `RUN_VECTORSTORE_TESTS` Chroma test: persistence + InMemory parity.
      Validated on the real corpus (201 NVDA+AVGO chunks: persist, filter, retrieve).

### P6 — Retrieval + ingest pipeline ✅
- [x] `rag/retriever.py` — `Retriever(embedder, store)`: embed query → over-fetch (`top_k×4`) →
      **dedup by normalized text** (kills verbatim boilerplate repeated across years/overlap) →
      `top_k` → `EvidenceSet`. NO LLM. `rag/pipeline.py` — `iter_filing_dirs` +
      `ingest_ticker(...)` (parse→chunk→embed→`store.add`; idempotent via upsert) → `IngestResult`.
      CLI `documents ingest --ticker/--all` and `rag query --ticker --question` (chunks + scores +
      citations; "no evidence" path). Tests (FakeEmbedder + InMemoryVectorStore): top-k/score
      order, dedup, empty-retrieval, filter pass-through, citations; ingest from a tmp filing dir,
      end-to-end ingest→retrieve, idempotent re-ingest. Validated on the real corpus (NVDA+AVGO →
      2062 chunks; query→cosine top-k verified `score == q·d` exactly).

### P7 — Grounded question answering ✅
- [x] `research/prompts.py` (versioned `research.v1`; placed in `research/` not `rag/` to keep
      `rag/` LLM-free) + `research/synthesis.py` `answer_question(question, evidence, *, llm)` —
      the single grounded call → `GroundedAnswer` (`schemas/research.py`). **Citation guard**:
      every cited marker (inline `[n]` *and* the `citations` list) must be a retrieved source
      `[1..N]`. **Number grounding** (reuses `NumberGrounding`): seeded from the source texts, no
      invented figures. One corrective retry, then raise. Empty evidence → "Insufficient evidence
      found." with **no LLM call**. CLI `rag query --answer`. Tests (canned `TextLLM`): citation
      resolution, empty-no-call, fabricated-citation retry→raise & retry→recover, invented-number
      retry→raise, grounded-number pass, LLM-signaled insufficiency, inline-marker out-of-range.

### P8 — Integrated research memo ✅
- [x] `research/memo.py` `build_memo(...)` + `render_memo_markdown` + `pipelines/research.py`
      `run_research` + CLI `research --ticker`. Gathers technicals (`compute_snapshot`) + baseline
      forecast + news (`summarize_news`) + **SEC evidence** (3 targeted retrievals merged/deduped),
      then **ONE** synthesis call → `ResearchMemo` (`schemas/research.py`): quant sections
      (Technical Indicators, Probability Scenarios) copied **verbatim from the models**; narrative
      (Executive Summary, Management Commentary, Business Drivers, Risk Factors, Bullish/Bearish
      Evidence, Uncertainty Notes, Recent News) from the LLM with SEC claims cited `[n]`. **No
      recommendation field.** Reuses P7's citation guard + `NumberGrounding` (seeded from forecast
      + snapshot + news + SEC texts), one retry then `MemoGuardError`. Tests (canned `TextLLM`):
      section assembly, single-LLM-call, citation guard, number grounding, empty-evidence graceful,
      Markdown export. **MVP COMPLETE (P0–P8).**

### P8.5 — Wire RAG into the chat agent ✅
> **Why.** P0–P8 built the SEC-grounded QA + memo, but they're reachable **only via the CLI**
> (`rag query`, `research`). The chat agent (`agent/`, used by the Streamlit chat) has 16 tools
> (prices, indicators, news, forecasts, backtest…) but **none touch the vector store** — so asking
> the agent "what are NVDA's risk factors?" today does *not* search the embedded filings. P8.5 adds
> **two tools** so the agent can (1) answer a specific filing question from the ingested SEC vectors,
> and (2) produce an integrated executive summary fusing filings + news + the forecast — both cited.

**Decision (locked): expose the guarded synthesis, not raw retrieval.** Both tools make their own
guarded LLM call(s) and return *cited, validated* output — the `summarize_news` pattern (a tool that
does its own synthesis). This keeps P7's **citation guard + number grounding** (and P8's, for the
summary) intact rather than handing raw chunks to the agent and hoping it cites correctly.

- [x] **Tool** `search_filings` in `agent/tools.py` — schema `{ticker, question}` (+ optional
      `top_k`). Handler `_tool_search_filings`: build a `Retriever(build_embedder, build_vector_store)`
      **memoized on the executor** (like `_backtest_cache`, so repeated filing questions don't rebuild
      the embedder/store), `ChunkFilter(ticker=...)`, call `answer_question(question, evidence,
      llm=self._llm)` → return `{answer, citations:[{marker,label,chunk_id}], insufficient_evidence,
      n_sources}`. Mirror `summarize_news`'s **no-LLM guard** (`self._llm is None → {"error": …}`).
      Never raises (dispatch already wraps to `{"error": …}`).
- [x] **Empty-store / not-ingested path.** If retrieval is empty → relay P7's
      "Insufficient evidence found." plus a hint that filings for the ticker aren't ingested
      (`documents ingest --ticker X`). The tool **does NOT** download/ingest on the fly (parse+chunk+
      embed ~1k chunks ≈ 60s — wrong for a synchronous chat turn). Background/auto-ingest of a
      watchlist is deferred to **P9**.
- [x] **Tool** `research_summary` in `agent/tools.py` (**required**, not optional) — schema
      `{ticker}` (+ optional `days`). Handler `_tool_research_summary`: call **P8's**
      `run_research(ticker, settings=self._settings, registry=self._registry, llm=self._llm)` →
      `ResearchMemo`, and return a **compact** dict (NOT the full Markdown — too long for a tool
      result): `executive_summary`, `business_drivers`, `risk_factors`, `bullish_evidence`,
      `bearish_evidence`, `uncertainty_notes`, `recent_news`, the headline `forecasts`
      (model / horizon / P(up) / E[r] / VaR95), key `technical_indicators`, and `citations`. This is
      the **integrated executive summary** (filings + news + forecast) the agent relays for
      "summarize / give me the full picture on TICKER / overview" requests. It is the **heaviest
      tool** (prices + forecast + retrieval + a news-summary call + the memo call ≈ 2 LLM calls,
      ~30–60s) — bound it like `run_backtest`, and apply the same **no-LLM guard**. Catch
      `ResearchPipelineError` → `{"error": …}` (it's a `RuntimeError`, outside the dispatch's caught
      tuple — extend the handler/tuple). Numbers are already from the models + P8's guards.
- [x] **Agent prompt** (`agent/prompts/agent.py`, bump `agent.v15 → v16`): add routing lines —
      *for a specific filing question (risk factors, business, products, MD&A, management commentary,
      accounting, legal) → `search_filings`; for an integrated picture / "executive summary" /
      "overview of TICKER" → `research_summary` (fuses filings + news + forecast).* Relay each tool's
      citations; do **not** answer filing questions from general knowledge. Reserve `research_summary`
      for explicit full-picture requests (it's the expensive path). Keep the router-not-calculator rule.
- [x] **Guards / invariants.** Both tools' outputs are already citation- + number-guarded (P7 for
      `search_filings`, P8 for `research_summary`), so the agent's existing **numeric-grounding guard**
      grounds their figures from the tool output (relaying SEC numbers stays safe), and non-advisory is
      preserved (neither recommends). Dependency direction stays downward (`agent/ → research/ → rag/`).
- [x] **Tests** (offline, fakes): tool-schema conformance for both. `_tool_search_filings` with a fake
      `Retriever` (InMemoryVectorStore + FakeEmbedder) + canned `TextLLM` → answer + resolved citations;
      empty-store → insufficient; `self._llm is None` → error. `_tool_research_summary` with `run_research`
      injected/monkeypatched to return a `ResearchMemo` → compact-dict shape (exec summary + sections +
      citations); no-LLM → error; `ResearchPipelineError` → `{"error": …}`. Agent numeric-grounding guard
      accepts both tools' grounded numbers; agent-loop tests that a filing question routes to
      `search_filings` and a "summarize TICKER" request routes to `research_summary`. **No live calls.**
- [x] **(Same change)** surface each tool's citations in the Streamlit chat answer rendering if not
      automatic. Prerequisite to *use* either tool: `documents download-sec` + `documents ingest` for
      the ticker (the chat answers from whatever is in `data/vectorstore`).

### P9 — Maturity / go-live (post-MVP)
> Begins only **after P8 is green**. The MVP is built + tested on local `fastembed` (free,
> unlimited) — these are the steps to take it to production scale + paid-quality embeddings
> once the pipeline (esp. chunking) is settled. A growing list; sub-milestones added over time.
- [ ] **9a — Switch embeddings to `voyage-4`.** Flip `embedding_provider` from `local` →
      `voyage` (`pip install -e ".[voyage]"`, set `VOYAGE_API_KEY`) and do the **final, one-time
      paid ingestion** of the settled corpus against the 200M-token free pool. Embed-once: don't
      switch until chunking is locked, so the free pool isn't burned on re-embeds. (Optionally A/B
      `voyage-4` vs `voyage-finance-2` first — see 9e.)
- [ ] **9b — Bulk historical download.** Pull 2–3 yrs of 10-K/10-Q/8-K for the full universe
      (deferred from P1; free, idempotent) once the download/metadata format is stable.
- [ ] **9c — Quarterly refresh scheduling.** Cron/CI (reuse the retrain-workflow pattern) to
      pull newly-filed documents + incrementally ingest them each quarter.
- [ ] **9d — Embedding spend guard.** `rag_max_embed_tokens` setting: estimate tokens before
      embedding and refuse/cap a run over the limit — a client-side hard ceiling independent of
      the provider dashboard.
- [ ] **9e — Retrieval-quality A/B.** Compare `fastembed` vs `voyage-4` vs `voyage-finance-2`
      on a small labeled query set (recall@k / MRR) to lock the production embedder.

---

## Cost (full 141-ticker universe)
| Setup | Ingestion embed | Retrieval | Per memo | Notes |
|---|---|---|---|---|
| **Local (chosen)** | **$0** (M2, minutes) | $0 | ~$0.02–0.05 | only the Claude synthesis call is paid |
| OpenAI embed (opt) | ~$0.7–2 one-time, **re-paid on each re-embed** | ~$0 | ~$0.02–0.05 | flip `embedding_provider=openai` |

## Effort (MVP, incremental + gated): ~7–9 focused days
P1 ~1.5d · P2 ~1d · P3 ~0.5d · P4 ~0.75d · P5 ~0.5d · P6 ~0.75d · P7 ~1d · P8 ~1.5d.

## Top risks → mitigations
1. **torch/lightgbm segfault** → fastembed (onnxruntime, no torch). 2. **SEC HTML variance**
(inline XBRL, exhibits) → robust HTML→text + soft-fail section detection + fixtures. 3. **CI
must stay offline** → FakeEmbedder + recorded EDGAR responses; no model download. 4. **SEC
fair-access** → throttle + `DiskCache`. 5. **Citation hallucination** → chunk-ID grounding
guard. 6. **Storage/repro** → gitignore `data/`; deterministic seeds. 7. **Context bloat** →
top-k cap + dedup + single synthesis call.

## Out of scope (V1+)
Earnings transcripts, investor decks, hybrid (BM25+vector) search, reranking, QoQ document
comparison, agentic/multi-step retrieval, contradiction detection, evidence scoring,
company/sector/portfolio reports, report memory.
