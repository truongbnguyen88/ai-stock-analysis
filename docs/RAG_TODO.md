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

Dependencies (add to `pyproject`): `chromadb`, `fastembed`; `openai` (optional, for the
OpenAI embedder). HTML→text reuses the present `lxml`. Heavy model loads are **lazy-imported**
so module import stays cheap and CI stays fast.

---

## Phases (each: state the step, smallest vertical slice, tests in the same change, `make check` green)

### P0 — Scaffolding & config
- [ ] Add deps (`chromadb`, `fastembed`), `settings` fields above, `.gitignore data/`,
      empty `documents/`, `rag/`, `research/` packages, and `schemas/documents.py` +
      `schemas/retrieval.py` (models only). Test: schema conformance + settings defaults.

### P1 — SEC document download
- [ ] `providers/sec_edgar.py` — EDGAR client (CIK lookup via `company_tickers.json`,
      submissions index, filing download) behind a Protocol; throttle + `DiskCache`; UA from
      settings. `documents/ticker_cik.py`, `documents/download.py`, `documents/manifest.py`
      (raw never overwritten; storage layout per the plan). CLI `documents download-sec`
      (`--ticker` / `--all` / `--forms` / `--limit`). Tests: ticker→CIK mapping, download
      via mocked EDGAR responses, manifest idempotency, rate-limit handling.

### P2 — Parsing
- [ ] `documents/parsers.py` — SEC HTML/TXT → normalized text + **section detection**
      (10-K/10-Q Item anchors: 1A Risk Factors, 7 MD&A; 8-K item headers). Emit
      `Document` + `DocumentMetadata` (ticker, type, source, source_url, filing_date,
      section, document_id, ingested_at). Tests: golden HTML fixture → expected sections +
      metadata; malformed/edge filings fail soft.

### P3 — Chunking
- [ ] `rag/chunking.py` — PURE section-aware chunker (~`rag_chunk_tokens`, ~`overlap`,
      never cross a section boundary; carry metadata onto every chunk). Tests: boundary
      preservation, size bounds (no tiny/giant chunks), no metadata loss, deterministic.

### P4 — Embeddings
- [ ] `rag/embeddings.py` — `Embedder` Protocol (`embed_documents`, `embed_query`,
      `dim`, `name`); `FastEmbedEmbedder` (default, lazy model load) + `OpenAIEmbedder`;
      `build_embedder(settings)` selector. Tests: `FakeEmbedder` determinism, selector
      routing, dim consistency. **No real model download in tests.**

### P5 — Vector store
- [ ] `rag/vector_store.py` — `VectorStore` Protocol (`add`, `query`, `count`, persistence)
      + `ChromaVectorStore` (persistent client, metadata filters). Tests: insert→count,
      metadata filter (ticker/type/date), top-k order by score, temp-dir isolation.

### P6 — Retrieval + ingest pipeline
- [ ] `rag/retriever.py` (filters + top-k + chunk dedup, NO LLM) + `rag/pipeline.py`
      (`ingest_ticker` once; `retrieve` for queries) + CLI `documents ingest` and
      `rag query --ticker --question`. Returns chunks + scores + citations. Tests: filter
      correctness, dedup, empty-retrieval path, end-to-end ingest→retrieve with FakeEmbedder.

### P7 — Grounded question answering
- [ ] `rag/prompts.py` (versioned) + `research/synthesis.py` (single call: question +
      retrieved evidence → cited answer) + **citation guard** in the synthesis path
      (cited sources ⊆ retrieved set; numbers grounded; "Insufficient evidence found."
      on empty). CLI `rag query` gains `--answer`. Tests: canned-LLM schema conformance,
      citation guard rejects fabricated source, empty-evidence message, no invented numbers.

### P8 — Integrated research memo
- [ ] `research/memo.py` + CLI `research --ticker` — gather technicals (`compute_snapshot`)
      + forecast (`run_forecast`) + news (`summarize_news`) + RAG evidence, then **ONE**
      synthesis call → memo (Executive Summary, Technical Indicators, Probability Scenarios,
      Recent News, Management Commentary, Business Drivers, Risk Factors, Bullish Evidence,
      Bearish Evidence, Uncertainty Notes, Source Citations). Numbers from modules; narrative
      cited. Tests: section assembly, numbers-grounding + citation guards on the memo,
      single-LLM-call assertion, Markdown export.

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
