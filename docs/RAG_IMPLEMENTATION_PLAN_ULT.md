# Ultimate RAG Implementation Plan for AI Stock Analysis Agent

This plan starts after the MVP RAG plan and the advanced RAG plan are complete:

- `docs/RAG_IMPLEMENTATION_PLAN.md`
- `docs/RAG_IMPLEMENTATION_PLAN_ADV.md`
- implementation roadmap: `docs/RAG_TODO.md`
- advanced implementation roadmap: `docs/ADVANCED_RAG_TODO.md`

The goal is to move from "strong local RAG implementation" to "production-grade, finance-aware RAG
system" while teaching the deeper concepts that still matter after vector search, hybrid retrieval,
reranking, agentic RAG, GraphRAG, and retrieval policy learning are understood.

This document is written for Claude Code. Claude Code should use it to plan, implement, test, and
teach each phase. Do not implement everything in one change. Produce or update a detailed TODO first,
then execute one independently shippable phase at a time.

---

## Goal

Extend the RAG system with the remaining advanced topics:

1. Answer-level RAG evaluation
2. Hard document RAG for tables, PDFs, XBRL, filings, and slide decks
3. Query routing and intent classification
4. Index lifecycle, freshness, and reproducibility
5. Context engineering and citation-preserving compression
6. RAG security and trust
7. Domain adaptation and synthetic data
8. Production observability

Each phase must both:

- improve the repository
- teach the concept clearly in docs

---

## Constraints

- Keep the existing MVP and advanced RAG behavior stable.
- Do not break existing CLI commands, chat-agent tools, reports, or evals.
- Keep the system local-first unless a phase explicitly requires an opt-in paid call.
- Preserve citation grounding and number grounding.
- Never let the LLM invent evidence, citations, SEC facts, market data, or probabilities.
- Keep every new feature default-off or gated until it has evaluation evidence.
- Add focused tests in the same change as implementation.
- Run `make check` after each implementation phase when feasible.
- Update docs after each phase:
  - implementation notes in `docs/rag_implementation_notes.md`
  - theory/teaching notes in `docs/rag_concepts.md`
  - roadmap/checklist in a new `docs/ULTIMATE_RAG_TODO.md`

---

## Required Claude Code Workflow

Before implementation:

1. Review the current repository structure and existing RAG modules.
2. Review `docs/RAG_TODO.md`, `docs/ADVANCED_RAG_TODO.md`, and `docs/rag_implementation_notes.md`.
3. Create `docs/ULTIMATE_RAG_TODO.md`.
4. For each phase, explain:
   - learning objective
   - user-facing value
   - implementation surface
   - tests
   - evaluation strategy
   - risks and rollback path
5. Implement phases one by one.

During implementation:

- Use existing abstractions first: `RetrievalSystem`, `EvidenceSet`, `ChunkFilter`, `TextLLM`,
  `NumberGrounding`, and the existing CLI patterns.
- Prefer small vertical slices over broad rewrites.
- Keep new paid or model-heavy features opt-in.
- Add offline tests using fakes before adding live/provider paths.

After each phase:

- Record what changed.
- Record what was learned.
- Record limitations and deferred work.
- Run relevant tests.

---

## Target Architecture

The final system should support this kind of flow:

```text
question
  -> intent/router
  -> retrieval strategy selection
  -> dense / hybrid / graph / agentic retrieval
  -> hard-document extractors when needed
  -> context packing / compression
  -> grounded synthesis
  -> answer-level evals
  -> logs, traces, drift checks, feedback
```

The key difference from earlier phases: retrieval quality is no longer only "did we get the right
chunks?" The system must also answer:

- Did the final answer faithfully use the chunks?
- Did it omit important evidence?
- Did it handle tables and structured finance data correctly?
- Did it choose the right retrieval strategy?
- Is the index fresh and reproducible?
- Can we audit why an answer was produced?
- Can we detect failures over time?

---

## U1 - Answer-Level RAG Evaluation

### Learning Objective

Learn the difference between retrieval evaluation and answer evaluation.

Retrieval metrics ask whether the right evidence was found. Answer-level metrics ask whether the final
answer is supported, complete, useful, and properly cited.

### User Value

The system should not only retrieve good chunks; it should produce answers that are faithful,
complete, citation-grounded, and regression-tested.

### Concepts to Teach

- faithfulness
- answer completeness
- citation precision and citation recall
- contradiction detection
- rubric-based evaluation
- LLM-as-judge risks and calibration
- golden-answer vs reference-free evaluation

### Implementation Requirements

Add an answer-evaluation framework that can evaluate generated answers against retrieved evidence.

Suggested modules:

```text
src/stock_agent/rag/answer_eval.py
src/stock_agent/schemas/answer_eval.py
configs/rag_answer_eval.example.json
```

Support metrics:

- citation_validity: every citation points to retrieved evidence
- citation_precision: cited chunks actually support the cited claim
- citation_recall: important supporting chunks were cited
- faithfulness: answer claims are supported by cited chunks
- completeness: answer covers required aspects
- contradiction_flag: answer contradicts retrieved evidence

Implementation should start with deterministic checks:

- citation IDs are valid
- answer contains no citations outside the evidence set
- cited chunks contain expected spans
- numeric claims pass existing number-grounding rules

Then add optional LLM-judge checks behind a flag:

```text
rag answer-eval --queries configs/rag_answer_eval.example.json --judge none
rag answer-eval --queries configs/rag_answer_eval.example.json --judge llm
```

LLM judge must be default-off and cost-gated.

### Tests

- deterministic citation-validity tests
- faithfulness parser tests with canned judge responses
- contradiction flag tests with fake evidence
- CLI smoke test using fake retriever and fake LLM

### Teaching Deliverable

Add a section to `docs/rag_concepts.md` explaining:

- retrieval eval vs answer eval
- why high Recall@K can still produce a bad answer
- why LLM judges need calibration
- how citation precision differs from citation recall

---

## U2 - Hard Document RAG for Finance Documents

### Learning Objective

Learn why real-world RAG is often limited by document extraction, not embeddings.

Finance documents contain tables, XBRL facts, PDFs, footnotes, slides, and layout-dependent meaning.
Naive text extraction loses important structure.

### User Value

The assistant should answer questions about financial tables, filing footnotes, presentation slides,
and structured SEC data more accurately.

### Concepts to Teach

- layout-aware parsing
- table extraction
- XBRL facts
- parent-child chunking for structured documents
- figure/table citations
- text vs structured retrieval
- SEC filing sections vs inline facts

### Implementation Requirements

Add a hard-document ingestion layer, starting with SEC tables and XBRL before arbitrary PDFs.

Suggested modules:

```text
src/stock_agent/documents/tables.py
src/stock_agent/documents/xbrl.py
src/stock_agent/rag/structured_chunks.py
src/stock_agent/schemas/structured_documents.py
```

Start with these capabilities:

1. Extract HTML tables from SEC filings.
2. Store table metadata:
   - ticker
   - form type
   - filing date
   - section
   - table caption or inferred title
   - source URL
   - row/column labels
3. Convert tables into both:
   - normalized structured rows
   - citation-friendly text summaries
4. Add table chunks to retrieval without breaking existing text chunks.
5. Add answer behavior that cites table evidence clearly.

Later optional additions:

- XBRL concept extraction
- PDF parsing for investor decks
- slide-level chunks
- image/chart extraction only if useful

### CLI Examples

```bash
python -m stock_agent documents ingest --ticker NVDA --include-tables
python -m stock_agent rag query --ticker NVDA --question "What changed in revenue by segment?"
```

### Tests

- parse small HTML table fixture
- preserve row/column labels
- produce stable table chunk IDs
- retrieve table chunks by ticker/date/section
- cite table chunks in grounded answers

### Teaching Deliverable

Add docs explaining:

- why tables are not just text
- when to use structured retrieval instead of vector retrieval
- how table citations should work
- limitations of PDF and slide extraction

---

## U3 - Query Routing and Intent Classification

### Learning Objective

Learn that a production RAG system should not always use the same retrieval path.

Some questions need vector search. Some need graph traversal. Some need tables. Some need market data.
Some should be refused or answered without retrieval.

### User Value

Users get faster, cheaper, more accurate answers because the system chooses the right tool for the
question.

### Concepts to Teach

- query intent classification
- router vs agent
- deterministic routing vs LLM routing
- fallback paths
- abstention
- retrieval strategy selection

### Implementation Requirements

Add a router that classifies questions into retrieval intents.

Suggested module:

```text
src/stock_agent/rag/router.py
src/stock_agent/schemas/rag_router.py
```

Initial intents:

- filing_fact
- filing_risk
- table_or_metric
- company_comparison
- temporal_change
- graph_relationship
- market_data
- news
- unsupported_or_financial_advice

Routing should start rule-based and deterministic:

- ticker mentions
- words like "table", "revenue", "segment", "cash flow"
- comparison words
- temporal words
- relationship words like supplier, customer, competitor
- financial-advice phrases

Add optional LLM classification later, default-off.

Router output should include:

- intent
- confidence
- selected retrieval strategy
- required filters
- reason string for trace/debug

### CLI Examples

```bash
python -m stock_agent rag route -q "Compare NVDA and AMD AI risks"
python -m stock_agent rag query --auto-route -q "Who are NVDA's main suppliers?"
```

### Tests

- deterministic intent classification examples
- unsupported financial advice routes to refusal/safety path
- table questions route to structured/table retrieval when enabled
- graph questions route to graph retrieval when graph is available
- low-confidence route falls back to default hybrid retrieval

### Teaching Deliverable

Explain:

- router vs ReAct agent
- why routing is cheaper and more predictable than always using an agent
- when routing should abstain

---

## U4 - Index Lifecycle, Freshness, and Reproducibility

### Learning Objective

Learn how real RAG systems maintain indexes over time.

Building an index once is easy. Keeping it fresh, reproducible, versioned, and debuggable is harder.

### User Value

The system should know what is indexed, when it was indexed, which embedding model created it, and
whether citations point to current or stale filings.

### Concepts to Teach

- index manifests
- embedding model versioning
- re-embedding migrations
- document deletion and supersession
- stale evidence detection
- reproducible chunk IDs
- index health checks

### Implementation Requirements

Add an index manifest and lifecycle commands.

Suggested modules:

```text
src/stock_agent/rag/index_manifest.py
src/stock_agent/rag/index_health.py
src/stock_agent/schemas/index_manifest.py
```

Manifest should record:

- corpus namespace
- embedding provider/model/dimension
- chunking settings
- retrieval mode
- indexed tickers
- document counts
- chunk counts
- table chunk counts
- sparse index status
- graph index status
- created_at / updated_at
- source document hashes

CLI:

```bash
python -m stock_agent rag index-status
python -m stock_agent rag index-health
python -m stock_agent rag reindex --ticker NVDA --reason embedding-model-change
python -m stock_agent rag stale-sources --ticker NVDA
```

Implementation should detect:

- vector chunks missing from sparse store
- sparse chunks missing from vector store
- graph provenance pointing to missing chunks
- source documents changed after indexing
- embedder namespace mismatch

### Tests

- manifest read/write
- stale source detection
- missing sparse/vector pair detection
- graph provenance dangling-reference detection
- reindex dry-run behavior

### Teaching Deliverable

Explain:

- why reproducibility matters in RAG
- why embedding migrations require new namespaces
- how stale citations happen
- how manifests make retrieval auditable

---

## U5 - Context Engineering and Citation-Preserving Compression

### Learning Objective

Learn how to turn retrieved evidence into the best possible prompt context.

RAG quality often fails after retrieval: too much context, duplicated chunks, bad ordering, lost
citations, or important evidence buried in the middle.

### User Value

Answers should become more focused, cheaper, and easier to audit.

### Concepts to Teach

- context packing
- deduplication
- quote extraction
- parent-child retrieval
- contextual compression
- lost-in-the-middle mitigation
- citation-preserving summarization

### Implementation Requirements

Add a context-packing layer between retrieval and synthesis.

Suggested modules:

```text
src/stock_agent/rag/context_pack.py
src/stock_agent/schemas/context_pack.py
```

Features:

1. Deduplicate near-identical chunks.
2. Group evidence by ticker, document, section, and date.
3. Preserve citation IDs through packing.
4. Optionally extract the most relevant quotes from long chunks.
5. Apply a token budget.
6. Prioritize:
   - cited table chunks
   - exact section matches
   - newer filings when question asks current state
   - older filings when question asks change over time
   - graph provenance chunks for relationship questions

Start deterministic. Optional LLM compression can be added later behind a flag.

### CLI Examples

```bash
python -m stock_agent rag query --ticker NVDA --question "What are the main AI risks?" --show-context
python -m stock_agent rag pack-context --query "Compare NVDA and AMD risks" --top-k 12
```

### Tests

- citation IDs remain stable after packing
- token budget enforced
- duplicate chunks removed
- quote extractor preserves source metadata
- grouped ordering is deterministic

### Teaching Deliverable

Explain:

- why retrieval is not the same as context construction
- how compression can break citations if done carelessly
- why deterministic compression should come before LLM compression

---

## U6 - RAG Security and Trust

### Learning Objective

Learn the security risks unique to RAG systems.

Retrieved text can contain malicious instructions, citation spoofing, misleading text, or poisoned
content. A RAG system must treat retrieved documents as untrusted data.

### User Value

The assistant should remain grounded and safe even when retrieved documents contain hostile,
irrelevant, or misleading instructions.

### Concepts to Teach

- prompt injection in retrieved documents
- corpus poisoning
- citation spoofing
- source allowlists
- trust levels
- permission-aware retrieval
- answer audit trails

### Implementation Requirements

Add RAG trust controls.

Suggested modules:

```text
src/stock_agent/rag/trust.py
src/stock_agent/schemas/trust.py
```

Features:

1. Source trust policy:
   - SEC filings: trusted primary source
   - company investor relations: trusted but promotional
   - news: third-party source
   - generated reports: derived source
2. Prompt-injection scanner for retrieved chunks.
3. Citation spoofing detection:
   - ignore citation-like text inside retrieved documents
   - generated answer citations must map to system-assigned evidence IDs
4. Retrieval allowlist by source type.
5. Audit trace for every answer:
   - query
   - selected retriever/router path
   - evidence IDs
   - source trust levels
   - guard outcomes

### Tests

- retrieved chunk containing "ignore previous instructions" is flagged
- answer citations cannot cite document-internal fake citation IDs
- low-trust sources can be excluded by policy
- audit trace records trust metadata

### Teaching Deliverable

Explain:

- why RAG documents are untrusted input
- why source citations are not enough
- how system-assigned citations prevent spoofing
- how trust differs from relevance

---

## U7 - Domain Adaptation and Synthetic Data

### Learning Objective

Learn how to improve retrieval quality using domain-specific data without immediately fine-tuning
models.

### User Value

The RAG system should get better at finance-specific questions, SEC wording, risk-factor language,
and ticker/entity terminology.

### Concepts to Teach

- synthetic query generation
- hard-negative mining
- embedding fine-tuning
- reranker fine-tuning
- evaluation before training
- domain drift

### Implementation Requirements

Start with data generation and evaluation, not fine-tuning.

Suggested modules:

```text
src/stock_agent/rag/synthetic_queries.py
src/stock_agent/rag/hard_negatives.py
```

Capabilities:

1. Generate candidate eval questions from filings.
2. Extract expected answer spans from source chunks.
3. Mine hard negatives:
   - same ticker, wrong section
   - same section, wrong ticker
   - semantically similar but unsupported chunk
4. Export training/eval data in JSONL.
5. Compare retrieval behavior before and after adding synthetic eval cases.

Fine-tuning is optional and deferred until the eval set proves a real failure pattern.

Potential future fine-tuning:

- train a local reranker on query, positive chunk, hard negative triples
- evaluate against current reranker
- promote only if measured improvement is clear

### CLI Examples

```bash
python -m stock_agent rag synth-queries --ticker NVDA --limit 25 --out data/eval/synth_nvda.jsonl
python -m stock_agent rag mine-negatives --queries data/eval/synth_nvda.jsonl
```

### Tests

- synthetic schema validation
- hard negative selection excludes positive chunk
- exported JSONL round-trips
- generated eval cases can run through existing eval harness

### Teaching Deliverable

Explain:

- why synthetic data can help and harm
- why hard negatives matter
- why eval must come before fine-tuning
- when not to fine-tune

---

## U8 - Production Observability

### Learning Objective

Learn how to monitor and debug a RAG system after it exists.

Without traces and metrics, RAG failures are hard to diagnose. The system needs visibility into
retrieval, context packing, synthesis, guardrails, and user feedback.

### User Value

Failures become actionable. The system can answer: what went wrong, where, and how often?

### Concepts to Teach

- retrieval traces
- latency metrics
- empty-result rates
- citation failure rates
- drift detection
- feedback loops
- regression dashboards

### Implementation Requirements

Add structured RAG tracing and metrics logs.

Suggested modules:

```text
src/stock_agent/rag/trace.py
src/stock_agent/rag/metrics.py
src/stock_agent/schemas/rag_trace.py
```

Trace fields:

- trace_id
- timestamp
- query
- router intent
- retrieval system name
- filters
- top_k/fetch_k
- retrieved chunk IDs
- packed context IDs
- synthesis model
- citation guard result
- number guard result
- latency by stage
- empty retrieval flag
- answer length
- optional user feedback

Logs should be JSONL under:

```text
data/retrieval_logs/
```

CLI:

```bash
python -m stock_agent rag trace-summary --days 7
python -m stock_agent rag failed-traces --reason citation_guard
python -m stock_agent rag feedback --trace-id TRACE_ID --rating good
```

### Tests

- trace serialization
- latency fields recorded
- summary aggregates empty-result rate
- failed trace filtering
- feedback update by trace ID

### Teaching Deliverable

Explain:

- why RAG needs observability
- common production failure metrics
- how traces support eval and future bandit/RL work

---

## Recommended Build Order

1. U8 - Production Observability
2. U1 - Answer-Level RAG Evaluation
3. U5 - Context Engineering
4. U4 - Index Lifecycle
5. U3 - Query Routing
6. U2 - Hard Document RAG
7. U6 - Security and Trust
8. U7 - Domain Adaptation

Reasoning:

- Observability first makes every later change measurable.
- Answer-level eval establishes quality gates.
- Context engineering improves the existing system without new ingestion risk.
- Index lifecycle prevents silent stale or broken retrieval.
- Routing becomes useful once there are several reliable paths.
- Hard-document RAG is high value but touches ingestion and schemas, so it should come after lifecycle.
- Security should be added before ingesting broader external sources.
- Domain adaptation should wait until logs and evals reveal real failure patterns.

---

## Deliverables Claude Code Should Produce

Before implementation:

1. `docs/ULTIMATE_RAG_TODO.md`
2. A phase-by-phase execution plan
3. A repo-specific dependency/risk review
4. A list of existing modules to reuse

For each phase:

1. Minimal implementation
2. Tests
3. CLI or config surface if appropriate
4. Documentation update
5. Verification result
6. Notes on what the concept means and how this repo implements it

---

## Definition of Done

The ultimate RAG track is done when:

- answer-level evals exist and can be run locally
- table/structured evidence can be retrieved and cited
- query routing can choose between major retrieval modes
- index state can be inspected, checked, and reproduced
- context packing preserves citations and token budgets
- RAG trust controls detect prompt injection and citation spoofing
- synthetic eval/hard-negative generation exists
- RAG traces and summaries make failures observable
- docs teach each concept clearly
- all features are tested and default-safe

The final system should remain an educational research assistant, not a financial advisor.
