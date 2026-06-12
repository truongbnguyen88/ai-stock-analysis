# Advanced RAG Roadmap for AI Stock Analysis Agent

We have completed the MVP RAG implementation for ai-stock-analysis-agent.

Current capabilities include:
- SEC filing download
- SEC parsing
- section-aware chunking
- local embeddings with fastembed
- ChromaDB vector store
- retrieval
- grounded question answering
- cited research memo generation
- chat-agent integration through search_filings

Now I want to plan the next phase: advanced RAG learning enhancements.

## Goal

Design a staged roadmap to improve the RAG system while teaching the following six advanced RAG concepts:

1. Hybrid Search
2. Reranking
3. Retrieval Evaluation
4. Agentic RAG
5. GraphRAG
6. Retrieval + RL

Do not implement yet. First produce a design and roadmap.

## Constraints

- Keep the existing MVP stable.
- Do not break existing CLI commands or chat-agent behavior.
- Keep local-first and cost-efficient design.
- Avoid unnecessary paid LLM calls.
- Preserve citation grounding and number grounding.
- Do not let the LLM invent evidence, citations, or probabilities.
- Each enhancement should be independently shippable.
- Each enhancement should include tests and evaluation strategy.

## Current RAG Architecture

The existing architecture roughly follows:

text SEC filings → parse → section-aware chunks → fastembed embeddings → ChromaDB → retriever → grounded QA → research memo / chat agent 

## Enhancement 1: Hybrid Search

Design a way to combine:
- dense vector search
- keyword/BM25 search

Questions to answer:
- Should BM25 use rank_bm25, SQLite FTS5, Tantivy, or another local option?
- How should vector and keyword scores be fused?
- Should we use Reciprocal Rank Fusion?
- How should metadata filters apply to both dense and sparse retrieval?
- How should citations remain unchanged?

Expected output:
- architecture
- module changes
- scoring/fusion design
- tests
- CLI examples
- evaluation metrics

## Enhancement 2: Reranking

Design a reranking layer after initial retrieval.

Pipeline:

text question → retrieve top 20-50 chunks → rerank → keep top 5-8 chunks → grounded answer 

Questions to answer:
- Should MVP reranker be local?
- Should we use a cross-encoder reranker?
- Should reranking be optional/configurable?
- How do we avoid heavy dependencies or torch conflicts?
- How do we compare retrieval quality before/after reranking?

Expected output:
- reranker interface
- local-first recommendation
- fallback no-op reranker
- tests
- evaluation design

## Enhancement 3: Retrieval Evaluation

Create a formal retrieval evaluation framework.

Design a small benchmark dataset:

json {   "question": "...",   "ticker": "NVDA",   "expected_document_type": "10-K",   "expected_section": "Item 1A Risk Factors",   "expected_chunk_ids": [...] } 

Metrics:
- Precision@K
- Recall@K
- MRR
- nDCG
- citation accuracy
- answer faithfulness

Questions to answer:
- How should we create the first hand-labeled benchmark?
- How should evals run in CI?
- Which metrics should be required for regression tests?
- How do we compare vector-only vs hybrid vs reranked retrieval?

Expected output:
- eval data format
- eval runner design
- metric definitions
- sample benchmark
- CLI examples

## Enhancement 4: Agentic RAG

Design a multi-step retrieval planner.

Example questions:
- “Compare NVDA and AMD AI risks.”
- “What changed in TSLA risk disclosures over the last three years?”
- “Explain why the model forecasts upside but filings show risk.”

The agent should:
- decompose complex questions
- run multiple retrievals
- compare evidence
- synthesize final answer with citations

Constraints:
- keep number of LLM calls controlled
- preserve citation grounding
- no autonomous financial advice
- no automatic ingestion during chat

Expected output:
- planner architecture
- tool calls
- prompt design
- safeguards
- tests
- examples

## Enhancement 5: GraphRAG

Design a lightweight knowledge graph layer.

Entities:
- companies
- products
- business segments
- competitors
- customers
- suppliers
- risks
- regulatory topics

Relationships:
- competes_with
- supplies_to
- depends_on
- exposed_to
- acquired
- operates_in
- mentions_risk

Questions to answer:
- Should graph extraction be manual, rules-based, or LLM-assisted?
- Should we use NetworkX, SQLite tables, or Neo4j later?
- How should graph retrieval combine with vector retrieval?
- What is the minimal GraphRAG MVP for this repo?

Expected output:
- graph schema
- extraction strategy
- storage choice
- retrieval integration
- example queries
- tests

## Enhancement 6: Retrieval + RL

Explore whether retrieval strategy can be optimized.

Research-style goal:
Learn how retrieval can be framed as a sequential decision problem.

Possible formulation:
- state: user question + metadata
- action: choose retrieval strategy, filters, top-k, reranker, document source
- reward: retrieval quality / answer faithfulness / citation correctness

Questions to answer:
- What is a practical non-overengineered MVP?
- Should this start as contextual bandits rather than full RL?
- What logs are needed to learn from retrieval outcomes?
- How can user feedback or benchmark results become reward signals?
- How does this connect to my broader interest in RL and decision systems?

Expected output:
- conceptual formulation
- MVP recommendation
- logging schema
- offline evaluation design
- possible bandit/RL roadmap

## Deliverables

Please produce:

1. A ranked roadmap of these six enhancements.
2. Recommended implementation order.
3. For each enhancement:
   - learning objective
   - user-facing value
   - architecture changes
   - new modules/files
   - tests
   - estimated complexity
   - risks
4. A suggested ADVANCED_RAG_TODO.md.
5. Suggested CLAUDE.md additions for advanced RAG work.
6. Clear recommendation on what to implement first.

Do not write implementation code yet.

Focus on design, sequencing, and learning value.