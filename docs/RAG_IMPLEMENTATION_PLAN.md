# RAG Implementation Plan for AI Stock Analysis Agent

## Background

This repository already supports:

- Historical market data retrieval
- Technical indicators (MA20, MA50, MA200, RSI, MACD)
- Volatility and return calculations
- News retrieval
- LLM-based news summarization
- Probabilistic upside/downside scenario modeling
- Markdown research report generation

The next phase is to evolve the project from a stock signal analyzer into an AI-powered equity research assistant by introducing Retrieval-Augmented Generation (RAG).

---

# Project Objective

Transform:

text AI Stock Analysis Agent 

into:

text AI Equity Research Assistant 

by adding:

- Financial document ingestion
- Embeddings
- Vector search
- Retrieval
- Grounded generation
- Evidence-based research reports

The existing quantitative analysis engine remains unchanged and becomes one component of the final research workflow.

---

# Guiding Principles

## Educational Use Only

This project is for:

- learning
- experimentation
- research

It is NOT intended to provide financial advice.

---

## Grounded Answers

The system must:

- retrieve evidence
- cite evidence
- explain evidence

The system must NOT:

- hallucinate evidence
- fabricate citations
- fabricate probabilities
- issue buy/sell recommendations

---

## Separation of Responsibilities

### Quantitative Models

Responsible for:

- technical indicators
- volatility metrics
- probability forecasts
- scenario modeling

### RAG Layer

Responsible for:

- document retrieval
- evidence retrieval
- citation generation

### LLM

Responsible for:

- summarization
- synthesis
- explanation
- research memo generation

The LLM must not override model-generated outputs.

---

# RAG Architecture

## Document Sources

### SEC Filings

Supported forms:

- 10-K
- 10-Q
- 8-K

### Earnings Call Transcripts

Examples:

- quarterly earnings calls
- analyst Q&A sessions

### Investor Presentations

Examples:

- investor day decks
- earnings presentations

### News Articles

Articles already collected through:

- Alpha Vantage
- Finnhub
- Marketaux

### Internal Research Reports

Previously generated reports may also be indexed.

---

# Repository Structure

text src/stock_agent/    documents/     sec_downloader.py     ticker_cik.py     parsers.py     metadata.py     manifest.py    rag/     chunking.py     embeddings.py     vector_store.py     retriever.py     pipeline.py     prompts.py    research/     synthesis.py     memo.py     evidence.py    reports/   models/   indicators/   news/  data/    raw/   processed/   vectorstore/   reports/ 

---

# Phase 1: SEC Document Downloading

## Objective

Download public SEC filings into local storage.

The downloader should:

- read tracked tickers from the existing repository
- resolve ticker → CIK
- download filings
- save metadata
- track download history

Supported forms:

- 10-K
- 10-Q
- 8-K

---

## CLI Examples

bash python -m stock_agent documents download-sec --ticker NVDA  python -m stock_agent documents download-sec \   --ticker AVGO \   --forms 10-K 10-Q  python -m stock_agent documents download-sec --all  python -m stock_agent documents download-sec \   --all \   --limit 2 

---

# Storage Layout

text data/    raw/      sec/        NVDA/          10-K/           2025-02-26/             filing.html             metadata.json          10-Q/           2025-05-28/             filing.html             metadata.json    processed/      sec/        NVDA/          10-K/           2025-02-26/             text.txt             chunks.jsonl    vectorstore/    reports/ 

Raw files should never be overwritten.

---

# Phase 2: Parsing

Convert:

- SEC HTML
- SEC TXT
- PDFs
- earnings transcripts
- investor presentations

into normalized text.

Metadata must include:

json {   "ticker": "NVDA",   "document_type": "10-K",   "source": "SEC",   "source_url": "...",   "filing_date": "...",   "section": "...",   "document_id": "...",   "ingested_at": "..." } 

---

# Phase 3: Chunking

Chunk documents for retrieval.

Goals:

- preserve context
- preserve section boundaries
- preserve management discussion sections
- preserve risk factor sections
- preserve earnings Q&A sections

Avoid:

- tiny chunks
- giant chunks
- metadata loss

---

# Phase 4: Embeddings

Create an embedding abstraction layer.

## Primary

OpenAI embeddings

## Optional

Sentence-transformers

The implementation should support future providers.

---

# Phase 5: Vector Store

Evaluate:

- ChromaDB
- FAISS
- LanceDB

Select one MVP solution.

Provide abstraction so the vector backend can be swapped later.

---

# Phase 6: Retrieval

Support:

- ticker filters
- document type filters
- date filters
- source filters
- top-k retrieval

Example:

bash python -m stock_agent rag query \   --ticker NVDA \   --question \   "What AI growth drivers were highlighted by management?" 

Return:

- retrieved chunks
- similarity scores
- citations

---

# Phase 7: Grounded Question Answering

Requirements:

- retrieve first
- answer second
- cite sources
- cite document dates
- cite sections when available

If evidence is insufficient:

text Insufficient evidence found. 

No hallucinations.

---

# Phase 8: Integrated Research Reports

Add:

bash python -m stock_agent research \   --ticker NVDA 

Workflow:

1. Technical analysis
2. Probability forecasts
3. News analysis
4. RAG retrieval
5. Evidence synthesis
6. Research memo generation

---

# Research Memo Sections

- Executive Summary
- Technical Indicators
- Probability Scenarios
- Recent News
- Management Commentary
- Business Drivers
- Risk Factors
- Bullish Evidence
- Bearish Evidence
- Uncertainty Notes
- Source Citations

---

# Testing Requirements

Add tests for:

- SEC downloading
- ticker → CIK mapping
- parsing
- metadata validation
- chunking
- embeddings
- vector insertion
- retrieval
- citation formatting
- empty retrieval handling

Use mocked SEC responses.

---

# MVP Scope

MVP includes:

- SEC downloading
- local document storage
- parsing
- chunking
- embeddings
- vector database
- retrieval
- cited answers
- integration with stock reports

No agentic workflows yet.

---

# V1 Scope

Add:

- earnings transcript ingestion
- investor presentation ingestion
- hybrid search
- reranking
- quarter-over-quarter document comparison
- advanced report generation

---

# Future Roadmap

Potential future capabilities:

- agentic retrieval planning
- multi-step retrieval
- contradiction detection
- evidence quality scoring
- company comparison reports
- sector-level research
- portfolio-level research
- memory of prior reports

---

# Deliverables

Before implementation:

1. Review current repository structure.
2. Identify ticker universe location.
3. Produce architecture proposal.
4. Recommend vector database.
5. Recommend embedding model.
6. Produce implementation roadmap.
7. Create RAG_TODO.md.
8. Update CLAUDE.md.
9. Estimate MVP effort.
10. Identify technical risks.

Do not begin implementation immediately.

Start with architecture review and planning.