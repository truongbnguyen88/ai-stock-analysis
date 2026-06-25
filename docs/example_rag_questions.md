# Example RAG questions — when to loop (agentic) vs. single-shot

Reference + worked catalog for the RAG layer: which SEC-filing questions need the **agentic** A4
loop (`research_multistep` / `rag ask`) and which stay on **single-shot** retrieval
(`search_filings` / `rag query`). Also a useful seed for the (deferred) multi-step eval set.

- Theory + mechanism → [rag_concepts.md §15](rag_concepts.md), [rag_implementation_notes.md §A4](rag_implementation_notes.md).
- All examples assume the relevant tickers' filings are **already ingested** (the tools never fetch
  on the fly). Tickers used: NVDA, AMD, INTC, TSLA, MU (Micron), TSM (TSMC), AAPL, MSFT, AVGO.

---

## The one decision rule

Route to **agentic / multi-hop** when answering requires **gathering and *connecting* evidence that a
single top-k retrieval can't return together** — because of one (or both) of these triggers:

- **(D) Disjoint-evidence** — the needed passages live in *different* filings / sections / periods /
  entities. One query returns a blurred mix or only one side.
- **(C) Conditional** — the *right* next query is only knowable *after* seeing an earlier result
  (a query that depends on a prior finding).

Otherwise — **one entity, one topic, one neighborhood of the corpus** — stay single-shot. The loop
would only add cost and latency.

> Rule of thumb: **"different filings/sections/periods/entities, or a follow-up that depends on a
> prior finding" → agentic; "one entity, one topic" → single-shot.**

---

## Questions that REQUIRE the agentic loop

Ten categories, each mapped to its trigger, with why one retrieval fails and several example
questions.

### 1. Comparative / multi-entity — trigger (D)
Compare two or more companies, segments, or products along some dimension. Each entity's evidence is
in a different filing; a query scoped to one ticker returns one side, an unscoped query blurs them.

- "Compare NVDA's and AMD's data-center risk factors."
- "How do NVDA, AMD, and INTC each describe their exposure to U.S.–China export controls?"
- "Contrast NVDA's and AVGO's customer-concentration disclosures."
- "Which of AAPL or MSFT discloses more supplier-concentration risk, and how do they differ?"
- "Compare the gross-margin commentary in NVDA's vs Micron's latest MD&A."

### 2. Temporal / change-over-time — trigger (D)
How a disclosure evolved across filings or years; the trend in a risk/segment narrative. Needs each
period retrieved *separately*, then contrasted.

- "How did TSLA's full-self-driving / autonomy risk language change from 2022 to 2025?"
- "What new risk factors did NVDA add between its FY2023 and FY2025 10-K?"
- "How has Micron's description of memory-market cyclicality shifted over the last three 10-Ks?"
- "Did INTC's foundry-strategy disclosure change after its 2024 reorganization?"
- "Track how NVDA's data-center revenue commentary evolved across the last four 10-Qs."

### 3. Bridging / discover-then-follow-up — trigger (C)
Step 1 surfaces entities (suppliers, customers, competitors, subsidiaries, acquired companies, named
products); step 2 queries *those specific names*. The second query does not exist until step 1 is read.

- "Which of NVDA's named suppliers also flag export-control or China-related risk in their own filings?"
- "NVDA names certain foundry partners — do those partners disclose capacity-constraint risk?"
- "Which companies that NVDA lists as competitors also claim AI-accelerator leadership in their filings?"
- "What products did AVGO acquire, and what risks do the acquired businesses carry?"
- "Identify TSLA's named lithium/battery suppliers, then check which flag raw-material concentration risk."

### 4. Compound / multi-part — trigger (D)
A single question bundling several sub-questions whose answers sit in *different sections* (Business /
Risk Factors / MD&A / liquidity).

- "What is NVDA's AI strategy, which stated risks most threaten it, and how is it funding the buildout?"
- "Summarize AMD's data-center strategy and the specific risks it ties to that strategy."
- "What are MSFT's main AI revenue drivers and the regulatory risks attached to them?"
- "Describe TSLA's energy-storage business and both its growth drivers and disclosed risks."

### 5. Aggregation / set-spanning ("list every…") — trigger (D)
The answer spans many disjoint passages that a single top-k would truncate; enumerate-all questions.

- "List every operating segment's stated headwinds in NVDA's latest 10-K."
- "Summarize all litigation and legal proceedings NVDA discloses."
- "What are all the material risks Micron groups under 'industry and market' risk?"
- "Enumerate each of AAPL's reportable segments and one risk specific to each."
- "Pull together every place NVDA mentions export controls across its latest 10-K and recent 8-Ks."

### 6. Causal / mechanism-chain — trigger (D), often (C)
Link a cause to an effect that live in different sections; retrieve both and connect them.

- "How does NVDA's export-control risk connect to its data-center revenue outlook?"
- "How does Micron's China-revenue concentration relate to the demand risks in its MD&A?"
- "Trace how TSLA's supply-chain risk feeds into its stated margin pressures."
- "How does INTC's capital-spending plan relate to the liquidity risks it discloses?"

### 7. Consistency / cross-section corroboration — trigger (D)
Does claim X in section A square with claim Y in section B? Pull both sides and compare.

- "Does NVDA management's MD&A optimism square with its risk-factor disclosures?"
- "Is AMD's stated revenue-recognition policy consistent with how it describes its segments?"
- "Does TSLA's growth narrative in MD&A match the demand risks in its risk factors?"
- "Do NVDA's forward-looking demand statements conflict with any cautionary supply-side language?"

### 8. Conditional / contingent — trigger (C)
The answer depends on a fact you must discover first, which then determines the next query.

- "If NVDA discloses any going-concern or debt-covenant language, what triggers it?"
- "Does Micron disclose any restructuring charges, and if so, what's the stated cause?"
- "If TSLA reports any material weakness in internal controls, what remediation does it describe?"
- "Does AMD mention any goodwill impairment, and if so, which segment and why?"

### 9. Decomposition of a vague / broad ask — trigger (D/C)
Open-ended questions that need several angles, each its own retrieval.

- "What should I understand about NVDA's competitive position?" (competitors, moat, customer
  concentration, supply risk, pricing power)
- "Give me the key qualitative picture of Micron's business and risks." (segments, cyclicality,
  China, capex)
- "What are the main things to know from TSLA's latest 10-K?"
- "What drives AMD's business and what could break it?"

### 10. Cross-document reconciliation (10-K + later 8-Ks / 10-Qs) — trigger (D), often (C)
Combine a baseline filing with later amendments/updates that live in separate documents.

- "Has anything in NVDA's risk factors been updated or superseded by its most recent 8-Ks?"
- "Do NVDA's recent 10-Qs revise the data-center outlook stated in the last 10-K?"
- "What did AMD's latest 8-K change relative to the guidance in its prior 10-Q?"
- "Reconcile TSLA's annual risk factors with any new risks raised in its interim filings this year."

---

## Questions that do NOT require the loop (single-shot `search_filings`)

One entity, one topic, one neighborhood of the corpus — a single top-k retrieval covers it. Using the
loop here only adds cost and latency.

| Question | Why single-shot is enough |
|---|---|
| "What are NVDA's risk factors?" | One entity, one section (Item 1A). |
| "What does NVDA's 10-K say about gross margin?" | One topic, one filing; lands in MD&A. |
| "How does management describe data-center demand?" | Single qualitative topic, one neighborhood. |
| "What is NVDA's revenue-recognition policy?" | One accounting-policy passage. |
| "Summarize NVDA's business / what it does." | One section (Item 1, Business). |
| "What legal proceedings does NVDA disclose?" | Single section (unless asked to *aggregate across filings* → then it's #5). |
| "What does NVDA say about its supply of HBM memory?" | One topic in one filing. |
| "What dividend or buyback policy does AAPL state?" | One capital-return passage. |

**Borderline calls (lean single-shot unless the question forces a second region):**

- "What are NVDA's *biggest* risks?" → single-shot (one section) — *unless* it asks to rank against
  another company or another year (then #1 or #2).
- "What does NVDA say about China?" → single-shot if you just want the mentions; agentic if it's
  "compare NVDA's vs AMD's China exposure" (#1) or "how did it change" (#2).
- "Tell me about NVDA's suppliers." → single-shot (just list them); agentic once you ask what *those*
  suppliers disclose (#3).

The tipping point is always the same: **does answering need a second, different region of the corpus,
or a query that depends on a first finding?** If yes → agentic; if no → single-shot.

---

## Using these for evaluation (note)

This catalog doubles as the seed for the deferred A4 **multi-step eval set** (reuse the A1 harness):
label each multi-hop question with the chunk_ids that *must* appear in the union for a correct answer,
then score whether the loop's accumulated evidence covers them (retrieval recall) and whether the
final cited answer is faithful (citation accuracy). The single-shot list is the negative control —
those should route to `search_filings`, and pushing them through the loop should not improve the answer.
