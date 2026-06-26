"""Agentic RAG (advanced-RAG A4) — the bounded ReAct controller for multi-hop SEC QA.

A self-contained, **bounded ReAct loop**: reason → retrieve → observe, with a *reflective* stop.
Each iteration is ONE cheap structured decision call (``_react_step`` → ``ReActStep``) that reasons
over the question + a compact summary of evidence-so-far and emits either **search** (retrieve more,
the *Act*) or **stop** (evidence sufficient, the reflective stop). The controller accumulates a
deduped ``RetrievedChunk`` union across steps and ends in the **existing** P7 guarded synthesis
(``answer_question``) over that union — "P7, but iterative".

This module is pure orchestration. It **reuses, never re-implements**:
- retrieval — ``build_retrieval_system`` (hybrid, $0/local) between steps;
- synthesis + guards — ``answer_question`` (citation guard + number grounding + the empty-evidence
  short-circuit that refuses with NO LLM call).

The loop's decisions emit only queries/filters — no numbers, no citations — so there is nothing to
hallucinate-ground inside the loop; the invariants are enforced once, at the terminal answer.

Budget: ≤ ``agentic_max_steps`` decision calls + 1 terminal answer ⇒ ≤4 LLM calls by default
(``agentic_max_steps=3``); all retrieval between steps is local/free. Bounded like ``run_backtest``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from stock_agent.llm.client import TextLLM
from stock_agent.logging_config import get_logger
from stock_agent.rag.read_path import build_retrieval_system
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.research._shared import loads_lenient
from stock_agent.research.bridge import bridge_candidates, is_bridging, load_alias_map
from stock_agent.research.prompts import REACT_SYSTEM, build_react_user
from stock_agent.research.synthesis import answer_question
from stock_agent.schemas.agentic import MultiStepAnswer, ReActStep, StepTrace
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

log = get_logger(__name__)

# Decision calls are cheap by construction (small structured output, no answer prose).
_DECISION_MAX_TOKENS = 400


def answer_multistep(
    question: str,
    *,
    settings: Settings,
    llm: TextLLM,
    retriever: RetrievalSystem | None = None,
    max_steps: int | None = None,
    per_step_k: int | None = None,
    max_evidence: int | None = None,
    alias_map: Mapping[str, Sequence[str]] | None = None,
) -> MultiStepAnswer:
    """Answer a multi-hop SEC question via a bounded ReAct loop ending in the guarded P7 synthesis.

    The loop runs up to ``max_steps`` cheap decision calls; each ``search`` retrieves more evidence
    (local, $0) and folds it into a deduped union capped at ``max_evidence``; a ``stop`` decision —
    or an empty/duplicate query (anti-loop) — ends gathering. For a *bridging* question a
    deterministic entity-bridge pass then adds the discovered related entity's filings (the pivot
    the LLM won't make; $0, gated by ``agentic_bridge_max_entities``; ``alias_map`` injectable for
    tests). The terminal ``answer_question`` runs the citation + number guards over the union and
    short-circuits to an honest refusal (no LLM call) when the union is empty. ``retriever`` is
    injectable for tests; defaults to the configured hybrid read path.
    """
    retriever = retriever or build_retrieval_system(settings)
    max_steps = max_steps if max_steps is not None else settings.agentic_max_steps
    per_step_k = per_step_k if per_step_k is not None else settings.agentic_per_step_k
    max_evidence = max_evidence if max_evidence is not None else settings.agentic_max_evidence

    evidence: list[RetrievedChunk] = []
    trace: list[StepTrace] = []
    # Anti-loop: retrieval requests already issued this run. The key is the EFFECTIVE request —
    # (query, scope) — NOT the query text alone, because a comparative multi-hop ("compare NVDA and
    # AMD risks") legitimately reuses one query string across different tickers; keying on the query
    # alone would mis-flag the second entity as a duplicate and stop before retrieving it.
    issued: set[tuple[str, str]] = set()

    for _ in range(max_steps):
        step = _react_step(llm, question, trace, evidence)
        if step.action == "stop":
            break
        query = (step.query or "").strip()
        if not query:
            log.info("agentic.anti_loop_break", reason="empty")
            break  # an empty query carries no new information -> terminate
        where = step.filter or (ChunkFilter(ticker=step.ticker) if step.ticker else None)
        # Scope-aware key: same query under a different ticker/filter is a NEW request, not a loop.
        key = (query, where.model_dump_json() if where is not None else "")
        if key in issued:
            log.info("agentic.anti_loop_break", reason="duplicate")
            break  # exact same (query, scope) again -> nothing new to gather
        issued.add(key)
        ev = retriever.retrieve(query, top_k=per_step_k, where=where)
        evidence = _dedup_union(evidence, ev.chunks)[:max_evidence]
        trace.append(
            StepTrace(
                thought=step.thought, query=query, ticker=step.ticker, n_retrieved=len(ev)
            )
        )

    n_loop_steps = len(trace)  # search hops the LLM chose (excludes the entity-bridge below)

    # Entity-bridge (the bridging fix): the loop won't pivot to a discovered related entity, so for
    # a bridging question fold in that entity's OWN filings deterministically ($0; no LLM). Gated by
    # config + the question shape, so single-entity runs are untouched.
    if settings.agentic_bridge_max_entities > 0 and is_bridging(question):
        amap = alias_map if alias_map is not None else load_alias_map()
        searched = {st.ticker for st in trace if st.ticker} | {rc.chunk.ticker for rc in evidence}
        for ticker, chunks in bridge_candidates(
            question, evidence,
            retriever=retriever, alias_map=amap, searched=searched,
            per_step_k=per_step_k, max_entities=settings.agentic_bridge_max_entities,
        ):
            evidence = _dedup_union(evidence, chunks)[:max_evidence]
            trace.append(
                StepTrace(
                    thought=f"entity-bridge: pivot to {ticker}'s own filings",
                    query=question, ticker=ticker, n_retrieved=len(chunks),
                )
            )

    # Terminal: the EXISTING guarded answer over the accumulated union (reused P7). Empty union ->
    # "Insufficient evidence found." with NO LLM call (handled inside answer_question).
    ans = answer_question(
        question, EvidenceSet(query=question, chunks=evidence), llm=llm
    )
    return MultiStepAnswer(
        answer=ans, trace=trace, n_steps=n_loop_steps, n_evidence=len(evidence), evidence=evidence
    )


def _react_step(
    llm: TextLLM, question: str, trace: list[StepTrace], evidence: list[RetrievedChunk]
) -> ReActStep:
    """One ReAct decision: a single cheap structured-output call → parsed ``ReActStep``.

    Parsed leniently (like P7's ``_RawAnswer``); a malformed/thin reply degrades to the schema
    default (``action="stop"``), which safely ends gathering rather than looping. A reply that does
    not parse at all (or fails schema validation) is treated the same way: one bad decision must not
    crash a multi-step run or discard evidence already gathered. (``pydantic.ValidationError`` and
    ``json``'s ``JSONDecodeError`` are both ``ValueError`` subclasses, so one except covers both.)
    """
    user = build_react_user(question, trace, evidence)
    raw = llm.complete_json(system=REACT_SYSTEM, user=user, max_tokens=_DECISION_MAX_TOKENS)
    try:
        return ReActStep.model_validate(loads_lenient(raw))
    except ValueError as exc:
        log.warning("agentic.decision_parse_failed", error=str(exc))
        return ReActStep()  # default action="stop": end gathering safely over what we have


def _dedup_union(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Append ``new`` chunks not already in ``existing`` (dedup by ``chunk_id``), order preserved.

    A chunk retrieved in multiple steps appears once; earlier (typically higher-scored) provenance
    is kept. Chunk identity is the stable key — verbatim-text dedup already happens inside each
    retriever's own ``retrieve`` call.
    """
    seen = {rc.chunk.chunk_id for rc in existing}
    out = list(existing)
    for rc in new:
        if rc.chunk.chunk_id in seen:
            continue
        seen.add(rc.chunk.chunk_id)
        out.append(rc)
    return out
