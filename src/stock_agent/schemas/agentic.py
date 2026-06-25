"""Agentic-RAG domain models (advanced-RAG A4) — the bounded ReAct loop's typed IO.

A4 is a self-contained, bounded **ReAct** loop (reason → retrieve → observe, with a reflective
stop) for multi-hop SEC QA — "P7, but iterative". Each iteration emits one ``ReActStep`` (a cheap
structured decision: search-more vs. stop); the controller accumulates a deduped ``RetrievedChunk``
union and ends in the **existing** guarded P7 synthesis (``answer_question``). These schemas are the
loop's IO contract: the per-step LLM decision, a transparent per-step trace, and the final result.

The loop's decisions emit only queries/filters — never numbers or citations — so there is nothing
to hallucinate-ground inside the loop; the reused P7 guards run only at the terminal answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_agent.schemas.research import GroundedAnswer
from stock_agent.schemas.retrieval import ChunkFilter


class ReActStep(BaseModel):
    """One ReAct decision: the model's reasoning plus the next action (search or stop).

    Parsed leniently from the decision LLM call (like P7's ``_RawAnswer``). ``action="search"``
    means "retrieve more" — ``query`` carries the next (possibly observation-dependent) search,
    optionally scoped by ``ticker`` or a full ``filter``. ``action="stop"`` is the reflective stop:
    the gathered evidence is judged sufficient. NO numbers, NO citations are ever emitted here.
    """

    thought: str = ""  # the reasoning for this step (transparency only; not grounded)
    # Default-stop is the safe terminal if a reply parses thin (ends gathering, never loops).
    action: Literal["search", "stop"] = "stop"
    query: str | None = None  # the next retrieval query when action="search"
    ticker: str | None = None  # convenience scope; folded into a ChunkFilter if no explicit filter
    filter: ChunkFilter | None = None  # full metadata predicate (overrides ticker when set)


class StepTrace(BaseModel):
    """A transparent record of one executed search step (debug + tool output + tests)."""

    thought: str
    query: str
    ticker: str | None = None
    n_retrieved: int  # chunks this step's retrieval returned (pre-union; provenance of the step)


class MultiStepAnswer(BaseModel):
    """The result of a bounded ReAct run: the terminal grounded answer plus the loop trace."""

    answer: GroundedAnswer  # the reused P7 guarded answer over the accumulated evidence union
    trace: list[StepTrace] = Field(default_factory=list)  # executed search steps, in order
    n_steps: int = 0  # number of search steps executed (== len(trace))
    n_evidence: int = 0  # size of the deduped evidence union handed to the terminal synthesis
