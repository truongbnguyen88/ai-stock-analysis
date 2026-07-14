"""Seating (A6.2-E7) — how a fan-out sweep's N branches become the capped synthesis union.

A sweep retrieves ``N`` branches (one per discovered candidate), but only a bounded union reaches
synthesis. *Seating* is the rule that picks which retrieved chunks survive that cap, and it is a
distinct decision from retrieval: E3 fixed **reachability** (can the policy search the target at
all?) and E6 fixed **branch retrieval** (does the target's branch surface the span?), but a chunk
that is retrieved and then evicted by the cap never reaches the metric — *a retrieval that never
reaches the synthesis context did not happen* (``rag_concepts.md`` §20.4).

**The defect this module fixes.** The E3 rule was ``breadth_first``: merge branches round-robin by
rank, cap at ``len(union) + N``. With a hop-1 union of 6 and N=19 that cap is exactly 25, and the
round-robin's first 19 entries are exactly every branch's rank-0 chunk — so the rule degenerates to
*"seat rank 0, or you do not exist."* Measured on the held-out fold, the target's span-bearing chunk
is its branch's rank-0 hit only 27.3% of the time, so **12 of the 21 episodes whose branch retrieved
the span were thrown away by the cap**. Worse, of the 25 chunks that did seat, 18 were rank-0 chunks
from companies that are not the target — the evidence budget was spent on noise.

**Why a cross-encoder is the right instrument.** ``breadth_first`` exists because RRF scores are
*rank-derived* (``1/(k+rank)``), so every branch's rank-1 chunk carries an identical score and
sorting the pool by score degenerates to insertion order — rank is the only cross-branch quantity
RRF offers. A cross-encoder does not share that defect: it scores ``(query, chunk)`` jointly on an
absolute scale, so its scores **are** comparable across branches. Pool every branch's chunks,
rescore
them against the topic query, and the target's chunk competes on merit instead of being locked out
by
"your branch already spent its one seat".

**The information argument, stated precisely.** E3 established ``I(Y; E₁) ≈ 0`` — hop-1 evidence
carries no signal about *which* candidate discloses the topic, so candidates cannot be ranked *a
priori* and must be swept. That says nothing about ``I(Y; E₂)``. Once each candidate's filings have
actually been searched for the topic, the retrieved content is precisely the observation that
discriminates: the target discloses the topic, most of the other N−1 do not. Seating is an **a
posteriori** decision, and the impossibility result does not reach it. ``top_branches`` exploits
this
directly — it discards all but the ``b`` best-scoring branches and *gains* coverage while shrinking
the context.

Label-free by construction: the reranker sees the query and the chunk texts, never the gold aspects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from stock_agent.rag.rerank import (
    _LOCAL_DEFAULT_MODEL,
    _VOYAGE_DEFAULT_MODEL,
    FastEmbedReranker,
    NoOpReranker,
    Reranker,
    VoyageReranker,
)
from stock_agent.schemas.retrieval import RetrievedChunk
from stock_agent.settings import Settings

# breadth_first  — the E3 rule; kept so every pre-E7 number stays reproducible (and it is the only
#                  correct rule when no reranker is configured, since RRF scores cannot be pooled).
# pooled_rerank  — rescore the pooled branches, take the best `cap` chunks.
# top_branches   — rescore, keep only the `b` best-scoring BRANCHES, `m` chunks each (smallest
#                  context; exploits the a posteriori argument above most aggressively).
SeatingRule = Literal["breadth_first", "pooled_rerank", "top_branches"]


def build_seating_reranker(settings: Settings) -> Reranker:
    """The cross-encoder used to SEAT a fan-out — deliberately separate from ``rerank_provider``.

    ``rerank_provider`` governs whether a *retrieval arm* reranks (A2, default-OFF because reranking
    is catastrophic on hop 1: it buries the paragraph that names the competitors). Seating is the
    opposite regime — the pool is already scoped to one topic across many companies, and reranking
    is what makes it work. Tying the two to one switch would force a single answer to two questions
    with **opposite** signs, so seating gets its own provider knob.
    """
    provider = settings.rl_seating_provider
    if provider == "local":
        return FastEmbedReranker(settings.rl_seating_model or _LOCAL_DEFAULT_MODEL)
    if provider == "voyage":
        return VoyageReranker(settings, model=settings.rl_seating_model or _VOYAGE_DEFAULT_MODEL)
    return NoOpReranker()


def effective_rule(rule: SeatingRule, reranker: Reranker | None) -> SeatingRule:
    """The rule that will actually run, given the reranker we actually have.

    **Call this once and feed the result to BOTH ``seat_branches`` and ``fanout_cap``.** The two
    must agree: the ordering and the cap are halves of one guarantee, and a reranked rule's flat cap
    is only safe *because* the reranked order puts the best chunks first. Pair a breadth-first
    ordering (the fallback when no reranker is available) with a reranked rule's flat cap and the
    sweep seats the alphabetically-first candidates and drops the rest — the E3 bug, rebuilt out of
    two individually reasonable halves.
    """
    if reranker is None or reranker.name == NoOpReranker.name:
        return "breadth_first"
    return rule


def merge_breadth_first(branches: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Round-robin by rank: every branch's rank-1 chunk, then every rank-2, … (the E3 merge).

    Single branch ⇒ its own order, unchanged. For a sweep the interleave is load-bearing: the union
    dedup appends in insertion order and is capped, so *concatenating* the branches would fill the
    union with the alphabetically-first candidates and starve the rest.
    """
    if len(branches) == 1:
        return list(branches[0])
    out: list[RetrievedChunk] = []
    for rank in range(max((len(b) for b in branches), default=0)):
        out.extend(b[rank] for b in branches if rank < len(b))
    return out


def _pool(branches: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Flatten the branches into one candidate list, preserving first-seen order (dedup by id)."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for branch in branches:
        for rc in branch:
            if rc.chunk.chunk_id not in seen:
                seen.add(rc.chunk.chunk_id)
                out.append(rc)
    return out


def seat_branches(
    branches: Sequence[Sequence[RetrievedChunk]],
    *,
    rule: SeatingRule,
    query: str,
    reranker: Reranker | None = None,
    top_branches: int = 3,
    per_branch: int = 3,
) -> list[RetrievedChunk]:
    """Order a sweep's branches into the candidate list the union cap will truncate.

    Returns chunks best-first; the caller applies the cap (which differs per rule — see
    ``fanout_cap``). ``rule="breadth_first"`` needs no reranker; the other two fall back to it when
    ``reranker is None``, so a misconfiguration degrades to the E3 behaviour rather than crashing.

    ``top_branches``/``per_branch`` (b, m) apply to ``rule="top_branches"`` only: keep the b
    branches
    whose best chunk scores highest, m chunks from each ⇒ at most b·m chunks regardless of N.

    **A no-op reranker must fall back to breadth-first, not to the pooled order.** The pooled order
    is
    branch-major (branch 0's chunks, then branch 1's, …), so seating it under a cap would spend the
    whole budget on the alphabetically-first candidates — reintroducing the exact slot bias E3
    exists to remove. A reranker that does not reorder therefore has to be treated as *no reranker
    at all*, or a misconfigured provider silently regresses the environment.
    """
    reordering = reranker is not None and reranker.name != NoOpReranker.name
    if rule == "breadth_first" or not reordering or not branches:
        return merge_breadth_first(branches)
    assert reranker is not None  # narrowed by `reordering`

    ranked = reranker.rerank(query, _pool(branches))
    if rule == "pooled_rerank":
        return ranked

    # top_branches: group the reranked pool by source company, preserving the reranked order within
    # each group. dict insertion order == descending best-score order, because `ranked` is sorted,
    # so
    # the FIRST time a ticker appears is at its best chunk — no separate max() pass is needed.
    by_ticker: dict[str | None, list[RetrievedChunk]] = {}
    for rc in ranked:
        by_ticker.setdefault(rc.chunk.ticker, []).append(rc)
    keep = list(by_ticker)[:top_branches]
    return [rc for ticker in keep for rc in by_ticker[ticker][:per_branch]]


def fanout_cap(
    *,
    rule: SeatingRule,
    n_union: int,
    n_branches: int,
    max_evidence: int,
    top_branches: int = 3,
    per_branch: int = 3,
) -> int:
    """The union cap for one fan-out step — the budget that makes each rule's guarantee true.

    ``breadth_first`` must widen to ``n_union + n_branches``: the E3 rule's guarantee is *one seat
    per
    branch*, and under a flat cap a 19-candidate sweep on a HARD episode (6 hop-1 chunks ⇒ 14 slots)
    would seat the first 14 candidates **alphabetically** and silently drop 5 — a smaller copy of
    the
    very slot bias E3 removes.

    The reranked rules make the opposite guarantee — *the best chunks seat, wherever they came
    from* —
    so they need no per-branch allowance and keep the flat ``max_evidence``. ``top_branches`` is
    bounded tighter still (``n_union + b·m``), which is why it can beat the E3 rule on coverage
    while
    handing the synthesis LLM a **smaller** context: it is a strict Pareto improvement, not a trade.
    """
    if rule == "breadth_first":
        return max(max_evidence, n_union + n_branches)
    if rule == "top_branches":
        return min(max_evidence, n_union + top_branches * per_branch)
    return max_evidence
