"""Reward oracle adapter for the retrieval policy (advanced-RAG A6.1c) — the ``r`` in the bandit.

A6.1 learns a policy offline against a **reward** for each (context, arm) pair. This module turns a
labeled benchmark query + a retrieval arm into a scalar reward, reusing the *existing* A1/A6.0
metrics so the reward can never drift from what those harnesses measure (leakage rule 3):

    r(x, a) = quality(a on x) - lambda_cost * cost(a)                      (lambda_f term deferred)

- **quality** is retrieval-only and **$0** (no synthesis): for a single-shot ``LabeledQuery`` it is
  ``evaluate_query(...).ndcg`` (graded nDCG@k); for a ``MultiHopQuery`` it is the **single-shot**
  aspect ``coverage`` of one ``retrieve()`` — exactly the single-shot baseline ``evaluate_multihop``
  computes, so reward == metric. Both ∈ [0, 1].
- **cost(a)** is a *static per-arm* latency/compute proxy (``DEFAULT_ARM_COSTS``): the reward-hack
  guard — an arm must earn its extra cost in quality or it loses. ``lambda_cost`` (small, default
  0.05) sets the price of compute in quality units.
- The **faithfulness** penalty ``lambda_f`` is deferred (it needs a synthesis call A6.1 never runs);
  the field/knob exists so A6.2 / opt-in synth-in-loop can switch it on.

Because the oracle is deterministic and $0, ``reward_matrix`` evaluates *every arm on every query* —
the full-information reward matrix ``R[i, j]``. That matrix is (i) the ground-truth policy value
used to check the off-policy estimators (A6.1d) and (ii) the pool a uniform logging policy samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from stock_agent.rag.eval import LabeledQuery, evaluate_query
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.settings import Settings

if TYPE_CHECKING:
    from stock_agent.schemas.documents import DocumentChunk

# Static per-arm cost proxy (relative latency/compute units): dense is free; hybrid adds a BM25
# pass; rerank adds a cross-encoder; graph adds traversal + scoped fetches. Tune/sensitivity-test in
# the verdict run (A6.1f). Keys are the CANONICAL arm names (build_named_system / "graph"), NOT
# ``system.name`` (which is decorated, e.g. "graph(hybrid)") — so cost lookup is by arm, not by the
# system's display name.
DEFAULT_ARM_COSTS: dict[str, float] = {
    "dense": 0.0,
    "hybrid": 0.1,
    "reranked": 0.3,
    "hybrid+rerank": 0.4,
    "graph": 0.3,
}


def _quality(
    system: RetrievalSystem,
    labeled: LabeledQuery | object,
    *,
    settings: Settings,
    corpus_chunks: Sequence[DocumentChunk] | None,
    top_k: int | None,
) -> float:
    """Retrieval-only quality ∈ [0, 1], dispatched on the label type (nDCG vs single-shot coverage).

    ``LabeledQuery`` → graded ``nDCG@k`` (needs ``corpus_chunks`` for the ideal-DCG pool + recall
    base, ticker-scoped inside ``evaluate_query``). ``MultiHopQuery`` → aspect ``coverage`` of one
    **unscoped** ``retrieve()`` (identical to ``evaluate_multihop``'s single-shot baseline).
    """
    k = top_k if top_k is not None else settings.rag_top_k
    if isinstance(labeled, LabeledQuery):
        if corpus_chunks is None:
            raise ValueError(
                "LabeledQuery reward needs corpus_chunks (ideal-DCG pool + recall base)"
            )
        return evaluate_query(system, labeled, top_k=k, corpus_chunks=corpus_chunks).ndcg
    # Lazy import: research.multistep_eval imports rag → a top-level import here would be circular
    # (same lazy pattern rag.read_path uses for graph). Reusing `coverage` keeps reward == metric.
    from stock_agent.research.multistep_eval import MultiHopQuery, coverage

    if isinstance(labeled, MultiHopQuery):
        evidence = system.retrieve(
            labeled.question, top_k=k
        )  # unscoped: matches the metric baseline
        frac, _, _ = coverage(evidence.chunks, labeled.aspects)
        return frac
    raise TypeError(f"unsupported label type for reward: {type(labeled).__name__}")


def reward(
    system: RetrievalSystem,
    labeled: LabeledQuery | object,
    *,
    arm: str,
    settings: Settings,
    corpus_chunks: Sequence[DocumentChunk] | None = None,
    lambda_cost: float | None = None,
    lambda_faithfulness: float = 0.0,
    arm_costs: Mapping[str, float] = DEFAULT_ARM_COSTS,
    top_k: int | None = None,
) -> float:
    """Composite reward for running ``arm`` (=``system``) on ``labeled``: quality − λ_c·cost.

    ``arm`` is the canonical arm name used for the cost lookup (decoupled from ``system.name``).
    ``lambda_cost`` defaults to ``settings.reward_lambda_cost``. The ``lambda_faithfulness`` term is
    plumbed but 0 by default (deferred — no synthesis runs in A6.1).
    """
    lam_c = lambda_cost if lambda_cost is not None else settings.reward_lambda_cost
    quality = _quality(system, labeled, settings=settings, corpus_chunks=corpus_chunks, top_k=top_k)
    cost = arm_costs.get(arm, 0.0)
    # lambda_faithfulness * 0 today (no guard-failure signal without synthesis); term kept visible.
    return quality - lam_c * cost - lambda_faithfulness * 0.0


def reward_matrix(
    systems: Mapping[str, RetrievalSystem],
    queries: Sequence[LabeledQuery | object],
    *,
    settings: Settings,
    corpus_chunks: Sequence[DocumentChunk] | None = None,
    lambda_cost: float | None = None,
    arm_costs: Mapping[str, float] = DEFAULT_ARM_COSTS,
    top_k: int | None = None,
) -> np.ndarray:
    """Full-information reward matrix ``R[i, j]`` = reward of arm ``j`` on query ``i`` (all $0).

    Column order follows ``systems`` insertion order (use that same order for the arm index in the
    policy/OPE code). This is the ground truth OPE is validated against and the pool the logging
    policy subsamples. ``float64`` array of shape ``(len(queries), len(systems))``.
    """
    arms = list(systems)
    matrix = np.zeros((len(queries), len(arms)), dtype=np.float64)
    for i, q in enumerate(queries):
        for j, arm in enumerate(arms):
            matrix[i, j] = reward(
                systems[arm],
                q,
                arm=arm,
                settings=settings,
                corpus_chunks=corpus_chunks,
                lambda_cost=lambda_cost,
                arm_costs=arm_costs,
                top_k=top_k,
            )
    return matrix
