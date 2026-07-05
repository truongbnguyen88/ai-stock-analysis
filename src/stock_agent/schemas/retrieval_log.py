"""Retrieval telemetry log entry (advanced-RAG A6.1a) — the on-disk record of one decision.

A6.1 treats retrieval as a *contextual bandit*: a featurized query (context ``x``) → a policy picks
one retrieval **arm** ``a`` → earns a reward ``r``. To learn/evaluate a policy **offline** we must
first LOG each decision as it happens. One ``RetrievalLogEntry`` is that record: the context, the
chosen arm, the **propensity** ``mu(a|x)`` under the logging policy (the crucial quantity that makes
off-policy evaluation *unbiased* — IPS/SNIPS/DR all divide by it), what was retrieved, and any
downstream outcome. Logging is **default-OFF** and append-only; it never changes retrieval, only
observes it (see ``rag/retrieval_log.py``).

Design notes:
- ``context`` is stored as a self-describing ``{feature_name: value}`` map (not a bare vector), so
  a raw JSONL line is human-readable and robust to featurizer version drift. The A6.1b featurizer
  serializes its ``ContextVector`` into this map at log time.
- ``propensity`` is ``None`` for a *deterministic* logging policy (no exploration) — such rows
  cannot support importance-weighted OPE and are skipped by the estimators; a uniform-random logger
  writes ``1/n_arms`` on every row (the exact, known ``mu`` A6.1 uses to synthesize its dataset).
- Numbers-from-models and grounding invariants are untouched: this only records ids/scores/outcomes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ScoredChunkRef(BaseModel):
    """A retrieved chunk reference — its id and relevance score (higher = closer).

    Only the id + score are logged (not the chunk text): the corpus is the source of truth, so a
    log stays small and the chunk can always be re-materialized by id for later analysis.
    """

    chunk_id: str
    score: float


class RetrievalLogEntry(BaseModel):
    """One logged retrieval decision — the atomic row of the A6.1 off-policy dataset.

    Fields split into (1) the bandit tuple ``(x, a, mu, r)`` needed for OPE — ``context`` (``x``),
    ``action`` (``a``), ``propensity`` (``mu(a|x)``), ``reward`` (``r``, optional/deferred); and
    (2) provenance/debug — ``query``/``ticker``, ``retrieved`` ids+scores, optional downstream
    ``answer`` + ``guard_ok`` (the P7 citation/number guard outcome, when a synthesis ran), and the
    ``seed`` of the logging policy for reproducibility.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str
    ticker: str | None = None
    context: dict[str, float] | None = None  # featurized context x (feature_name -> value)
    action: str  # chosen arm name (a lattice/graph system name)
    propensity: float | None = None  # mu(a|x); None if the logging policy was deterministic
    retrieved: list[ScoredChunkRef] = Field(default_factory=list)
    answer: str | None = None  # optional downstream synthesis text (usually absent in $0 A6.1 logs)
    guard_ok: bool | None = None  # citation/number guard outcome, if a synthesis ran (else None)
    reward: float | None = None  # optional reward/feedback (deferred: A6.1 computes r offline)
    seed: int | None = None  # logging-policy seed (reproducibility)
