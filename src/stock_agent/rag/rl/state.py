"""MDP state featurizer for full retrieval RL (advanced-RAG A6.2a) — the sequential state ``s_t``.

A6.1's bandit context ``x`` (``rag/policy_features.featurize``) describes the *query* and is static
across an episode. A6.2 is sequential: the policy must condition on **what it has gathered so far**,
so the state adds a **dynamic evidence-summary block** to the static query block:

    s_t = featurize(query)                     # 11-dim static block (A6.1b, reused verbatim)
          ⊕ [step_idx, budget_remaining,       # where we are in the horizon
             n_chunks, n_tickers_covered,      # size / breadth of the accumulated union
             n_sections_covered,
             n_discovered_unretrieved,         # THE crux: entities named in the union but not yet
             last_new_chunks]                  # retrieved against + novelty of the last action

**Leakage line (Q4.3, non-negotiable).** The state is **label-free**: it is a pure function of the
trajectory-so-far (query text + accumulated evidence + the alias map / graph universe) and **never**
reads the gold aspects/stratum. `featurize_state` therefore takes NO aspects. The reward (union
*coverage* against the aspects) is label-using but lives **only inside the simulator** — if any
label-derived quantity leaked into the state, the frozen policy could not run on a real user query.

``n_discovered_unretrieved`` is the encoded form of "NVDA's filing just named Micron, but I have not
pulled Micron's filing yet" — the signal that makes the bridge decision learnable. It reuses
``research.bridge.mentioned_tickers`` (the SAME resolver A4/A5 bridge on) so the policy's notion of
a "discovered entity" cannot drift from the retriever's. That helper is imported lazily inside
``discovered_unretrieved_entities`` because ``research.bridge`` imports ``rag`` (a top-level import
here would be circular) — the same pattern ``rag.policy_features`` uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stock_agent.rag.policy_features import FEATURE_NAMES, ContextVector
from stock_agent.schemas.retrieval import RetrievedChunk

# The dynamic evidence-summary block appended to the static query features. Append-only: a trained
# policy's weights index into ``STATE_FEATURE_NAMES`` (= FEATURE_NAMES + this), so reordering or
# inserting silently remaps every stored weight. Raw counts (unstandardized) — the policy/training
# layer standardizes, exactly like the static block.
EVIDENCE_FEATURE_NAMES: tuple[str, ...] = (
    "step_idx",  # t: how many search steps already taken (0 at reset)
    "budget_remaining",  # max_steps - t: search steps left before the horizon forces a stop
    "n_chunks",  # size of the deduped evidence union
    "n_tickers_covered",  # distinct tickers present in the union (breadth across companies)
    "n_sections_covered",  # distinct filing sections present in the union (breadth across sections)
    "n_discovered_unretrieved",  # entities NAMED in the union but not yet retrieved (bridge signal)
    "last_new_chunks",  # novelty: chunks the LAST action added to the union (0 at reset)
)

# Full state feature order = static query block (A6.1b) ⊕ dynamic evidence block. This tuple defines
# the state dimension a policy is trained against.
STATE_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + EVIDENCE_FEATURE_NAMES
N_STATE_FEATURES: int = len(STATE_FEATURE_NAMES)


@dataclass(frozen=True)
class EvidenceSummary:
    """Aggregate description of the accumulated evidence union (the dynamic part of the state).

    All fields are raw counts, computed from the union alone except ``n_discovered_unretrieved``
    (needs the env's ``searched`` bookkeeping) and ``last_new_chunks`` (needs the last transition's
    delta) — both supplied by the environment, which owns that trajectory state.
    """

    n_chunks: int
    n_tickers_covered: int
    n_sections_covered: int
    n_discovered_unretrieved: int
    last_new_chunks: int


def discovered_unretrieved_entities(
    union: Sequence[RetrievedChunk],
    *,
    alias_map: Mapping[str, Sequence[str]],
    searched: set[str],
) -> set[str]:
    """Tickers **named in the union's text but not yet retrieved against** — the bridge signal.

    Resolves company names in the concatenated union text to tickers via the same
    ``research.bridge.mentioned_tickers`` A4/A5 use, then subtracts ``searched`` (tickers already
    scoped by a retrieval). The result drives both the ``n_discovered_unretrieved`` state
    feature and the env's discovered-entity action slots, so they stay consistent by construction.
    Empty union ⇒ empty set. ``searched`` holds upper-cased tickers (matches ``mentioned_tickers``).
    """
    if not union:
        return set()
    # Lazy import: research.bridge imports rag, so a top-level import here is circular (same hatch
    # policy_features uses). Reusing it keeps "discovered" identical to production bridging.
    from stock_agent.research.bridge import mentioned_tickers

    union_text = " ".join(rc.chunk.text for rc in union)
    return mentioned_tickers(union_text, alias_map) - searched


def summarize_evidence(
    union: Sequence[RetrievedChunk],
    *,
    n_discovered_unretrieved: int,
    last_new_chunks: int,
) -> EvidenceSummary:
    """Reduce the evidence union (+ two env-supplied scalars) to an ``EvidenceSummary``.

    ``n_tickers_covered`` / ``n_sections_covered`` count *distinct* non-empty ``ticker`` /
    ``section`` across the union chunks; ``None``/empty sections are ignored (unlabeled chunks do
    not inflate breadth). ``n_discovered_unretrieved`` / ``last_new_chunks`` pass through from env.
    """
    tickers = {rc.chunk.ticker for rc in union if rc.chunk.ticker}
    sections = {rc.chunk.section for rc in union if rc.chunk.section}
    return EvidenceSummary(
        n_chunks=len(union),
        n_tickers_covered=len(tickers),
        n_sections_covered=len(sections),
        n_discovered_unretrieved=n_discovered_unretrieved,
        last_new_chunks=last_new_chunks,
    )


def featurize_state(
    static: ContextVector,
    union: Sequence[RetrievedChunk],
    *,
    step_idx: int,
    max_steps: int,
    n_discovered_unretrieved: int,
    last_new_chunks: int,
) -> np.ndarray:
    """Assemble the full MDP state vector ``s_t`` (static query block ⊕ dynamic evidence block).

    Args:
        static: the A6.1b ``ContextVector`` for the episode's question (computed once at ``reset``;
            constant for the whole episode).
        union: the deduped evidence accumulated so far (empty at ``reset``).
        step_idx: search steps already taken this episode (``t``; 0 at reset).
        max_steps: the horizon ``T`` (``budget_remaining = max_steps - step_idx``).
        n_discovered_unretrieved: count from ``discovered_unretrieved_entities`` (env-computed).
        last_new_chunks: chunks the last action added to the union (0 at reset).

    Returns:
        A 1-D float64 ``np.ndarray`` of length ``N_STATE_FEATURES`` aligned to
        ``STATE_FEATURE_NAMES``. **Label-free** — no aspects are read anywhere in this function.
    """
    summary = summarize_evidence(
        union,
        n_discovered_unretrieved=n_discovered_unretrieved,
        last_new_chunks=last_new_chunks,
    )
    evidence_block = np.array(
        [
            float(step_idx),
            float(max_steps - step_idx),
            float(summary.n_chunks),
            float(summary.n_tickers_covered),
            float(summary.n_sections_covered),
            float(summary.n_discovered_unretrieved),
            float(summary.last_new_chunks),
        ],
        dtype=np.float64,
    )
    return np.concatenate([static.values.astype(np.float64), evidence_block])
