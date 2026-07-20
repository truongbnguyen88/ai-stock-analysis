"""A6.2a — MDP state featurizer: dynamic evidence block, discovered-entity signal, leakage guard.

The headline test reproduces the Q4 worked ``s_0 -> s_1`` transition from
``docs/rl_rag_pre_questions.md`` (an NVDA bridging episode): after one self-ticker retrieval the
union grows to 6 chunks over 1 ticker / 1 section, ``budget`` drops 3->2, and — the crux —
``n_discovered_unretrieved`` goes 0->1 (a supplier is now named but not pulled). Offline, no model.
"""

from __future__ import annotations

import inspect
from datetime import date

from stock_agent.rag.policy_features import FEATURE_NAMES, N_FEATURES, featurize
from stock_agent.rag.rl.state import (
    EVIDENCE_FEATURE_NAMES,
    N_STATE_FEATURES,
    STATE_FEATURE_NAMES,
    discovered_unretrieved_entities,
    featurize_state,
    summarize_evidence,
)
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import RetrievedChunk

ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"], "TSM": ["Taiwan Semiconductor", "TSMC"]}
UNIVERSE = {"NVDA", "MU", "TSM"}


def _rc(
    cid: str,
    text: str,
    *,
    ticker: str = "NVDA",
    section: str | None = "Item 1A. Risk Factors",
) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=cid,
        document_id=cid.rsplit(":", 1)[0],
        chunk_index=int(cid.rsplit(":", 1)[1]),
        text=text,
        ticker=ticker,
        document_type="10-K",
        source="SEC",
        source_url="https://sec.gov/x",
        filing_date=date(2026, 2, 25),
        section=section,
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


# ---- feature-order / dimension invariants -------------------------------------
def test_state_feature_names_layout() -> None:
    # State = the 11-dim static query block (A6.1b) then the 7-dim dynamic evidence block, in order.
    assert STATE_FEATURE_NAMES == FEATURE_NAMES + EVIDENCE_FEATURE_NAMES
    assert N_STATE_FEATURES == N_FEATURES + len(EVIDENCE_FEATURE_NAMES) == 18
    assert EVIDENCE_FEATURE_NAMES == (
        "step_idx",
        "budget_remaining",
        "n_chunks",
        "n_tickers_covered",
        "n_sections_covered",
        "n_discovered_unretrieved",
        "last_new_chunks",
    )


def test_state_static_block_is_the_a6_1_context() -> None:
    # The first N_FEATURES dims of the state must be exactly featurize(query).values (no drift).
    static = featurize(
        "What are NVIDIA's risks?", ticker="NVDA", alias_map=ALIAS, graph_universe=UNIVERSE
    )
    s = featurize_state(
        static, [], step_idx=0, max_steps=3, n_discovered_unretrieved=0, last_new_chunks=0
    )
    assert s.shape == (N_STATE_FEATURES,)
    assert list(s[:N_FEATURES]) == list(static.values)


# ---- the Q4 worked s_0 -> s_1 transition --------------------------------------
def test_q4_worked_transition_evidence_block() -> None:
    static = featurize(
        "Which memory supplier NVIDIA depends on discloses Chinese cybersecurity restrictions?",
        ticker="NVDA",
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
    )

    # s_0: nothing gathered yet -> evidence block all zeros except budget = max_steps.
    s0 = featurize_state(
        static, [], step_idx=0, max_steps=3, n_discovered_unretrieved=0, last_new_chunks=0
    )
    assert list(s0[N_FEATURES:]) == [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # s_1: one self-ticker hop pulled 6 NVDA / Item 1A chunks; one text names "Micron" -> MU is now
    # discovered-but-unretrieved. step 0->1, budget 3->2, n_chunks 0->6, tickers/sections ->1,
    # n_discovered_unretrieved 0->1, last_new_chunks ->6.
    union = [_rc(f"NVDA:10-K:2026:{i}", "NVIDIA relies on Micron for HBM.") for i in range(6)]
    s1 = featurize_state(
        static, union, step_idx=1, max_steps=3, n_discovered_unretrieved=1, last_new_chunks=6
    )
    assert list(s1[N_FEATURES:]) == [1.0, 2.0, 6.0, 1.0, 1.0, 1.0, 6.0]
    # Static block is unchanged across the episode (question is constant).
    assert list(s1[:N_FEATURES]) == list(s0[:N_FEATURES])


# ---- discovered-entity resolution (the bridge signal) -------------------------
def test_discovered_unretrieved_excludes_searched() -> None:
    # Union text names Micron + Taiwan Semiconductor; NVDA already searched -> {MU, TSM} discovered.
    union = [_rc("NVDA:10-K:2026:0", "NVIDIA depends on Micron and Taiwan Semiconductor.")]
    disc = discovered_unretrieved_entities(union, alias_map=ALIAS, searched={"NVDA"})
    assert disc == {"MU", "TSM"}
    # Once MU is searched it drops out of the discovered set.
    got = discovered_unretrieved_entities(union, alias_map=ALIAS, searched={"NVDA", "MU"})
    assert got == {"TSM"}


def test_discovered_unretrieved_empty_union() -> None:
    assert discovered_unretrieved_entities([], alias_map=ALIAS, searched={"NVDA"}) == set()


# ---- evidence summary counting --------------------------------------------------
def test_summarize_evidence_distinct_tickers_and_sections() -> None:
    union = [
        _rc("NVDA:10-K:2026:0", "a", ticker="NVDA", section="Item 1A. Risk Factors"),
        _rc("NVDA:10-K:2026:1", "b", ticker="NVDA", section="Item 1. Business"),
        _rc("MU:10-K:2026:0", "c", ticker="MU", section="Item 1A. Risk Factors"),
        _rc("MU:10-K:2026:1", "d", ticker="MU", section=None),  # None section ignored in breadth
    ]
    summ = summarize_evidence(union, n_discovered_unretrieved=2, last_new_chunks=1)
    assert summ.n_chunks == 4
    assert summ.n_tickers_covered == 2  # NVDA, MU
    assert summ.n_sections_covered == 2  # Item 1A + Item 1 (the None section excluded from breadth)
    assert summ.n_discovered_unretrieved == 2
    assert summ.last_new_chunks == 1


# ---- leakage guard (state must be label-free) ----------------------------------
def test_state_featurizer_is_label_free() -> None:
    # Structural guard: no aspect/label/stratum/coverage parameter may enter the state featurizer.
    params = set(inspect.signature(featurize_state).parameters)
    forbidden = {"aspect", "aspects", "label", "labels", "stratum", "coverage", "reward"}
    assert params.isdisjoint(forbidden), f"state featurizer leaks a label: {params & forbidden}"
