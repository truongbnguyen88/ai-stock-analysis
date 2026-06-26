"""Deterministic entity-bridge for the A4 loop — the cross-entity pivot the LLM won't make itself.

The ReAct loop reliably gathers a company's OWN filings, but — measured twice on the live model
(react.v1 + a few-shot v2), at temperature 0 — it **refuses to pivot its search to a discovered
related entity**: asked "which of NVDA's suppliers also discloses X in its *own* filings", every hop
stayed `ticker=NVDA`, so the union never reached the supplier's filings and a *bridging* question
never realized its multi-hop gain (see `rag_implementation_notes.md` §A4). Prompting did not fix it;
the control policy needs structural help.

This module supplies that pivot deterministically and **$0** (retrieval only, no extra LLM call):
after the loop, for a bridging-style question it resolves company NAMES mentioned in the gathered
evidence to tickers via the shared alias map, ranks the *named-but-unsearched* entities by how well
their OWN filings match the question, and returns the top few to fold in. The eval-driven
fix for the bridging negative; gated to bridging questions so single-entity runs are untouched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from stock_agent.logging_config import get_logger
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.schemas.retrieval import ChunkFilter, RetrievedChunk

log = get_logger(__name__)

_ALIAS_PATH = Path("configs/ticker_aliases.json")

# Question cues that signal a BRIDGE — the answer lives in a *related* entity's own filings, not the
# subject's. Deliberately narrow (relationship words + "its/their own ... filings") so single-entity
# questions (e.g. "NVDA's data-center demand and competition") do not trigger the pivot.
_BRIDGE_CUES: tuple[str, ...] = (
    "supplier", "suppliers", "customer", "customers", "vendor", "partner", "partners",
    "its own", "their own", "in its filings", "in their filings", "in its own", "in their own",
)

# Bound the number of candidate entities we score per bridge (a chunk can name many companies);
# keeps the pass cheap and the union focused even before the top-N cut.
_MAX_CANDIDATES = 8


def load_alias_map(path: Path = _ALIAS_PATH) -> dict[str, list[str]]:
    """Load ``{ticker: [name aliases]}`` from JSON (drops ``_comment`` keys and empty lists).

    Mirrors ``news.gdelt_ingest.load_alias_map`` (kept local so ``research/`` does not depend on
    ``news/``). Returns an empty map if the file is missing — the bridge then simply no-ops.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        log.warning("bridge.alias_map_unavailable", path=str(path))
        return {}
    return {
        ticker: [a for a in aliases if a.strip()]
        for ticker, aliases in raw.items()
        if not ticker.startswith("_") and isinstance(aliases, list) and aliases
    }


def is_bridging(question: str) -> bool:
    """Whether ``question`` looks like a bridge (answer lies in a related entity's own filings)."""
    q = question.lower()
    return any(cue in q for cue in _BRIDGE_CUES)


def mentioned_tickers(text: str, alias_map: Mapping[str, Sequence[str]]) -> set[str]:
    """Tickers whose company name appears in ``text`` (case-insensitive, word-boundary)."""
    low = text.lower()
    found: set[str] = set()
    for ticker, names in alias_map.items():
        if any(re.search(rf"\b{re.escape(name.lower())}\b", low) for name in names):
            found.add(ticker)
    return found


def bridge_candidates(
    question: str,
    evidence: Sequence[RetrievedChunk],
    *,
    retriever: RetrievalSystem,
    alias_map: Mapping[str, Sequence[str]],
    searched: set[str],
    per_step_k: int,
    max_entities: int,
) -> list[tuple[str, list[RetrievedChunk]]]:
    """Ranked ``(ticker, chunks)`` to fold into the union — discovered entities' OWN filings ($0).

    No-ops (returns ``[]``) unless ``max_entities > 0`` and the question is bridging. Otherwise it
    resolves the company names in ``evidence`` to tickers, drops those already ``searched``, and for
    each remaining candidate retrieves its own filings *with the question* — keeping the top
    ``max_entities`` by best chunk score (the entity whose filings most match the question; e.g. for
    a regulatory-restriction question, the supplier that discloses one). Pure retrieval; the
    caller folds the chunks into the deduped, capped union and records the trace.
    """
    if max_entities <= 0 or not is_bridging(question):
        return []
    union_text = " ".join(rc.chunk.text for rc in evidence)
    candidates = sorted(mentioned_tickers(union_text, alias_map) - searched)[:_MAX_CANDIDATES]
    scored: list[tuple[float, str, list[RetrievedChunk]]] = []
    for ticker in candidates:
        ev = retriever.retrieve(question, top_k=per_step_k, where=ChunkFilter(ticker=ticker))
        if ev.chunks:  # skip entities with no filings in the corpus
            scored.append((max(rc.score for rc in ev.chunks), ticker, list(ev.chunks)))
    scored.sort(key=lambda row: -row[0])  # most question-relevant entity first
    return [(ticker, chunks) for _, ticker, chunks in scored[:max_entities]]
