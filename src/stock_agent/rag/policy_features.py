"""Query featurizer for the retrieval policy (advanced-RAG A6.1b) — the bandit context ``x``.

A contextual bandit picks a retrieval arm from a *context*: a numeric vector describing the query.
This module maps a raw question (+ its scoped ticker, the alias map, and the graph universe) to a
fixed-length ``ContextVector`` of **deploy-time signals only**. It is a pure function — no model, no
network, no I/O — so it is cheap enough to call per query at serving time and fully golden-testable.

**Leakage rule 1 (non-negotiable).** The featurizer is *label-free*: it never reads the gold
``stratum`` / ``relation`` / ``qtype`` of a benchmark row (those are the answer key). Every feature
is derivable from the query text + the alias map + ``graph_universe.txt`` at deploy time, so a
policy trained on these features is actually deployable. The ``qtype_*`` one-hot here is a **cheap
keyword heuristic**, not the labeled question type.

Feature semantics reuse the *canonical* production helpers (``research.bridge.is_bridging`` /
``mentioned_tickers``) so the ``is_bridging`` / ``n_entities`` features mean exactly what A4/A5
bridging means — preventing a drift where the policy's notion of "bridging" diverges from the
retriever's. They are imported lazily inside ``featurize`` because ``research.bridge`` imports
``rag`` (a top-level import here would be circular) — the same lazy-import pattern
``rag.read_path.build_graph_system`` uses to reach ``graph``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Stable feature order — a trained policy's coefficients index into THIS tuple. Treat as
# append-only: reordering or inserting silently remaps every stored weight. `bias` is the constant
# intercept term (1.0) linear policies (LinUCB / ridge q̂) expect.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",  # constant 1.0 — intercept for the linear policies (A6.1e)
    "n_tokens",  # query length in whitespace tokens (raw; the policy layer standardizes)
    "has_ticker",  # 1.0 if retrieval is ticker-scoped OR a company is named in the query
    "n_entities",  # # distinct tickers named in the query text (multi-entity / compare signal)
    "is_bridging",  # 1.0 if a bridge cue is present (answer lies in a related entity's own filings)
    "qtype_risk",  # ─┐
    "qtype_financial",  # │ cheap keyword one-hot (exactly one is 1.0); NOT the gold qtype label
    "qtype_business",  # │
    "qtype_overview",  # │ default bucket
    "qtype_bridging",  # ─┘ set when is_bridging fires
    "in_graph_universe",  # 1.0 if the scoped ticker has a built graph → the graph arm can traverse
)

N_FEATURES = len(FEATURE_NAMES)

# Cheap, label-free keyword cues for the qtype one-hot (deploy-time text heuristics). Priority order
# on a tie: bridging > risk > financial > business > overview. Substring match on the lowercased
# query (so "risks", "regulatory" both hit risk; "margin", "revenue" hit financial).
_RISK_CUES: tuple[str, ...] = (
    "risk", "warn", "concern", "threat", "uncertain", "adverse", "litigation",
    "regulat", "exposure", "liabilit", "disrupt",
)
_FINANCIAL_CUES: tuple[str, ...] = (
    "revenue", "margin", "profit", "earnings", "cash flow", "guidance", "capital",
    "debt", "cost", "expense", "gross", "operating income", "eps", "dividend",
)
_BUSINESS_CUES: tuple[str, ...] = (
    "product", "segment", "customer", "supplier", "manufactur", "market", "strategy",
    "acquisition", "operation", "compet", "partner", "platform",
)


@dataclass(frozen=True)
class ContextVector:
    """A featurized query: the numeric vector ``values`` plus its stable feature ``names``.

    ``values`` is a 1-D float ``np.ndarray`` aligned to ``names`` (default ``FEATURE_NAMES``).
    ``as_map`` serializes it to the ``{name: value}`` form the telemetry log (A6.1a) stores.
    """

    values: np.ndarray
    names: tuple[str, ...] = FEATURE_NAMES

    def as_map(self) -> dict[str, float]:
        """Self-describing ``{name: value}`` dict — the form ``RetrievalLogEntry.context`` uses."""
        return {n: float(v) for n, v in zip(self.names, self.values, strict=True)}


def _cue_hit(text: str, cues: Sequence[str]) -> bool:
    return any(cue in text for cue in cues)


def _qtype_index(low_query: str, is_bridge: bool) -> int:
    """Index (into the qtype one-hot block) of the single active question type.

    Priority: bridging > risk > financial > business > overview (default). Returns the offset within
    the ``qtype_*`` sub-block (0=risk, 1=financial, 2=business, 3=overview, 4=bridging).
    """
    if is_bridge:
        return 4  # qtype_bridging
    if _cue_hit(low_query, _RISK_CUES):
        return 0
    if _cue_hit(low_query, _FINANCIAL_CUES):
        return 1
    if _cue_hit(low_query, _BUSINESS_CUES):
        return 2
    return 3  # qtype_overview (default)


def featurize(
    query: str,
    *,
    ticker: str | None = None,
    alias_map: Mapping[str, Sequence[str]] | None = None,
    graph_universe: Collection[str] | None = None,
) -> ContextVector:
    """Map a question to its bandit context vector (label-free, deploy-time signals only).

    Args:
        query: the raw question text.
        ticker: the scoped ticker if retrieval is ticker-filtered (drives ``has_ticker`` /
            ``in_graph_universe``); ``None`` for an unscoped query.
        alias_map: ``{ticker: [names]}`` for detecting companies named *in the text*
            (``n_entities`` / ``has_ticker``); ``None`` disables mention detection.
        graph_universe: tickers with a built graph (``configs/graph_universe.txt``); ``None`` ⇒
            ``in_graph_universe`` is 0 (the graph arm would degrade to hybrid).

    Returns:
        A ``ContextVector`` over ``FEATURE_NAMES`` (numpy float64).
    """
    # Lazy import: research.bridge imports rag → a top-level import here is circular. Reusing the
    # canonical helpers keeps the features' meaning identical to production bridging (no drift).
    from stock_agent.research.bridge import is_bridging, mentioned_tickers

    low = query.lower()
    named = mentioned_tickers(query, alias_map) if alias_map else set()
    n_entities = float(len(named))
    has_ticker = 1.0 if (ticker or named) else 0.0
    is_bridge = is_bridging(query)

    # Primary ticker = the explicit scope, else the lexicographically-first named ticker (a stable,
    # deterministic pick when several are named — only used for the graph-universe membership flag).
    primary = ticker or (min(named) if named else None)
    universe = {t.upper() for t in graph_universe} if graph_universe is not None else set()
    in_universe = 1.0 if (primary and primary.upper() in universe) else 0.0

    values = np.zeros(N_FEATURES, dtype=np.float64)
    values[0] = 1.0  # bias
    values[1] = float(len(query.split()))  # n_tokens
    values[2] = has_ticker
    values[3] = n_entities
    values[4] = 1.0 if is_bridge else 0.0
    values[5 + _qtype_index(low, is_bridge)] = 1.0  # exactly one qtype one-hot
    values[10] = in_universe
    return ContextVector(values=values)
