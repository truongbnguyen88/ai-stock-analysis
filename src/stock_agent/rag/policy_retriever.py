"""Policy-driven retrieval (advanced-RAG A6.1f) — serve the contextual bandit at query time.

``PolicyRetriever`` is the read-path integration of A6.1: it wraps the trained ``Policy`` (A6.1e) in
the ``RetrievalSystem`` contract so any consumer (``ToolExecutor``, ``build_retrieval_system``, the
``rag query`` CLI) can retrieve *through* the bandit without knowing it exists. Per query it:

    featurize(query)  →  policy.act(x) → (arm index, propensity)  →  build/reuse that arm
        →  delegate ``retrieve`` to the chosen arm  →  log the decision (A6.1a, gated)

The class **always adapts**; whether it is wired in at all is gated upstream by
``settings.adaptive_retrieval`` (default-OFF ⇒ the read path is byte-identical to A5.3). It touches
no LLM and produces no numbers of its own — it only *chooses* which existing retrieval config runs,
so every A6.1 invariant (numbers-from-models, grounding, non-advisory) is untouched.

Two design points:

- **Arm executor.** The 5 arms are the 4 lattice configs (``build_named_system``) plus the A5
  ``graph`` arm (``build_graph_system``); ``build_arm_system`` dispatches on the name. It is
  injected (``build_arm=``) so tests supply fakes and never load an embedder / open the graph DB.
  Built arms are **cached per instance** (memoized) — the embedder/store for an arm loads once.
- **Context sources.** ``featurize`` needs the scoped ticker (read from the query ``where`` filter),
  the alias map, and the graph universe. ``alias_map``/``graph_universe`` default to ``None``
  meaning "lazy-load the real ones"; passing them explicitly (even empty) uses exactly those (tests
  avoid I/O this way). Loading is lazy to keep this module free of a top-level ``research`` import
  (cycle guard, mirroring ``policy_features``).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

import structlog

from stock_agent.rag.policy import ARMS, Policy
from stock_agent.rag.policy_features import featurize
from stock_agent.rag.read_path import build_graph_system, build_named_system
from stock_agent.rag.retrieval_log import log_retrieval
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet
from stock_agent.schemas.retrieval_log import RetrievalLogEntry, ScoredChunkRef
from stock_agent.settings import Settings

log = structlog.get_logger(__name__)

_DEFAULT_GRAPH_UNIVERSE = Path("configs/graph_universe.txt")


def build_arm_system(name: str, settings: Settings) -> RetrievalSystem:
    """Executor for a single arm name: the A5 ``graph`` arm, else a lattice config.

    Dispatches the 5-arm action space (``ARMS``): ``"graph"`` → ``build_graph_system`` (traversal ⊕
    hybrid), everything else → ``build_named_system`` (dense/reranked/hybrid/hybrid+rerank). This is
    the default ``build_arm`` for ``PolicyRetriever`` — a superset of plain ``build_named_system``
    because the 5th (graph) arm needs the graph factory.
    """
    if name == "graph":
        return build_graph_system(settings)
    return build_named_system(name, settings)


class PolicyRetriever:
    """A ``RetrievalSystem`` that selects its retrieval arm per query via a bandit ``Policy``.

    Satisfies the ``RetrievalSystem`` Protocol (``name`` + ``retrieve``), so it is a drop-in for the
    fixed read path. ``policy`` is any A6.1e policy; its action indices index ``arm_names`` (which
    MUST be the ``ARMS`` order the policy was trained on). Logging is gated by
    ``settings.retrieval_logging`` (A6.1a) — off ⇒ ``retrieve`` writes nothing.
    """

    def __init__(
        self,
        policy: Policy,
        settings: Settings,
        *,
        build_arm: Callable[[str, Settings], RetrievalSystem] = build_arm_system,
        alias_map: Mapping[str, Sequence[str]] | None = None,
        graph_universe: Collection[str] | None = None,
        graph_universe_path: Path = _DEFAULT_GRAPH_UNIVERSE,
        arm_names: Sequence[str] = ARMS,
    ) -> None:
        self._policy = policy
        self._settings = settings
        self._build_arm = build_arm
        self._arm_names = tuple(arm_names)
        self._alias_map = alias_map
        self._graph_universe = graph_universe
        self._graph_universe_path = graph_universe_path
        self._arm_cache: dict[str, RetrievalSystem] = {}
        self._name = f"policy({policy.name})"

    @property
    def name(self) -> str:
        """Identifier for eval reports — ``policy(<policy name>)`` (read-only)."""
        return self._name

    def _sources(self) -> tuple[Mapping[str, Sequence[str]], Collection[str]]:
        """Resolve (alias map, graph universe); lazy-load the real ones only when ``None``.

        Lazy + local import so importing this module never pulls ``research`` / ticker I/O (cycle
        guard). Explicit (even empty) sources passed at construction are used verbatim — an empty
        alias map disables mention detection, an empty universe forces ``in_graph_universe=0``.
        """
        if self._alias_map is None:
            from stock_agent.research.bridge import load_alias_map

            self._alias_map = load_alias_map()
        if self._graph_universe is None:
            from stock_agent.documents.ticker_cik import load_universe

            self._graph_universe = load_universe(self._graph_universe_path)
        return self._alias_map, self._graph_universe

    def _arm(self, name: str) -> RetrievalSystem:
        """Build (once) and cache the retrieval system for arm ``name`` — memoized per instance."""
        if name not in self._arm_cache:
            self._arm_cache[name] = self._build_arm(name, self._settings)
        return self._arm_cache[name]

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        """Featurize → pick an arm via the policy → delegate retrieval → log the decision.

        The scoped ticker (for the context features) is read from ``where.ticker``. The returned
        ``EvidenceSet`` is exactly what the chosen arm produced — this wrapper adds no chunks and
        reorders nothing; it only routes.
        """
        ticker = where.ticker if where is not None else None
        alias_map, universe = self._sources()
        ctx = featurize(query, ticker=ticker, alias_map=alias_map, graph_universe=universe)
        action, propensity = self._policy.act(ctx.values)
        arm_name = self._arm_names[action]
        evidence = self._arm(arm_name).retrieve(query, top_k=top_k, where=where)
        self._log(query, ticker, ctx.as_map(), arm_name, propensity, evidence)
        return evidence

    def _log(
        self,
        query: str,
        ticker: str | None,
        context: dict[str, float],
        arm_name: str,
        propensity: float,
        evidence: EvidenceSet,
    ) -> None:
        """Append the bandit decision to the A6.1a JSONL log (no-op when logging is off)."""
        log_retrieval(
            RetrievalLogEntry(
                query=query,
                ticker=ticker,
                context=context,
                action=arm_name,
                propensity=propensity,
                retrieved=[
                    ScoredChunkRef(chunk_id=rc.chunk.chunk_id, score=rc.score)
                    for rc in evidence.chunks
                ],
            ),
            settings=self._settings,
        )
