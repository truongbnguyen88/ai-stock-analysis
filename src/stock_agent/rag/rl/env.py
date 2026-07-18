"""Deterministic, ``$0`` retrieval MDP + transition cache for full retrieval RL (adv-RAG A6.2c).

Wraps the A6.2a state featurizer and A6.2b action space into a minimal Gym-style ``reset``/``step``
environment. One episode = one labeled ``MultiHopQuery``; a step runs the **real** retriever against
the **real** corpus with a **templated** (no-LLM) query, folds the chunks into the deduped union
(the A4 loop's own ``_dedup_union``, so the union is byte-identical to the ReAct loop's), and
returns the shaped reward. No LLM anywhere in the loop, no RNG in the env — so rollouts are
reproducible and free, which is exactly what on-policy PPO needs.

**Reward (potential-based shaping, Ng–Harada).** With Φ(s) = ``coverage(union, aspects)``:

    search step a_t:  r_t = γ·Φ(s_{t+1}) − Φ(s_t) − λ_c·cost(arm)
    STOP:             r   = 0                              (terminal; no retrieval, no cost)

The discounted shaping telescopes to ``γ^n·Φ(s_n) − Φ(s_0)`` (Φ(s_0)=0), i.e. the densified terminal
coverage — same objective as the sparse terminal reward, credited the step it is earned. Φ (the
**only** place the gold aspects enter) lives here in the simulator; the *state* never sees a label
(leakage line Q4.3), so a policy trained here runs unchanged on a real, unlabeled query.

**Transition cache (§3a — the tractability lynchpin).** Retrieval is ``$0`` but slow, and PPO does
thousands of rollouts. A transition is a pure function of ``(arm, scope_ticker, query)`` (the
templated query already carries the question), so results memoize on first hit; every later
rollout/epoch is a table lookup. That is what turns "``$0`` but slow" into "``$0`` and fast".
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stock_agent.rag.policy_features import ContextVector, featurize
from stock_agent.rag.read_path import build_graph_system, build_named_system
from stock_agent.rag.rerank import Reranker
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.rag.reward import DEFAULT_ARM_COSTS
from stock_agent.rag.rl.action import (
    GRAPH_ARM,
    Action,
    DiscoveredEntity,
    RetrievalRequest,
    ScopeKind,
    action_label,
    action_to_requests,
    discovered_scope_query,
    is_legal,
)
from stock_agent.rag.rl.seating import (
    SeatingRule,
    build_seating_reranker,
    effective_rule,
    fanout_cap,
    merge_breadth_first,
    seat_branches,
)
from stock_agent.rag.rl.state import discovered_unretrieved_entities, featurize_state
from stock_agent.schemas.retrieval import ChunkFilter, RetrievedChunk
from stock_agent.settings import Settings

if TYPE_CHECKING:
    from stock_agent.research.multistep_eval import MultiHopQuery

_DEFAULT_GRAPH_UNIVERSE = Path("configs/graph_universe.txt")

# A retriever factory maps a canonical arm name to a built RetrievalSystem (tests inject a fake).
RetrieverFactory = Callable[[str], RetrievalSystem]

# A query writer rewrites the *text* of a templated request — and nothing else. Default (``None``) =
# use the template verbatim: deterministic, $0, no LLM (the training/eval contract). The A6.2g paid
# sim-to-real run injects an **LLM** writer here, so the real path shares every other mechanism with
# the simulator (union dedup, discovered set, coverage oracle, cost accounting) and differs in
# exactly ONE variable — the query string. That is what makes the measured gap attributable.
# NOTE: an LLM writer makes the env non-deterministic; never pass one during training.
QueryWriter = Callable[[RetrievalRequest, "MultiHopQuery", Sequence[RetrievedChunk]], str]


def default_retriever_factory(settings: Settings) -> RetrieverFactory:
    """Production arm builder: ``"graph"`` → ``build_graph_system``, else ``build_named_system``.

    Mirrors ``PolicyRetriever``'s ``build_arm`` convention (canonical arm name in, one shared
    ``Settings``). The env memoizes the built system per arm, so each backend loads at most once.
    """

    def factory(arm: str) -> RetrievalSystem:
        if arm == GRAPH_ARM:
            return build_graph_system(settings)
        return build_named_system(arm, settings)

    return factory


class TransitionCache:
    """Memoize retrieval results by ``(arm, scope_ticker, query)`` — the deterministic transition.

    Retrieval is a pure function of the arm, the scope ticker, and the (templated) query string, so
    the same request always yields the same chunks: cache once, look up forever. Tracks ``hits`` /
    ``misses`` so a test can assert the cache is actually saving work (miss count == distinct
    requests). Not thread-safe (the env is single-threaded); persist per run for reproducibility.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], list[RetrievedChunk]] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        arm: str,
        scope_ticker: str | None,
        query: str,
        compute: Callable[[], list[RetrievedChunk]],
    ) -> list[RetrievedChunk]:
        """Return cached chunks for the request, computing (and storing) them on the first miss."""
        key = (arm, scope_ticker or "", query)
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        chunks = compute()
        self._store[key] = chunks
        return chunks


@dataclass(frozen=True)
class StepInfo:
    """Per-step telemetry (logging + tests); never consumed by the reward or the learner's math."""

    action_label: str
    arm: str | None  # None for STOP or a masked no-op
    scope_ticker: str | None  # the single scope; None for STOP/masked and for a multi-branch fanout
    n_new_chunks: int  # chunks this step added to the deduped union
    coverage: float  # Φ(s_{t+1}) = aspect coverage of the union after the step
    n_discovered: int  # raw count of named-but-unretrieved entities (state feature, uncapped)
    step_idx: int
    cost: float  # arm cost proxy charged this step; a fanout pays it once PER BRANCH
    masked: bool  # True iff a non-STOP action expanded to no request (illegal discovered slot)
    stopped: bool  # True iff this was the STOP action
    scope_tickers: tuple[str, ...] = ()  # every scope this step searched (len > 1 ⇒ fanout sweep)


class RagRetrievalEnv:
    """Finite-horizon retrieval MDP over one benchmark of labeled multi-hop questions.

    Deterministic and ``$0``: ``reset(i)`` loads episode ``i`` and returns ``s_0``; ``step(a)`` runs
    the templated retrieval for action ``a``, folds it into the union, and returns
    ``(s_next, reward, done, info)``. The caller (the trainer) owns episode iteration order and any
    seeded shuffling — the env holds no RNG (determinism invariant). ``retriever_factory``,
    ``alias_map`` and ``graph_universe`` are injectable (fakes offline; real backends in prod).
    """

    def __init__(
        self,
        episodes: Sequence[MultiHopQuery],
        *,
        settings: Settings,
        action_space: Sequence[Action],
        retriever_factory: RetrieverFactory | None = None,
        alias_map: Mapping[str, Sequence[str]] | None = None,
        graph_universe: Collection[str] | None = None,
        graph_universe_path: Path = _DEFAULT_GRAPH_UNIVERSE,
        gamma: float = 1.0,
        lambda_cost: float | None = None,
        arm_costs: Mapping[str, float] = DEFAULT_ARM_COSTS,
        max_steps: int | None = None,
        per_step_k: int | None = None,
        max_evidence: int | None = None,
        query_writer: QueryWriter | None = None,
        seating_rule: SeatingRule | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("env needs at least one episode")
        if not action_space or not action_space[0].is_stop:
            raise ValueError("action_space[0] must be STOP")
        self.episodes = list(episodes)
        self.settings = settings
        self.action_space = list(action_space)
        self._factory = retriever_factory or default_retriever_factory(settings)
        self._query_writer = query_writer  # None ⇒ templated queries (deterministic, $0)
        self._alias_map = alias_map
        self._graph_universe = graph_universe
        self._graph_universe_path = graph_universe_path
        self.gamma = gamma
        self.lambda_cost = lambda_cost if lambda_cost is not None else settings.reward_lambda_cost
        self.arm_costs = dict(arm_costs)
        self.max_steps = max_steps if max_steps is not None else settings.agentic_max_steps
        self.per_step_k = per_step_k if per_step_k is not None else settings.agentic_per_step_k
        self.max_evidence = (
            max_evidence if max_evidence is not None else settings.agentic_max_evidence
        )
        # E7 seating: how a fan-out's branches become the capped union (see rag/rl/seating.py).
        # The reranker is built LAZILY, on the first fan-out that needs one — constructing it here
        # would download a cross-encoder in CI, where every env is built with fake retrievers.
        self.seating_rule: SeatingRule = seating_rule or settings.rl_seating_rule
        self._reranker = reranker
        self._reranker_built = reranker is not None
        # Discovered slots the action space actually addresses (single source of truth = the space).
        slots = [
            a.scope_slot
            for a in self.action_space
            if a.scope_kind is ScopeKind.DISCOVERED and a.scope_slot is not None
        ]
        self.n_discovered_slots = (max(slots) + 1) if slots else 0

        self._systems: dict[str, RetrievalSystem] = {}  # built once per arm, across episodes
        self._cache = TransitionCache()  # shared across episodes (key carries the question)
        # E7 seating rerank is part of the deterministic transition too, but the TransitionCache
        # (§3a) predates E7 and only covers *retrieval*. The cross-encoder recompute — not the
        # cached retrieval — dominates a fan-out's wall-clock (~1.4s/sweep on the real fold), so
        # it gets its own memo, keyed on the seating fn's pure inputs (rule, pooled query, branch
        # chunk-ids). That is what makes thousands of on-policy rollouts over the seated env
        # feasible (the same tractability argument as the retrieval cache). See `_seat`.
        self._seat_cache: dict[tuple[str, str, tuple[str, ...]], list[RetrievedChunk]] = {}
        self.seating_hits = 0
        self.seating_misses = 0

        # Per-episode trajectory state (populated by reset()).
        self._episode: MultiHopQuery | None = None
        self._static: ContextVector | None = None
        self._union: list[RetrievedChunk] = []
        self._scoped: set[str] = set()
        self._discovered: list[DiscoveredEntity] = []
        self._n_discovered_raw = 0
        self._step_idx = 0
        self._phi = 0.0
        self._done = True

    # ---- properties ---------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        """Number of actions in the (fixed) action space."""
        return len(self.action_space)

    @property
    def cache(self) -> TransitionCache:
        """The transition cache (exposed for hit/miss assertions and per-run persistence)."""
        return self._cache

    @property
    def evidence(self) -> list[RetrievedChunk]:
        """The current deduped evidence union — the terminal synthesis input in the A6.2g paid run.

        A copy, so a caller cannot mutate the trajectory state the reward depends on.
        """
        return list(self._union)

    # ---- lazy source resolution (mirrors PolicyRetriever._sources) -----------------
    def _sources(self) -> tuple[Mapping[str, Sequence[str]], Collection[str]]:
        """Resolve (alias map, graph universe), lazy-loading the real ones only when ``None``.

        Lazy + local import so importing this module never pulls ``research`` / ticker I/O (cycle
        guard, same as ``policy_retriever``). Explicit (even empty) sources are used verbatim.
        """
        if self._alias_map is None:
            from stock_agent.research.bridge import load_alias_map

            self._alias_map = load_alias_map()
        if self._graph_universe is None:
            from stock_agent.documents.ticker_cik import load_universe

            self._graph_universe = (
                load_universe(self._graph_universe_path)
                if self._graph_universe_path.exists()
                else set()
            )
        return self._alias_map, self._graph_universe

    # ---- reset / step -------------------------------------------------------------
    def reset(self, episode_idx: int) -> np.ndarray:
        """Load episode ``episode_idx`` and return the initial state ``s_0`` (empty-union state)."""
        if not 0 <= episode_idx < len(self.episodes):
            raise IndexError(f"episode_idx {episode_idx} out of range [0, {len(self.episodes)})")
        episode = self.episodes[episode_idx]
        alias_map, universe = self._sources()
        self._episode = episode
        self._static = featurize(
            episode.question, ticker=episode.seed, alias_map=alias_map, graph_universe=universe
        )
        self._union = []
        self._scoped = set()
        self._step_idx = 0
        self._done = False
        self._recompute_discovered()  # empty union → no discovered entities
        self._phi = self._coverage()  # coverage of the empty union = 0.0
        return self._state(last_new_chunks=0)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        """Apply ``action_space[action_idx]``; return ``(s_next, reward, done, info)``.

        STOP terminates with reward 0. A search action increments the step counter (so the horizon
        bounds the episode), runs the templated retrieval (via the cache), re-derives the discovered
        set + coverage, and returns the shaped reward. A search action whose discovered slot is
        unpopulated is a no-op (0 new chunks, 0 cost) — dominated, and never chosen under masking.
        """
        if self._done or self._episode is None:
            raise RuntimeError("step() on a done/uninitialized env; call reset() first")
        action = self.action_space[action_idx]
        phi_before = self._phi

        if action.is_stop:
            self._done = True
            info = StepInfo(
                action_label=action_label(action), arm=None, scope_ticker=None, n_new_chunks=0,
                coverage=phi_before, n_discovered=self._n_discovered_raw, step_idx=self._step_idx,
                cost=0.0, masked=False, stopped=True,
            )
            return self._state(last_new_chunks=0), 0.0, True, info

        self._step_idx += 1  # every non-STOP action consumes budget (guarantees termination)
        reqs = action_to_requests(
            action, self._episode.question, self_ticker=self._episode.seed,
            discovered=self._discovered,
        )
        if self._query_writer is not None:
            # Rewrite ONLY the query text — arm and scope stay the policy's decision. An empty/blank
            # rewrite falls back to the template (never issue a contentless query).
            reqs = [self._rewrite(r) for r in reqs]
        n_new, cost, masked = 0, 0.0, not reqs
        if reqs:
            # One retrieval per branch (1 for self/disc, N for a fanout sweep), then SEATED into the
            # capped union — see rag/rl/seating.py for why the rule is load-bearing (the E3 rule
            # silently discarded every target chunk that was not its branch's rank-0 hit).
            branches = [self._retrieve(r) for r in reqs]
            before = len(self._union)
            seated = self._seat(branches, reqs)
            self._union = self._dedup_union(self._union, seated)[
                : self._evidence_cap(n_branches=len(branches))
            ]
            n_new = len(self._union) - before
            self._scoped.update(r.scope_ticker for r in reqs if r.scope_ticker)
            cost = sum(self.arm_costs.get(r.arm, 0.0) for r in reqs)  # a sweep pays PER branch

        self._recompute_discovered()
        phi_after = self._phi = self._coverage()
        # Potential-based shaping (Ng–Harada); cost charged separately. STOP handled above with r=0.
        reward = (self.gamma * phi_after - phi_before) - self.lambda_cost * cost
        self._done = self._step_idx >= self.max_steps
        scopes = tuple(r.scope_ticker for r in reqs if r.scope_ticker)
        info = StepInfo(
            action_label=action_label(action),
            arm=(reqs[0].arm if reqs else None),
            scope_ticker=(scopes[0] if len(scopes) == 1 else None),
            n_new_chunks=n_new, coverage=phi_after, n_discovered=self._n_discovered_raw,
            step_idx=self._step_idx, cost=cost, masked=masked, stopped=False,
            scope_tickers=scopes,
        )
        return self._state(last_new_chunks=n_new), reward, self._done, info

    def _rewrite(self, req: RetrievalRequest) -> RetrievalRequest:
        """Apply the injected ``QueryWriter`` to one request's query text (arm/scope untouched)."""
        assert self._query_writer is not None and self._episode is not None
        written = self._query_writer(req, self._episode, list(self._union)).strip()
        return replace(req, query=written) if written else req

    def _seat(
        self, branches: Sequence[Sequence[RetrievedChunk]], reqs: Sequence[RetrievalRequest]
    ) -> list[RetrievedChunk]:
        """Order the step's retrieved chunks for the union (E7). Single-branch steps are untouched.

        A self/discovered action retrieves ONE branch, and its order is the retriever's own ranking
        —
        there is nothing to seat, so it bypasses the seating rule entirely and every A4/A5 baseline
        and pre-E7 single-branch number stays byte-identical. Only a fan-out has the cross-branch
        problem the seating rule exists to solve.
        """
        if len(branches) <= 1:
            return merge_breadth_first(branches)
        rule = self._effective_rule()
        # The pooled query is the TOPIC, with no candidate-name prefix: a name-prefixed query
        # would tilt the cross-encoder toward one branch and destroy exactly the cross-branch
        # comparability that makes pooled seating work. Every branch's request carries the same
        # topic clause (E6), so any of them supplies it.
        query = discovered_scope_query(self._episode.question, "") if self._episode else ""
        # Memoize the seating rerank (the fan-out wall-clock driver) exactly as retrieval is cached.
        # seat_branches is a pure function of (branches, rule, query, reranker config); rule/query
        # are captured in the key and the reranker config is fixed per env, so the flattened branch
        # chunk-ids identify the result uniquely — no collision is possible (equal inputs ⇒ equal
        # output).
        key = (rule, query, tuple(rc.chunk.chunk_id for branch in branches for rc in branch))
        cached = self._seat_cache.get(key)
        if cached is not None:
            self.seating_hits += 1
            return cached
        self.seating_misses += 1
        seated = seat_branches(
            branches,
            rule=rule,
            query=query,
            reranker=self._seating_reranker(),
            top_branches=self.settings.rl_seating_top_branches,
            per_branch=self.settings.rl_seating_per_branch,
        )
        self._seat_cache[key] = seated
        return seated

    def _seating_reranker(self) -> Reranker | None:
        """The seating cross-encoder, built on first use (never at construction — see __init__)."""
        if not self._reranker_built:
            self._reranker_built = True
            if self.seating_rule != "breadth_first":
                self._reranker = build_seating_reranker(self.settings)
        return self._reranker

    def _effective_rule(self) -> SeatingRule:
        """The seating rule that will actually run — the SINGLE source of truth for order AND cap.

        Both ``_seat`` and ``_evidence_cap`` read this. They must never disagree: a reranked rule's
        flat cap is safe only because the reranked order puts the best chunks first, so pairing it
        with the breadth-first fallback order would drop candidates alphabetically (see
        ``seating.effective_rule``).
        """
        return effective_rule(self.seating_rule, self._seating_reranker())

    def _evidence_cap(self, *, n_branches: int) -> int:
        """The union cap for this step — delegated to the seating rule, which defines the guarantee.

        Each rule earns a different budget: ``breadth_first`` must widen to ``len(union) +
        n_branches`` (its guarantee is *one seat per branch*, and a flat cap would seat the first 14
        candidates **alphabetically** and drop the rest — the very slot bias E3 removes), while the
        reranked rules guarantee *the best chunks seat, wherever they came from* and so need no
        per-branch allowance. See ``seating.fanout_cap``.
        """
        if n_branches <= 1:  # unchanged from pre-E7 (byte-identical single-branch behaviour)
            return max(self.max_evidence, len(self._union) + n_branches)
        return fanout_cap(
            rule=self._effective_rule(),
            n_union=len(self._union),
            n_branches=n_branches,
            max_evidence=self.max_evidence,
            top_branches=self.settings.rl_seating_top_branches,
            per_branch=self.settings.rl_seating_per_branch,
        )

    def legal_mask(self) -> np.ndarray:
        """Boolean mask (len == ``n_actions``): which actions are executable at the current state.

        STOP and self-scope are always legal; a discovered slot is legal iff it indexes an
        actually-discovered entity. The policy uses this to mask illegal actions (``-inf`` logit).
        """
        n = len(self._discovered)
        return np.array([is_legal(a, n_discovered=n) for a in self.action_space], dtype=bool)

    # ---- internals ----------------------------------------------------------------
    def _retrieve(self, req: RetrievalRequest) -> list[RetrievedChunk]:
        """Run (or look up) the templated retrieval for ``req`` via the transition cache."""

        def compute() -> list[RetrievedChunk]:
            system = self._system(req.arm)
            where = ChunkFilter(ticker=req.scope_ticker) if req.scope_ticker else None
            return list(system.retrieve(req.query, top_k=self.per_step_k, where=where).chunks)

        return self._cache.get_or_compute(req.arm, req.scope_ticker, req.query, compute)

    def _system(self, arm: str) -> RetrievalSystem:
        """Build (once) and cache the retrieval system for ``arm`` — memoized across episodes."""
        if arm not in self._systems:
            self._systems[arm] = self._factory(arm)
        return self._systems[arm]

    def _recompute_discovered(self) -> None:
        """Re-derive the discovered-unretrieved entities from the current union (label-free).

        ``searched`` = tickers already scoped by a retrieval ∪ tickers already present in the union
        (matches the A4 bridge's own definition), so an entity leaves the discovered set once its
        filings are gathered — including every branch a fanout swept.

        The list is **complete** (not truncated to the J action slots) and lexicographic by ticker:
        deterministic and label-free. Slot actions still address ``[0..J-1]``, i.e. exactly the
        entities they addressed pre-E3, so their behaviour is unchanged; fanout needs the whole
        list, and truncating it here is what made the oracle action unreachable in the first place.
        """
        alias_map, _ = self._sources()
        searched = self._scoped | {rc.chunk.ticker for rc in self._union if rc.chunk.ticker}
        tickers = sorted(
            discovered_unretrieved_entities(self._union, alias_map=alias_map, searched=searched)
        )
        self._n_discovered_raw = len(tickers)
        self._discovered = [
            DiscoveredEntity(ticker=t, name=self._display_name(t, alias_map)) for t in tickers
        ]

    @staticmethod
    def _display_name(ticker: str, alias_map: Mapping[str, Sequence[str]]) -> str:
        """Primary alias for ``ticker`` (the retrieval-relevant name); falls back to the ticker."""
        names = alias_map.get(ticker) or []
        return names[0] if names else ticker

    def _coverage(self) -> float:
        """Φ(s) = aspect coverage of the current union (the ONLY place gold aspects enter).

        Lazy import: ``research.multistep_eval`` imports ``rag`` (a top-level import here would be
        circular) — the same hatch ``rag.reward`` uses. Reusing ``coverage`` keeps reward == metric.
        """
        from stock_agent.research.multistep_eval import coverage

        assert self._episode is not None  # guarded by callers (reset/step)
        frac, _, _ = coverage(self._union, self._episode.aspects)
        return frac

    @staticmethod
    def _dedup_union(
        existing: list[RetrievedChunk], new: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Fold ``new`` into ``existing`` using the A4 loop's own union dedup (by ``chunk_id``).

        Deliberate reuse (not reimplementation): A6.2 *is* the A4 loop as RL, and the A6.2g
        comparison against that loop only holds if the accumulated union is built identically.
        """
        from stock_agent.research.agentic import _dedup_union

        return _dedup_union(existing, new)

    def _state(self, *, last_new_chunks: int) -> np.ndarray:
        """Assemble ``s_t`` from the static block + the current evidence summary (A6.2a)."""
        assert self._static is not None  # populated by reset()
        return featurize_state(
            self._static,
            self._union,
            step_idx=self._step_idx,
            max_steps=self.max_steps,
            n_discovered_unretrieved=self._n_discovered_raw,
            last_new_chunks=last_new_chunks,
        )
