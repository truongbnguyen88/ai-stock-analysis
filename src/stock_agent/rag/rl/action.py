"""Discrete action space + query templates for full retrieval RL (advanced-RAG A6.2b).

The MDP action ``a_t`` jointly answers **which retriever / where to point it / whether to stop**:

    a_t ∈ {STOP} ∪ {(config c, scope σ)}      c ∈ arms,  σ ∈ {self-ticker, discovered-entity_j}

This strictly contains A6.1 (config-only bandit) and A4/A5 (scope + stop). The space is a **fixed
ordered list** (STOP at index 0) — a trained policy's action logits index into it, so the order is
load-bearing and frozen in checkpoints (a test pins it).

**Deterministic, LLM-free transition (Q3 template B).** An action expands to a ``RetrievalRequest``
via a pure template — no decision-call LLM — which is what makes ``$0`` reproducible rollouts (and
on-policy PPO) possible:

    self-scope         → query = question,              scope = self_ticker
    discovered-entity#j → query = "{entity_name} {question}",  scope = that entity's ticker

**Leakage line (Q4.3).** The template is **label-free**: it reads only the question text and the
discovered entity's alias name, never the gold aspect spans. (The pre-questions Q3 illustration
``"Micron critical information infrastructure"`` prepended an *aspect span* — a gold label — which
would be undeployable; template B replaces the topic phrase with the plain question so the frozen
policy can build the exact same query on a real, unlabeled user question.) The hard
``ChunkFilter(ticker=...)`` the env builds from ``scope_ticker`` does the actual scoping; the name
prefix is a soft ranking signal (filings say "Micron", not "MU", so the name is the useful token).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from stock_agent.rag.read_path import LATTICE_SYSTEMS

# The executable retrieval arms an action may select — the A6.1 config space plus the A5 graph arm,
# i.e. LATTICE_SYSTEMS ∪ {graph}. Single source of truth is read_path; re-derived here only so a
# bad arm in a custom action space fails fast at construction, not later at retrieval time.
GRAPH_ARM = "graph"
KNOWN_ARMS: tuple[str, ...] = LATTICE_SYSTEMS + (GRAPH_ARM,)

# The two named presets (§3b of docs/a6_2_plan.md):
#  - "pruned" (default): the arms that actually win — hybrid (A6.1) and graph (A5) — so ~149 train
#    episodes × T=3 are not spread across a 16-way space that is mostly never-optimal.
#  - "full": every arm, kept as an ablation.
CONFIGS_PRUNED: tuple[str, ...] = ("hybrid", GRAPH_ARM)
CONFIGS_FULL: tuple[str, ...] = KNOWN_ARMS
_NAMED_CONFIGS: dict[str, tuple[str, ...]] = {"pruned": CONFIGS_PRUNED, "full": CONFIGS_FULL}

# Discovered-entity action slots (cap J = agentic_bridge_max_entities). A slot indexes the state's
# label-free, lexicographically-sorted discovered list, so "slot j" names a stable entity per step.
DEFAULT_DISCOVERED_SLOTS = 2


class ScopeKind(StrEnum):
    """Where a search points: the episode's own subject ticker, or a discovered-entity slot."""

    SELF = "self"
    DISCOVERED = "discovered"


@dataclass(frozen=True)
class Action:
    """One MDP action: STOP, or a ``(config, scope)`` search.

    STOP   — ``config is None`` and no scope (the learned reflective stop).
    SEARCH — ``config`` is an arm in ``KNOWN_ARMS``; ``scope_kind`` is ``SELF`` (``scope_slot``
    None) or ``DISCOVERED`` (``scope_slot`` = the discovered-list index targeted, 0..J-1).

    Frozen ⇒ hashable, so actions are usable as cache / anti-loop keys downstream. ``__post_init__``
    rejects malformed combinations at construction (STOP-with-scope, discovered-without-slot, …).
    """

    config: str | None = None
    scope_kind: ScopeKind | None = None
    scope_slot: int | None = None

    def __post_init__(self) -> None:
        if self.config is None:  # STOP: must carry no scope
            if self.scope_kind is not None or self.scope_slot is not None:
                raise ValueError("STOP action must have no scope")
            return
        if self.config not in KNOWN_ARMS:
            raise ValueError(f"unknown arm '{self.config}'; choose from {KNOWN_ARMS}")
        if self.scope_kind is ScopeKind.SELF:
            if self.scope_slot is not None:
                raise ValueError("self-scope action must not carry a discovered slot")
        elif self.scope_kind is ScopeKind.DISCOVERED:
            if self.scope_slot is None or self.scope_slot < 0:
                raise ValueError("discovered-scope action needs a non-negative slot")
        else:
            raise ValueError("search action needs a scope_kind (SELF or DISCOVERED)")

    @property
    def is_stop(self) -> bool:
        """True for the terminal STOP action (no retrieval runs)."""
        return self.config is None


STOP = Action()  # the canonical terminal action; index 0 in every action space


@dataclass(frozen=True)
class DiscoveredEntity:
    """A named-but-unretrieved entity a discovered-scope action can target.

    ``ticker`` scopes the retrieval (the hard ``ChunkFilter``); ``name`` is the display alias the
    query template prepends to the question. The env builds these from
    ``discovered_unretrieved_entities`` in a **label-free lexicographic-by-ticker** order, so slot
    indices are deterministic and reproducible across rollouts.
    """

    ticker: str
    name: str


@dataclass(frozen=True)
class RetrievalRequest:
    """The templated, LLM-free retrieval an action expands to (executed by the env in A6.2c).

    ``arm`` names the ``RetrievalSystem`` to build; ``query`` is the deterministic template string;
    ``scope_ticker`` becomes ``ChunkFilter(ticker=...)`` (``None`` ⇒ unscoped). The env keys its
    anti-loop set on ``(query, scope)`` exactly as the A4 loop does.
    """

    arm: str
    query: str
    scope_ticker: str | None


def build_action_space(
    configs: Sequence[str], *, n_discovered_slots: int = DEFAULT_DISCOVERED_SLOTS
) -> list[Action]:
    """STOP + ``configs`` × (self-ticker + J discovered slots), in a pinned **config-major** order.

    Order (load-bearing — a policy's action logits index into it): index 0 = STOP, then for each
    config in ``configs`` order: (config, self), (config, disc#0), …, (config, disc#J-1).
    Reordering silently remaps every trained weight, so this order is asserted in tests and frozen
    in checkpoints. Raises on an unknown arm or a negative slot count.
    """
    if n_discovered_slots < 0:
        raise ValueError("n_discovered_slots must be >= 0")
    unknown = [c for c in configs if c not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {KNOWN_ARMS}")
    space: list[Action] = [STOP]
    for config in configs:
        space.append(Action(config=config, scope_kind=ScopeKind.SELF))
        for slot in range(n_discovered_slots):
            space.append(Action(config=config, scope_kind=ScopeKind.DISCOVERED, scope_slot=slot))
    return space


def named_action_space(
    name: str = "pruned", *, n_discovered_slots: int = DEFAULT_DISCOVERED_SLOTS
) -> list[Action]:
    """Build a preset space: ``"pruned"`` (2 arms, 7 actions) or ``"full"`` (5 arms, 16 actions)."""
    if name not in _NAMED_CONFIGS:
        raise ValueError(f"unknown action space '{name}'; choose from {sorted(_NAMED_CONFIGS)}")
    return build_action_space(_NAMED_CONFIGS[name], n_discovered_slots=n_discovered_slots)


def is_legal(action: Action, *, n_discovered: int) -> bool:
    """Whether ``action`` is executable at the current state.

    STOP and self-scope are always legal; a discovered slot is legal only when it indexes an
    actually-discovered entity (``scope_slot < n_discovered``). The policy masks illegal actions
    (``-inf`` logit) and the env treats a masked pick as a no-op — see ``action_to_request``.
    """
    if action.is_stop or action.scope_kind is ScopeKind.SELF:
        return True
    return action.scope_slot is not None and action.scope_slot < n_discovered


# Hop-1 query decomposition (A6.2 E2). A bridge question — "Among the competitors NVIDIA names in
# its filings, which one discloses {topic}?" — asks TWO things at once, and searching its full text
# answers the wrong one: the topic terms dominate the query, so the retriever returns the seed's
# own {topic} paragraphs instead of the paragraph that NAMES the competitors. The naming chunk is
# what hop 1 exists to find (it is the only source of the entities hop 2 can bridge to), so the
# self-scoped query for a bridge question must ask for the RELATION, not the topic. Measured on the
# 35 held-out HARD episodes: the full-question template found the naming chunk 48.6% of the time,
# these focused queries find it 100%.
#
# Cues are matched against the question TEXT only (never generation metadata), so the deployed
# policy builds the identical query on a real, unlabeled user question — the A6.2b invariant.
_RELATION_FOCUS: tuple[tuple[str, str], ...] = (
    ("competitor", "competitors competition compete"),
    ("compete", "competitors competition compete"),
    ("dependenc", "suppliers supply depend key dependencies"),
    ("supplier", "suppliers supply depend key dependencies"),
    ("vendor", "suppliers supply depend key dependencies"),
    ("customer", "customers"),
    ("partner", "partners collaborations"),
)


def self_scope_query(question: str) -> str:
    """The hop-1 (self-scoped) query: the RELATION for a bridge question, else the question itself.

    Gated on the production ``research.bridge.is_bridging`` so the simulator's notion of "this is a
    bridge" cannot drift from the one A4/A5 actually deploy. A non-bridge (e.g. CTRL single-entity)
    question is searched verbatim — unchanged behaviour, which is what the CTRL stratum regresses.
    """
    from stock_agent.research.bridge import is_bridging  # lazy: research imports rag (circular)

    if not is_bridging(question):
        return question
    low = question.lower()
    for cue, focus in _RELATION_FOCUS:
        if cue in low:
            return focus
    return question  # a bridge cue with no known relation ⇒ fall back to the whole question


def action_to_request(
    action: Action,
    question: str,
    *,
    self_ticker: str | None,
    discovered: Sequence[DiscoveredEntity],
) -> RetrievalRequest | None:
    """Expand an action to its templated ``RetrievalRequest``; ``None`` for STOP or a masked slot.

    Templates (deterministic, NO LLM, label-free — the frozen policy runs these verbatim at deploy):
      - self-scope: ``query = self_scope_query(question)`` (the relation for a bridge question — see
        ``_RELATION_FOCUS``), ``scope = self_ticker`` (``None`` ⇒ unscoped).
      - discovered#j: ``query = "{name} {question}"``, ``scope = that entity's ticker``. If slot j
        is not (yet) populated (``j >= len(discovered)``) the action is illegal ⇒ ``None``.
    """
    if action.config is None:  # STOP (narrows action.config to str below for mypy)
        return None
    if action.scope_kind is ScopeKind.SELF:
        return RetrievalRequest(
            arm=action.config, query=self_scope_query(question), scope_ticker=self_ticker
        )
    slot = action.scope_slot
    if slot is None or slot >= len(discovered):
        return None  # masked: this discovered slot has no entity at the current state
    entity = discovered[slot]
    return RetrievalRequest(
        arm=action.config, query=f"{entity.name} {question}", scope_ticker=entity.ticker
    )


def action_label(action: Action) -> str:
    """Short stable label for logs / test pinning: ``STOP``, ``hybrid@self``, ``graph@disc1``, …."""
    if action.is_stop:
        return "STOP"
    if action.scope_kind is ScopeKind.SELF:
        return f"{action.config}@self"
    return f"{action.config}@disc{action.scope_slot}"
