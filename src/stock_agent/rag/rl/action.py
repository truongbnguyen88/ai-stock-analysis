"""Discrete action space + query templates for full retrieval RL (advanced-RAG A6.2b).

The MDP action ``a_t`` jointly answers **which retriever / where to point it / whether to stop**:

    a_t ∈ {STOP} ∪ {(config c, scope σ)}   c ∈ arms,  σ ∈ {self, discovered_j, **fanout**}

This strictly contains A6.1 (config-only bandit) and A4/A5 (scope + stop). The space is a **fixed
ordered list** (STOP at index 0) — a trained policy's action logits index into it, so the order is
load-bearing and frozen in checkpoints (a test pins it).

**``fanout`` (A6.2 E3) is the action the retracted A6.2 verdict was missing.** The pre-E3 space
could only point hop 2 at discovered slot 0 or 1 — the alphabetically-first two of ~9.3 candidates
on a held-out HARD episode (max 19) — so the oracle action was on the menu for only ρ ≈ 11.4% of
states. That capped *any* policy at ≈0.26 coverage while the scripted A4/A5 baseline already scored
0.271: the null result was an artifact of the action space, not evidence about RL.

``fanout`` sweeps every candidate in one action and **fixes reachability** (the target is now a
discovered candidate on 78.6% of held-out bridge episodes, up from 11.4%). It does **not**, on its
own, deliver a large coverage win — measured, `sweep(hybrid)` scores 0.500 HARD coverage vs
`react(hybrid)`'s 0.486, and loses on *return* once its N × arm cost is charged. The bottleneck
simply moved downstream, to hop-2 retrieval and the evidence budget:

    reachable (target discovered)        78.6%
    → its branch RETRIEVED the evidence  50.0%   (E6 raised this from 33.3%)
    → that chunk SEATED in the union     21.4%   ← the sweep seats ~1 chunk per branch

An earlier estimate of "A2 → 68.6%, ceiling ≈0.84" was **wrong**: it measured retrieval into the
branch and never asked whether the chunk survived the capped union. Do not restore it.

**Deterministic, LLM-free transition (Q3 template B).** An action expands to a ``RetrievalRequest``
via a pure template — no decision-call LLM — which is what makes ``$0`` reproducible rollouts (and
on-policy PPO) possible. A bridge question asks **two** things at once, and each hop wants a
different one, so each hop searches its own half (E2 / E6 — see ``self_scope_query`` and
``discovered_scope_query``):

    self-scope (hop 1)  → query = the RELATION  ("competitors competition compete")
    discovered / fanout → query = "{entity_name} {TOPIC}"   (hop 2: the topic, not the relation)
    (non-bridge questions are searched verbatim in both cases)

**Leakage line (Q4.3).** The template is **label-free**: it reads only the question text and the
discovered entity's alias name, never the gold aspect spans. (The pre-questions Q3 illustration
``"Micron critical information infrastructure"`` prepended an *aspect span* — a gold label — which
would be undeployable; template B replaces the topic phrase with the plain question so the frozen
policy can build the exact same query on a real, unlabeled user question.) The hard
``ChunkFilter(ticker=...)`` the env builds from ``scope_ticker`` does the actual scoping; the name
prefix is a soft ranking signal (filings say "Micron", not "MU", so the name is the useful token).
"""

from __future__ import annotations

import re
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
    """Where a search points: the episode's own ticker, one discovered slot, or *all* candidates."""

    SELF = "self"
    DISCOVERED = "discovered"
    FANOUT = "fanout"  # sweep EVERY discovered candidate in one action (A6.2 E3)


@dataclass(frozen=True)
class Action:
    """One MDP action: STOP, or a ``(config, scope)`` search.

    STOP   — ``config is None`` and no scope (the learned reflective stop).
    SEARCH — ``config`` is an arm in ``KNOWN_ARMS``; ``scope_kind`` is ``SELF`` (``scope_slot``
    None), ``DISCOVERED`` (``scope_slot`` = the discovered-list index targeted, 0..J-1), or
    ``FANOUT`` (no slot — sweeps *every* discovered candidate; expands to N requests).

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
        elif self.scope_kind is ScopeKind.FANOUT:
            if self.scope_slot is not None:
                raise ValueError("fanout action must not carry a discovered slot")
        else:
            raise ValueError("search action needs a scope_kind (SELF, DISCOVERED or FANOUT)")

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
    configs: Sequence[str],
    *,
    n_discovered_slots: int = DEFAULT_DISCOVERED_SLOTS,
    fanout: bool = True,
) -> list[Action]:
    """STOP + ``configs`` × (self + J discovered slots + fanout), in a pinned config-major order.

    Order (load-bearing — a policy's action logits index into it): index 0 = STOP, then for each
    config in ``configs`` order: (config, self), (config, disc#0), …, (config, disc#J-1),
    (config, fanout). Reordering silently remaps every trained weight, so this order is asserted in
    tests and frozen in checkpoints. Raises on an unknown arm or a negative slot count.

    ``fanout=False`` reproduces the pre-E3 space exactly (the A6.2c-f layout), which is what the
    E3 ablation and the retracted-verdict checkpoints need.
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
        if fanout:
            space.append(Action(config=config, scope_kind=ScopeKind.FANOUT))
    return space


def named_action_space(
    name: str = "pruned",
    *,
    n_discovered_slots: int = DEFAULT_DISCOVERED_SLOTS,
    fanout: bool = True,
) -> list[Action]:
    """Build a preset space: ``"pruned"`` (2 arms, 9 actions) or ``"full"`` (5 arms, 21 actions)."""
    if name not in _NAMED_CONFIGS:
        raise ValueError(f"unknown action space '{name}'; choose from {sorted(_NAMED_CONFIGS)}")
    return build_action_space(
        _NAMED_CONFIGS[name], n_discovered_slots=n_discovered_slots, fanout=fanout
    )


def is_legal(action: Action, *, n_discovered: int) -> bool:
    """Whether ``action`` is executable at the current state.

    STOP and self-scope are always legal; a discovered slot is legal only when it indexes an
    actually-discovered entity (``scope_slot < n_discovered``); fanout needs **at least one**
    candidate to sweep. The policy masks illegal actions (``-inf`` logit) and the env treats a
    masked pick as a no-op — see ``action_to_requests``.
    """
    if action.is_stop or action.scope_kind is ScopeKind.SELF:
        return True
    if action.scope_kind is ScopeKind.FANOUT:
        return n_discovered >= 1
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


# Hop-2 query decomposition (A6.2 E6) — the mirror of E2. A bridge question asks two things, and
# hop 2 wants the OTHER one. Sent whole into a candidate's filings, the question is mostly
# scaffolding about the *seed* and the *relation*:
#
#   "Analog Devices Among the competitors AMD names in its SEC filings, which one discloses
#    customer concentration in its own filings?"
#
# — 20-odd tokens of "competitors / AMD / which one / SEC filings" around the four that matter.
# Retrieval inside the RIGHT company then fails: measured on the held-out fold, only 33.3% of
# targets had their evidence retrieved even when the sweep reached them. Hop 2 must search the
# TOPIC, not the relation. Cues are matched against question TEXT only (never generation metadata),
# so a deployed policy builds the identical query on a real, unlabeled question.
_TOPIC_CUES: tuple[str, ...] = (
    "discloses",
    "disclose",
    "warns about",
    "warns of",
    "warn about",
    "mentions",
    "reports",
    "describes",
    "flags",
)
# The trailing "... in its own filings?" clause is scaffolding too — it belongs to the question
# frame, not to the topic being searched for.
_TOPIC_TAIL = re.compile(
    r"\s+in (?:its|their) own (?:sec )?filings\b.*$|\s*[?.]+\s*$", re.IGNORECASE
)


def discovered_scope_query(question: str, entity_name: str) -> str:
    """The hop-2 query for one candidate: ``"{name} {topic}"`` on a bridge question, else the whole.

    Strips the bridge frame ("Among the competitors X names …, which one discloses …") down to the
    topic clause, and prefixes the candidate's display alias (filings say "Micron", not "MU", so the
    name is the useful ranking token). Non-bridge questions are searched verbatim, and a bridge
    question with no recognizable topic cue falls back to the whole question — never a guess.

    Label-free: reads ``question`` and the discovered entity's alias only, never the gold aspects.

    **Benchmark caveat (do not over-read the lift this produces).** The generated benchmark builds
    its questions by interpolating the gold A2 span verbatim as ``{topic}``, so on *this* corpus the
    extracted clause recovers the gold span exactly, and hop-2 retrieval looks easier than it will
    be on a real user question whose topic is paraphrased. The measured gain is therefore an upper
    bound; the paid LLM-query-writer (sim-to-real) run is what arbitrates the real number.
    """
    from stock_agent.research.bridge import is_bridging  # lazy: research imports rag (circular)

    if not is_bridging(question):
        return f"{entity_name} {question}"
    low = question.lower()
    for cue in _TOPIC_CUES:
        idx = low.find(f" {cue} ")
        if idx < 0:
            continue
        topic = _TOPIC_TAIL.sub("", question[idx + len(cue) + 2 :]).strip()
        if topic:
            return f"{entity_name} {topic}"
    return f"{entity_name} {question}"  # bridge frame we do not recognize ⇒ keep today's behaviour


def _discovered_request(arm: str, question: str, entity: DiscoveredEntity) -> RetrievalRequest:
    """The per-candidate hop-2 request. Shared by DISCOVERED and FANOUT — deliberately identical.

    Because a fanout branch builds the *same* ``(arm, scope_ticker, query)`` triple a discovered
    slot action would, the two share entries in the env's ``TransitionCache``: a fanout over N
    candidates is N cache lookups once those retrievals have been seen, which is what keeps a
    sweep affordable inside a PPO training loop.
    """
    return RetrievalRequest(
        arm=arm, query=discovered_scope_query(question, entity.name), scope_ticker=entity.ticker
    )


def action_to_requests(
    action: Action,
    question: str,
    *,
    self_ticker: str | None,
    discovered: Sequence[DiscoveredEntity],
) -> list[RetrievalRequest]:
    """Expand an action to the retrievals it runs. Empty list ⇒ STOP, or a masked (illegal) action.

    Templates (deterministic, NO LLM, label-free — the frozen policy runs these verbatim at deploy):
      - self-scope: ``query = self_scope_query(question)`` (the relation for a bridge question —
        see ``_RELATION_FOCUS``), ``scope = self_ticker`` (``None`` ⇒ unscoped). One request.
      - discovered#j: ``query = "{name} {question}"``, ``scope = that entity's ticker``. One
        request, or none if slot j is unpopulated (``j >= len(discovered)``).
      - fanout: one such request **per discovered candidate**, in the caller's (lexicographic,
        label-free) order. None if nothing has been discovered yet.

    **Why fanout exists (A6.2 E3).** Hop-1 evidence says *which* companies the seed relates to, but
    carries essentially no signal about *which of them* discloses the topic — measured on held-out
    HARD, no label-free ordering beat chance at putting the target in the top 2 (alphabetical 42.9%,
    mention-count 40%, chunk-score 40%, **random 40%**). A ranking-shaped action ("probe candidate
    #j") therefore cannot be chosen well by *any* policy; the only reliable move is to **sweep**
    every candidate. Fanout makes that move expressible in a single action — and its cost, N × the
    arm's cost, is what the policy has to learn to price.
    """
    if action.config is None:  # STOP (narrows action.config to str below for mypy)
        return []
    if action.scope_kind is ScopeKind.SELF:
        return [
            RetrievalRequest(
                arm=action.config, query=self_scope_query(question), scope_ticker=self_ticker
            )
        ]
    if action.scope_kind is ScopeKind.FANOUT:
        return [_discovered_request(action.config, question, e) for e in discovered]
    slot = action.scope_slot
    if slot is None or slot >= len(discovered):
        return []  # masked: this discovered slot has no entity at the current state
    return [_discovered_request(action.config, question, discovered[slot])]


def action_to_request(
    action: Action,
    question: str,
    *,
    self_ticker: str | None,
    discovered: Sequence[DiscoveredEntity],
) -> RetrievalRequest | None:
    """Single-request convenience over ``action_to_requests``; ``None`` for STOP or a masked slot.

    Raises on a FANOUT action rather than silently returning only its first branch — a sweep that
    quietly collapses to one candidate is precisely the bug E3 removes, so it must fail loudly.
    """
    if action.scope_kind is ScopeKind.FANOUT:
        raise ValueError("fanout expands to N requests; call action_to_requests()")
    reqs = action_to_requests(
        action, question, self_ticker=self_ticker, discovered=discovered
    )
    return reqs[0] if reqs else None


def action_label(action: Action) -> str:
    """Short stable label for logs / tests: ``STOP``, ``hybrid@self``, ``graph@fanout``, …."""
    if action.is_stop:
        return "STOP"
    if action.scope_kind is ScopeKind.SELF:
        return f"{action.config}@self"
    if action.scope_kind is ScopeKind.FANOUT:
        return f"{action.config}@fanout"
    return f"{action.config}@disc{action.scope_slot}"
