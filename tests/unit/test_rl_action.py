"""A6.2b — action space + templates: pinned ordering, label-free query templates, slot masking.

The action space is a fixed ordered list (STOP at 0) a policy's logits index into, so the order
is a load-bearing invariant pinned here. The discovered-scope template is asserted label-free (it
prepends the entity's *alias name* + the plain question, never a gold aspect span) so the frozen
policy can build the identical query on a real, unlabeled question. Pure functions — no model/IO.
"""

from __future__ import annotations

import pytest

from stock_agent.rag.read_path import LATTICE_SYSTEMS
from stock_agent.rag.rl.action import (
    CONFIGS_FULL,
    CONFIGS_PRUNED,
    KNOWN_ARMS,
    STOP,
    Action,
    DiscoveredEntity,
    ScopeKind,
    action_label,
    action_to_request,
    action_to_requests,
    build_action_space,
    discovered_scope_query,
    is_legal,
    named_action_space,
    self_scope_query,
)

MU = DiscoveredEntity(ticker="MU", name="Micron")
TSM = DiscoveredEntity(ticker="TSM", name="Taiwan Semiconductor")
QUESTION = "Which memory supplier NVIDIA depends on discloses Chinese cybersecurity restrictions?"
# The generator's real bridge frame — both hops are asserted against THIS, not a paraphrase.
BRIDGE_Q = (
    "Among the companies NVIDIA names as key dependencies in its SEC filings, "
    "which one warns about export license requirements in its own filings?"
)


# ---- known arms = LATTICE_SYSTEMS ∪ {graph} (single source of truth is read_path) --------------
def test_known_arms_are_lattice_plus_graph() -> None:
    assert KNOWN_ARMS == LATTICE_SYSTEMS + ("graph",)
    assert CONFIGS_PRUNED == ("hybrid", "graph")
    assert CONFIGS_FULL == KNOWN_ARMS


# ---- pinned action-space ordering (reordering silently remaps trained weights) ------------------
def test_pruned_action_space_layout() -> None:
    space = named_action_space("pruned")  # 2 arms × (self + 2 disc + fanout) + STOP = 9
    assert [action_label(a) for a in space] == [
        "STOP",
        "hybrid@self",
        "hybrid@disc0",
        "hybrid@disc1",
        "hybrid@fanout",
        "graph@self",
        "graph@disc0",
        "graph@disc1",
        "graph@fanout",
    ]
    assert space[0] is STOP and space[0].is_stop


def test_full_action_space_layout() -> None:
    space = named_action_space("full")  # 5 arms × (self + 2 disc + fanout) + STOP = 21
    assert len(space) == 21
    assert space[0].is_stop
    # config-major order: the whole hybrid block precedes the graph block, etc.
    assert action_label(space[1]) == "dense@self"
    assert action_label(space[-1]) == "graph@fanout"
    # exactly one STOP; all non-STOP actions distinct.
    assert sum(a.is_stop for a in space) == 1
    assert len(set(space)) == len(space)


def test_named_action_space_dispatch_and_unknown() -> None:
    assert len(named_action_space("pruned")) == 9
    assert len(named_action_space("full")) == 21
    with pytest.raises(ValueError, match="unknown action space"):
        named_action_space("enormous")


def test_fanout_can_be_disabled_reproducing_the_pre_E3_space() -> None:
    """The A6.2c-f layout, for the E3 ablation and for loading pre-E3 checkpoints."""
    space = named_action_space("pruned", fanout=False)
    assert [action_label(a) for a in space] == [
        "STOP",
        "hybrid@self",
        "hybrid@disc0",
        "hybrid@disc1",
        "graph@self",
        "graph@disc0",
        "graph@disc1",
    ]


def test_slot_count_scales_space() -> None:
    # J=0 -> self + fanout only (fanout does not index slots, so it survives J=0).
    assert [
        action_label(a)
        for a in build_action_space(("hybrid",), n_discovered_slots=0)
    ] == ["STOP", "hybrid@self", "hybrid@fanout"]
    # J=3 -> self + 3 disc + fanout per config.
    assert len(build_action_space(("hybrid",), n_discovered_slots=3)) == 1 + 1 + 3 + 1


def test_build_action_space_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_action_space(("hybrid", "quantum"))
    with pytest.raises(ValueError, match=">= 0"):
        build_action_space(("hybrid",), n_discovered_slots=-1)


# ---- Action self-validation ---------------------------------------------------------------------
def test_action_post_init_validation() -> None:
    # STOP must carry no scope.
    with pytest.raises(ValueError, match="STOP action must have no scope"):
        Action(scope_kind=ScopeKind.SELF)
    # discovered needs a non-negative slot; self must not carry one.
    with pytest.raises(ValueError, match="non-negative slot"):
        Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED)
    with pytest.raises(ValueError, match="must not carry a discovered slot"):
        Action(config="hybrid", scope_kind=ScopeKind.SELF, scope_slot=0)
    # a search action needs a scope_kind and a known arm.
    with pytest.raises(ValueError, match="needs a scope_kind"):
        Action(config="hybrid")
    with pytest.raises(ValueError, match="unknown arm"):
        Action(config="quantum", scope_kind=ScopeKind.SELF)


def test_actions_are_hashable() -> None:
    # frozen ⇒ usable as set/dict keys (env cache + anti-loop rely on this).
    a = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    assert a in {a}
    assert a == Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)


# ---- templates: STOP, self-scope, discovered-scope ----------------------------------------------
def test_stop_expands_to_no_request() -> None:
    assert action_to_request(STOP, QUESTION, self_ticker="NVDA", discovered=[]) is None


def test_self_scope_query_asks_for_the_RELATION_on_a_bridge_question() -> None:
    """E2: hop 1 of a bridge must find the chunk that NAMES the related companies.

    Searching the whole question lets the topic terms ("Chinese cybersecurity restrictions")
    dominate and returns the seed's own topic paragraphs — the naming chunk, which is the only
    source of the entities hop 2 can bridge to, is never retrieved. 48.6% → 100% on held-out HARD.
    """
    a = Action(config="hybrid", scope_kind=ScopeKind.SELF)
    req = action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[])
    assert req is not None
    assert req.arm == "hybrid"
    assert req.query == "suppliers supply depend key dependencies"  # the RELATION, not the topic
    assert req.scope_ticker == "NVDA"
    # no seed ticker ⇒ unscoped self search (degrades gracefully).
    req2 = action_to_request(a, QUESTION, self_ticker=None, discovered=[])
    assert req2 is not None and req2.scope_ticker is None


def test_self_scope_query_leaves_a_non_bridge_question_verbatim() -> None:
    """CTRL (single-entity) questions are unchanged — the regression the CTRL stratum guards."""
    ctrl = "What does Micron disclose about DRAM pricing in its SEC filings?"
    a = Action(config="hybrid", scope_kind=ScopeKind.SELF)
    req = action_to_request(a, ctrl, self_ticker="MU", discovered=[])
    assert req is not None and req.query == ctrl


def test_self_scope_query_is_label_free() -> None:
    """The decomposition reads the question TEXT only — never stratum/relation/target metadata."""
    bridge_q = "Among the competitors NVIDIA names, which one discloses X in its own filings?"
    assert self_scope_query(bridge_q) == "competitors competition compete"
    # a bridge cue with no known relation word ⇒ fall back to the whole question (never a guess)
    odd = "Which entity in its own filings discusses X?"
    assert self_scope_query(odd) == odd


def test_discovered_scope_template_prepends_name_and_scopes_by_ticker() -> None:
    a = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    req = action_to_request(a, BRIDGE_Q, self_ticker="NVDA", discovered=[MU, TSM])
    assert req is not None
    assert req.arm == "graph"
    assert req.query == "Micron export license requirements"  # {name} {TOPIC} — E6
    assert req.scope_ticker == "MU"
    # slot 1 targets the second discovered entity.
    a1 = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    req1 = action_to_request(a1, BRIDGE_Q, self_ticker="NVDA", discovered=[MU, TSM])
    assert req1 is not None and req1.scope_ticker == "TSM"
    assert req1.query == "Taiwan Semiconductor export license requirements"


# ---- E6: hop 2 searches the TOPIC, not the relation (the mirror of E2) ---------------------------
def test_discovered_scope_query_strips_the_bridge_frame_to_the_topic() -> None:
    """Sent whole, the question drowns the target's filing in scaffolding about the SEED.

    "Analog Devices Among the competitors AMD names in its SEC filings, which one discloses customer
    concentration in its own filings?" is ~20 tokens of "competitors / AMD / which one / SEC
    filings" around the 2 that matter. Measured on the held-out fold, retrieval inside the RIGHT
    company found the evidence only 33.3% of the time. Hop 2 must search the topic.
    """
    competes = (
        "Among the competitors AMD names in its SEC filings, "
        "which one discloses customer concentration in its own filings?"
    )
    got = discovered_scope_query(competes, "Analog Devices")
    assert got == "Analog Devices customer concentration"
    depends = (
        "Among the companies NVIDIA names as key dependencies in its SEC filings, "
        "which one warns about export license requirements in its own filings?"
    )
    assert discovered_scope_query(depends, "Micron") == "Micron export license requirements"


def test_discovered_scope_query_leaves_a_non_bridge_question_verbatim() -> None:
    """CTRL (single-entity) questions keep the old template — the CTRL stratum guards this."""
    ctrl = "What does Micron disclose about DRAM pricing in its SEC filings?"
    assert discovered_scope_query(ctrl, "Micron") == f"Micron {ctrl}"


def test_discovered_scope_query_falls_back_on_an_unrecognized_bridge_frame() -> None:
    """A bridge cue with no topic cue ⇒ send the whole question. Never guess at a topic."""
    odd = "Which entity in its own filings is the relevant competitor here?"
    assert discovered_scope_query(odd, "Micron") == f"Micron {odd}"


def test_hop1_and_hop2_split_the_bridge_question_between_them() -> None:
    """The two halves of a bridge question, each going to the hop that wants it (E2 + E6)."""
    assert self_scope_query(BRIDGE_Q) == "suppliers supply depend key dependencies"  # the RELATION
    assert discovered_scope_query(BRIDGE_Q, "Micron") == "Micron export license requirements"


# ---- E3: fanout sweeps EVERY candidate ----------------------------------------------------------
def test_fanout_expands_to_one_request_per_candidate() -> None:
    """The whole point of E3: hop 2 reaches every candidate, not the alphabetically-first J."""
    a = Action(config="hybrid", scope_kind=ScopeKind.FANOUT)
    reqs = action_to_requests(a, QUESTION, self_ticker="NVDA", discovered=[MU, TSM])
    assert [r.scope_ticker for r in reqs] == ["MU", "TSM"]  # caller's order, preserved
    assert all(r.arm == "hybrid" for r in reqs)


def test_fanout_branch_matches_the_discovered_slot_template_so_the_cache_is_shared() -> None:
    """A fanout branch must build the SAME (arm, scope, query) triple a disc-slot action would.

    That identity is what lets the env's TransitionCache serve a sweep as N lookups — the property
    that keeps fanout affordable inside a PPO training loop.
    """
    fan = Action(config="hybrid", scope_kind=ScopeKind.FANOUT)
    disc0 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    disc1 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    kw = {"self_ticker": "NVDA", "discovered": [MU, TSM]}
    swept = action_to_requests(fan, QUESTION, **kw)  # type: ignore[arg-type]
    assert swept[0] == action_to_request(disc0, QUESTION, **kw)  # type: ignore[arg-type]
    assert swept[1] == action_to_request(disc1, QUESTION, **kw)  # type: ignore[arg-type]


def test_fanout_with_nothing_discovered_is_masked() -> None:
    # No candidates ⇒ no requests (a CTRL question never discovers one, so fanout is a no-op there).
    a = Action(config="hybrid", scope_kind=ScopeKind.FANOUT)
    assert action_to_requests(a, QUESTION, self_ticker="NVDA", discovered=[]) == []
    assert not is_legal(a, n_discovered=0)
    assert is_legal(a, n_discovered=1)  # one candidate is enough to sweep
    assert is_legal(a, n_discovered=19)  # ... and the sweep is NOT capped at the J slots


def test_fanout_rejects_a_slot_and_action_to_request_refuses_to_collapse_it() -> None:
    with pytest.raises(ValueError, match="fanout action must not carry a discovered slot"):
        Action(config="hybrid", scope_kind=ScopeKind.FANOUT, scope_slot=0)
    # Silently returning only the first branch would be exactly the bug E3 removes ⇒ fail loudly.
    a = Action(config="hybrid", scope_kind=ScopeKind.FANOUT)
    with pytest.raises(ValueError, match="expands to N requests"):
        action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[MU, TSM])


def test_discovered_slot_out_of_range_is_masked() -> None:
    # slot 1 with only one discovered entity ⇒ no request (env no-op / policy masks it).
    a = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    assert action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[MU]) is None
    # slot 0 with an empty discovered list ⇒ also masked.
    a0 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    assert action_to_request(a0, QUESTION, self_ticker="NVDA", discovered=[]) is None


def test_query_template_is_label_free() -> None:
    # The discovered-scope query is built from the entity's alias + text lifted out of the
    # QUESTION — never from the gold aspects. Guards the leakage line: the frozen policy builds
    # the identical query on a real, unlabeled question, where no spans exist to read.
    a = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    req = action_to_request(a, BRIDGE_Q, self_ticker="NVDA", discovered=[MU])
    assert req is not None
    assert req.query == f"{MU.name} export license requirements"
    assert req.query == discovered_scope_query(BRIDGE_Q, MU.name)  # one template, one source


# ---- legality mask (matches action_to_request's None <-> masked contract) ------------------------
def test_is_legal_matches_request_masking() -> None:
    self_a = Action(config="hybrid", scope_kind=ScopeKind.SELF)
    disc0 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    disc1 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    # STOP + self always legal, independent of discoveries.
    assert is_legal(STOP, n_discovered=0)
    assert is_legal(self_a, n_discovered=0)
    # discovered slot legal iff it indexes an actually-discovered entity.
    assert not is_legal(disc0, n_discovered=0)
    assert is_legal(disc0, n_discovered=1)
    assert not is_legal(disc1, n_discovered=1)
    assert is_legal(disc1, n_discovered=2)
    # is_legal and action_to_request agree on the masking boundary.
    for n in range(3):
        for act in (disc0, disc1):
            legal = is_legal(act, n_discovered=n)
            disc = [MU, TSM][:n]
            req = action_to_request(act, QUESTION, self_ticker="NVDA", discovered=disc)
            assert legal == (req is not None)
