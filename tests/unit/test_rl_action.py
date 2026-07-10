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
    build_action_space,
    is_legal,
    named_action_space,
)

MU = DiscoveredEntity(ticker="MU", name="Micron")
TSM = DiscoveredEntity(ticker="TSM", name="Taiwan Semiconductor")
QUESTION = "Which memory supplier NVIDIA depends on discloses Chinese cybersecurity restrictions?"


# ---- known arms = LATTICE_SYSTEMS ∪ {graph} (single source of truth is read_path) --------------
def test_known_arms_are_lattice_plus_graph() -> None:
    assert KNOWN_ARMS == LATTICE_SYSTEMS + ("graph",)
    assert CONFIGS_PRUNED == ("hybrid", "graph")
    assert CONFIGS_FULL == KNOWN_ARMS


# ---- pinned action-space ordering (reordering silently remaps trained weights) ------------------
def test_pruned_action_space_layout() -> None:
    space = named_action_space("pruned")  # 2 arms × (self + 2 disc) + STOP = 7
    assert [action_label(a) for a in space] == [
        "STOP",
        "hybrid@self",
        "hybrid@disc0",
        "hybrid@disc1",
        "graph@self",
        "graph@disc0",
        "graph@disc1",
    ]
    assert space[0] is STOP and space[0].is_stop


def test_full_action_space_layout() -> None:
    space = named_action_space("full")  # 5 arms × 3 scopes + STOP = 16
    assert len(space) == 16
    assert space[0].is_stop
    # config-major order: the whole hybrid block precedes the graph block, etc.
    assert action_label(space[1]) == "dense@self"
    assert action_label(space[-1]) == "graph@disc1"
    # exactly one STOP; all non-STOP actions distinct.
    assert sum(a.is_stop for a in space) == 1
    assert len(set(space)) == len(space)


def test_named_action_space_dispatch_and_unknown() -> None:
    assert len(named_action_space("pruned")) == 7
    assert len(named_action_space("full")) == 16
    with pytest.raises(ValueError, match="unknown action space"):
        named_action_space("enormous")


def test_slot_count_scales_space() -> None:
    # J=0 -> only self scopes; J=3 -> self + 3 disc per config.
    assert [action_label(a) for a in build_action_space(("hybrid",), n_discovered_slots=0)] == [
        "STOP",
        "hybrid@self",
    ]
    assert len(build_action_space(("hybrid",), n_discovered_slots=3)) == 1 + 1 + 3


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


def test_self_scope_template_is_plain_question() -> None:
    a = Action(config="hybrid", scope_kind=ScopeKind.SELF)
    req = action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[])
    assert req is not None
    assert req.arm == "hybrid"
    assert req.query == QUESTION  # plain question, no prefix
    assert req.scope_ticker == "NVDA"
    # no seed ticker ⇒ unscoped self search (degrades gracefully).
    req2 = action_to_request(a, QUESTION, self_ticker=None, discovered=[])
    assert req2 is not None and req2.scope_ticker is None


def test_discovered_scope_template_prepends_name_and_scopes_by_ticker() -> None:
    a = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    req = action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[MU, TSM])
    assert req is not None
    assert req.arm == "graph"
    assert req.query == f"Micron {QUESTION}"  # {name} {question}: label-free (name, not a span)
    assert req.scope_ticker == "MU"
    # slot 1 targets the second discovered entity.
    a1 = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    req1 = action_to_request(a1, QUESTION, self_ticker="NVDA", discovered=[MU, TSM])
    assert req1 is not None and req1.scope_ticker == "TSM"
    assert req1.query == f"Taiwan Semiconductor {QUESTION}"


def test_discovered_slot_out_of_range_is_masked() -> None:
    # slot 1 with only one discovered entity ⇒ no request (env no-op / policy masks it).
    a = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=1)
    assert action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[MU]) is None
    # slot 0 with an empty discovered list ⇒ also masked.
    a0 = Action(config="hybrid", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    assert action_to_request(a0, QUESTION, self_ticker="NVDA", discovered=[]) is None


def test_query_template_is_label_free() -> None:
    # The discovered-scope query must contain ONLY the entity name + verbatim question — never a
    # gold aspect span. Guards the leakage line: frozen policy builds this on an unlabeled query.
    a = Action(config="graph", scope_kind=ScopeKind.DISCOVERED, scope_slot=0)
    req = action_to_request(a, QUESTION, self_ticker="NVDA", discovered=[MU])
    assert req is not None
    assert req.query == f"{MU.name} {QUESTION}"


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
