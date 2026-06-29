"""A6.0 question templates — turn a verified graph edge + a topic into a labeled question.

Pure construction (no I/O, no graph, no LLM): the generator (``research/multistep_gen``) resolves
the surface names + topic from the A5 graph and calls these formatters. Two shapes:

- a **bridge** question (2 aspects): the seed names the target (A1), and the target's OWN filing
  discloses the topic (A2) — the multi-hop test. Present-in-seed (A1) and present-in-target (A2) are
  guaranteed upstream by the A5.1 edge guards; the generator's span-isolation probe adds only the
  absent-in-seed check on A2 (→ ``HARD`` vs ``MED``).
- a **control** question (1 aspect): a single entity's own disclosure of a topic — single-hop, the
  regression guard proving the loop/graph does not hurt easy questions.

Phrasing stays *generic* across targets (the graph relation does not distinguish a foundry from a
memory supplier), trading a little naturalness for uniform, $0 templating — the deliberate
sim-to-real tradeoff (an optional LLM-paraphrase pass is deferred, per ADVANCED_RAG_TODO §A6.0).
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_agent.research.multistep_eval import Aspect

# Relational framing per bridge relation: {seed} = the subject company's display name; {topic} = the
# bridged entity's own risk/topic phrase (also the A2 span). Only the two company→company relations
# are bridgeable into a "which related company …" question.
_BRIDGE_FRAME: dict[str, str] = {
    "depends_on": (
        "Among the companies {seed} names as key dependencies in its SEC filings, "
        "which one warns about {topic} in its own filings?"
    ),
    "competes_with": (
        "Among the competitors {seed} names in its SEC filings, "
        "which one discloses {topic} in its own filings?"
    ),
}

# The word used for the bridged entity in the aspect names (so labels read naturally).
_RELATION_OBJECT_WORD: dict[str, str] = {
    "depends_on": "dependency",
    "competes_with": "competitor",
}

_CONTROL_FRAME = "What does {company} disclose about {topic} in its SEC filings?"

BRIDGE_RELATIONS: tuple[str, ...] = tuple(_BRIDGE_FRAME)


@dataclass(frozen=True)
class FilledQuestion:
    """A template's output — the question text, its aspects, and the relation/qtype metadata.

    The generator wraps this into a ``MultiHopQuery`` after the span-isolation probe assigns the
    stratum and it attaches the seed/target/group provenance.
    """

    question: str
    aspects: list[Aspect]
    relation: str
    qtype: str


def fill_bridge(
    *, seed_name: str, target_surfaces: list[str], topic: str, relation: str
) -> FilledQuestion:
    """Build a 2-aspect bridge question for ``(seed) -[relation]-> (target)`` about ``topic``.

    - **A1** ("seed names the target"): spans = the target's surface forms as the seed's filing
      writes them (e.g. ``["Taiwan Semiconductor", "TSMC"]``) — guaranteed present in the seed by
      the bridge edge's A5.1 guard.
    - **A2** ("target's own disclosure"): span = the ``topic`` phrase (a risk/topic node name) —
      guaranteed present in the target by the ``mentions_risk``/``exposed_to`` edge's guard.

    Raises ``ValueError`` on an unsupported relation or empty ``target_surfaces`` (the A1 source).
    """
    if relation not in _BRIDGE_FRAME:
        raise ValueError(
            f"unsupported bridge relation {relation!r}; expected one of {list(_BRIDGE_FRAME)}"
        )
    if not target_surfaces:
        raise ValueError("target_surfaces must be non-empty (it is the A1 span source)")
    object_word = _RELATION_OBJECT_WORD[relation]
    question = _BRIDGE_FRAME[relation].format(seed=seed_name, topic=topic)
    aspects = [
        Aspect(name=f"{seed_name} names the {object_word}", spans=list(target_surfaces)),
        Aspect(name=f"that {object_word}'s own {topic} disclosure", spans=[topic]),
    ]
    return FilledQuestion(question=question, aspects=aspects, relation=relation, qtype="bridging")


def fill_control(*, company_name: str, topic: str) -> FilledQuestion:
    """Build a 1-aspect single-entity CONTROL question about ``company``'s own ``topic`` disclosure.

    Single-hop by construction (the answer lives in the company's own filing), so single-shot
    retrieval should already cover it — the regression guard that graph/agentic must not degrade.
    """
    question = _CONTROL_FRAME.format(company=company_name, topic=topic)
    aspects = [Aspect(name=f"{company_name}'s {topic} disclosure", spans=[topic])]
    return FilledQuestion(
        question=question, aspects=aspects, relation="single-entity", qtype="risk"
    )
