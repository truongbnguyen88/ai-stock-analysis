"""A6.2g — sim-to-real gap harness: the query-writer hook, call accounting, and the measured gap.

No network, no API key: a canned ``FakeLLM`` plays both the query writer and the terminal synthesis.
Pins the invariant that makes the experiment interpretable — the LLM rewrites the query **text** and
nothing else (arm, scope and the stop decision stay the policy's) — plus the billing bound, the
template fallback on a malformed completion, and the gap arithmetic.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from stock_agent.rag.rl.action import named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv, RetrieverFactory
from stock_agent.rag.rl.rleval import (
    OneShotArmPolicy,
    ReActBridgePolicy,
    SweepBridgePolicy,
    run_episode,
)
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.research.prompts import RL_QUERY_SYSTEM
from stock_agent.research.rl_simreal import (
    DryRunLLM,
    LLMQueryWriter,
    RealRow,
    estimate_llm_calls,
    format_simreal_markdown,
    head_to_head,
    measure_llm_calls,
    run_real_policy,
    run_sim_to_real,
)
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

S = Settings(_env_file=None)
ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"]}
UNIVERSE = {"NVDA", "MU"}


def _rc(cid: str, text: str, *, ticker: str) -> RetrievedChunk:
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
        section="Item 1A. Risk Factors",
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


NVDA_CHUNK = _rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", ticker="NVDA")
MU_CHUNK = _rc("MU:0", "Micron cites critical information infrastructure rules.", ticker="MU")
# The "better query" the LLM writes surfaces an extra chunk the template never reaches — this is the
# whole point of the experiment, so the fake corpus must be able to express it.
MU_DEEP = _rc("MU:1", "Micron discloses export license requirements for China.", ticker="MU")


class FakeSystem:
    """Ticker-scoped retriever whose MU results depend on the query text (template vs LLM)."""

    name = "hybrid"

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        ticker = where.ticker if where is not None else None
        chunks: list[RetrievedChunk] = []
        if ticker == "NVDA":
            chunks = [NVDA_CHUNK]
        elif ticker == "MU":
            chunks = [MU_CHUNK]
            if "export license" in query.lower():  # only a well-phrased query finds the deep chunk
                chunks = [MU_CHUNK, MU_DEEP]
        return EvidenceSet(query=query, chunks=chunks[:top_k])


def _factory() -> RetrieverFactory:
    return lambda arm: FakeSystem()


EP = MultiHopQuery(
    question="Which memory supplier NVIDIA depends on discloses Chinese export rules?",
    aspects=[
        Aspect(name="A1 names supplier", spans=["Micron"]),
        Aspect(name="A2 export rules", spans=["export license requirements"]),
    ],
    seed="NVDA",
    stratum="HARD",
    relation="depends_on",
    group_id="NVDA:MU",
)


class FakeLLM:
    """Canned TextLLM: query calls get a query, the synthesis call gets a grounded answer."""

    def __init__(self, query: str = "Micron export license requirements China") -> None:
        self.query = query
        self.systems: list[str] = []
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        self.systems.append(system)
        if "search query" in system:  # RL_QUERY_SYSTEM
            return json.dumps({"query": self.query})
        return json.dumps(  # the terminal synthesis (guarded downstream)
            {"answer": "Micron discloses export license requirements [2].", "citations": [2]}
        )


def _env(writer: object | None = None) -> RagRetrievalEnv:
    return RagRetrievalEnv(
        [EP],
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=_factory(),
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.05,
        query_writer=writer,  # type: ignore[arg-type]
    )


RNG = np.random.default_rng(0)


# ---- the env hook: the LLM rewrites the TEXT and nothing else ------------------------------------
def test_query_writer_replaces_only_the_query_text() -> None:
    seen: list[tuple[str, str | None, str]] = []

    def writer(req, episode, evidence):  # type: ignore[no-untyped-def]
        seen.append((req.arm, req.scope_ticker, req.query))
        return "REWRITTEN " + req.query

    env = _env(writer)
    policy = ReActBridgePolicy("hybrid", action_space=env.action_space)
    out = run_episode(env, policy, 0, rng=RNG)
    # the policy's decisions are untouched: same arms, same scopes, same stop
    assert out.actions == ("hybrid@self", "hybrid@disc0", "STOP")
    assert [(arm, scope) for arm, scope, _ in seen] == [("hybrid", "NVDA"), ("hybrid", "MU")]
    # …and the templates it was handed are the current ones (which it then rewrote). A bridge
    # question asks two things and each hop takes the half it needs: hop 1 the RELATION (E2),
    # hop 2 the entity's name + the TOPIC (E6) — never the whole question.
    assert seen[0][2] == "suppliers supply depend key dependencies"
    assert seen[1][2] == "Micron Chinese export rules"


def test_blank_rewrite_falls_back_to_the_template() -> None:
    env = _env(lambda req, ep, ev: "   ")  # a malformed/empty completion degrades to the sim
    out = run_episode(env, OneShotArmPolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert out.n_retrievals == 1 and out.coverage == pytest.approx(0.5)  # template still retrieved


def test_default_env_is_still_templated_and_free() -> None:
    env = _env()  # no writer ⇒ the training/eval contract is unchanged
    out = run_episode(env, OneShotArmPolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert out.n_llm_calls == 0 and out.coverage == pytest.approx(0.5)


# ---- the writer ---------------------------------------------------------------------------------
def test_writer_prompts_with_scope_and_anti_loop_list_and_counts_calls() -> None:
    llm = FakeLLM()
    writer = LLMQueryWriter(llm, alias_map=ALIAS)
    env = _env(writer)
    run_episode(env, ReActBridgePolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert writer.calls == 2  # one per search hop; STOP costs nothing
    assert writer.issued == [llm.query, llm.query]
    writer.reset()
    assert writer.issued == []  # per-episode anti-loop list clears; billing does not
    assert writer.calls == 2


def test_writer_returns_empty_on_malformed_json() -> None:
    class Broken:
        def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
            return "not json at all"

    writer = LLMQueryWriter(Broken(), alias_map=ALIAS)
    from stock_agent.rag.rl.action import RetrievalRequest

    req = RetrievalRequest(arm="hybrid", query="template", scope_ticker="MU")
    assert writer(req, EP, []) == ""  # → env falls back to the template


# ---- the measured gap ---------------------------------------------------------------------------
def test_sim_to_real_measures_a_positive_coverage_gap() -> None:
    # The LLM's query surfaces MU_DEEP (the template cannot), so the real run covers A2 and the sim
    # does not: the $0 verdict UNDERSTATED this policy.
    llm = FakeLLM()
    writer = LLMQueryWriter(llm, alias_map=ALIAS)
    sim_env, real_env = _env(), _env(writer)
    policy = ReActBridgePolicy("hybrid", action_space=sim_env.action_space)

    report = run_sim_to_real(
        policy, [EP], sim_env=sim_env, real_env=real_env, writer=writer, llm=llm
    )
    assert report.n_episodes == 1
    row = report.episodes[0]
    assert row.coverage_sim == pytest.approx(0.5)  # template: A1 only
    assert row.coverage_real == pytest.approx(1.0)  # LLM query: A1 + A2
    assert report.coverage_gap == pytest.approx(0.5)
    assert row.actions_sim == row.actions_real and report.same_action_rate == 1.0
    assert row.queries == (llm.query, llm.query)  # the audit trail of what was actually issued
    assert row.n_llm_calls == 3 and report.total_llm_calls == 3  # 2 query hops + 1 synthesis
    assert not row.insufficient and report.refusal_rate == 0.0
    assert "UNDERSTATED" in format_simreal_markdown(report)


def test_estimate_llm_calls_is_the_naive_step_floor() -> None:
    # the cheap a-priori floor (kept for reference): max_steps writer calls + 1 synthesis / episode
    assert estimate_llm_calls(24, 3) == 96


def test_measure_llm_calls_counts_fanout_branches_not_steps() -> None:
    # The fix: a sweep issues one retrieval REQUEST per discovered candidate and the writer bills
    # each, so a fan-out over 3 candidates is 3 calls in ONE step — the branch fan-out the naive
    # step bound (1*(3+1)=4) never sees. Counted on a $0 dry rollout, so it reflects the real spend.
    seed_chunk = _rc(
        "NVDA:9", "NVIDIA relies on Micron, Samsung, and Broadcom for components.", ticker="NVDA"
    )

    class ThreeSupplierSystem:
        name = "hybrid"

        def retrieve(
            self, query: str, *, top_k: int, where: ChunkFilter | None = None
        ) -> EvidenceSet:
            t = where.ticker if where is not None else None
            hit = (
                seed_chunk
                if t == "NVDA"
                else _rc(f"{t}:0", f"{t} discloses a risk.", ticker=t or "")
            )
            return EvidenceSet(query=query, chunks=[hit][:top_k])

    alias = {"NVDA": ["NVIDIA"], "MU": ["Micron"], "SMSN": ["Samsung"], "AVGO": ["Broadcom"]}
    ep = MultiHopQuery(
        question="Which supplier NVIDIA depends on discloses export rules?",
        aspects=[Aspect(name="A1", spans=["Micron"])],
        seed="NVDA",
        stratum="HARD",
        relation="depends_on",
        group_id="NVDA:MU",
    )
    writer = LLMQueryWriter(DryRunLLM(), alias_map=alias)  # $0: returns "" ⇒ env keeps the template
    env = RagRetrievalEnv(
        [ep],
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=lambda arm: ThreeSupplierSystem(),
        alias_map=alias,
        graph_universe={"NVDA", "MU", "SMSN", "AVGO"},
        gamma=1.0,
        lambda_cost=0.05,
        seating_rule="breadth_first",  # no cross-encoder ⇒ never downloads a model in CI
        query_writer=writer,
    )
    sweep = SweepBridgePolicy("hybrid", action_space=env.action_space)  # self → fanout → STOP

    n = measure_llm_calls(sweep, [ep], env=env, writer=writer)
    assert n == 5  # 1 self + 3 fan-out branches + 1 synthesis
    assert n > estimate_llm_calls(1, S.agentic_max_steps)  # 4: the naive step bound undercounts
    # A baseline-only run (rl-h2h) pays no synthesis ⇒ one fewer.
    writer.calls = 0
    assert measure_llm_calls(sweep, [ep], env=env, writer=writer, include_synthesis=False) == 4


# ---- the real-env head-to-head -------------------------------------------------------------------
def _real_rows(returns: list[float], *, groups: list[str], calls: int = 0) -> list[RealRow]:
    return [
        RealRow(
            question=f"q{i}",
            stratum="HARD",
            group_id=g,
            coverage=r,
            total_return=r,
            actions=("hybrid@self", "STOP"),
            queries=(),
            n_llm_calls=calls,
        )
        for i, (r, g) in enumerate(zip(returns, groups, strict=True))
    ]


def test_head_to_head_pairs_episodes_and_bills_only_the_baseline() -> None:
    # A beats B on every episode ⇒ Δ > 0 and the CI must clear 0 ⇒ the gate PASSES.
    a = _real_rows([0.6, 0.8, 0.5, 0.9], groups=["g1", "g1", "g2", "g2"])
    b = _real_rows([0.4, 0.5, 0.3, 0.6], groups=["g1", "g1", "g2", "g2"], calls=2)
    rep = head_to_head(a, b, name_a="rl", name_b="react:hybrid")
    assert rep.delta == pytest.approx(0.25)  # mean of (.2, .3, .2, .3)
    assert rep.ci_low > 0 and rep.wins
    assert rep.total_llm_calls == 8  # only the baseline's hops bill; the rl side was already paid
    n, cov = rep.strata["HARD"]
    assert n == 4 and cov == pytest.approx(0.25)


def test_head_to_head_refuses_a_misaligned_pairing() -> None:
    # The whole comparison is a paired difference; silently zipping different episodes would look
    # like a result. Guard it.
    a = _real_rows([0.6, 0.8], groups=["g1", "g1"])
    b = _real_rows([0.4, 0.5], groups=["g1", "g1"])
    shuffled = [b[1], b[0]]
    object.__setattr__(shuffled[0], "question", "other")
    with pytest.raises(ValueError, match="misalignment"):
        head_to_head(a, shuffled, name_a="rl", name_b="react:hybrid")
    with pytest.raises(ValueError, match="unpaired"):
        head_to_head(a, b[:1], name_a="rl", name_b="react:hybrid")


def test_head_to_head_ci_resamples_groups_not_rows() -> None:
    # Both questions of a bridge pair share a corpus; a row bootstrap would treat them as two
    # independent draws and understate the CI. One group here ⇒ the resample can only ever pick that
    # group, so the CI collapses onto the point estimate — the signature of a GROUP bootstrap.
    a = _real_rows([0.6, 0.6], groups=["g1", "g1"])
    b = _real_rows([0.4, 0.4], groups=["g1", "g1"])
    rep = head_to_head(a, b, name_a="rl", name_b="react:hybrid")
    assert rep.ci_low == pytest.approx(0.2) and rep.ci_high == pytest.approx(0.2)


def test_run_real_policy_scores_the_baseline_without_paying_for_synthesis() -> None:
    llm = FakeLLM()
    writer = LLMQueryWriter(llm, alias_map=ALIAS)
    env = _env(writer)
    rows = run_real_policy(
        ReActBridgePolicy("hybrid", action_space=env.action_space),
        [EP],
        real_env=env,
        writer=writer,
    )
    assert len(rows) == 1
    assert rows[0].n_llm_calls == 2  # 2 query hops, NO synthesis call (the return needs no LLM)
    assert rows[0].coverage == pytest.approx(1.0)  # the LLM query reaches MU_DEEP
    assert rows[0].group_id == "NVDA:MU"  # carried through for the group bootstrap
    assert {s for s in llm.systems} == {RL_QUERY_SYSTEM}  # synthesis never invoked
