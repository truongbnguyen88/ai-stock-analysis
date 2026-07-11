"""A6.2c — retrieval MDP env + TransitionCache: worked trajectory, shaping, cache, sentinel.

Deterministic fakes only (a ticker-scoped ``FakeSystem`` + canned ``MultiHopQuery`` episodes — no
embedder, no corpus, no LLM). Pins the reward math (potential shaping − cost), the NVDA→MU worked
transition (Q4), the Ng–Harada telescoping identity, cache hit/miss accounting, the reward-hacking
sentinel, discovered-slot masking, and the env-level leakage guard (state is label-independent).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stock_agent.rag.policy_features import N_FEATURES
from stock_agent.rag.rl.action import Action, ScopeKind, named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv, RetrieverFactory, TransitionCache
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

S = Settings(_env_file=None)
ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"], "TSM": ["Taiwan Semiconductor", "TSMC"]}
UNIVERSE = {"NVDA", "MU", "TSM"}

# pruned action-space indices (pinned in A6.2b): 0 STOP, 1 hybrid@self, 2 hybrid@disc0,
# 3 hybrid@disc1, 4 graph@self, 5 graph@disc0, 6 graph@disc1.
STOP_A, HY_SELF, HY_DISC0, GR_SELF = 0, 1, 2, 4


def _rc(
    cid: str, text: str, *, ticker: str, section: str = "Item 1A. Risk Factors"
) -> RetrievedChunk:
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
        section=section,
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


class FakeSystem:
    """RetrievalSystem returning canned chunks scoped by ``where.ticker`` (the env scoping path)."""

    def __init__(self, name: str, by_ticker: dict[str | None, list[RetrievedChunk]]) -> None:
        self.name = name
        self._by_ticker = by_ticker

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        ticker = where.ticker if where is not None else None
        return EvidenceSet(query=query, chunks=list(self._by_ticker.get(ticker, []))[:top_k])


def _factory(mapping: dict[str, FakeSystem]) -> RetrieverFactory:
    return lambda arm: mapping[arm]


# The NVDA→MU bridge episode: A1 (NVDA names the supplier) + A2 (the supplier's own CAC disclosure).
NVDA_CHUNK = _rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", ticker="NVDA")
MU_CHUNK = _rc(
    "MU:0", "Micron cites critical information infrastructure review rules.", ticker="MU"
)
EP = MultiHopQuery(
    question="Which memory supplier NVIDIA depends on discloses Chinese cybersecurity rules?",
    aspects=[
        Aspect(name="A1 names supplier", spans=["Micron"]),
        Aspect(name="A2 supplier CAC", spans=["critical information infrastructure"]),
    ],
    seed="NVDA",
    stratum="HARD",
    relation="depends_on",
    group_id="NVDA:MU",
)
HYBRID = FakeSystem("hybrid", {"NVDA": [NVDA_CHUNK], "MU": [MU_CHUNK]})


def _bridge_env(**overrides: object) -> RagRetrievalEnv:
    kwargs: dict[str, object] = dict(
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=_factory({"hybrid": HYBRID}),
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.05,
    )
    kwargs.update(overrides)
    return RagRetrievalEnv([EP], **kwargs)  # type: ignore[arg-type]


# ---- the Q4 worked NVDA→MU trajectory -----------------------------------------------------------
def test_worked_bridge_trajectory() -> None:
    env = _bridge_env()
    assert env.n_discovered_slots == 2  # pruned space addresses disc0 + disc1

    s0 = env.reset(0)
    assert list(s0[N_FEATURES:]) == [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # empty union

    s1, r1, done1, i1 = env.step(HY_SELF)  # self-ticker NVDA: A1 covered, MU discovered
    assert not done1
    assert list(s1[N_FEATURES:]) == [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert i1.coverage == 0.5 and i1.n_new_chunks == 1 and i1.n_discovered == 1
    assert i1.arm == "hybrid" and i1.scope_ticker == "NVDA"
    assert r1 == pytest.approx(0.5 - 0.05 * 0.1)  # shaping 0.5 − λ_c·cost(hybrid)=0.1

    s2, r2, done2, i2 = env.step(HY_DISC0)  # bridge to MU: A2 covered, no entity left
    assert i2.scope_ticker == "MU" and i2.coverage == 1.0
    assert list(s2[N_FEATURES:]) == [2.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0]
    assert r2 == pytest.approx((1.0 - 0.5) - 0.05 * 0.1)

    s3, r3, done3, i3 = env.step(STOP_A)
    assert done3 and r3 == 0.0 and i3.stopped
    # static block is constant across the whole episode (question fixed).
    assert list(s3[:N_FEATURES]) == list(s0[:N_FEATURES])


def test_rollout_is_deterministic() -> None:
    env = _bridge_env()
    env.reset(0)
    a = [env.step(HY_SELF), env.step(HY_DISC0)]
    env.reset(0)
    b = [env.step(HY_SELF), env.step(HY_DISC0)]
    for (sa, ra, _, _), (sb, rb, _, _) in zip(a, b, strict=True):
        assert np.array_equal(sa, sb) and ra == rb


# ---- potential-based shaping telescopes to γ^n·Φ(s_n) (Ng–Harada) --------------------------------
def test_shaping_telescopes_to_discounted_terminal_coverage() -> None:
    env = _bridge_env(gamma=0.9, lambda_cost=0.0)  # isolate the shaping term (no cost)
    env.reset(0)
    _, r0, _, _ = env.step(HY_SELF)  # 0.9·0.5 − 0
    _, r1, _, i1 = env.step(HY_DISC0)  # 0.9·1.0 − 0.5
    discounted_return = r0 + 0.9 * r1  # Σ_t γ^t r_t
    # Ng–Harada: Σ γ^t (γΦ(s_{t+1})−Φ(s_t)) = γ^n Φ(s_n) − Φ(s_0), Φ(s_0)=0, n=2.
    assert discounted_return == pytest.approx((0.9**2) * i1.coverage)


# ---- transition cache: hit/miss accounting + anti-loop no-op -------------------------------------
def test_cache_counts_and_horizon_and_antiloop() -> None:
    env = _bridge_env()
    env.reset(0)
    env.step(HY_SELF)  # (hybrid, NVDA, question) — miss
    _, _, _, i_dup = env.step(HY_SELF)  # exact repeat — cache HIT, dedup adds nothing
    assert i_dup.n_new_chunks == 0
    _, _, done, _ = env.step(HY_SELF)  # 3rd non-STOP step hits the horizon (max_steps=3)
    assert done
    assert env.cache.misses == 1 and env.cache.hits == 2  # one distinct request, two repeats

    # a distinct (arm, scope, query) is a fresh miss; a later identical request across resets hits.
    env.reset(0)
    env.step(HY_SELF)  # HIT again (key persists across episodes/resets)
    env.step(HY_DISC0)  # (hybrid, MU, "Micron …") — new MISS
    assert env.cache.misses == 2


def test_cache_get_or_compute_memoizes() -> None:
    cache = TransitionCache()
    calls = {"n": 0}

    def compute() -> list[RetrievedChunk]:
        calls["n"] += 1
        return [NVDA_CHUNK]

    first = cache.get_or_compute("hybrid", "NVDA", "q", compute)
    second = cache.get_or_compute("hybrid", "NVDA", "q", compute)
    assert first == second and calls["n"] == 1
    assert cache.hits == 1 and cache.misses == 1


# ---- reward-hacking sentinel: high-recall junk earns 0 coverage, taxed by cost -------------------
def test_reward_hacking_sentinel_scores_negative() -> None:
    junk = [_rc(f"J:{i}", "weather is sunny; markets open", ticker="NVDA") for i in range(3)]
    noisy = FakeSystem("graph-noisy", {"NVDA": junk})
    env = _bridge_env(retriever_factory=_factory({"hybrid": HYBRID, "graph": noisy}))
    env.reset(0)
    _, r, _, info = env.step(GR_SELF)  # graph@self: dumps 3 chunks, none covering an aspect
    assert info.n_new_chunks == 3 and info.coverage == 0.0
    assert r < 0 and r == pytest.approx(0.0 - 0.05 * 0.3)  # only the graph cost (0.3) shows up


# ---- discovered-slot masking (legal_mask ↔ no-op step) -------------------------------------------
def test_discovered_slot_masking() -> None:
    env = _bridge_env()
    env.reset(0)
    m0 = env.legal_mask()
    assert m0[STOP_A] and m0[HY_SELF] and m0[GR_SELF]  # STOP + self scopes always legal
    assert not m0[HY_DISC0]  # nothing discovered yet ⇒ disc slots illegal

    _, r, _, info = env.step(HY_DISC0)  # illegal slot ⇒ no-op step (budget spent, nothing gained)
    assert info.masked and info.n_new_chunks == 0 and r == pytest.approx(0.0)

    env.reset(0)
    env.step(HY_SELF)  # discovers MU
    assert env.legal_mask()[HY_DISC0]  # disc0 now indexes an entity ⇒ legal


# ---- env-level leakage guard: the state is label-independent -------------------------------------
def test_state_is_label_independent() -> None:
    # Same question/seed/corpus, different gold aspects ⇒ identical states but different rewards.
    ep_other = EP.model_copy(update={"aspects": [Aspect(name="x", spans=["absent-span-zzz"])]})
    env_a = _bridge_env()
    env_b = RagRetrievalEnv(
        [ep_other],
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=_factory({"hybrid": HYBRID}),
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.05,
    )
    env_a.reset(0)
    env_b.reset(0)
    sa, ra, _, _ = env_a.step(HY_SELF)
    sb, rb, _, _ = env_b.step(HY_SELF)
    assert np.array_equal(sa, sb)  # state never reads the labels
    assert ra != rb  # reward does (coverage 0.5 vs 0.0)


# ---- construction / lifecycle guards ------------------------------------------------------------
def test_env_guards() -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        RagRetrievalEnv([], settings=S, action_space=named_action_space("pruned"))
    with pytest.raises(ValueError, match="must be STOP"):
        RagRetrievalEnv(
            [EP], settings=S, action_space=[Action(config="hybrid", scope_kind=ScopeKind.SELF)]
        )
    env = _bridge_env()
    with pytest.raises(IndexError):
        env.reset(99)
    env.reset(0)
    env.step(STOP_A)  # terminates the episode
    with pytest.raises(RuntimeError, match="reset"):
        env.step(HY_SELF)  # stepping a done env must fail loudly
