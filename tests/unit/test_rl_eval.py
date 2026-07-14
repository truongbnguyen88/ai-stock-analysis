"""A6.2g — held-out RL eval: baseline scripts, metrics, the paired CI, and the verdict gates.

Deterministic fakes only (ticker-scoped ``FakeSystem`` + canned ``MultiHopQuery`` episodes — no
embedder, no corpus, no LLM). Pins: each baseline's trajectory (``react`` ≡
``bridge_expert_rollout``, i.e. the A4/A5 controller), the bandit's restriction to the MDP's arm
menu, the sentinel's budget-spamming under the cost tax, seed-averaged metrics, the group-leakage
guard, and every branch of the pre-registered promote rule.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stock_agent.rag.policy import ARMS
from stock_agent.rag.policy_eval import Dataset, fit_bandit_baseline
from stock_agent.rag.policy_features import N_FEATURES
from stock_agent.rag.rl.action import named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv, RetrieverFactory
from stock_agent.rag.rl.reinforce import bridge_expert_rollout
from stock_agent.rag.rl.rleval import (
    AlwaysSearchPolicy,
    BanditOneShotPolicy,
    GreedyPolicy,
    OneShotArmPolicy,
    RandomActionPolicy,
    ReActBridgePolicy,
    ScriptedPolicy,
    _step_idx,
    assert_group_disjoint,
    build_baselines,
    evaluate_policy,
    evaluate_rl,
    format_report_markdown,
    policy_name,
    run_episode,
    verdict,
)
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

S = Settings(_env_file=None)
ALIAS = {
    "NVDA": ["NVIDIA"],
    "MU": ["Micron"],
    "TSM": ["Taiwan Semiconductor", "TSMC"],
    "ASML": ["ASML"],
}
UNIVERSE = {"NVDA", "MU", "TSM", "ASML"}

# pruned action-space indices (pinned in A6.2b): 0 STOP, 1 hybrid@self, 2 hybrid@disc0,
# 3 hybrid@disc1, 4 graph@self, 5 graph@disc0, 6 graph@disc1.
STOP_A, HY_SELF, HY_DISC0, GR_SELF = 0, 1, 2, 4


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


class FakeSystem:
    """RetrievalSystem returning canned chunks scoped by ``where.ticker`` (the env scoping path)."""

    def __init__(self, name: str, by_ticker: dict[str | None, list[RetrievedChunk]]) -> None:
        self.name = name
        self._by_ticker = by_ticker

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        ticker = where.ticker if where is not None else None
        return EvidenceSet(query=query, chunks=list(self._by_ticker.get(ticker, []))[:top_k])


# Two bridge chains (NVDA→MU, TSM→ASML) + two single-hop CTRL questions over the same corpus.
CORPUS: dict[str | None, list[RetrievedChunk]] = {
    "NVDA": [_rc("NVDA:0", "NVIDIA relies on Micron for HBM memory.", ticker="NVDA")],
    "MU": [
        _rc(
            "MU:0",
            "Micron cites critical information infrastructure review rules.",
            ticker="MU",
        )
    ],
    "TSM": [_rc("TSM:0", "Taiwan Semiconductor buys ASML lithography systems.", ticker="TSM")],
    "ASML": [
        _rc("ASML:0", "ASML discloses export license requirements for EUV tools.", ticker="ASML")
    ],
}
HYBRID = FakeSystem("hybrid", CORPUS)
JUNK = FakeSystem(  # the costly arm (graph, cost 0.3) that retrieves nothing useful
    "graph",
    {
        t: [_rc(f"J:{i}", "weather is sunny; markets open", ticker=t)]
        for i, t in enumerate(UNIVERSE)
    },
)

EP_NVDA_MU = MultiHopQuery(
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
EP_TSM_ASML = MultiHopQuery(
    question="Which lithography supplier TSMC depends on discloses export licence requirements?",
    aspects=[
        Aspect(name="A1 names supplier", spans=["ASML"]),
        Aspect(name="A2 supplier export", spans=["export license requirements"]),
    ],
    seed="TSM",
    stratum="HARD",
    relation="depends_on",
    group_id="TSM:ASML",
)
EP_CTRL_NVDA = MultiHopQuery(  # single-hop: the seed's own filing already answers it
    question="Which memory supplier does NVIDIA name?",
    aspects=[Aspect(name="A1", spans=["Micron"])],
    seed="NVDA",
    stratum="CTRL",
    relation="depends_on",
    group_id="CTRL:NVDA",
)
EP_CTRL_MU = MultiHopQuery(
    question="What cybersecurity review rules does Micron cite?",
    aspects=[Aspect(name="A1", spans=["critical information infrastructure"])],
    seed="MU",
    stratum="CTRL",
    relation="depends_on",
    group_id="CTRL:MU",
)
EPISODES = [EP_NVDA_MU, EP_TSM_ASML, EP_CTRL_NVDA, EP_CTRL_MU]
TRAIN_IDX, TEST_IDX = [2, 3], [0, 1]  # CTRL pair trains; the two HARD bridges are held out


def _factory(mapping: dict[str, FakeSystem]) -> RetrieverFactory:
    return lambda arm: mapping[arm]


def _env(**overrides: object) -> RagRetrievalEnv:
    kwargs: dict[str, object] = dict(
        settings=S,
        action_space=named_action_space("pruned"),
        retriever_factory=_factory({"hybrid": HYBRID, "graph": JUNK}),
        alias_map=ALIAS,
        graph_universe=UNIVERSE,
        gamma=1.0,
        lambda_cost=0.05,
    )
    kwargs.update(overrides)
    return RagRetrievalEnv(EPISODES, **kwargs)  # type: ignore[arg-type]


RNG = np.random.default_rng(0)


class SmartBridgePolicy(ScriptedPolicy):
    """Stand-in for a *good* learned policy: self-search, bridge once, STOP (no wasted budget)."""

    name = "smart"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        step = _step_idx(x)
        if step == 0:
            return HY_SELF
        if step == 1 and mask is not None and bool(mask[HY_DISC0]):
            return HY_DISC0
        return STOP_A


# ---- the baseline scripts ------------------------------------------------------------------------
def test_oneshot_policy_searches_once_then_stops() -> None:
    env = _env()
    out = run_episode(env, OneShotArmPolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert out.actions == ("hybrid@self", "STOP")
    assert out.n_retrievals == 1 and out.stopped
    assert out.coverage == pytest.approx(0.5)  # only NVDA's own filing ⇒ A1 covered, A2 not
    assert out.arm_cost == pytest.approx(0.1)
    assert out.total_return == pytest.approx(0.5 - 0.05 * 0.1)
    assert out.n_llm_calls == 0  # the $0 sim is LLM-free by construction


def test_react_policy_reproduces_the_a4_bridge_expert() -> None:
    # The strong baseline must BE the A4/A5 controller, not an approximation of it.
    env = _env()
    expert = bridge_expert_rollout(env, 0, prod_arm="hybrid")
    out = run_episode(env, ReActBridgePolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert out.actions[: expert.length] == ("hybrid@self", "hybrid@disc0", "STOP")
    assert out.total_return == pytest.approx(expert.total_return)
    assert out.coverage == pytest.approx(1.0)  # the bridge pulls MU's own filing ⇒ A2 covered


def test_react_bridges_twice_on_a_two_hop_chain() -> None:
    env = _env()
    out = run_episode(env, ReActBridgePolicy("hybrid", action_space=env.action_space), 1, rng=RNG)
    assert out.actions == ("hybrid@self", "hybrid@disc0", "STOP")
    assert out.coverage == pytest.approx(1.0)  # TSM → ASML


def test_random_policy_only_samples_legal_actions_and_is_seeded() -> None:
    env = _env()
    policy = RandomActionPolicy(env.n_actions)
    first = run_episode(env, policy, 0, rng=np.random.default_rng(7))
    again = run_episode(env, policy, 0, rng=np.random.default_rng(7))
    assert first.actions == again.actions  # seeded ⇒ reproducible
    # disc slots are illegal before anything is discovered, so no episode may open with one.
    assert first.actions[0] in {"STOP", "hybrid@self", "graph@self"}


def test_sentinel_never_stops_and_pays_the_cost_tax() -> None:
    env = _env()
    sentinel = AlwaysSearchPolicy(action_space=env.action_space, arm_costs=env.arm_costs)
    assert sentinel.arm == "graph"  # the costliest arm in the space (0.3 vs hybrid's 0.1)
    out = run_episode(env, sentinel, 0, rng=RNG)
    assert out.actions == ("graph@self",) * 3  # never STOPs; the horizon terminates it
    assert not out.stopped and out.n_retrievals == 3
    assert out.coverage == 0.0 and out.arm_cost == pytest.approx(0.9)
    assert out.total_return == pytest.approx(-0.05 * 0.9)  # junk recall, taxed ⇒ strictly negative

    react = run_episode(env, ReActBridgePolicy("hybrid", action_space=env.action_space), 0, rng=RNG)
    assert out.total_return < react.total_return


class FakeScorer:
    """A6.1-shaped arm scorer: fixed per-arm scores, recording the contexts it is asked about."""

    name = "linucb(alpha=1)"

    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)
        self.seen: list[np.ndarray] = []

    def ucb_scores(self, x: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(x, dtype=np.float64))
        return self.scores


def test_bandit_argmax_is_restricted_to_the_action_space_menu() -> None:
    env = _env()
    # ARMS order: dense, reranked, hybrid, hybrid+rerank, graph. `dense` scores highest but the
    # pruned space has no dense@self action, so the bandit must fall back to its best AVAILABLE arm.
    scorer = FakeScorer([10.0, 5.0, 1.0, 4.0, 2.0])
    policy = BanditOneShotPolicy(scorer, arm_names=ARMS, action_space=env.action_space)
    out = run_episode(env, policy, 0, rng=RNG)
    assert out.actions == ("graph@self", "STOP")  # graph (2.0) beats hybrid (1.0); dense is masked


def test_bandit_scores_the_standardized_static_block() -> None:
    env = _env()
    mean = np.full(11, 0.5)
    std = np.full(11, 2.0)
    scorer = FakeScorer([0.0, 0.0, 1.0, 0.0, 0.0])  # hybrid wins ⇒ hybrid@self
    policy = BanditOneShotPolicy(
        scorer, arm_names=ARMS, action_space=env.action_space, context_mean=mean, context_std=std
    )
    state = env.reset(0)
    assert policy.greedy_action(state) == HY_SELF
    expected = (np.asarray(state[:11], dtype=np.float64) - mean) / std
    assert np.allclose(scorer.seen[0], expected)  # the A6.1 (mu, sigma) scale, not the raw state


def test_fitted_linucb_plugs_into_the_bandit_baseline() -> None:
    # The exact CLI path: A6.1 Dataset → fit_bandit_baseline → LinUCB(d=11) → BanditOneShotPolicy,
    # which scores the 18-dim RL state's 11-dim static block. Two separate contracts are pinned.
    rng = np.random.default_rng(0)
    n = 60
    contexts = rng.normal(size=(n, N_FEATURES))
    reward_matrix = np.zeros((n, len(ARMS)))
    reward_matrix[:, ARMS.index("graph")] = 3.0 * contexts[:, 0]  # graph pays off along feature 0
    dataset = Dataset(
        contexts=contexts,
        reward_matrix=reward_matrix,
        groups=np.arange(n) // 2,  # 30 groups of 2 (bridge-pair shaped)
        strata=np.array(["HARD"] * n),
        arm_names=ARMS,
    )
    # alpha=0 ⇒ no optimism bonus ⇒ a pure ridge argmax, so the fit (not exploration) is under test.
    linucb, mean, std = fit_bandit_baseline(dataset, alpha=0.0, seed=1)
    assert linucb.d == N_FEATURES and linucb.n_arms == len(ARMS)
    assert mean.shape == (N_FEATURES,) and std.shape == (N_FEATURES,)

    # (1) the fit recovers the signal in-distribution: a context with a large feature 0 ⇒ graph.
    x = np.zeros(N_FEATURES)
    x[0] = 2.0
    assert int(np.argmax(linucb.ucb_scores(x))) == ARMS.index("graph")

    # (2) the plumbing: fed a real 18-dim env state it scores the static block and returns an action
    # from the MDP's restricted menu (never `dense`, which the pruned space cannot run), then STOPs.
    env = _env()
    policy = BanditOneShotPolicy(
        linucb,
        arm_names=ARMS,
        action_space=env.action_space,
        context_mean=mean,
        context_std=std,
    )
    out = run_episode(env, policy, 0, rng=RNG)
    assert out.actions[0] in {"hybrid@self", "graph@self"} and out.actions[1] == "STOP"
    assert out.n_retrievals == 1


def test_unknown_arm_rejected() -> None:
    env = _env()
    with pytest.raises(ValueError, match="no self-scope action"):
        OneShotArmPolicy("dense", action_space=env.action_space)


# ---- metrics -------------------------------------------------------------------------------------
def test_evaluate_policy_metrics_and_seed_averaging() -> None:
    env = _env()
    policy = OneShotArmPolicy("hybrid", action_space=env.action_space)
    metrics, returns = evaluate_policy(env, TEST_IDX, policy, seeds=(0, 1, 2))
    assert metrics.n_episodes == 2 and metrics.n_seeds == 3
    assert returns.shape == (2,)
    # deterministic policy ⇒ seed-averaging changes nothing; both HARD bridges cover 1 of 2 aspects.
    assert metrics.mean_coverage == pytest.approx(0.5)
    assert metrics.mean_return == pytest.approx(0.5 - 0.005)
    assert metrics.stop_rate == pytest.approx(1.0) and metrics.mean_llm_calls == 0.0


def test_greedy_wrapper_matches_the_underlying_argmax() -> None:
    env = _env()
    inner = SmartBridgePolicy()
    wrapped = GreedyPolicy(inner, name="rl-reinforce(greedy)")
    a = run_episode(env, inner, 0, rng=np.random.default_rng(1))
    b = run_episode(env, wrapped, 0, rng=np.random.default_rng(99))
    assert a.actions == b.actions and a.total_return == pytest.approx(b.total_return)


def test_build_baselines_covers_the_four_reference_controllers() -> None:
    env = _env()
    baselines = build_baselines(
        env, bandit=FakeScorer([1, 1, 1, 1, 1]), bandit_arm_names=ARMS
    )
    names = [policy_name(p) for p in baselines]
    assert names == [
        "fixed(graph)",
        "fixed(hybrid)",
        "react(hybrid)",
        "react(graph)",
        "bandit(linucb(alpha=1))",
        "random",
    ]


# ---- folds + leakage -----------------------------------------------------------------------------
def test_group_leakage_guard_rejects_a_straddling_pair() -> None:
    env = _env()
    assert_group_disjoint(env, TRAIN_IDX, TEST_IDX)  # the canonical split is clean
    with pytest.raises(ValueError, match="group leakage"):
        assert_group_disjoint(env, [0, 2], [0, 1])  # episode 0's bridge pair is on both sides


# ---- the verdict ---------------------------------------------------------------------------------
def test_evaluate_rl_promotes_a_strictly_better_learned_policy() -> None:
    env = _env()
    report = evaluate_rl(
        env,
        train_indices=TRAIN_IDX,
        test_indices=TEST_IDX,
        learned=SmartBridgePolicy(),
        baselines=[
            OneShotArmPolicy("hybrid", action_space=env.action_space),
            RandomActionPolicy(env.n_actions),
        ],
        sentinel=AlwaysSearchPolicy(action_space=env.action_space, arm_costs=env.arm_costs),
        learned_name="rl-reinforce",
        seeds=(0, 1),
        n_boot=200,
    )
    assert report.best_baseline == "fixed(hybrid)"
    assert report.learned == "rl-reinforce(greedy)"
    # bridging lifts coverage 0.5 → 1.0 on both held-out episodes, for one extra hybrid hop (0.005).
    assert report.delta_return == pytest.approx(0.5 - 0.05 * 0.1)
    assert report.delta_ci[0] > 0 and report.delta_p_positive == 1.0
    assert report.promote and report.rationale.startswith("PROMOTE")
    assert not report.sentinel_beats_learned
    # both folds are scored, so the overfit gap is a real number (CTRL train fold is easier here).
    assert len(report.train_metrics) == 2 and report.overfit_gap == pytest.approx(
        report.train_metrics[0].mean_return - report.test_metrics[0].mean_return
    )


def test_report_and_markdown_expose_the_reward_identity_and_the_ctrl_row() -> None:
    # A CTRL-only held-out fold: bridging cannot lift coverage on a single-hop control question, so
    # the extra hop is pure cost — the learned policy must LOSE, and the CTRL row must show it.
    env = _env()
    report = evaluate_rl(
        env,
        train_indices=[0, 1],
        test_indices=[2, 3],
        learned=ReActBridgePolicy("hybrid", action_space=env.action_space),
        baselines=[OneShotArmPolicy("hybrid", action_space=env.action_space)],
        seeds=(0,),
        n_boot=100,
    )
    # γ=1 ⇒ the shaping telescopes: return == terminal coverage − λ_c·Σcost, for EVERY candidate.
    for m in report.test_metrics:
        assert m.mean_return == pytest.approx(m.mean_coverage - 0.05 * m.mean_arm_cost)
    ctrl = [s for s in report.per_stratum if s.stratum == "CTRL"]
    assert len(ctrl) == 1 and ctrl[0].n == 2 and ctrl[0].delta < 0
    assert not report.promote
    md = format_report_markdown(report)
    assert "| stratum | n | learned | baseline | Δ |" in md and "| CTRL | 2 |" in md


def test_generalization_gap_catches_an_edge_that_does_not_transfer() -> None:
    # The real A6.2g run's failure mode: the policy BEATS the baseline on the fold it trained on
    # (bridging pays on the HARD episodes) and LOSES on held-out (bridging is pure cost on CTRL).
    # The absolute train−test gaps miss this — the folds differ in difficulty — so the paired delta
    # must be reported on BOTH folds. Train = the two HARD bridges; test = the two CTRL controls.
    env = _env()
    report = evaluate_rl(
        env,
        train_indices=[0, 1],  # HARD: bridging lifts coverage 0.5 → 1.0
        test_indices=[2, 3],  # CTRL: single-hop, so the extra bridge hop is pure cost
        learned=SmartBridgePolicy(),
        baselines=[OneShotArmPolicy("hybrid", action_space=env.action_space)],
        seeds=(0,),
        n_boot=100,
    )
    assert report.delta_train > 0  # the policy looked good on train …
    assert report.delta_return < 0  # … and lost on held-out
    assert report.generalization_gap == pytest.approx(report.delta_train - report.delta_return)
    assert report.generalization_gap > 0 and not report.promote
    assert "generalization gap" in format_report_markdown(report)


def test_evaluate_rl_rejects_a_learned_policy_that_ties_the_baseline() -> None:
    env = _env()
    report = evaluate_rl(
        env,
        train_indices=TRAIN_IDX,
        test_indices=TEST_IDX,
        learned=OneShotArmPolicy("hybrid", action_space=env.action_space),
        baselines=[
            OneShotArmPolicy("hybrid", action_space=env.action_space),
            RandomActionPolicy(env.n_actions),
        ],
        seeds=(0,),
        n_boot=100,
    )
    assert report.delta_return == pytest.approx(0.0)
    assert not report.promote and "≤" in report.rationale


def test_sentinel_blocks_promotion_when_the_cost_tax_is_off() -> None:
    # lambda_cost = 0 ⇒ retrieval is free ⇒ spamming the budget is never punished. The learned
    # policy still beats every baseline, but the sentinel matches it, so the verdict is blocked.
    env = _env(lambda_cost=0.0, retriever_factory=_factory({"hybrid": HYBRID, "graph": HYBRID}))
    report = evaluate_rl(
        env,
        train_indices=TRAIN_IDX,
        test_indices=TEST_IDX,
        learned=SmartBridgePolicy(),
        baselines=[
            OneShotArmPolicy("hybrid", action_space=env.action_space),
            RandomActionPolicy(env.n_actions),
        ],
        sentinel=AlwaysSearchPolicy(action_space=env.action_space, arm_costs=env.arm_costs),
        seeds=(0,),
        n_boot=100,
    )
    assert report.delta_return > 0 and report.delta_ci[0] > 0  # every other gate passes …
    assert report.sentinel_beats_learned and not report.promote  # … but the tripwire fires
    assert report.rationale.startswith("REJECT: the reward-hacking sentinel")


@pytest.mark.parametrize(
    ("delta", "ci", "ctrl", "hacked", "expect", "reason"),
    [
        (0.05, (0.01, 0.09), False, False, True, "PROMOTE"),
        (-0.02, (-0.05, 0.01), False, False, False, "≤"),  # negative point estimate
        (0.05, (-0.01, 0.11), False, False, False, "CI"),  # CI straddles 0
        (0.05, (0.01, 0.09), True, False, False, "CTRL"),  # control stratum pays for the gain
        (0.05, (0.01, 0.09), False, True, False, "sentinel"),  # reward-hacking tripwire
    ],
)
def test_verdict_gate_matrix(
    delta: float,
    ci: tuple[float, float],
    ctrl: bool,
    hacked: bool,
    expect: bool,
    reason: str,
) -> None:
    promote, rationale = verdict(
        delta, ci, ctrl_regression=ctrl, sentinel_beats_learned=hacked
    )
    assert promote is expect
    assert reason in rationale
