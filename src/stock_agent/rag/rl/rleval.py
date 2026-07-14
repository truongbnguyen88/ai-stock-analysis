"""Held-out eval of a frozen RL policy vs the baseline suite (adv-RAG A6.2g) — `rag rl-eval`.

The verdict harness. A checkpointed policy (A6.2f) is replayed **greedily** (deploy semantics) on
the group-wise held-out fold of the *same* deterministic ``$0`` MDP it trained in, against
reference controllers expressed as policies over that same MDP — so every candidate sees the same
action menu, horizon, and reward, and the only thing that differs is the decision rule:

| baseline | what it stands for | script in the MDP |
|---|---|---|
| ``fixed(arm)`` | the A5.3 production pipeline (one-shot) | ``arm@self`` → STOP |
| ``bandit(linucb)`` | the A6.1 one-shot router (REJECTed) | contextual ``arm@self`` → STOP |
| ``react(arm)`` | A4 loop + A5 bridge (pre-E3 strong one) | ``arm@self`` → ``arm@disc0`` → STOP |
| ``sweep(arm)`` | **the E3 strong one** — scripted fanout | ``arm@self`` → ``arm@fanout`` → STOP |
| ``random`` | the do-nothing-smart floor | uniform over the legal actions |

``sweep`` exists because E3's fanout action is *trivially scriptable*: once "sweep every candidate"
is expressible, a two-line script plays it, with no learning. Giving the learner fanout while the
baseline stays on ``disc0`` would manufacture a win out of an action-space asymmetry — the same
class of error that invalidated the original A6.2 verdict. RL must beat ``sweep``, not ``react``.

plus a **reward-hacking sentinel** (``sentinel(always-search)``): never STOP, spend the whole budget
on the costliest arm. It is not a baseline to beat — it is a *tripwire*. If mindless budget-spamming
scores at or above the learned policy, the cost tax ``λ_c`` is not doing its job and the verdict is
blocked regardless of the delta.

**Why the baselines are scripts over the state, not separate code paths.** Each baseline reads only
the raw state (``step_idx``, the legal mask) and returns an action index, so it satisfies the same
``RolloutPolicy`` surface as the learned policy and runs through the same ``env.step``. That is what
makes the comparison honest: identical union dedup, identical coverage oracle, identical cost
accounting. ``react`` in particular is asserted (in tests) to emit the *same trajectory* as
``bridge_expert_rollout`` — the A4/A5 controller — so "the strong baseline" is not a re-modeled
approximation of production, it *is* the scripted production controller.

**Statistics.** The headline is the paired per-episode return delta ``Δ_i = R_learned(i) −
R_bestbaseline(i)``; its CI comes from the **group** bootstrap (``bootstrap_delta_stats``, reused
from A6.1) — resampling bridge *pairs*, not rows, because the benchmark's variants within a pair are
near-copies (a row bootstrap would understate the variance). The pre-registered promote rule (below)
also requires no CTRL regression and a clean sentinel, mirroring the A6.1 verdict discipline.

**What the ``$0`` sim cannot tell you.** Queries here are templated (no LLM), so ``n_llm_calls`` is
0 for *every* candidate by construction and the faithfulness floor never fires. The A4 loop's real
advantage — an LLM writing better query strings each hop — is invisible in the sim. That gap is
exactly what the paid sim-to-real run (A6.2g, separate) measures; the numbers here are the ``$0``
sim's, and the report says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from stock_agent.rag.ope import bootstrap_delta_stats
from stock_agent.rag.policy_features import N_FEATURES
from stock_agent.rag.rl.action import Action, action_label
from stock_agent.rag.rl.env import RagRetrievalEnv
from stock_agent.rag.rl.reinforce import STOP_INDEX, RolloutPolicy
from stock_agent.rag.rl.state import STATE_FEATURE_NAMES

# The policies below read the step counter straight out of the raw state; resolve its slot by NAME
# (never a magic index) so an append to STATE_FEATURE_NAMES can't silently repoint it.
STEP_IDX_FEATURE: int = STATE_FEATURE_NAMES.index("step_idx")

_STRATA = ("HARD", "MED", "CTRL")


def _step_idx(x: np.ndarray) -> int:
    """The env's step counter read back out of the raw (unstandardized) state vector."""
    return int(round(float(x[STEP_IDX_FEATURE])))


def _labels(action_space: Sequence[Action]) -> list[str]:
    """Action labels in space order (``STOP``, ``hybrid@self``, ``hybrid@disc0``, …)."""
    return [action_label(a) for a in action_space]


# ---- baseline policies (scripts over the raw state; all satisfy ``RolloutPolicy``) ---------------
class ScriptedPolicy:
    """A deterministic MDP policy: ``act`` is ``greedy_action`` (propensity 1 ⇒ log-prob 0).

    Subclasses implement ``greedy_action(state, mask)``, reading the **raw** (unstandardized) state
    — the harness always runs on the raw env, and a loaded checkpoint standardizes internally
    (``StandardizedPolicy``). Holding no RNG makes every scripted baseline seed-independent, so
    seed-averaging them is exact rather than noisy.
    """

    name: str

    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        return self.greedy_action(x, mask=mask), 0.0

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        raise NotImplementedError


class OneShotArmPolicy(ScriptedPolicy):
    """Baseline (i) — the fixed production pipeline: one self-scoped search with ``arm``, then STOP.

    This is the A5.3 non-agentic read path expressed in the MDP (and the thing `fixed(graph)` /
    `fixed(hybrid)` mean in the report). Raises if ``arm`` has no self-scope action in the space.
    """

    def __init__(
        self, arm: str, *, action_space: Sequence[Action], name: str | None = None
    ) -> None:
        labels = _labels(action_space)
        if f"{arm}@self" not in labels:
            raise ValueError(f"arm {arm!r} has no self-scope action in this action space")
        self.arm = arm
        self.self_idx = labels.index(f"{arm}@self")
        self.name = name or f"fixed({arm})"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        return self.self_idx if _step_idx(x) == 0 else STOP_INDEX


class ArmScorer(Protocol):
    """The one thing the bandit baseline needs from an A6.1 policy: per-arm scores for a context.

    Satisfied by ``rag.policy.LinUCB`` (``ucb_scores`` = ridge mean + optimism bonus). Deliberately
    narrower than the A6.1 ``Policy`` protocol: ``prob`` is a one-hot on the *unrestricted* argmax,
    which cannot be re-argmaxed over a restricted arm menu (below) without collapsing to ties.
    """

    name: str

    def ucb_scores(self, x: np.ndarray) -> np.ndarray: ...


class BanditOneShotPolicy(ScriptedPolicy):
    """Baseline (ii) — the A6.1 contextual bandit: pick the arm from ``s_0``, search self, STOP.

    The context is the state's **static block** (``x[:N_FEATURES]``), which is byte-identical to the
    A6.1 context (the env builds it with the same ``policy_features.featurize`` call) — then
    standardized with the bandit's own frozen train-fold ``(μ, σ)``, because that is the scale it
    was fit on.

    **Restricted menu.** The bandit's argmax runs only over arms the MDP actually offers (e.g. the
    pruned space's ``{hybrid, graph}``), so it plays the *same* action menu as the learned policy —
    an unrestricted bandit picking ``dense`` would be scored on an action its opponent cannot take.
    Ties break to the lowest arm index (deterministic).
    """

    def __init__(
        self,
        scorer: ArmScorer,
        *,
        arm_names: Sequence[str],
        action_space: Sequence[Action],
        context_mean: np.ndarray | None = None,
        context_std: np.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        labels = _labels(action_space)
        # arm index (A6.1 column order) → MDP action index, for the arms present in BOTH.
        self.menu: dict[int, int] = {
            a: labels.index(f"{arm}@self")
            for a, arm in enumerate(arm_names)
            if f"{arm}@self" in labels
        }
        if not self.menu:
            raise ValueError("no A6.1 arm has a self-scope action in this action space")
        self.scorer = scorer
        self.mean = context_mean
        self.std = context_std
        self.name = name or f"bandit({scorer.name})"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        if _step_idx(x) != 0:
            return STOP_INDEX
        context = np.asarray(x[:N_FEATURES], dtype=np.float64)
        if self.mean is not None and self.std is not None:
            context = (context - self.mean) / self.std
        scores = np.asarray(self.scorer.ucb_scores(context), dtype=np.float64)
        best_arm = max(self.menu, key=lambda a: float(scores[a]))  # insertion order ⇒ ties → lowest
        return self.menu[best_arm]


class ReActBridgePolicy(ScriptedPolicy):
    """Baseline (iii, the strong one) — the A4 ReAct loop + A5 entity bridge, scripted in the MDP.

    Self-scoped search with the production arm, then pull the first discovered-but-unretrieved
    entity (``disc0`` — each bridge consumes it, so the slot re-points to the next discovery) while
    budget and discoveries last, then STOP. Identical control flow to ``bridge_expert_rollout``,
    hence to `research/agentic.py` + `research/bridge.py`; a test pins the two to one trajectory.

    Its LLM-written queries are the one thing the sim replaces with a template — the sim-to-real
    gap the paid run measures.
    """

    def __init__(
        self, arm: str, *, action_space: Sequence[Action], name: str | None = None
    ) -> None:
        labels = _labels(action_space)
        if f"{arm}@self" not in labels:
            raise ValueError(f"arm {arm!r} has no self-scope action in this action space")
        self.arm = arm
        self.self_idx = labels.index(f"{arm}@self")
        disc0 = f"{arm}@disc0"
        self.disc0_idx = labels.index(disc0) if disc0 in labels else None
        self.name = name or f"react({arm})"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        if _step_idx(x) == 0:
            return self.self_idx  # step 1: the A4 self-scoped search
        if self.disc0_idx is not None and (mask is None or bool(mask[self.disc0_idx])):
            return self.disc0_idx  # A5 bridge: pivot to a discovered entity's own filings
        return STOP_INDEX  # nothing left to bridge ⇒ reflective stop


class SweepBridgePolicy(ScriptedPolicy):
    """Baseline (v, the E3 strong one) — self-scoped search, ONE fanout sweep, STOP.

    **This, not ``react(arm)``, is what a learned policy must beat to justify RL after E3.** Fanout
    makes the correct hop-2 move *expressible*; it also makes it **trivially scriptable** — "search
    the seed, then sweep every candidate it named" requires no learning at all. Handing the learner
    a fanout action while leaving the baseline stuck on ``disc0`` would manufacture a win out of an
    action-space asymmetry, which is the same class of error that invalidated the A6.2 verdict in
    the first place. So the baseline gets the sweep too.

    What that leaves for a learner to actually win on is the *cost* side: knowing **when not to
    sweep** — a CTRL question with nothing to bridge to, an episode already covered by hop 1, or a
    candidate set so large the sweep's N × arm-cost outweighs the coverage it can buy. If no such
    policy exists, RL genuinely has nothing to add here, and that is a real finding.

    The sweep is self-terminating: it marks every candidate searched, so the discovered set drains
    and the fanout action is masked illegal on the next step ⇒ STOP.
    """

    def __init__(
        self, arm: str, *, action_space: Sequence[Action], name: str | None = None
    ) -> None:
        labels = _labels(action_space)
        for needed in (f"{arm}@self", f"{arm}@fanout"):
            if needed not in labels:
                raise ValueError(f"arm {arm!r} has no {needed!r} action in this action space")
        self.arm = arm
        self.self_idx = labels.index(f"{arm}@self")
        self.fanout_idx = labels.index(f"{arm}@fanout")
        self.name = name or f"sweep({arm})"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        if _step_idx(x) == 0:
            return self.self_idx  # step 1: the A4 self-scoped search (E2 relation query)
        if mask is None or bool(mask[self.fanout_idx]):
            return self.fanout_idx  # step 2: sweep EVERY candidate the seed named
        return STOP_INDEX  # nothing discovered (CTRL), or the sweep already drained it


class RandomActionPolicy:
    """Baseline (iv) — uniform over the *legal* actions (the floor any learner must clear).

    The only stochastic candidate, so it is the one that actually needs the seed-averaging.
    ``greedy_action`` is the argmax of a uniform distribution = the first legal action (STOP); it is
    degenerate by construction and the harness never uses it — ``random`` is always *sampled*.
    """

    def __init__(self, n_actions: int, *, name: str = "random") -> None:
        self.n_actions = n_actions
        self.name = name

    def _legal(self, mask: np.ndarray | None) -> np.ndarray:
        return np.flatnonzero(mask) if mask is not None else np.arange(self.n_actions)

    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        legal = self._legal(mask)
        return int(rng.choice(legal)), float(-np.log(len(legal)))

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        return int(self._legal(mask)[0])


class AlwaysSearchPolicy(ScriptedPolicy):
    """The **reward-hacking sentinel** — never STOP; burn the whole budget on the costliest arm.

    Prefers a discovered slot when one is legal (maximum recall), else self-scope, and never stops,
    so the horizon terminates it. Under the cost tax ``λ_c·cost(arm)`` a policy that keeps paying
    for retrieval after coverage has saturated must score *below* one that stops. If this sentinel
    ties or beats the learned policy, the reward is hackable (or the learned policy has learned to
    hack it) and ``evaluate_rl`` blocks promotion.
    """

    def __init__(
        self,
        *,
        action_space: Sequence[Action],
        arm_costs: Mapping[str, float],
        name: str | None = None,
    ) -> None:
        arms = sorted({a.config for a in action_space if a.config is not None})
        if not arms:
            raise ValueError("action space has no search actions")
        self.arm = max(arms, key=lambda c: arm_costs.get(c, 0.0))  # sorted ⇒ ties → lowest name
        labels = _labels(action_space)
        self.self_idx = labels.index(f"{self.arm}@self")
        disc0 = f"{self.arm}@disc0"
        self.disc0_idx = labels.index(disc0) if disc0 in labels else None
        self.name = name or "sentinel(always-search)"

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        if self.disc0_idx is not None and (mask is None or bool(mask[self.disc0_idx])):
            return self.disc0_idx
        return self.self_idx


class GreedyPolicy:
    """Run any ``RolloutPolicy`` under **deploy semantics**: argmax, never sample.

    The learned policy's headline row. A deployed checkpoint takes its best action, not a sample
    from π, so the verdict is computed on this wrapper — which also makes the learned policy's
    returns deterministic (no seed noise in the paired CI). The sampled row is reported alongside as
    the on-policy value ``V(π)`` the objective actually optimizes.
    """

    def __init__(self, policy: RolloutPolicy, *, name: str) -> None:
        self.policy = policy
        self.name = name

    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        return self.policy.greedy_action(x, mask=mask), 0.0

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        return self.policy.greedy_action(x, mask=mask)


# ---- rollout + metrics ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeOutcome:
    """One episode's telemetry under one policy (the row the aggregate metrics average over)."""

    episode_idx: int
    total_return: float  # Σ_t r_t (undiscounted) — with γ=1 this is Φ(s_T) − λ_c·Σ_t cost(a_t)
    coverage: float  # Φ(s_T): terminal aspect coverage of the union — the quality half of reward
    n_retrievals: int  # search steps actually issued (STOP and masked no-ops excluded)
    arm_cost: float  # Σ_t cost(arm_t): the latency/compute proxy the reward taxes
    n_llm_calls: int  # 0 by construction in the $0 sim (templated queries) — see module docstring
    stopped: bool  # ended on an explicit STOP (vs the horizon forcing termination)
    actions: tuple[str, ...]


def run_episode(
    env: RagRetrievalEnv, policy: RolloutPolicy, episode_idx: int, *, rng: np.random.Generator
) -> EpisodeOutcome:
    """Roll one episode and record its telemetry (``reinforce.rollout`` + the env's ``StepInfo``).

    Same loop as the training rollout — mask, act, step — but it keeps the ``StepInfo`` the learner
    discards (coverage, cost, stop/mask flags), which is what the eval reports.
    """
    state = env.reset(episode_idx)
    total_return = 0.0
    arm_cost = 0.0
    n_retrievals = 0
    coverage = 0.0
    stopped = False
    labels: list[str] = []
    done = False
    while not done:
        mask = env.legal_mask()
        action, _ = policy.act(state, rng=rng, mask=mask)
        state, reward, done, info = env.step(int(action))
        total_return += reward
        arm_cost += info.cost
        coverage = info.coverage
        labels.append(info.action_label)
        if info.stopped:
            stopped = True
        elif not info.masked:
            n_retrievals += 1  # a masked action burns budget but issues no retrieval
    return EpisodeOutcome(
        episode_idx=episode_idx,
        total_return=total_return,
        coverage=coverage,
        n_retrievals=n_retrievals,
        arm_cost=arm_cost,
        n_llm_calls=0,
        stopped=stopped,
        actions=tuple(labels),
    )


@dataclass(frozen=True)
class PolicyMetrics:
    """One candidate's held-out metrics, averaged over episodes × seeds."""

    name: str
    mean_return: float
    mean_coverage: float
    mean_retrievals: float
    mean_arm_cost: float
    mean_llm_calls: float
    stop_rate: float  # fraction of episodes ended by an explicit STOP (vs the horizon)
    n_episodes: int
    n_seeds: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_return": self.mean_return,
            "mean_coverage": self.mean_coverage,
            "mean_retrievals": self.mean_retrievals,
            "mean_arm_cost": self.mean_arm_cost,
            "mean_llm_calls": self.mean_llm_calls,
            "stop_rate": self.stop_rate,
            "n_episodes": self.n_episodes,
            "n_seeds": self.n_seeds,
        }


def policy_name(policy: RolloutPolicy) -> str:
    """A candidate's report label — its ``name`` attribute, else its class name (checkpoints)."""
    return str(getattr(policy, "name", policy.__class__.__name__))


def evaluate_policy(
    env: RagRetrievalEnv,
    indices: Sequence[int],
    policy: RolloutPolicy,
    *,
    seeds: Sequence[int] = (0,),
    name: str | None = None,
) -> tuple[PolicyMetrics, np.ndarray]:
    """Seed-averaged metrics for ``policy`` over ``indices``, plus the per-episode mean returns.

    The returned ``[len(indices)]`` array is the paired-delta input: entry ``i`` is episode
    ``indices[i]``'s return averaged over ``seeds`` (exact, not noisy, for the deterministic
    candidates — they ignore the RNG). Each seed gets a fresh ``default_rng(seed)``, so a stochastic
    policy's rollouts are reproducible.
    """
    if not indices:
        raise ValueError("evaluate_policy needs at least one episode index")
    if not seeds:
        raise ValueError("evaluate_policy needs at least one seed")
    returns = np.zeros((len(seeds), len(indices)), dtype=np.float64)
    outcomes: list[EpisodeOutcome] = []
    for s, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for i, episode_idx in enumerate(indices):
            outcome = run_episode(env, policy, int(episode_idx), rng=rng)
            returns[s, i] = outcome.total_return
            outcomes.append(outcome)
    metrics = PolicyMetrics(
        name=name or policy_name(policy),
        mean_return=float(np.mean(returns)),
        mean_coverage=float(np.mean([o.coverage for o in outcomes])),
        mean_retrievals=float(np.mean([o.n_retrievals for o in outcomes])),
        mean_arm_cost=float(np.mean([o.arm_cost for o in outcomes])),
        mean_llm_calls=float(np.mean([o.n_llm_calls for o in outcomes])),
        stop_rate=float(np.mean([1.0 if o.stopped else 0.0 for o in outcomes])),
        n_episodes=len(indices),
        n_seeds=len(seeds),
    )
    return metrics, returns.mean(axis=0)


# ---- fold bookkeeping (groups / strata / the leakage guard) --------------------------------------
def _group_keys(env: RagRetrievalEnv, indices: Sequence[int]) -> list[str]:
    """Split key per episode: the bridge pair's ``group_id``, else the question (mirrors A6.1)."""
    return [(env.episodes[i].group_id or env.episodes[i].question) for i in indices]


def fold_keys(
    env: RagRetrievalEnv, indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(groups, strata)`` for the bootstrap: int-encoded bridge pairs + HARD/MED/CTRL labels."""
    _, groups = np.unique(np.array(_group_keys(env, indices)), return_inverse=True)
    strata = np.array([env.episodes[i].stratum or "NA" for i in indices])
    return groups.astype(int), strata


def assert_group_disjoint(
    env: RagRetrievalEnv, train_indices: Sequence[int], test_indices: Sequence[int]
) -> None:
    """Fail loud if any bridge pair straddles the folds (the D2 group-split leakage invariant).

    The eval is only meaningful if no held-out question is a near-copy of a trained one; the two
    benchmark files are produced by ``split_multihop``, and this re-checks that on the loaded data
    rather than trusting the filenames.
    """
    overlap = set(_group_keys(env, train_indices)) & set(_group_keys(env, test_indices))
    if overlap:
        raise ValueError(
            f"train/test folds share {len(overlap)} group(s) — group leakage: {sorted(overlap)[:3]}"
        )


# ---- report -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StratumDelta:
    """Per-stratum return gap, learned − best baseline (the CTRL row is the regression guard)."""

    stratum: str
    n: int
    learned: float
    baseline: float
    delta: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "n": self.n,
            "learned": self.learned,
            "baseline": self.baseline,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class RLEvalReport:
    """The A6.2g verdict: every candidate's held-out value, the paired delta, and promote/reject."""

    algo: str
    action_space: str
    seeds: tuple[int, ...]
    n_train: int
    n_test: int
    gamma: float
    lambda_cost: float
    learned: str
    best_baseline: str
    test_metrics: list[PolicyMetrics]
    train_metrics: list[PolicyMetrics]  # learned + best baseline on the TRAIN fold (overfit check)
    overfit_gap: float  # train − test mean return, learned policy
    baseline_gap: float  # same for the best baseline — the fold-difficulty reference for that gap
    delta_train: float  # the SAME paired delta, measured on the train fold
    generalization_gap: float  # delta_train − delta_return: how much of the edge is fold-specific
    delta_return: float
    delta_ci: tuple[float, float]
    delta_p_positive: float  # exact one-sided group-bootstrap P(Δ>0) — reported, not gated
    per_stratum: list[StratumDelta]
    sentinel: str
    sentinel_return: float
    sentinel_beats_learned: bool
    promote: bool
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "action_space": self.action_space,
            "seeds": list(self.seeds),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "gamma": self.gamma,
            "lambda_cost": self.lambda_cost,
            "learned": self.learned,
            "best_baseline": self.best_baseline,
            "test_metrics": [m.to_json_dict() for m in self.test_metrics],
            "train_metrics": [m.to_json_dict() for m in self.train_metrics],
            "overfit_gap": self.overfit_gap,
            "baseline_gap": self.baseline_gap,
            "delta_train": self.delta_train,
            "generalization_gap": self.generalization_gap,
            "delta_return": self.delta_return,
            "delta_ci": list(self.delta_ci),
            "delta_p_positive": self.delta_p_positive,
            "per_stratum": [s.to_json_dict() for s in self.per_stratum],
            "sentinel": {
                "name": self.sentinel,
                "mean_return": self.sentinel_return,
                "beats_learned": self.sentinel_beats_learned,
            },
            "promote": self.promote,
            "rationale": self.rationale,
        }


def verdict(
    delta: float,
    ci: tuple[float, float],
    *,
    ctrl_regression: bool,
    sentinel_beats_learned: bool,
    learned: str = "learned",
    best_baseline: str = "best baseline",
    sentinel: str = "sentinel",
) -> tuple[bool, str]:
    """The **pre-registered** promote rule (pure): all four gates, plus the reason it failed.

        PROMOTE iff Δ > 0 AND CI_low > 0 AND no CTRL regression AND the sentinel does not beat it.

    Same discipline as the A6.1 verdict (a positive point estimate is not enough — the group
    bootstrap must exclude 0, and the control stratum must not pay for the gain), with the extra
    reward-hacking gate A6.2's sequential action space needs: if budget-spamming scores at or above
    the learned policy, the cost tax failed and the delta means nothing.
    """
    lo, hi = ci
    if sentinel_beats_learned:
        return False, (
            f"REJECT: the reward-hacking sentinel {sentinel} scores ≥ {learned} — the cost tax "
            f"does not dominate budget-spamming, so the delta is not interpretable."
        )
    if delta <= 0:
        return False, f"REJECT: {learned} return ≤ {best_baseline} (Δ={delta:+.4f})."
    if lo <= 0:
        return False, (
            f"REJECT: Δ={delta:+.4f} does not clear the group-bootstrap 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]."
        )
    if ctrl_regression:
        return False, f"REJECT: Δ={delta:+.4f} clears the CI but regresses the CTRL stratum."
    return True, (
        f"PROMOTE: {learned} beats {best_baseline} by Δ={delta:+.4f}, "
        f"CI [{lo:+.4f}, {hi:+.4f}] > 0, no CTRL regression, sentinel clean."
    )


def evaluate_rl(
    env: RagRetrievalEnv,
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    learned: RolloutPolicy,
    baselines: Sequence[RolloutPolicy],
    sentinel: RolloutPolicy | None = None,
    learned_name: str = "learned",
    algo: str = "unknown",
    action_space: str = "unknown",
    seeds: Sequence[int] = (0, 1, 2),
    n_boot: int = 1000,
    seed: int = 42,
) -> RLEvalReport:
    """Evaluate the frozen policy against the baselines on the held-out fold and apply the verdict.

    Steps: (1) group-leakage guard on the two folds; (2) score every candidate on the test fold —
    the learned policy twice, greedy (deploy semantics ⇒ the headline) and sampled (the on-policy
    value); (3) pick the best baseline **by held-out mean return**; (4) paired per-episode delta
    learned(greedy) − best baseline with a group-bootstrap CI; (5) per-stratum deltas (CTRL is the
    regression guard); (6) re-score learned + best baseline on the *train* fold for the overfit gap;
    (7) the pre-registered rule:

        PROMOTE iff Δ > 0 AND CI_low > 0 AND no CTRL regression AND the sentinel does not beat it.

    ``env`` must be built over ``train + test`` episodes (one env ⇒ one shared ``TransitionCache``,
    so the baselines' retrievals are the learned policy's cached ones — identical transitions).
    """
    if not baselines:
        raise ValueError("evaluate_rl needs at least one baseline")
    if not seeds:
        raise ValueError("evaluate_rl needs at least one seed")
    names = [policy_name(p) for p in baselines]
    if len(set(names)) != len(names):  # names key the metrics/returns dicts — a dup would overwrite
        raise ValueError(f"baseline names must be unique; got {names}")
    assert_group_disjoint(env, train_indices, test_indices)
    groups, strata = fold_keys(env, test_indices)

    # The headline row: the checkpoint under deploy semantics (greedy ⇒ deterministic ⇒ one seed is
    # exact). The sampled row reports the on-policy value V(π) the objective actually maximizes.
    greedy = GreedyPolicy(learned, name=f"{learned_name}(greedy)")
    learned_metrics, learned_returns = evaluate_policy(
        env, test_indices, greedy, seeds=(seeds[0],)
    )
    sampled_metrics, _ = evaluate_policy(
        env, test_indices, learned, seeds=seeds, name=f"{learned_name}(sampled)"
    )

    test_metrics = [learned_metrics, sampled_metrics]
    baseline_metrics: dict[str, PolicyMetrics] = {}
    baseline_returns: dict[str, np.ndarray] = {}
    for policy in baselines:
        metrics, returns = evaluate_policy(env, test_indices, policy, seeds=seeds)
        test_metrics.append(metrics)
        baseline_metrics[metrics.name] = metrics
        baseline_returns[metrics.name] = returns

    best_baseline = max(baseline_returns, key=lambda name: float(baseline_returns[name].mean()))
    best_returns = baseline_returns[best_baseline]

    # Paired per-episode delta. Both sides are deterministic in practice (greedy learned policy; the
    # scripted baselines ignore the RNG), so the CI carries only the episode/group resampling
    # variance the group bootstrap exists to measure. The one exception is `random`: its per-episode
    # return is a seed-average, and if it ever won the best-baseline slot the CI would ignore that
    # Monte-Carlo noise — harmless, since a random policy losing to everything is the whole point.
    deltas = learned_returns - best_returns
    delta = float(deltas.mean())
    lo, hi, p_positive = bootstrap_delta_stats(
        lambda idx: float(deltas[idx].mean()),
        len(deltas),
        groups=groups,
        n_boot=n_boot,
        seed=seed,
    )

    per_stratum: list[StratumDelta] = []
    for s in _STRATA:
        idx = np.flatnonzero(strata == s)
        if idx.size == 0:
            continue
        per_stratum.append(
            StratumDelta(
                stratum=s,
                n=int(idx.size),
                learned=float(learned_returns[idx].mean()),
                baseline=float(best_returns[idx].mean()),
                delta=float(deltas[idx].mean()),
            )
        )

    # Overfit check: the same two policies re-scored on the fold the learner actually trained on. A
    # large learned gap next to a small baseline gap = memorization; a large gap on BOTH just means
    # the held-out fold is harder (the baselines never saw it either), which is why both are shown.
    train_learned, _ = evaluate_policy(env, train_indices, greedy, seeds=(seeds[0],))
    best_policy = next(p for p in baselines if policy_name(p) == best_baseline)
    train_baseline, _ = evaluate_policy(env, train_indices, best_policy, seeds=seeds)
    # The paired delta on the TRAIN fold. This — not the absolute gaps above — is the diagnostic
    # that matters: the folds differ in difficulty, so a policy can score HIGHER on test in absolute
    # terms while its *edge over the baseline* collapses. (The mean of paired deltas is the
    # difference of the means, so this costs no extra rollouts.) delta_train >> delta_return ⇒ the
    # edge was fold-specific.
    delta_train = train_learned.mean_return - train_baseline.mean_return

    sentinel_name, sentinel_return = "none", float("-inf")
    if sentinel is not None:
        sentinel_metrics, _ = evaluate_policy(env, test_indices, sentinel, seeds=(seeds[0],))
        test_metrics.append(sentinel_metrics)
        sentinel_name, sentinel_return = sentinel_metrics.name, sentinel_metrics.mean_return
    sentinel_beats_learned = sentinel_return >= learned_metrics.mean_return

    ctrl_regression = any(s.stratum == "CTRL" and s.delta < -1e-9 for s in per_stratum)
    promote, rationale = verdict(
        delta,
        (lo, hi),
        ctrl_regression=ctrl_regression,
        sentinel_beats_learned=sentinel_beats_learned,
        learned=learned_metrics.name,
        best_baseline=best_baseline,
        sentinel=sentinel_name,
    )
    return RLEvalReport(
        algo=algo,
        action_space=action_space,
        seeds=tuple(int(s) for s in seeds),
        n_train=len(train_indices),
        n_test=len(test_indices),
        gamma=env.gamma,
        lambda_cost=env.lambda_cost,
        learned=learned_metrics.name,
        best_baseline=best_baseline,
        test_metrics=test_metrics,
        train_metrics=[train_learned, train_baseline],
        overfit_gap=train_learned.mean_return - learned_metrics.mean_return,
        baseline_gap=train_baseline.mean_return - baseline_metrics[best_baseline].mean_return,
        delta_train=delta_train,
        generalization_gap=delta_train - delta,
        delta_return=delta,
        delta_ci=(lo, hi),
        delta_p_positive=p_positive,
        per_stratum=per_stratum,
        sentinel=sentinel_name,
        sentinel_return=sentinel_return,
        sentinel_beats_learned=sentinel_beats_learned,
        promote=promote,
        rationale=rationale,
    )


def build_baselines(
    env: RagRetrievalEnv,
    *,
    react_arms: Sequence[str] = ("hybrid", "graph"),
    bandit: ArmScorer | None = None,
    bandit_arm_names: Sequence[str] = (),
    context_mean: np.ndarray | None = None,
    context_std: np.ndarray | None = None,
) -> list[RolloutPolicy]:
    """The baseline suite over ``env``'s action space (the CLI's default candidate set).

    One ``fixed(arm)`` per arm the space offers (their max is "the best fixed pipeline"), one
    ``react(arm)`` per production arm in ``react_arms`` the space offers, one ``sweep(arm)`` per
    production arm **when the space has a fanout action** (E3 — the strong baseline a learner must
    beat; see ``SweepBridgePolicy``), the A6.1 bandit when a fitted scorer is supplied, and
    ``random``. The sentinel is built separately (it is not a baseline to beat — see
    ``AlwaysSearchPolicy``).
    """
    arms = sorted({a.config for a in env.action_space if a.config is not None})
    labels = _labels(env.action_space)
    policies: list[RolloutPolicy] = [
        OneShotArmPolicy(arm, action_space=env.action_space) for arm in arms
    ]
    policies += [
        ReActBridgePolicy(arm, action_space=env.action_space) for arm in react_arms if arm in arms
    ]
    policies += [
        SweepBridgePolicy(arm, action_space=env.action_space)
        for arm in react_arms
        if arm in arms and f"{arm}@fanout" in labels
    ]
    if bandit is not None:
        policies.append(
            BanditOneShotPolicy(
                bandit,
                arm_names=bandit_arm_names,
                action_space=env.action_space,
                context_mean=context_mean,
                context_std=context_std,
            )
        )
    policies.append(RandomActionPolicy(env.n_actions))
    return policies


def format_report_markdown(report: RLEvalReport) -> str:
    """Compact console/report table: every candidate's held-out metrics + the verdict line."""
    lines = [
        f"RL held-out eval — algo={report.algo}, action_space={report.action_space}, "
        f"n_train={report.n_train}, n_test={report.n_test}, seeds={list(report.seeds)}",
        f"gamma={report.gamma}, lambda_cost={report.lambda_cost}",
        "",
        "| policy | return | coverage | retrievals | arm cost | LLM calls | stop rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in report.test_metrics:
        lines.append(
            f"| {m.name} | {m.mean_return:+.4f} | {m.mean_coverage:.3f} | {m.mean_retrievals:.2f} "
            f"| {m.mean_arm_cost:.3f} | {m.mean_llm_calls:.1f} | {m.stop_rate:.2f} |"
        )
    lo, hi = report.delta_ci
    lines += [
        "",
        f"Δ(return) {report.learned} − {report.best_baseline} = {report.delta_return:+.4f} "
        f"[95% group-bootstrap CI {lo:+.4f}, {hi:+.4f}], P(Δ>0) = {report.delta_p_positive:.3f}",
    ]
    if report.per_stratum:  # the CTRL row is a promote gate — always show it, never only in JSON
        lines += [
            "",
            "| stratum | n | learned | baseline | Δ |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {s.stratum} | {s.n} | {s.learned:+.4f} | {s.baseline:+.4f} | {s.delta:+.4f} |"
            for s in report.per_stratum
        ]
    lines += [
        "",
        f"Paired Δ vs {report.best_baseline}: train {report.delta_train:+.4f} → "
        f"held-out {report.delta_return:+.4f} "
        f"(generalization gap {report.generalization_gap:+.4f} — the edge that did NOT transfer)",
        f"Absolute return, train − test: learned {report.overfit_gap:+.4f}, "
        f"best baseline {report.baseline_gap:+.4f} "
        f"(a gap on BOTH = fold difficulty, not memorization)",
        f"Sentinel {report.sentinel}: return {report.sentinel_return:+.4f} "
        f"(beats learned: {report.sentinel_beats_learned})",
        "",
        report.rationale,
    ]
    return "\n".join(lines)
