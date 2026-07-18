"""Sim-to-real gap for the frozen RL policy (adv-RAG A6.2g) — the `rag rl-simreal` engine.

The `$0` simulator that trained and judged the policy (A6.2c–g) issues **templated** queries: a
self-scoped hop searches the plain question, a bridge hop searches ``"{entity} {question}"``.
Reality does not: the A4 loop has an **LLM write the query text** at each hop, and a better-phrased
query retrieves different chunks. So every number in the `$0` verdict is measured under one
systematic distortion, and the honest thing to do is **measure it** rather than argue about it.

**The experiment.** Run the *same frozen policy* over the *same held-out episodes* twice:

| run | who picks arm / scope / stop | who writes the query text | cost |
|---|---|---|---|
| sim | the policy | the A6.2b template | `$0` |
| real | the policy | **the LLM** (`RL_QUERY_SYSTEM`) | ~1 call per hop |

Everything else is held fixed *by construction*: both runs use the same `RagRetrievalEnv`, so they
share the union dedup, the discovered-entity bookkeeping, the coverage oracle, the cost table and
the horizon. The only free variable is the query string (the env's ``query_writer`` hook), so the
difference in outcome **is** the sim-to-real gap — not a confound.

$$\\text{gap} = \\overline{\\Phi}_{\\text{real}} - \\overline{\\Phi}_{\\text{sim}}$$

A **positive** gap means the simulator *understated* the policy (LLM queries retrieve better than
the template, so the `$0` verdict is conservative). A **negative** gap means the simulator
*flattered* it — the policy's decisions were tuned against template retrieval and do not survive
real query text, which is the failure mode this run exists to catch.

**The terminal answer (only here).** The `$0` sim never synthesizes, so the faithfulness floor of
the reward is inert there. This run does the real thing: the accumulated union goes through the
existing guarded ``answer_question`` (citation + `NumberGrounding` guards), and the report carries
the refusal rate. Grounding is unchanged — the LLM still writes no numbers of its own.

Cost is bounded and reported *before* the first call: at most ``max_steps`` query-writing calls plus
one synthesis per episode. The CLI refuses to spend without an explicit ``--yes``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from stock_agent.llm.client import TextLLM
from stock_agent.logging_config import get_logger
from stock_agent.rag.ope import bootstrap_delta_stats
from stock_agent.rag.rl.action import RetrievalRequest
from stock_agent.rag.rl.env import RagRetrievalEnv
from stock_agent.rag.rl.reinforce import RolloutPolicy
from stock_agent.rag.rl.rleval import GreedyPolicy, run_episode
from stock_agent.research._shared import loads_lenient
from stock_agent.research.multistep_eval import MultiHopQuery, coverage
from stock_agent.research.prompts import RL_QUERY_SYSTEM, build_rl_query_user
from stock_agent.research.synthesis import answer_question
from stock_agent.schemas.retrieval import EvidenceSet, RetrievedChunk

log = get_logger(__name__)

_QUERY_MAX_TOKENS = 200  # the writer emits one short JSON object; cap it hard (this is the $ knob)


class LLMQueryWriter:
    """The env ``query_writer`` that asks the LLM to phrase the query for the policy's chosen scope.

    The policy has already decided *which* arm, *where* to point it, and *whether* to stop; this
    writer only turns that decision into search text. It cannot change the arm or the scope — those
    are read off the ``RetrievalRequest`` and passed back untouched — so the paid run tests the
    policy's *decisions* under real query text, not a different policy.

    Tracks ``calls`` (billing) and the per-episode ``issued`` list (fed to the prompt as "do not
    repeat", mirroring the A4 loop's anti-loop discipline). A malformed or empty completion returns
    ``""``, and the env falls back to the template — a bad LLM response degrades to the sim, never
    to a contentless query.
    """

    def __init__(
        self, llm: TextLLM, *, alias_map: Mapping[str, Sequence[str]] | None = None
    ) -> None:
        self.llm = llm
        self.calls = 0
        self.issued: list[str] = []
        self._alias_map = alias_map or {}

    def reset(self) -> None:
        """Clear the per-episode anti-loop list (call between episodes; ``calls`` keeps rising)."""
        self.issued = []

    def _scope_name(self, ticker: str | None) -> str | None:
        if not ticker:
            return None
        names = self._alias_map.get(ticker) or []
        return names[0] if names else ticker

    def __call__(
        self,
        request: RetrievalRequest,
        episode: MultiHopQuery,
        evidence: Sequence[RetrievedChunk],
    ) -> str:
        user = build_rl_query_user(
            episode.question,
            scope_ticker=request.scope_ticker,
            scope_name=self._scope_name(request.scope_ticker),
            issued=self.issued,
            evidence=evidence,
        )
        self.calls += 1
        raw = self.llm.complete_json(
            system=RL_QUERY_SYSTEM, user=user, max_tokens=_QUERY_MAX_TOKENS
        )
        # A malformed reply must NOT kill a paid run mid-flight (the calls already spent would be
        # wasted): degrade to the template, log it, and keep going. `loads_lenient` RAISES on
        # unparseable text, so both failure shapes are caught here.
        try:
            payload = loads_lenient(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("rl_simreal.query_writer_unparseable", scope=request.scope_ticker)
            return ""
        query = str(payload.get("query", "")).strip()
        if not query:
            log.info("rl_simreal.query_writer_empty", scope=request.scope_ticker)
            return ""
        self.issued.append(query)
        return query


@dataclass(frozen=True)
class EpisodeGap:
    """One held-out question, run twice (templated vs LLM-written queries) — the paired row."""

    question: str
    stratum: str
    coverage_sim: float
    coverage_real: float
    return_sim: float
    return_real: float
    steps_sim: int
    steps_real: int
    actions_sim: tuple[str, ...]
    actions_real: tuple[str, ...]
    queries: tuple[str, ...]  # what the LLM actually wrote (the audit trail for the gap)
    n_llm_calls: int
    insufficient: bool  # the guarded synthesis refused (empty/ungrounded evidence)
    answer: str

    @property
    def coverage_gap(self) -> float:
        return self.coverage_real - self.coverage_sim

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "stratum": self.stratum,
            "coverage_sim": self.coverage_sim,
            "coverage_real": self.coverage_real,
            "coverage_gap": self.coverage_gap,
            "return_sim": self.return_sim,
            "return_real": self.return_real,
            "steps_sim": self.steps_sim,
            "steps_real": self.steps_real,
            "actions_sim": list(self.actions_sim),
            "actions_real": list(self.actions_real),
            "queries": list(self.queries),
            "n_llm_calls": self.n_llm_calls,
            "insufficient": self.insufficient,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class SimRealReport:
    """The measured sim-to-real gap + the grounding outcome of the terminal (paid) synthesis."""

    n_episodes: int
    mean_coverage_sim: float
    mean_coverage_real: float
    coverage_gap: float  # real − sim: > 0 ⇒ the $0 verdict was conservative
    mean_return_sim: float
    mean_return_real: float
    return_gap: float
    same_action_rate: float  # fraction of episodes whose action SEQUENCE real queries left intact
    refusal_rate: float  # guarded synthesis said "Insufficient evidence found."
    total_llm_calls: int
    episodes: list[EpisodeGap]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "mean_coverage_sim": self.mean_coverage_sim,
            "mean_coverage_real": self.mean_coverage_real,
            "coverage_gap": self.coverage_gap,
            "mean_return_sim": self.mean_return_sim,
            "mean_return_real": self.mean_return_real,
            "return_gap": self.return_gap,
            "same_action_rate": self.same_action_rate,
            "refusal_rate": self.refusal_rate,
            "total_llm_calls": self.total_llm_calls,
            "episodes": [e.to_json_dict() for e in self.episodes],
        }


class DryRunLLM:
    """A ``$0`` stand-in used ONLY by ``rag rl-simreal --dry-run`` to exercise the harness.

    Returns an empty query (so the env falls back to the template ⇒ the "real" run *is* the sim ⇒
    the measured gap must come out **exactly 0**, which is the harness self-test) and a canned
    refusal for the synthesis. Never touches the network, so a dry run costs nothing and still
    proves the whole path — env wiring, call accounting, report shape — before money is spent.
    """

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        if "search query" in system:  # RL_QUERY_SYSTEM ⇒ "" ⇒ the env keeps the template
            return json.dumps({"query": ""})
        return json.dumps({"answer": "Insufficient evidence found.", "insufficient_evidence": True})


def estimate_llm_calls(n_episodes: int, max_steps: int) -> int:
    """Naive per-STEP call bound: ``max_steps`` query-writing calls + 1 synthesis, per episode.

    A cheap a-priori floor, kept for reference. It is **not** the real count for any policy that
    fans out: a fan-out step issues one retrieval request *per discovered candidate* and the writer
    bills each, so this undercounts every sweep (a 9-candidate sweep is 9 writer calls in one step,
    not one). Use ``measure_llm_calls`` for the honest, fan-out-aware number the spend gate shows.
    """
    return n_episodes * (max_steps + 1)


def measure_llm_calls(
    policy: RolloutPolicy,
    episodes: Sequence[MultiHopQuery],
    *,
    env: RagRetrievalEnv,
    writer: LLMQueryWriter,
    include_synthesis: bool = True,
) -> int:
    """Fan-out-aware LLM-call count for a paid run, measured on a ``$0`` dry rollout.

    Rolls the greedy policy through ``env`` (which must carry ``writer`` as its ``query_writer``,
    and ``writer`` must wrap a ``$0`` stand-in LLM, ``DryRunLLM``) and counts the writer's *actual*
    calls. Because the writer bills once per retrieval **request**, a fan-out step contributes one
    call per swept candidate — exactly the branch fan-out ``estimate_llm_calls`` misses. Adds one
    synthesis per non-empty union when ``include_synthesis`` (``rl-simreal`` pays it; the ``rl-h2h``
    baseline side does not). Costs nothing beyond the retrieval the paid run runs anyway.

    An **estimate, not a guarantee**: the paid run's LLM-written queries change the hop-1 union, so
    the hop-2 discovered set (and thus the fan-out width) can differ. Always far closer to the true
    spend than the naive step bound — this is the number the ``--yes`` gate should quote.
    """
    greedy = GreedyPolicy(policy, name="dry-count")
    rng = np.random.default_rng(0)  # greedy ignores it; kept for the rollout signature
    total = 0
    for i in range(len(episodes)):
        writer.reset()
        before = writer.calls
        run_episode(env, greedy, i, rng=rng)
        total += (writer.calls - before) + (1 if include_synthesis and env.evidence else 0)
    return total


def run_sim_to_real(
    policy: RolloutPolicy,
    episodes: Sequence[MultiHopQuery],
    *,
    sim_env: RagRetrievalEnv,
    real_env: RagRetrievalEnv,
    writer: LLMQueryWriter,
    llm: TextLLM,
    seed: int = 0,
) -> SimRealReport:
    """Roll the frozen policy over ``episodes`` twice — templated vs LLM-written — and measure the
    gap.

    ``sim_env`` and ``real_env`` must be built over the **same** episode list with the **same**
    action space, arms, costs and horizon; the only difference is that ``real_env`` carries
    ``writer`` as its ``query_writer``. Both rollouts are **greedy** (deploy semantics — the same
    policy the `$0` verdict gated on). The terminal guarded synthesis runs on the real union only.

    Labels (``aspects``) are used **only** to score coverage after the fact — never by the policy.
    """
    if len(sim_env.episodes) != len(episodes) or len(real_env.episodes) != len(episodes):
        raise ValueError("sim_env and real_env must be built over exactly these episodes")
    greedy = GreedyPolicy(policy, name="rl(greedy)")
    rng = np.random.default_rng(seed)  # unused by a greedy policy; kept for the rollout signature

    rows: list[EpisodeGap] = []
    for i, episode in enumerate(episodes):
        sim = run_episode(sim_env, greedy, i, rng=rng)

        writer.reset()  # per-episode anti-loop list; the call counter keeps accruing (billing)
        calls_before = writer.calls
        real = run_episode(real_env, greedy, i, rng=rng)
        union = real_env.evidence
        queries = tuple(writer.issued)

        # The terminal answer — the ONLY place the faithfulness guards can fire (the sim never
        # synthesizes). Empty union ⇒ answer_question refuses without paying for a call.
        answer = answer_question(
            episode.question, EvidenceSet(query=episode.question, chunks=union), llm=llm
        )
        n_calls = (writer.calls - calls_before) + (0 if not union else 1)

        cov_real, _, _ = coverage(union, episode.aspects)
        rows.append(
            EpisodeGap(
                question=episode.question,
                stratum=episode.stratum or "NA",
                coverage_sim=sim.coverage,
                coverage_real=cov_real,
                return_sim=sim.total_return,
                return_real=real.total_return,
                steps_sim=sim.n_retrievals,
                steps_real=real.n_retrievals,
                actions_sim=sim.actions,
                actions_real=real.actions,
                queries=queries,
                n_llm_calls=n_calls,
                insufficient=answer.insufficient_evidence,
                answer=answer.answer,
            )
        )
        log.info(
            "rl_simreal.episode",
            i=i,
            stratum=rows[-1].stratum,
            coverage_sim=round(sim.coverage, 3),
            coverage_real=round(cov_real, 3),
            llm_calls=n_calls,
        )

    cov_sim = float(np.mean([r.coverage_sim for r in rows]))
    cov_real = float(np.mean([r.coverage_real for r in rows]))
    ret_sim = float(np.mean([r.return_sim for r in rows]))
    ret_real = float(np.mean([r.return_real for r in rows]))
    return SimRealReport(
        n_episodes=len(rows),
        mean_coverage_sim=cov_sim,
        mean_coverage_real=cov_real,
        coverage_gap=cov_real - cov_sim,
        mean_return_sim=ret_sim,
        mean_return_real=ret_real,
        return_gap=ret_real - ret_sim,
        same_action_rate=float(np.mean([r.actions_sim == r.actions_real for r in rows])),
        refusal_rate=float(np.mean([r.insufficient for r in rows])),
        total_llm_calls=sum(r.n_llm_calls for r in rows),
        episodes=rows,
    )


# ---- the real-env head-to-head (A6.2g addendum) --------------------------------------------------
# The gap above is measured for ONE policy: it says the simulator is conservative, not that the
# learned policy is good. The promote rule gates on Δ = V(learned) − V(best baseline), and the $0
# eval measured that under TEMPLATED queries only. The obvious objection to the REJECT is "the
# baseline was never given real query text either" — so give it real query text and re-measure Δ.
# Both candidates then face the same environment and the comparison is apples-to-apples.
#
# Cost note: the RETURN needs no LLM (the env computes r from the coverage oracle + the cost table),
# so this run pays ONLY for the baseline's query hops — the terminal synthesis is skipped. The
# learned policy's real returns are read back from the already-paid `rl_simreal.json`.


@dataclass(frozen=True)
class RealRow:
    """One episode under one policy in the **real** env (LLM-written queries) — the paired unit."""

    question: str
    stratum: str
    group_id: str  # the bridge pair: the bootstrap resamples THESE, not rows (A6.1 D2)
    coverage: float
    total_return: float
    actions: tuple[str, ...]
    queries: tuple[str, ...]
    n_llm_calls: int


def run_real_policy(
    policy: RolloutPolicy,
    episodes: Sequence[MultiHopQuery],
    *,
    real_env: RagRetrievalEnv,
    writer: LLMQueryWriter,
    seed: int = 0,
) -> list[RealRow]:
    """Roll a policy greedily through the **real** (LLM-query) env; no synthesis, so no synth spend.

    Used for the baseline side of the head-to-head. The reward — hence the return the promote rule
    compares — is produced by the env from the gold-aspect oracle and the arm-cost table, both of
    which are `$0`; only the query writer bills. Labels score coverage; the policy never sees them.
    """
    if len(real_env.episodes) != len(episodes):
        raise ValueError("real_env must be built over exactly these episodes")
    greedy = GreedyPolicy(policy, name="baseline(greedy)")
    rng = np.random.default_rng(seed)  # unused by a greedy policy; kept for the rollout signature
    rows: list[RealRow] = []
    for i, episode in enumerate(episodes):
        writer.reset()
        before = writer.calls
        out = run_episode(real_env, greedy, i, rng=rng)
        cov, _, _ = coverage(real_env.evidence, episode.aspects)
        rows.append(
            RealRow(
                question=episode.question,
                stratum=episode.stratum or "NA",
                group_id=episode.group_id or episode.question,
                coverage=cov,
                total_return=out.total_return,
                actions=out.actions,
                queries=tuple(writer.issued),
                n_llm_calls=writer.calls - before,
            )
        )
        log.info(
            "rl_simreal.baseline_episode",
            i=i,
            stratum=rows[-1].stratum,
            coverage=round(cov, 3),
            llm_calls=rows[-1].n_llm_calls,
        )
    return rows


@dataclass(frozen=True)
class HeadToHeadReport:
    """Paired Δ = V(A) − V(B) with both policies run under **real** query text.

    ``delta`` is the mean of the per-episode differences (paired: the same question, the same env,
    so episode difficulty — the dominant variance term — cancels). The CI is a **group** bootstrap:
    it resamples bridge *pairs*, because the two questions of a pair share a corpus and are not
    independent draws. Same estimator and the same pre-registered gates as the `$0` eval, so the two
    verdicts are directly comparable.
    """

    name_a: str
    name_b: str
    n_episodes: int
    mean_return_a: float
    mean_return_b: float
    delta: float  # paired mean of (a − b): > 0 ⇒ A wins
    ci_low: float
    ci_high: float
    p_delta_positive: float
    mean_coverage_a: float
    mean_coverage_b: float
    strata: dict[str, tuple[int, float]]  # stratum → (n, paired Δ)
    total_llm_calls: int  # spent by THIS run (the baseline's query hops only)
    rows: list[dict[str, Any]]

    @property
    def wins(self) -> bool:
        """The promote gate this run can speak to: a strictly positive delta whose CI clears 0."""
        return self.delta > 0.0 and self.ci_low > 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n_episodes": self.n_episodes,
            "mean_return_a": self.mean_return_a,
            "mean_return_b": self.mean_return_b,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_delta_positive": self.p_delta_positive,
            "mean_coverage_a": self.mean_coverage_a,
            "mean_coverage_b": self.mean_coverage_b,
            "strata": {k: {"n": n, "delta": d} for k, (n, d) in self.strata.items()},
            "total_llm_calls": self.total_llm_calls,
            "rows": self.rows,
        }


def head_to_head(
    rows_a: Sequence[RealRow],
    rows_b: Sequence[RealRow],
    *,
    name_a: str,
    name_b: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> HeadToHeadReport:
    """Pair two real-env runs episode-by-episode and bootstrap the delta over bridge groups.

    Hard-fails if the two runs are not over the same episodes in the same order — the pairing (and
    therefore the whole comparison) is meaningless otherwise, and a silent misalignment would look
    like a result.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(f"unpaired runs: {len(rows_a)} vs {len(rows_b)} episodes")
    for a, b in zip(rows_a, rows_b, strict=True):
        if a.question != b.question:
            raise ValueError(f"episode misalignment: {a.question!r} vs {b.question!r}")

    pairs = list(zip(rows_a, rows_b, strict=True))
    deltas = np.array([a.total_return - b.total_return for a, b in pairs])
    _, groups = np.unique(np.array([a.group_id for a in rows_a]), return_inverse=True)
    lo, hi, p_pos = bootstrap_delta_stats(
        lambda idx: float(deltas[idx].mean()),
        len(deltas),
        groups=groups.astype(int),
        n_boot=n_boot,
        seed=seed,
    )

    strata: dict[str, tuple[int, float]] = {}
    for name in sorted({a.stratum for a in rows_a}):
        sel = [i for i, a in enumerate(rows_a) if a.stratum == name]
        strata[name] = (len(sel), float(deltas[sel].mean()))

    return HeadToHeadReport(
        name_a=name_a,
        name_b=name_b,
        n_episodes=len(rows_a),
        mean_return_a=float(np.mean([a.total_return for a in rows_a])),
        mean_return_b=float(np.mean([b.total_return for b in rows_b])),
        delta=float(deltas.mean()),
        ci_low=lo,
        ci_high=hi,
        p_delta_positive=p_pos,
        mean_coverage_a=float(np.mean([a.coverage for a in rows_a])),
        mean_coverage_b=float(np.mean([b.coverage for b in rows_b])),
        strata=strata,
        total_llm_calls=sum(b.n_llm_calls for b in rows_b),
        rows=[
            {
                "question": a.question,
                "stratum": a.stratum,
                "return_a": a.total_return,
                "return_b": b.total_return,
                "delta": a.total_return - b.total_return,
                "coverage_a": a.coverage,
                "coverage_b": b.coverage,
                "actions_b": list(b.actions),
                "queries_b": list(b.queries),
            }
            for a, b in zip(rows_a, rows_b, strict=True)
        ],
    )


def format_h2h_markdown(report: HeadToHeadReport) -> str:
    """Console summary of the real-env head-to-head + the gate it can speak to."""
    lines = [
        f"Real-env head-to-head — {report.n_episodes} episodes, "
        f"{report.total_llm_calls} LLM calls (baseline side only)",
        "",
        "| policy | coverage Φ | return (real queries) |",
        "|---|---|---|",
        f"| {report.name_a} | {report.mean_coverage_a:.3f} | {report.mean_return_a:+.4f} |",
        f"| {report.name_b} | {report.mean_coverage_b:.3f} | {report.mean_return_b:+.4f} |",
        "",
        f"Paired Δ = {report.delta:+.4f}  "
        f"95% group-bootstrap CI [{report.ci_low:+.4f}, {report.ci_high:+.4f}]  "
        f"P(Δ>0) = {report.p_delta_positive:.3f}",
        "",
        "| stratum | n | paired Δ |",
        "|---|---|---|",
        *[f"| {k} | {n} | {d:+.4f} |" for k, (n, d) in report.strata.items()],
        "",
        f"Gate (Δ > 0 and CI_low > 0): {'PASS' if report.wins else 'FAIL'}",
    ]
    return "\n".join(lines)


def format_simreal_markdown(report: SimRealReport) -> str:
    """Console summary: the gap, its sign, the action-stability check, and the grounding outcome."""
    sign = (
        "the $0 sim UNDERSTATED the policy (LLM queries retrieve better than the template)"
        if report.coverage_gap > 0
        else "the $0 sim FLATTERED the policy (its decisions do not fully survive real query text)"
        if report.coverage_gap < 0
        else "no gap"
    )
    lines = [
        f"Sim-to-real gap — {report.n_episodes} episodes, {report.total_llm_calls} LLM calls",
        "",
        "| metric | $0 sim (templated) | real (LLM-written) | gap |",
        "|---|---|---|---|",
        f"| coverage Φ(s_T) | {report.mean_coverage_sim:.3f} | {report.mean_coverage_real:.3f} "
        f"| {report.coverage_gap:+.3f} |",
        f"| return | {report.mean_return_sim:+.4f} | {report.mean_return_real:+.4f} "
        f"| {report.return_gap:+.4f} |",
        "",
        f"Action sequence unchanged by real queries: {report.same_action_rate:.0%} of episodes",
        f"Guarded synthesis refused (insufficient evidence): {report.refusal_rate:.0%}",
        "",
        f"Interpretation: {sign}.",
    ]
    return "\n".join(lines)


def write_report(report: SimRealReport, path: Any) -> None:
    """Persist the full report (per-episode rows incl. the LLM's actual queries) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
