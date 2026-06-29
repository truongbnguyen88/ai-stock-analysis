"""Multi-step (agentic A4) evaluation — does the ReAct loop's *union* cover a multi-hop question?

The A1 harness (``rag/eval.py``) scores ONE retrieval (a ranked list). A4 is different: it gathers a
**union of evidence across several hops** and ends in a cited answer. So this module measures the
property A4 exists for — that the accumulated union covers evidence a single retrieval misses — and
quantifies the **gain over a single-shot baseline** rather than asserting it.

**Unit of relevance: aspects (chunking-invariant).** A multi-hop question is labeled with K
``Aspect``s — one per hop/entity — each carrying answer-bearing **spans** (phrases the right
passage must contain), exactly the A1 ``relevant_spans`` philosophy but grouped by hop. An aspect is
*covered* iff some union chunk contains any of its spans; **aspect coverage** = covered / K. Choose
spans jointly entity-and-topic specific, so covering an aspect really means the right hop's evidence
was gathered (a generic span shared across aspects would over-credit).

**Metrics per question:**
- ``multistep_coverage`` — aspect coverage of the loop's accumulated union (the headline).
- ``single_shot_coverage`` — aspect coverage of ONE ``retrieve`` over the same question (baseline).
- ``coverage_gain`` = multistep − single_shot — the empirical value of the extra hops.
- ``citation_accuracy`` — fraction of the answer's citations whose chunk contains a relevant span
  (None when the answer makes no citations — excluded from the mean, like A1).
- ``n_steps`` / ``n_evidence`` — loop behavior.

Layering: depends on ``research/agentic`` (the loop, LLM) + ``rag`` schemas — never inverted. The
metric functions are pure; ``evaluate_multihop`` needs an ``llm`` + ``retriever`` (injected: a
canned ``TextLLM`` + fake store offline; the real Anthropic client + corpus in the CLI).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from stock_agent.llm.client import TextLLM
from stock_agent.rag.read_path import build_retrieval_system
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.research.agentic import answer_multistep
from stock_agent.schemas.research import GroundedAnswer
from stock_agent.schemas.retrieval import RetrievedChunk
from stock_agent.settings import Settings

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace (mirrors ``rag.eval._normalize`` so span matching agrees)."""
    return _WS.sub(" ", text).strip().lower()


class Aspect(BaseModel):
    """One hop/entity the multi-hop answer must cover, identified by answer-bearing spans.

    Covered iff some union chunk contains any span. Pick spans that are specific to THIS hop (e.g. a
    company name + a topic phrase), so coverage means the right evidence was gathered.
    """

    name: str
    spans: list[str] = Field(min_length=1)


Stratum = Literal["HARD", "MED", "CTRL"]


class MultiHopQuery(BaseModel):
    """A labeled multi-hop question: the prompt + the aspects the answer's evidence must cover.

    The optional fields below are **A6.0 generation metadata** (graph-mined benchmark); they default
    so the hand-written P-set / A4 examples (which omit them) still validate unchanged. ``stratum``
    is the difficulty/role bucket (HARD bridge / MED co-disclosed / CTRL single-entity).
    ``group_id`` is the leakage-safe split key (the bridge pair, so its variants never split folds).
    """

    question: str
    aspects: list[Aspect] = Field(min_length=1)
    max_steps: int | None = None  # optional per-query cap (else the agentic_max_steps default)

    # --- A6.0 metadata (optional; absent on the hand-written set) ---
    stratum: Stratum | None = None
    relation: str | None = None  # "depends_on" / "competes_with" / "single-entity"
    qtype: str | None = None  # "bridging" / "risk" / "business" / "overview"
    seed: str | None = None  # the subject ticker (the filing the question starts from)
    target: str | None = None  # the bridged-to ticker (None for CTRL single-entity)
    group_id: str | None = None  # leakage-safe split key (bridge pair, or seed for CTRL)
    generated: bool = False  # True iff produced by research/multistep_gen (vs hand-written)


class MultiHopReport(BaseModel):
    """Per-question multi-step eval result (union coverage vs. the single-shot baseline)."""

    question: str
    n_steps: int
    n_evidence: int
    multistep_coverage: float
    single_shot_coverage: float
    coverage_gain: float
    covered_aspects: list[str]
    missed_aspects: list[str]
    citation_accuracy: float | None
    insufficient_evidence: bool


class MultiHopSummary(BaseModel):
    """Aggregate (mean) multi-step metrics across the labeled set."""

    n_queries: int
    mean_multistep_coverage: float
    mean_single_shot_coverage: float
    mean_coverage_gain: float
    mean_citation_accuracy: float | None  # over questions that cited (None if none did)
    mean_n_steps: float
    mean_n_evidence: float


# ---- pure metrics -------------------------------------------------------------
def spans_present(texts: Sequence[str], spans: Sequence[str]) -> bool:
    """True iff any of ``spans`` appears as a normalized substring of any of ``texts``.

    The single span-matching primitive — **shared** by the coverage metric (``_aspect_covered``) and
    the A6.0 span-isolation probe (``research/multistep_gen``). Sharing it is a correctness
    requirement, not DRY convenience: if the generator's "present/absent" test and the eval's
    "covered" test ever diverged, a row labeled HARD/absent-in-seed could be scored *covered* by the
    single-shot baseline at eval time — a corrupt reward label. One function ⇒ probe == metric.
    """
    haystacks = [_normalize(t) for t in texts]
    return any(_normalize(span) in hay for span in spans for hay in haystacks)


def _aspect_covered(chunks: Sequence[RetrievedChunk], spans: Sequence[str]) -> bool:
    """True iff some chunk's text contains any of ``spans`` (case/whitespace-insensitive)."""
    return spans_present([rc.chunk.text for rc in chunks], spans)


def coverage(
    chunks: Sequence[RetrievedChunk], aspects: Sequence[Aspect]
) -> tuple[float, list[str], list[str]]:
    """Aspect coverage of ``chunks``: (fraction covered, covered names, missed names).

    An aspect is covered iff some chunk contains any of its spans. Returns 0.0 for an empty aspect
    list (degenerate / mislabeled query).
    """
    covered: list[str] = []
    missed: list[str] = []
    for aspect in aspects:
        (covered if _aspect_covered(chunks, aspect.spans) else missed).append(aspect.name)
    frac = len(covered) / len(aspects) if aspects else 0.0
    return frac, covered, missed


def multihop_citation_accuracy(
    answer: GroundedAnswer, union: Sequence[RetrievedChunk], aspects: Sequence[Aspect]
) -> float | None:
    """Citation precision of the terminal answer: fraction pointing to a chunk with a relevant span.

    Relevant = the cited chunk (resolved within the union) contains any aspect span. Returns None
    when the answer makes no citations (an honest refusal / no claim), so the aggregate excludes it.
    """
    if not answer.citations:
        return None
    all_spans = [span for aspect in aspects for span in aspect.spans]
    by_id = {rc.chunk.chunk_id: rc.chunk for rc in union}
    good = 0
    for cite in answer.citations:
        chunk = by_id.get(cite.chunk_id)
        if chunk is not None and any(_normalize(s) in _normalize(chunk.text) for s in all_spans):
            good += 1
    return good / len(answer.citations)


# ---- evaluation ---------------------------------------------------------------
def evaluate_multihop(
    query: MultiHopQuery,
    *,
    settings: Settings,
    llm: TextLLM,
    retriever: RetrievalSystem,
    top_k: int | None = None,
) -> MultiHopReport:
    """Run the A4 loop on ``query`` and score the union vs. a single-shot baseline.

    The loop's accumulated union (``MultiStepAnswer.evidence``) gives the multi-step coverage; one
    ``retrieve`` of the same question (``top_k`` defaults to ``settings.rag_top_k``; unscoped — the
    same starting point the loop has) gives the baseline. ``retriever`` is injected for both so the
    comparison holds the retrieval backend fixed (only the *number of hops* differs).
    """
    result = answer_multistep(
        query.question,
        settings=settings,
        llm=llm,
        retriever=retriever,
        max_steps=query.max_steps,
    )
    ms_cov, covered, missed = coverage(result.evidence, query.aspects)

    k = top_k if top_k is not None else settings.rag_top_k
    baseline = retriever.retrieve(query.question, top_k=k)
    ss_cov, _, _ = coverage(baseline.chunks, query.aspects)

    cite_acc = multihop_citation_accuracy(result.answer, result.evidence, query.aspects)
    return MultiHopReport(
        question=query.question,
        n_steps=result.n_steps,
        n_evidence=result.n_evidence,
        multistep_coverage=ms_cov,
        single_shot_coverage=ss_cov,
        coverage_gain=ms_cov - ss_cov,
        covered_aspects=covered,
        missed_aspects=missed,
        citation_accuracy=cite_acc,
        insufficient_evidence=result.answer.insufficient_evidence,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_multihop_set(
    queries: Sequence[MultiHopQuery],
    *,
    settings: Settings,
    llm: TextLLM,
    retriever: RetrievalSystem | None = None,
    top_k: int | None = None,
) -> tuple[list[MultiHopReport], MultiHopSummary]:
    """Score a labeled multi-hop set → (per-question reports, aggregate summary).

    ``retriever`` defaults to the configured read path (hybrid in prod); inject a fake offline. The
    citation-accuracy mean excludes questions whose answer made no citation (value None).
    """
    retriever = retriever or build_retrieval_system(settings)
    reports = [
        evaluate_multihop(q, settings=settings, llm=llm, retriever=retriever, top_k=top_k)
        for q in queries
    ]
    cited = [r.citation_accuracy for r in reports if r.citation_accuracy is not None]
    summary = MultiHopSummary(
        n_queries=len(reports),
        mean_multistep_coverage=_mean([r.multistep_coverage for r in reports]),
        mean_single_shot_coverage=_mean([r.single_shot_coverage for r in reports]),
        mean_coverage_gain=_mean([r.coverage_gain for r in reports]),
        mean_citation_accuracy=_mean(cited) if cited else None,
        mean_n_steps=_mean([float(r.n_steps) for r in reports]),
        mean_n_evidence=_mean([float(r.n_evidence) for r in reports]),
    )
    return reports, summary


def format_multihop_markdown(
    reports: Sequence[MultiHopReport], summary: MultiHopSummary
) -> str:
    """Render the per-question table + the aggregate line as GitHub-flavored Markdown."""
    if not reports:
        return "_no multi-hop queries evaluated_"
    lines = [
        "| question | steps | union | multi-cov | single-cov | gain | cite-acc | missed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        cite = "—" if r.citation_accuracy is None else f"{r.citation_accuracy:.2f}"
        missed = ", ".join(r.missed_aspects) if r.missed_aspects else "—"
        q = r.question if len(r.question) <= 60 else r.question[:57] + "…"
        lines.append(
            f"| {q} | {r.n_steps} | {r.n_evidence} | {r.multistep_coverage:.2f} | "
            f"{r.single_shot_coverage:.2f} | {r.coverage_gain:+.2f} | {cite} | {missed} |"
        )
    s = summary
    cite = "—" if s.mean_citation_accuracy is None else f"{s.mean_citation_accuracy:.2f}"
    lines += [
        "",
        f"**Mean** over {s.n_queries}: multi-cov {s.mean_multistep_coverage:.2f} vs "
        f"single-cov {s.mean_single_shot_coverage:.2f} (**gain {s.mean_coverage_gain:+.2f}**); "
        f"cite-acc {cite}; {s.mean_n_steps:.1f} steps, {s.mean_n_evidence:.1f} evidence chunks.",
    ]
    return "\n".join(lines)
