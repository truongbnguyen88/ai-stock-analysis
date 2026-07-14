"""A6.0 multi-hop benchmark generator — mine the A5 graph into labeled, corpus-verified questions.

Builds the **episode catalog** the A6 retrieval-RL phases train/evaluate on: graph-mined bridge
pairs × topic phrases → `MultiHopQuery` rows, each auto-verified against the *ingested* corpus. The
generator (A6.0c, ``generate_multihop``) sits on three primitives defined here:

- **the span-isolation probe** (``classify_stratum``) — the correctness lynchpin. It reuses the
  eval's ``spans_present`` (probe == metric, see ``multistep_eval``) to decide a bridge's stratum:
  A2 absent in the seed ⇒ ``HARD`` (a genuine bridge), present in both ⇒ ``MED`` (co-disclosed),
  absent in the target ⇒ discard (unanswerable). Present-in-seed (A1) / present-in-target (A2) are
  otherwise guaranteed by the A5.1 edge guards, so the probe only adds the absent-in-seed test.
- **the group-wise split** (``split_multihop``, deliverable D2) — splits by ``group_id`` (the bridge
  pair), never by row, so a pair's near-duplicate variants can't straddle train/test (anti-leakage /
  anti-pseudo-replication). A6.1f / A6.2 eval protocols MUST use this, not a row-wise split.

All offline / $0: graph traversal + a corpus grep, no LLM, no network.
"""

from __future__ import annotations

import random

from pydantic import BaseModel

from stock_agent.graph.store import GraphStore
from stock_agent.rag.sparse_store import SparseStore
from stock_agent.research.multistep_eval import MultiHopQuery, Stratum, spans_present
from stock_agent.research.multistep_templates import fill_bridge, fill_control
from stock_agent.schemas.graph import Relation
from stock_agent.schemas.retrieval import ChunkFilter

# Company→company relations are bridgeable into a "which related company…" question; the topic
# relations point at the risk/regulatory node whose name becomes the A2 span. Typed as Relation so
# they pass straight to GraphStore.neighbors.
_BRIDGE_RELS: tuple[Relation, ...] = ("depends_on", "competes_with")
_TOPIC_RELS: tuple[Relation, ...] = ("mentions_risk", "exposed_to")

_STRATA: tuple[Stratum, ...] = ("HARD", "MED", "CTRL")

# Topics so generic they appear in nearly every filing — they would all classify MED (co-disclosed)
# and add no bridge signal. Normalized (lowercased) match.
_DEFAULT_STOPLIST: frozenset[str] = frozenset(
    {
        "competition",
        "risk",
        "risks",
        "regulation",
        "regulations",
        "uncertainty",
        "general economic conditions",
        "economic conditions",
        "litigation",
        "cybersecurity",
        "operations",
    }
)

# Default benchmark composition (HARD-weighted; CTRL is the regression control). Final counts float
# to clean supply — these are *targets*, never filler (deliverable D1: count is an output).
_DEFAULT_SHARES: dict[Stratum, float] = {"HARD": 0.48, "MED": 0.28, "CTRL": 0.24}


def ticker_texts(sparse: SparseStore, ticker: str) -> list[str]:
    """All chunk texts for ``ticker`` (exhaustive scan; absent-in-seed needs every chunk)."""
    return [c.text for c in sparse.iter_chunks(ChunkFilter(ticker=ticker))]


def classify_stratum(
    *, a2_spans: list[str], seed_texts: list[str], target_texts: list[str]
) -> Stratum | None:
    """Span-isolation probe (pure): the bridge's difficulty stratum, or ``None`` to discard.

    - A2 **absent in the target** → ``None``: the topic isn't actually in the bridged company's own
      filings, so the question is unanswerable (e.g. a stale alias or a topic mined from a different
      filing year). Drop it rather than emit a label nothing can satisfy.
    - A2 **absent in the seed** → ``"HARD"``: a genuine bridge — single-shot on the seed cannot
      cover A2; only retrieving the *target's* filing does.
    - A2 **present in both** → ``"MED"``: co-disclosed; single-shot may already cover A2, so the
      extra hop is worth less (still a valid, easier multi-hop item).

    Uses ``spans_present`` (the exact eval-metric matcher) so the label agrees with how coverage is
    later scored.
    """
    if not spans_present(target_texts, a2_spans):
        return None
    return "MED" if spans_present(seed_texts, a2_spans) else "HARD"


def _group_of(query: MultiHopQuery) -> str:
    """The split key: the explicit ``group_id`` (bridge pair) or, lacking one, the question."""
    return query.group_id or query.question


def split_multihop(
    queries: list[MultiHopQuery], *, test_frac: float, seed: int
) -> tuple[list[MultiHopQuery], list[MultiHopQuery]]:
    """Group-wise train/test split (deliverable D2): partition **by ``group_id``**, not by row.

    All rows sharing a ``group_id`` (the bridge pair) land on the *same* side, so the held-out set
    is never near-copies of training rows (the pseudo-replication / leakage trap of a row split).
    Deterministic given ``seed``. ``test_frac`` is the fraction of *groups* (not rows) held out.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    groups = sorted({_group_of(q) for q in queries})  # sorted → deterministic before the shuffle
    random.Random(seed).shuffle(groups)
    n_test = round(len(groups) * test_frac)
    test_groups = set(groups[:n_test])
    train = [q for q in queries if _group_of(q) not in test_groups]
    test = [q for q in queries if _group_of(q) in test_groups]
    return train, test


# ---- the generator (A6.0c, deliverable D1) -----------------------------------
class SupplyReport(BaseModel):
    """Per-stratum supply accounting (deliverable D1) — the count is an **output**, not an input.

    ``distinct`` = genuinely distinct, deduped, stoplist-/probe-filtered questions the graph yields
    *before* any per-(seed,relation) cap or target sampling (the true clean supply). ``emitted`` =
    what was actually written after the cap + target sampling. If ``emitted`` < a requested target,
    the graph simply did not supply more clean questions — we cap, never dilute (grow the graph
    universe instead). The discard counters explain where candidate triples were dropped.
    """

    seeds: list[str]
    distinct: dict[str, int]  # clean supply per stratum (pre-cap)
    emitted: dict[str, int]  # written per stratum (post-cap, post-sample)
    target_count: int | None
    per_seed_relation_cap: int | None
    rng_seed: int
    discarded_absent_in_target: int  # A2 not in the target's own filings (unanswerable)
    discarded_stoplist: int  # topic too generic
    discarded_duplicate: int  # (seed, relation, target, topic) already emitted

    def render(self) -> str:
        """One-line-per-stratum human summary for the CLI."""
        lines = [f"Clean supply over {len(self.seeds)} seeds (rng_seed={self.rng_seed}):"]
        for s in _STRATA:
            d, e = self.distinct.get(s, 0), self.emitted.get(s, 0)
            lines.append(f"  {s:<5} distinct={d:>4}  emitted={e:>4}")
        lines.append(
            f"  discarded: absent-in-target={self.discarded_absent_in_target}, "
            f"stoplist={self.discarded_stoplist}, duplicate={self.discarded_duplicate}"
        )
        if self.target_count is not None:
            shortfall = self.target_count - sum(self.emitted.values())
            if shortfall > 0:
                lines.append(
                    f"  NOTE: requested {self.target_count}, emitted {sum(self.emitted.values())} "
                    f"(short by {shortfall}); clean supply is the ceiling — grow the universe."
                )
        return "\n".join(lines)


def _bridge_pair(seed: str, target: str) -> str:
    """Order-independent group key for a (seed, target) pair → leakage-safe split unit (D2)."""
    return "|".join(sorted((seed, target)))


def _sort_key(q: MultiHopQuery) -> tuple[str, str, str, str, str]:
    """Stable ordering so generation/sampling are reproducible regardless of graph row order."""
    return (q.stratum or "", q.seed or "", q.relation or "", q.target or "", q.question)


def _display_name(graph: GraphStore, ticker: str, alias_map: dict[str, list[str]]) -> str:
    """A presentable company name for the question (surface form), falling back to the ticker."""
    entity = graph.get_entity(ticker)
    if entity is not None and entity.name:
        return entity.name
    aliases = alias_map.get(ticker)
    return aliases[0].title() if aliases else ticker


def _target_surfaces(graph: GraphStore, ticker: str, alias_map: dict[str, list[str]]) -> list[str]:
    """A1 spans for a bridged target: graph surface name ∪ alias surfaces (deduped, ordered).

    These are how the *seed's* filing refers to the target; present-in-seed is guaranteed by the
    bridge edge's A5.1 guard. More surfaces ⇒ A1 covered whenever any naming variant is retrieved.
    """
    surfaces: list[str] = []
    entity = graph.get_entity(ticker)
    if entity is not None and entity.name:
        surfaces.append(entity.name)
    for alias in alias_map.get(ticker, []):
        if alias not in surfaces:
            surfaces.append(alias)
    return surfaces


def generate_multihop(
    graph: GraphStore,
    sparse: SparseStore,
    alias_map: dict[str, list[str]],
    *,
    seeds: list[str],
    seed: int = 0,
    target_count: int | None = None,
    per_seed_relation_cap: int | None = None,
    strata_shares: dict[Stratum, float] | None = None,
    stoplist: frozenset[str] = _DEFAULT_STOPLIST,
    min_topic_chars: int = 3,
) -> tuple[list[MultiHopQuery], SupplyReport]:
    """Mine ``seeds`` × the A5 graph into stratified, corpus-verified ``MultiHopQuery`` rows.

    For each seed: enumerate its bridge edges (``depends_on`` / ``competes_with``); for each target,
    enumerate the target's own risk/topic edges, turn each (seed, target, topic) into a 2-aspect
    bridge question, and let the span-isolation probe assign HARD/MED or discard. Each seed also
    yields single-entity CTRL questions over its *own* topics (the regression control). Dedup by
    (seed, relation, target, topic); drop stoplist/short topics. Returns the rows plus a D1
    ``SupplyReport``. Deterministic given the injected stores + ``seed`` (drives target sampling).
    """
    shares = strata_shares or _DEFAULT_SHARES
    rng = random.Random(seed)
    text_cache: dict[str, list[str]] = {}

    def texts(ticker: str) -> list[str]:
        if ticker not in text_cache:
            text_cache[ticker] = ticker_texts(sparse, ticker)
        return text_cache[ticker]

    candidates: list[MultiHopQuery] = []
    seen: set[tuple[str, str, str, str]] = set()  # (seed, relation, target, normalized-topic)
    n_absent_target = n_stoplist = n_duplicate = 0

    def topic_ok(topic: str) -> bool:
        norm = topic.strip().lower()
        return len(norm) >= min_topic_chars and norm not in stoplist

    for seed_ticker in sorted(set(seeds)):  # NB: distinct from ``seed`` (the int RNG seed)
        seed_name = _display_name(graph, seed_ticker, alias_map)

        # --- bridge questions: seed -[depends_on|competes_with]-> target, about target's topic ---
        for bridge in graph.neighbors(
            seed_ticker, relations=list(_BRIDGE_RELS), hops=1, direction="out"
        ):
            target = bridge.object
            surfaces = _target_surfaces(graph, target, alias_map)
            if not surfaces or target == seed_ticker:
                continue  # no A1 span source, or a self-loop
            for topic_edge in graph.neighbors(
                target, relations=list(_TOPIC_RELS), hops=1, direction="out"
            ):
                topic_entity = graph.get_entity(topic_edge.object)
                if topic_entity is None:
                    continue
                topic = topic_entity.name
                if not topic_ok(topic):
                    n_stoplist += 1
                    continue
                key = (seed_ticker, str(bridge.relation), target, topic.strip().lower())
                if key in seen:
                    n_duplicate += 1
                    continue
                seen.add(key)
                stratum = classify_stratum(
                    a2_spans=[topic], seed_texts=texts(seed_ticker), target_texts=texts(target)
                )
                if stratum is None:
                    n_absent_target += 1
                    continue
                fq = fill_bridge(
                    seed_name=seed_name,
                    seed_ticker=seed_ticker,
                    target_surfaces=surfaces,
                    target_ticker=target,
                    topic=topic,
                    relation=bridge.relation,
                )
                candidates.append(
                    MultiHopQuery(
                        question=fq.question,
                        aspects=fq.aspects,
                        stratum=stratum,
                        relation=fq.relation,
                        qtype=fq.qtype,
                        seed=seed_ticker,
                        target=target,
                        group_id=_bridge_pair(seed_ticker, target),
                        generated=True,
                    )
                )

        # --- CTRL questions: seed's OWN topics (single-hop regression control) ---
        seed_texts = texts(seed_ticker)
        for topic_edge in graph.neighbors(
            seed_ticker, relations=list(_TOPIC_RELS), hops=1, direction="out"
        ):
            topic_entity = graph.get_entity(topic_edge.object)
            if topic_entity is None:
                continue
            topic = topic_entity.name
            if not topic_ok(topic):
                n_stoplist += 1
                continue
            key = (seed_ticker, "single-entity", seed_ticker, topic.strip().lower())
            if key in seen:
                n_duplicate += 1
                continue
            seen.add(key)
            if not spans_present(seed_texts, [topic]):  # corpus may lag the graph → verify present
                n_absent_target += 1
                continue
            fq = fill_control(company_name=seed_name, company_ticker=seed_ticker, topic=topic)
            candidates.append(
                MultiHopQuery(
                    question=fq.question,
                    aspects=fq.aspects,
                    stratum="CTRL",
                    relation=fq.relation,
                    qtype=fq.qtype,
                    seed=seed_ticker,
                    target=None,
                    group_id=seed_ticker,
                    generated=True,
                )
            )

    candidates.sort(key=_sort_key)
    distinct = {s: sum(1 for q in candidates if q.stratum == s) for s in _STRATA}

    # Per-(seed, relation) cap (diversity guard) — applied AFTER counting distinct supply.
    if per_seed_relation_cap is not None:
        counts: dict[tuple[str | None, str | None], int] = {}
        pool: list[MultiHopQuery] = []
        for q in candidates:
            ck = (q.seed, q.relation)
            if counts.get(ck, 0) < per_seed_relation_cap:
                pool.append(q)
                counts[ck] = counts.get(ck, 0) + 1
    else:
        pool = list(candidates)

    # Target sampling per stratum (never exceeds available → never dilutes).
    if target_count is None:
        emitted_qs = pool
    else:
        emitted_qs = []
        for s in _STRATA:
            avail = [q for q in pool if q.stratum == s]
            want = round(target_count * shares.get(s, 0.0))
            k = min(want, len(avail))
            emitted_qs.extend(rng.sample(avail, k) if k < len(avail) else avail)
    emitted_qs.sort(key=_sort_key)

    report = SupplyReport(
        seeds=sorted(set(seeds)),
        distinct=distinct,
        emitted={s: sum(1 for q in emitted_qs if q.stratum == s) for s in _STRATA},
        target_count=target_count,
        per_seed_relation_cap=per_seed_relation_cap,
        rng_seed=seed,
        discarded_absent_in_target=n_absent_target,
        discarded_stoplist=n_stoplist,
        discarded_duplicate=n_duplicate,
    )
    return emitted_qs, report
