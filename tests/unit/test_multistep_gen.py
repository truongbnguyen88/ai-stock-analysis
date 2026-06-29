"""A6.0b — span-isolation probe + group-wise split (offline, deterministic)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from stock_agent.graph.store import SqliteGraphStore
from stock_agent.rag.sparse_store import InMemoryBM25Store
from stock_agent.research.multistep_eval import Aspect, MultiHopQuery
from stock_agent.research.multistep_gen import (
    classify_stratum,
    generate_multihop,
    split_multihop,
    ticker_texts,
)
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.graph import Edge, Entity


def _chunk(idx: int, text: str, *, ticker: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{ticker}:10-K:2026-02-25:{idx}",
        document_id=f"{ticker}:10-K:2026-02-25",
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type="10-K",
        source="SEC",
        source_url=f"https://sec.gov/{ticker}.htm",
        filing_date=date(2026, 2, 25),
        section="Item 1A. Risk Factors",
    )


# ---- the span-isolation probe (pure) -----------------------------------------
def test_classify_hard_when_absent_in_seed() -> None:
    # A2 = "earthquake": present in the target (TSM), absent in the seed (NVDA) → genuine bridge.
    stratum = classify_stratum(
        a2_spans=["earthquake"],
        seed_texts=["NVDA relies on Taiwan Semiconductor for fabrication."],
        target_texts=["TSMC warns about earthquake risk in Taiwan."],
    )
    assert stratum == "HARD"


def test_classify_med_when_co_disclosed() -> None:
    # A2 present in BOTH seed and target → single-shot may already cover it → MED.
    stratum = classify_stratum(
        a2_spans=["export control"],
        seed_texts=["NVDA discusses export control exposure to China."],
        target_texts=["The supplier also faces export control limits."],
    )
    assert stratum == "MED"


def test_classify_discard_when_absent_in_target() -> None:
    # A2 not in the target's own filings → unanswerable → discard (None).
    stratum = classify_stratum(
        a2_spans=["earthquake"],
        seed_texts=["NVDA names Taiwan Semiconductor."],
        target_texts=["TSMC discusses capital expenditure and pricing."],
    )
    assert stratum is None


def test_classify_uses_normalized_matching() -> None:
    # case/whitespace-insensitive (probe == metric) — uppercase + collapsed run still matches.
    stratum = classify_stratum(
        a2_spans=["critical information infrastructure"],
        seed_texts=["nothing relevant here"],
        target_texts=["subject to CRITICAL   INFORMATION\nINFRASTRUCTURE review"],
    )
    assert stratum == "HARD"


def test_ticker_texts_scans_only_that_ticker() -> None:
    store = InMemoryBM25Store()
    store.add(
        [
            _chunk(0, "NVDA depends on TSMC.", ticker="NVDA"),
            _chunk(0, "TSMC earthquake risk.", ticker="TSM"),
        ]
    )
    assert ticker_texts(store, "TSM") == ["TSMC earthquake risk."]
    assert classify_stratum(
        a2_spans=["earthquake"],
        seed_texts=ticker_texts(store, "NVDA"),
        target_texts=ticker_texts(store, "TSM"),
    ) == "HARD"


# ---- group-wise split (D2) ---------------------------------------------------
def _q(question: str, group_id: str) -> MultiHopQuery:
    return MultiHopQuery(
        question=question,
        aspects=[Aspect(name="a", spans=["x"])],
        group_id=group_id,
    )


def test_split_keeps_a_group_on_one_side() -> None:
    # Two questions share group "MU|NVDA"; they must never straddle the split.
    queries = [
        _q("q1", "MU|NVDA"),
        _q("q2", "MU|NVDA"),
        _q("q3", "AMD|NVDA"),
        _q("q4", "INTC|NVDA"),
        _q("q5", "TSM|NVDA"),
    ]
    train, test = split_multihop(queries, test_frac=0.4, seed=0)
    assert len(train) + len(test) == len(queries)
    # no overlap of group ids across the two sides
    train_groups = {q.group_id for q in train}
    test_groups = {q.group_id for q in test}
    assert train_groups.isdisjoint(test_groups)
    # the shared-group pair lands together
    sides = {q.question: ("test" if q in test else "train") for q in queries}
    assert sides["q1"] == sides["q2"]


def test_split_is_deterministic_under_seed() -> None:
    queries = [_q(f"q{i}", f"G{i}|NVDA") for i in range(10)]
    a = split_multihop(queries, test_frac=0.3, seed=7)
    b = split_multihop(queries, test_frac=0.3, seed=7)
    assert [q.question for q in a[1]] == [q.question for q in b[1]]
    # a different seed generally yields a different test set (not a hard guarantee, but holds here)
    c = split_multihop(queries, test_frac=0.3, seed=99)
    assert {q.question for q in a[1]} != {q.question for q in c[1]}


def test_split_rejects_bad_fraction() -> None:
    import pytest

    with pytest.raises(ValueError, match="test_frac"):
        split_multihop([_q("q", "g")], test_frac=0.0, seed=0)


# ---- the generator over a hand-built graph + corpus (A6.0c) -------------------
def _edge(subj: str, rel: str, obj: str) -> Edge:
    return Edge(
        subject=subj, relation=rel, object=obj, provenance=[f"{subj}:10-K:2026-02-25:0"],
        filing_date=date(2026, 2, 25), source_url=f"https://sec.gov/{subj}.htm", confidence=0.9,
    )


def _risk(node_id: str, name: str) -> Entity:
    return Entity(id=node_id, name=name, type="risk")


def _build(tmp_path: Path) -> tuple[SqliteGraphStore, InMemoryBM25Store, dict[str, list[str]]]:
    """NVDA --depends_on--> {MU, TSM}; targets carry own risks; NVDA carries its own (for CTRL)."""
    g = SqliteGraphStore(tmp_path / "graph.db")
    g.add_entities(
        [
            Entity(id="NVDA", name="NVIDIA", type="company", ticker="NVDA"),
            Entity(id="MU", name="Micron", type="company", ticker="MU"),
            Entity(id="TSM", name="Taiwan Semiconductor", type="company", ticker="TSM"),
            _risk("nand", "NAND oversupply"),
            _risk("quake", "earthquake"),
            _risk("dcd", "data center demand"),  # NVDA's own → CTRL
            _risk("comp", "competition"),  # stoplisted
            _risk("flood", "flooding"),  # absent in TSM corpus → discarded
        ]
    )
    g.add_edges(
        [
            _edge("NVDA", "depends_on", "MU"),
            _edge("NVDA", "depends_on", "TSM"),
            _edge("MU", "mentions_risk", "nand"),
            _edge("TSM", "mentions_risk", "quake"),
            _edge("TSM", "mentions_risk", "flood"),  # flooding absent from TSM's chunks
            _edge("MU", "mentions_risk", "comp"),  # competition → stoplist
            _edge("NVDA", "mentions_risk", "dcd"),  # NVDA's own → CTRL
        ]
    )
    sparse = InMemoryBM25Store()
    sparse.add(
        [
            _chunk(0, "NVIDIA depends on Micron and Taiwan Semiconductor.", ticker="NVDA"),
            _chunk(1, "Strong data center demand drove revenue.", ticker="NVDA"),
            _chunk(0, "Micron warns about NAND oversupply and competition.", ticker="MU"),
            _chunk(0, "TSMC warns about earthquake risk in Taiwan.", ticker="TSM"),
        ]
    )
    alias_map = {"NVDA": ["nvidia"], "MU": ["micron"], "TSM": ["taiwan semiconductor", "tsmc"]}
    return g, sparse, alias_map


def test_generate_strata_and_provenance(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    queries, report = generate_multihop(g, sparse, alias_map, seeds=["NVDA"], seed=0)

    # HARD: NVDA→MU (NAND oversupply) + NVDA→TSM (earthquake); CTRL: NVDA's own data center demand.
    assert report.distinct == {"HARD": 2, "MED": 0, "CTRL": 1}
    assert report.emitted == {"HARD": 2, "MED": 0, "CTRL": 1}
    # discards: flooding absent in TSM, competition stoplisted.
    assert report.discarded_absent_in_target == 1
    assert report.discarded_stoplist == 1

    by_stratum = {s: [q for q in queries if q.stratum == s] for s in ("HARD", "MED", "CTRL")}
    hard_targets = {q.target for q in by_stratum["HARD"]}
    assert hard_targets == {"MU", "TSM"}
    for q in by_stratum["HARD"]:
        assert q.generated is True and q.seed == "NVDA" and len(q.aspects) == 2
        assert q.group_id == "|".join(sorted(("NVDA", q.target)))  # type: ignore[arg-type]
        assert q.relation == "depends_on" and q.qtype == "bridging"
    ctrl = by_stratum["CTRL"][0]
    assert ctrl.target is None and ctrl.group_id == "NVDA" and len(ctrl.aspects) == 1


def test_generate_med_when_topic_co_disclosed(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    # Add a topic present in BOTH MU and NVDA → MED, not HARD.
    g.add_entities([_risk("asp", "average selling price")])
    g.add_edges([_edge("MU", "mentions_risk", "asp")])
    sparse.add(
        [
            _chunk(2, "NVIDIA notes average selling price pressure.", ticker="NVDA"),
            _chunk(1, "Micron faces average selling price declines.", ticker="MU"),
        ]
    )
    _, report = generate_multihop(g, sparse, alias_map, seeds=["NVDA"], seed=0)
    assert report.distinct["MED"] == 1
    assert report.distinct["HARD"] == 2  # unchanged


def test_generate_dedups_same_topic_via_two_relations(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    # TSM exposed_to the SAME 'quake' node it already mentions_risk → must dedup to one question.
    g.add_edges([_edge("TSM", "exposed_to", "quake")])
    _, report = generate_multihop(g, sparse, alias_map, seeds=["NVDA"], seed=0)
    assert report.distinct["HARD"] == 2  # not 3
    assert report.discarded_duplicate == 1


def test_generate_per_seed_relation_cap(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    queries, report = generate_multihop(
        g, sparse, alias_map, seeds=["NVDA"], seed=0, per_seed_relation_cap=1
    )
    assert report.distinct["HARD"] == 2  # supply unchanged
    # cap → at most 1 depends_on question emitted
    assert sum(1 for q in queries if q.relation == "depends_on") == 1
    assert report.emitted["HARD"] == 1


def test_generate_target_count_caps_never_dilutes(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    # Request 100 but only 3 clean questions exist → emit 3, report the shortfall ceiling.
    queries, report = generate_multihop(
        g, sparse, alias_map, seeds=["NVDA"], seed=0, target_count=100
    )
    assert sum(report.emitted.values()) == 3
    assert len(queries) == 3


def test_generate_is_deterministic(tmp_path: Path) -> None:
    g, sparse, alias_map = _build(tmp_path)
    a, _ = generate_multihop(g, sparse, alias_map, seeds=["NVDA"], seed=0)
    b, _ = generate_multihop(g, sparse, alias_map, seeds=["NVDA"], seed=0)
    assert [q.question for q in a] == [q.question for q in b]
    # every generated row re-validates as a MultiHopQuery (schema round-trip)
    assert all(MultiHopQuery.model_validate(q.model_dump()) for q in a)
