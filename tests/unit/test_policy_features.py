"""A6.1b — query featurizer golden vectors + flag correctness (label-free; offline, CI).

Every expected vector is hand-computed against FEATURE_NAMES:
    [bias, n_tokens, has_ticker, n_entities, is_bridging,
     qtype_risk, qtype_financial, qtype_business, qtype_overview, qtype_bridging,
     in_graph_universe]
"""

from __future__ import annotations

from stock_agent.rag.policy_features import FEATURE_NAMES, N_FEATURES, featurize

# Small deterministic fixtures (no file I/O, no model).
ALIAS = {"NVDA": ["NVIDIA"], "AMD": ["AMD", "Advanced Micro Devices"], "MU": ["Micron"]}
UNIVERSE = {"NVDA", "AMD", "MU"}


def _vec(*args, **kwargs) -> list[float]:
    return featurize(*args, **kwargs).values.tolist()


def test_risk_scoped_single_entity() -> None:
    # "What are NVIDIA's key risks?" -> 5 tokens; NVIDIA named -> NVDA; risk cue; NVDA in universe.
    got = _vec(
        "What are NVIDIA's key risks?", ticker="NVDA", alias_map=ALIAS, graph_universe=UNIVERSE
    )
    assert got == [1, 5, 1, 1, 0, 1, 0, 0, 0, 0, 1]


def test_bridging_beats_business_in_onehot() -> None:
    # "supplier" is both a bridge and a business cue -> bridging wins the one-hot; is_bridging=1.
    q = "Which supplier that NVIDIA depends on warns about earthquakes?"  # 9 tokens
    got = _vec(q, ticker="NVDA", alias_map=ALIAS, graph_universe=UNIVERSE)
    assert got == [1, 9, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_financial_unscoped_no_entity() -> None:
    # No ticker, no company named -> has_ticker=0, n_entities=0; "gross"/"margin" -> financial.
    got = _vec("How did gross margin trend?", alias_map=ALIAS, graph_universe=UNIVERSE)
    assert got == [1, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_overview_default_and_out_of_universe_ticker() -> None:
    # Scoped to AAPL (not in universe / alias map) -> has_ticker=1 via scope; overview default.
    got = _vec("Tell me about the company", ticker="AAPL", alias_map=ALIAS, graph_universe=UNIVERSE)
    assert got == [1, 5, 1, 0, 0, 0, 0, 0, 1, 0, 0]


def test_multi_entity_compare_counts_named_tickers() -> None:
    # Two companies named -> n_entities=2; unscoped, primary = min(named)="AMD" (in universe).
    q = "Compare NVIDIA and Advanced Micro Devices margins"  # 7 tokens
    got = _vec(q, alias_map=ALIAS, graph_universe=UNIVERSE)
    assert got == [1, 7, 1, 2, 0, 0, 1, 0, 0, 0, 1]


def test_graph_universe_none_zeroes_in_universe() -> None:
    # Even with a scoped ticker, no universe supplied -> in_graph_universe=0 (graph would degrade).
    got = _vec("What are NVIDIA's risks?", ticker="NVDA", alias_map=ALIAS, graph_universe=None)
    assert got[FEATURE_NAMES.index("in_graph_universe")] == 0.0


def test_no_alias_map_disables_mentions() -> None:
    # No alias map -> cannot detect named companies -> n_entities=0; scope still sets has_ticker.
    got = featurize("Compare NVIDIA and AMD", ticker="NVDA")
    assert got.values[FEATURE_NAMES.index("n_entities")] == 0.0
    assert got.values[FEATURE_NAMES.index("has_ticker")] == 1.0


def test_exactly_one_qtype_hot_and_stable_shape() -> None:
    for q in ["risks?", "revenue growth", "product roadmap", "hello", "who are its suppliers"]:
        cv = featurize(q, alias_map=ALIAS, graph_universe=UNIVERSE)
        onehot = [
            cv.values[FEATURE_NAMES.index(f"qtype_{t}")]
            for t in ("risk", "financial", "business", "overview", "bridging")
        ]
        assert sum(onehot) == 1.0, q
        assert len(cv.values) == N_FEATURES
        assert cv.names == FEATURE_NAMES


def test_determinism_and_as_map() -> None:
    a = featurize(
        "What are NVIDIA's risks?", ticker="NVDA", alias_map=ALIAS, graph_universe=UNIVERSE
    )
    b = featurize(
        "What are NVIDIA's risks?", ticker="NVDA", alias_map=ALIAS, graph_universe=UNIVERSE
    )
    assert a.values.tolist() == b.values.tolist()
    m = a.as_map()
    assert list(m.keys()) == list(FEATURE_NAMES)
    assert m["has_ticker"] == 1.0
