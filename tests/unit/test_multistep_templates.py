"""A6.0a — question templates + the shared span primitive + schema back-compat (offline, pure)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_agent.research.multistep_eval import MultiHopQuery, spans_present
from stock_agent.research.multistep_templates import (
    BRIDGE_RELATIONS,
    fill_bridge,
    fill_control,
)


def test_fill_bridge_depends_on_golden() -> None:
    fq = fill_bridge(
        seed_name="NVIDIA",
        target_surfaces=["Taiwan Semiconductor", "TSMC"],
        topic="earthquake",
        relation="depends_on",
    )
    assert fq.question == (
        "Among the companies NVIDIA names as key dependencies in its SEC filings, "
        "which one warns about earthquake in its own filings?"
    )
    assert fq.relation == "depends_on"
    assert fq.qtype == "bridging"
    # A1 = the seed names the target (its surface forms); A2 = the target's own topic span.
    assert [a.spans for a in fq.aspects] == [["Taiwan Semiconductor", "TSMC"], ["earthquake"]]
    assert fq.aspects[0].name == "NVIDIA names the dependency"
    assert fq.aspects[1].name == "that dependency's own earthquake disclosure"


def test_fill_bridge_competes_with_golden() -> None:
    fq = fill_bridge(
        seed_name="NVIDIA",
        target_surfaces=["Advanced Micro Devices"],
        topic="7 nm",
        relation="competes_with",
    )
    assert fq.question == (
        "Among the competitors NVIDIA names in its SEC filings, "
        "which one discloses 7 nm in its own filings?"
    )
    assert fq.aspects[0].name == "NVIDIA names the competitor"
    assert fq.aspects[1].spans == ["7 nm"]


def test_fill_bridge_rejects_unsupported_relation() -> None:
    with pytest.raises(ValueError, match="unsupported bridge relation"):
        fill_bridge(
            seed_name="NVDA", target_surfaces=["Micron"], topic="x", relation="mentions_risk"
        )


def test_fill_bridge_rejects_empty_surfaces() -> None:
    with pytest.raises(ValueError, match="target_surfaces must be non-empty"):
        fill_bridge(seed_name="NVDA", target_surfaces=[], topic="x", relation="depends_on")


def test_fill_control_golden() -> None:
    fq = fill_control(company_name="Micron", topic="DRAM")
    assert fq.question == "What does Micron disclose about DRAM in its SEC filings?"
    assert fq.relation == "single-entity"
    assert len(fq.aspects) == 1
    assert fq.aspects[0].spans == ["DRAM"]


def test_bridge_relations_are_company_to_company() -> None:
    assert set(BRIDGE_RELATIONS) == {"depends_on", "competes_with"}


# ---- the probe ≡ metric primitive --------------------------------------------
def test_spans_present_matches_normalized_substring() -> None:
    texts = ["TSMC warns about EARTHQUAKE risk in Taiwan.", "Unrelated text."]
    assert spans_present(texts, ["earthquake"]) is True  # case-insensitive
    assert spans_present(texts, ["political stability"]) is False
    # whitespace is collapsed on both sides → multi-word spans match across newlines/runs
    assert spans_present(
        ["critical   information\ninfrastructure"], ["critical information infrastructure"]
    )
    assert spans_present([], ["x"]) is False
    assert spans_present(["anything"], []) is False


# ---- schema back-compat ------------------------------------------------------
def test_existing_committed_set_still_validates() -> None:
    """The hand-written set (no A6.0 fields) must still load — defaults fill the new optionals."""
    path = Path("configs/rag_eval_multistep.json")
    rows = json.loads(path.read_text())
    queries = [MultiHopQuery.model_validate(r) for r in rows]
    assert queries, "committed multistep set should be non-empty"
    q0 = queries[0]
    assert q0.stratum is None and q0.relation is None and q0.generated is False
    assert q0.group_id is None and q0.seed is None and q0.target is None


def test_generated_style_row_round_trips() -> None:
    row = {
        "question": "Q?",
        "aspects": [{"name": "a1", "spans": ["Micron"]}, {"name": "a2", "spans": ["NAND"]}],
        "stratum": "HARD",
        "relation": "depends_on",
        "qtype": "bridging",
        "seed": "NVDA",
        "target": "MU",
        "group_id": "MU|NVDA",
        "generated": True,
    }
    q = MultiHopQuery.model_validate(row)
    assert q.stratum == "HARD" and q.target == "MU" and q.generated is True
    assert json.loads(q.model_dump_json())["group_id"] == "MU|NVDA"
