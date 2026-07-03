"""Display-state (de)serialization + citation collection — pure, no UI runtime (R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stock_agent.ui.state import (
    deserialize_messages,
    serialize_messages,
    sources_from_tool_results,
)
from stock_agent.viz.charts import ChartSpec


def _chart() -> ChartSpec:
    return ChartSpec(
        title="P(up)",
        kind="bar",
        data=pd.DataFrame({"bucket": ["down", "up"], "p": [0.42, 0.58]}),
        x="bucket",
        y="p",
        y_is_percent=True,
    )


def test_serialize_flattens_charts_and_keeps_sources() -> None:
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "hi", "charts": [], "sources": []},
        {
            "role": "assistant",
            "content": "answer",
            "charts": [_chart()],
            "sources": [{"marker": 1, "label": "NVDA 10-K"}],
        },
    ]
    out = serialize_messages(msgs)
    # Empty charts/sources are omitted (matches pre-R1 on-disk shape).
    assert out[0] == {"role": "user", "content": "hi"}
    # Charts flattened to dicts; sources preserved verbatim.
    assert isinstance(out[1]["charts"][0], dict)
    assert out[1]["charts"][0]["title"] == "P(up)"
    assert out[1]["sources"] == [{"marker": 1, "label": "NVDA 10-K"}]


def test_round_trip_rebuilds_chartspec() -> None:
    original: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a", "charts": [_chart()], "sources": []},
    ]
    restored = deserialize_messages(serialize_messages(original))
    spec = restored[0]["charts"][0]
    assert isinstance(spec, ChartSpec)
    assert spec.title == "P(up)"
    assert spec.y_is_percent is True
    pd.testing.assert_frame_equal(spec.data, _chart().data)


def test_deserialize_defaults_for_missing_keys() -> None:
    restored = deserialize_messages([{"role": "assistant", "content": "a"}])
    assert restored[0]["charts"] == []
    assert restored[0]["sources"] == []
    assert restored[0]["tiles"] == []  # R4: tolerant of pre-tiles threads


def test_tiles_round_trip_and_empty_omitted() -> None:
    tiles = [{"label": "P(up)", "value": "58%", "sub": "20d", "tone": "indigo"}]
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "hi", "tiles": []},  # empty -> omitted on disk
        {"role": "assistant", "content": "a", "tiles": tiles},
    ]
    ser = serialize_messages(msgs)
    assert "tiles" not in ser[0]  # empty tiles not persisted (like charts/sources)
    assert ser[1]["tiles"] == tiles  # plain dicts survive verbatim
    restored = deserialize_messages(ser)
    assert restored[1]["tiles"] == tiles


@dataclass
class _Inv:
    """Minimal ToolInvocation stand-in (structurally satisfies the state Protocol)."""

    result: dict[str, Any]


def test_sources_dedup_and_order() -> None:
    invs = [
        _Inv(result={"citations": [
            {"marker": 1, "label": "NVDA 10-K"},
            {"marker": 2, "label": "AMD 10-Q"},
        ]}),
        _Inv(result={"citations": [{"marker": 1, "label": "NVDA 10-K"}]}),  # dup dropped
        _Inv(result={"answer": "no citations here"}),  # non-citation tool ignored
    ]
    out = sources_from_tool_results(invs)
    assert out == [
        {"marker": 1, "label": "NVDA 10-K"},
        {"marker": 2, "label": "AMD 10-Q"},
    ]


def test_sources_empty_when_no_citations() -> None:
    assert sources_from_tool_results([_Inv(result={"error": "boom"})]) == []
