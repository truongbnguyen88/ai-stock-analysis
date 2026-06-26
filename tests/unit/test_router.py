"""Hybrid routing (deterministic fast-path + LLM-routing fallback) — offline, no network.

Deterministic dispatch is verified to (a) hit the right tool with the right structured args and
(b) make NO routing LLM call (a fake ToolLLM that explodes if used is injected). The LLM path is
verified to delegate to ``run_agent`` with a canned ``ToolLLM``.
"""

from __future__ import annotations

from typing import Any

import pytest

from stock_agent.agent.router import (
    ROUTE_NAMES,
    ROUTES,
    Router,
    RouterError,
)
from stock_agent.agent.runtime import ToolResponse
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor


# ---- doubles -----------------------------------------------------------------
class _RecordingExecutor:
    """Captures (tool_name, args) and returns a canned result; never touches a network/LLM."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._result = result if result is not None else {"ok": True}

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        return dict(self._result)


class _ExplodingLLM:
    """A ToolLLM that must NEVER be called — proves the deterministic path does no LLM routing."""

    def create(self, **_: Any) -> ToolResponse:
        raise AssertionError("the deterministic path must not call the routing LLM")


class _FinalLLM:
    """A ToolLLM that returns a final answer immediately (no tool_uses) — for the LLM path."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ToolResponse:
        self.calls += 1
        return ToolResponse(
            text=self.text, tool_uses=[], stop_reason="end_turn", assistant_content=[]
        )


def _router(executor: Any, *, tool_llm: Any = None) -> Router:
    # The fakes are structural stand-ins for ToolExecutor / ToolLLM (duck-typed at runtime).
    return Router(executor, tool_llm=tool_llm)


# ---- registry well-formedness ------------------------------------------------
def test_registry_well_formed() -> None:
    tool_names = {t["name"] for t in TOOL_SCHEMAS}
    assert tuple(ROUTES) == ROUTE_NAMES  # names tuple mirrors the dict
    for name, spec in ROUTES.items():
        assert spec.name == name  # key == route name
        assert spec.tool in tool_names, f"{name} -> unknown tool {spec.tool}"  # catches typos
        assert spec.blurb.strip()
        assert set(spec.needs) <= {"ticker", "question", "topic"}


# ---- deterministic dispatch: right tool + right args, NO LLM ------------------
def test_forecast_route_builds_args_and_skips_llm() -> None:
    ex = _RecordingExecutor({"prob_up": 0.6})
    r = _router(ex, tool_llm=_ExplodingLLM())  # exploding LLM proves no routing call
    out = r.run("ignored text", route="forecast", ticker="nvda", horizon=20)
    assert ex.calls == [("run_forecast", {"ticker": "NVDA", "horizon_days": 20})]
    assert out.mode == "deterministic" and out.route == "forecast"
    assert out.tool_calls == ["run_forecast"]
    assert out.structured == {"prob_up": 0.6}


def test_ticker_is_normalized() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("", route="technicals", ticker="  nvda ")
    assert ex.calls[0][1]["ticker"] == "NVDA"


def test_forecast_defaults_horizon_and_omits_unset_model() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("", route="forecast", ticker="NVDA")  # no horizon, no model
    name, args = ex.calls[0]
    assert name == "run_forecast"
    assert args == {"ticker": "NVDA", "horizon_days": 20}  # default horizon, model omitted


def test_optional_params_passed_through_only_when_set() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("", route="news", ticker="NVDA", days=7)
    assert ex.calls[0] == ("summarize_news", {"ticker": "NVDA", "days": 7})
    ex2 = _RecordingExecutor()
    _router(ex2).run("", route="news", ticker="NVDA")  # days unset -> omitted
    assert ex2.calls[0] == ("summarize_news", {"ticker": "NVDA"})


def test_big_move_passes_model_and_horizon() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("", route="big_move", ticker="NVDA", horizon=30, model="lightgbm")
    assert ex.calls[0] == (
        "get_large_move",
        {"ticker": "NVDA", "horizon_days": 30, "model": "lightgbm"},
    )


# ---- request text becomes the question/topic ---------------------------------
def test_filings_uses_request_as_question() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("What are the risk factors?", route="filings", ticker="NVDA")
    assert ex.calls[0] == (
        "search_filings",
        {"ticker": "NVDA", "question": "What are the risk factors?"},
    )


def test_multistep_uses_request_as_question() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("Compare NVDA and AMD risks", route="filings_multihop")
    assert ex.calls[0] == ("research_multistep", {"question": "Compare NVDA and AMD risks"})


def test_theme_news_uses_request_as_topic() -> None:
    ex = _RecordingExecutor()
    _router(ex).run("robotics", route="theme_news", days=10)
    assert ex.calls[0] == ("analyze_topic_news", {"topic": "robotics", "days": 10})


# ---- validation errors -------------------------------------------------------
def test_unknown_route_errors() -> None:
    with pytest.raises(RouterError, match="unknown route"):
        _router(_RecordingExecutor()).run("", route="nope", ticker="NVDA")


def test_missing_ticker_errors() -> None:
    with pytest.raises(RouterError, match="requires a ticker"):
        _router(_RecordingExecutor()).run("", route="forecast")


def test_blank_question_errors() -> None:
    with pytest.raises(RouterError, match="requires a question"):
        _router(_RecordingExecutor()).run("   ", route="filings", ticker="NVDA")


def test_blank_topic_errors() -> None:
    with pytest.raises(RouterError, match="requires a topic"):
        _router(_RecordingExecutor()).run("", route="theme_news")


# ---- render -------------------------------------------------------------------
def test_render_prefers_answer_then_error() -> None:
    answered = _RecordingExecutor({"answer": "Revenue rose [1].", "citations": []})
    assert _router(answered).run("q", route="filings", ticker="NVDA").text == "Revenue rose [1]."
    errored = _RecordingExecutor({"error": "no data"})
    assert _router(errored).run("", route="technicals", ticker="NVDA").text == "Error: no data"


def test_render_numeric_falls_back_to_json() -> None:
    ex = _RecordingExecutor({"prob_up": 0.6})
    text = _router(ex).run("", route="forecast", ticker="NVDA").text
    assert '"prob_up": 0.6' in text  # numeric tools surface their structured result verbatim


# ---- LLM path: delegates to run_agent ----------------------------------------
def test_auto_route_delegates_to_run_agent() -> None:
    ex = _RecordingExecutor()
    llm = _FinalLLM("Here is the answer.")
    out = _router(ex, tool_llm=llm).run("anything complex", route=None)
    assert out.mode == "llm"
    assert out.text == "Here is the answer."
    assert llm.calls == 1
    assert ex.calls == []  # the final answer used no tools


def test_explicit_auto_string_is_llm_path() -> None:
    llm = _FinalLLM("ok")
    out = _router(_RecordingExecutor(), tool_llm=llm).run("q", route="auto")
    assert out.mode == "llm" and llm.calls == 1


def test_llm_path_without_tool_llm_errors() -> None:
    with pytest.raises(RouterError, match="requires a tool LLM"):
        _router(_RecordingExecutor()).run("q")  # route=None, no tool_llm configured


# ---- the real ToolExecutor satisfies the Router's expectations ---------------
def test_router_accepts_real_executor_type() -> None:
    # Construction-only smoke check: a real ToolExecutor is a valid Router dependency (no calls).
    from stock_agent.settings import Settings

    r = Router(ToolExecutor(Settings(_env_file=None), llm=None))
    assert isinstance(r, Router)
