"""Agent tool-use loop tests with a scripted fake ToolLLM (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from stock_agent.agent.runtime import (
    AgentGroundingError,
    ToolResponse,
    ToolUse,
    run_agent,
)
from stock_agent.agent.tools import ToolExecutor
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


class FakeToolLLM:
    """Returns scripted ToolResponses in order; records the last messages list."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0
        self._last_messages: list[dict[str, object]] = []

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        self._last_messages = list(messages)
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _executor(n_bars: int = 60) -> ToolExecutor:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=100.0 + i + 0.5,
            low=100.0 + i - 0.5,
            close=100.0 + i,
        )
        for i in range(n_bars)
    ]
    fake = FakeProvider("fake", prices=PriceSeries(ticker="NVDA", bars=bars))
    registry = ProviderRegistry([fake], Settings(_env_file=None, provider_price_priority="fake"))
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


def test_tool_loop_executes_then_answers() -> None:
    script = [
        ToolResponse(
            text="",
            tool_uses=[
                ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA", "horizon_days": 20})
            ],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        _final("Analysis complete for NVDA over a 20 trading-day horizon."),  # no decimals/percents
    ]
    llm = FakeToolLLM(script)
    result = run_agent("forecast NVDA 20 days", llm=llm, executor=_executor())
    assert "run_forecast" in result.tool_calls
    assert result.iterations == 2
    assert "NVDA" in result.text
    # messages contains the full turn (user + tool calls + assistant) for next-turn history.
    assert any(m["role"] == "assistant" for m in result.messages)


def test_stateful_history_is_threaded() -> None:
    # Turn 1: simple answer with no tools.
    t1 = _final("NVDA closed at a specific price yesterday.")
    result1 = run_agent("What did NVDA close at?", llm=FakeToolLLM([t1]), executor=_executor())

    # Turn 2: pass turn-1 messages as history; the fake LLM receives a longer transcript.
    t2 = _final("Based on my earlier answer, NVDA is still the ticker.")
    llm2 = FakeToolLLM([t2])
    run_agent(
        "What ticker were we discussing?",
        llm=llm2,
        executor=_executor(),
        history=result1.messages,
    )
    # The messages passed to the LLM on turn 2 must include the prior user + assistant turns.
    assert len(llm2._last_messages) > 2  # history + new user message


def test_tool_results_are_surfaced_for_charting() -> None:
    # The structured result of each tool call this turn is exposed on AgentResult
    # (for the UI to chart), in call order, with name + input + result.
    script = [
        ToolResponse(
            text="",
            tool_uses=[
                ToolUse(id="1", name="run_forecast", input={"ticker": "NVDA", "horizon_days": 20})
            ],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        _final("Forecast done for NVDA over twenty trading days."),  # no decimals
    ]
    result = run_agent("forecast NVDA", llm=FakeToolLLM(script), executor=_executor())
    assert [inv.name for inv in result.tool_results] == ["run_forecast"]
    inv = result.tool_results[0]
    assert inv.input == {"ticker": "NVDA", "horizon_days": 20}
    assert inv.result.get("model_name") == "historical_sim"  # the real tool result dict


def test_history_tool_numbers_stay_grounded_next_turn() -> None:
    # Regression: a figure a tool produced in an earlier turn (e.g. a news stat
    # like "up 193%") must remain groundable in a later turn — otherwise the guard
    # wrongly blocks the agent for referencing real prior-turn numbers.
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "summarize MU news"},
        {"role": "assistant", "content": [{"type": "text", "text": "MU news summarized."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "x",
                    "content": '{"overview": "MU is up 193% over the past year.", "bullish": []}',
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "MU is up 193% this year."}]},
    ]
    # New turn restates 193% from history WITHOUT calling a tool -> must not be blocked.
    llm = FakeToolLLM([_final("As the news noted, MU is up 193% over the past year.")])
    result = run_agent("how much is MU up?", llm=llm, executor=_executor(), history=history)
    assert "193%" in result.text


def test_number_absent_from_history_still_blocked() -> None:
    # The seeding must not over-ground: a figure NO tool ever produced is still caught.
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": '{"overview": "neutral"}'}
            ],
        },
    ]
    script = [_final("There is a 77.7% chance of a rally."), _final("I cannot give that figure.")]
    result = run_agent(
        "will it rally?", llm=FakeToolLLM(script), executor=_executor(), history=history
    )
    assert "77.7%" not in result.text  # fabricated -> retried away


def test_fabricated_number_triggers_retry_then_succeeds() -> None:
    script = [
        _final("There is a 92.5% chance of a rally."),  # ungrounded (no tools called)
        _final("I cannot provide that figure without running the forecast."),
    ]
    llm = FakeToolLLM(script)
    result = run_agent("will NVDA go up?", llm=llm, executor=_executor())
    assert llm.calls == 2  # retried once
    assert "92.5%" not in result.text


def test_persistent_fabrication_raises() -> None:
    script = [_final("A 92.5% chance."), _final("Still 92.5% likely.")]
    with pytest.raises(AgentGroundingError):
        run_agent("will it go up?", llm=FakeToolLLM(script), executor=_executor())


def test_run_forecast_tool_routes_model_selection() -> None:
    ex = _executor(n_bars=260)  # enough history for Monte Carlo
    # Default model is the baseline.
    base = ex.execute("run_forecast", {"ticker": "NVDA", "horizon_days": 20})
    assert base.get("model_name") == "historical_sim"
    # Monte Carlo needs no trained artifact — routing should pick it (one of the two
    # promoted variants; GARCH routes through the identical code path).
    mc = ex.execute(
        "run_forecast", {"ticker": "NVDA", "horizon_days": 20, "model": "monte_carlo_bootstrap"}
    )
    assert "error" not in mc
    assert mc.get("model_name") == "monte_carlo_bootstrap"
    # ML without a trained artifact falls back to the baseline but keeps its name.
    ml = ex.execute("run_forecast", {"ticker": "NVDA", "horizon_days": 20, "model": "lightgbm"})
    assert "error" not in ml
    assert ml.get("model_name") == "ml_lightgbm"


def _executor_with_news() -> ToolExecutor:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
        )
        for i in range(60)
    ]
    arts = [
        Article(
            title="NVDA earnings beat estimates",
            url="https://x.com/1",
            source="alpha_vantage",
            published_at=datetime.now(UTC) - timedelta(days=1),
            sentiment=0.4,
        ),
        Article(
            title="NVDA faces regulatory probe",
            url="https://x.com/2",
            source="alpha_vantage",
            published_at=datetime.now(UTC) - timedelta(days=2),
            sentiment=-0.2,
        ),
    ]
    fake = FakeProvider(
        "fake",
        prices=PriceSeries(ticker="NVDA", bars=bars),
        news=NewsBundle(ticker="NVDA", articles=arts),
    )
    registry = ProviderRegistry(
        [fake],
        Settings(_env_file=None, provider_price_priority="fake", provider_news_priority="fake"),
    )
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def test_get_news_sentiment_tool_uses_av_by_default() -> None:
    r = _executor_with_news().execute("get_news_sentiment", {"ticker": "NVDA", "days": 14})
    assert "error" not in r
    assert r["sentiment_source"] == "alpha_vantage"
    assert r["article_count"] == 2.0
    assert r["avg_sentiment"] == pytest.approx((0.4 - 0.2) / 2)  # AV scores averaged
    assert r["has_earnings"] == 1.0  # "earnings" in a title
    assert r["has_regulatory"] == 1.0  # "regulatory" in a title


def test_compute_indicators_surfaces_data_warnings() -> None:
    # 60 bars dated 2024 loaded as "recent" → stale_data warning reaches the agent.
    r = _executor(n_bars=60).execute("compute_indicators", {"ticker": "NVDA"})
    assert "data_warnings" in r
    assert any("stale_data" in w for w in r["data_warnings"])


def test_get_earnings_context_tool() -> None:
    # FakeProvider has no earnings capability → registry raises → graceful empty context.
    r = _executor().execute("get_earnings_context", {"ticker": "NVDA", "horizon_days": 20})
    assert "error" not in r
    assert r["ticker"] == "NVDA"
    assert r["horizon_days"] == 20
    assert r["next_earnings_date"] is None  # fake provider supplies no earnings dates
