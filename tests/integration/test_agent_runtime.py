"""Agent tool-use loop tests with a scripted fake ToolLLM (no network)."""

from __future__ import annotations

from datetime import date, timedelta

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
from stock_agent.settings import Settings


class FakeToolLLM:
    """Returns scripted ToolResponses in order, ignoring the transcript."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _executor() -> ToolExecutor:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=100.0 + i + 0.5,
            low=100.0 + i - 0.5,
            close=100.0 + i,
        )
        for i in range(60)
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
