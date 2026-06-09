"""LLM topic keyword expansion (Enhancement C precision) — fake LLM, no network."""

from __future__ import annotations

from stock_agent.news.topic_expand import expand_topic_keywords


class FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        return self._payload


def test_expansion_prepends_phrase_and_dedupes() -> None:
    llm = FakeLLM('{"keywords": ["Iran-Israel conflict", "Iranian strikes", "iran war"]}')
    out = expand_topic_keywords("iran war", llm)
    assert out[0] == "iran war"  # original phrase first
    assert "Iran-Israel conflict" in out and "Iranian strikes" in out
    assert sum(1 for k in out if k.lower() == "iran war") == 1  # duplicate dropped


def test_expansion_caps_keyword_count() -> None:
    many = ", ".join(f'"k{i}"' for i in range(20))
    out = expand_topic_keywords("topic", FakeLLM('{"keywords": [' + many + "]}"), max_keywords=4)
    assert len(out) == 4


def test_expansion_falls_back_on_bad_json() -> None:
    out = expand_topic_keywords("robotics", FakeLLM("not json at all"))
    assert out == ("robotics",)  # safe no-op


def test_expansion_falls_back_on_missing_keywords_key() -> None:
    out = expand_topic_keywords("robotics", FakeLLM('{"other": []}'))
    assert out == ("robotics",)
