"""Production-integration plumbing for the insider feature: config gate, hardened
provider construction, and the retry helper (all offline)."""

from __future__ import annotations

import pytest

from stock_agent.data.insider import build_hardened_sec_provider, with_retry
from stock_agent.providers.base import ProviderUnavailable, SymbolNotFound
from stock_agent.settings import Settings


def test_feature_groups_csv_property() -> None:
    assert Settings(model_feature_groups="insider, relstr").feature_groups == ["insider", "relstr"]
    assert Settings(model_feature_groups="").feature_groups == []
    assert Settings(model_feature_groups="insider").feature_groups == ["insider"]


def test_build_hardened_provider_requires_user_agent() -> None:
    s = Settings(sec_user_agent=None)
    assert build_hardened_sec_provider(s) is None
    s2 = Settings(sec_user_agent="test-agent contact@example.com")
    provider = build_hardened_sec_provider(s2)
    assert provider is not None
    provider.close()  # owns a pooled httpx.Client


def test_with_retry_returns_on_first_success() -> None:
    assert with_retry(lambda: 42, what="x", retries=3) == 42


def test_with_retry_recovers_after_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_agent.data.insider.time.sleep", lambda _s: None)  # no real waiting
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderUnavailable("sec_edgar", "timeout")
        return "ok"

    assert with_retry(flaky, what="dl", retries=5) == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_with_retry_reraises_after_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_agent.data.insider.time.sleep", lambda _s: None)

    def always_fail() -> str:
        raise ProviderUnavailable("sec_edgar", "down")

    with pytest.raises(ProviderUnavailable):
        with_retry(always_fail, what="dl", retries=2)


def test_with_retry_does_not_retry_symbol_not_found() -> None:
    calls = {"n": 0}

    def missing() -> str:
        calls["n"] += 1
        raise SymbolNotFound("sec_edgar", "no CIK")

    with pytest.raises(SymbolNotFound):
        with_retry(missing, what="list", retries=5)
    assert calls["n"] == 1  # not transient → no retry
