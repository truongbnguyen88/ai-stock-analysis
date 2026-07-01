"""GDELT DOC normalization + fetch via httpx.MockTransport (no live calls)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderRateLimit
from stock_agent.providers.gdelt_doc import (
    GdeltDocProvider,
    _articles_from_payload,
    _parse_seendate,
    _tone_from_payload,
)
from stock_agent.settings import Settings


class _FakeClock:
    """Manually-advanced monotonic clock for deterministic throttle tests (no real waiting)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt

_PAYLOAD: dict[str, Any] = {
    "articles": [
        {
            "url": "https://site.com/newest",
            "title": "Humanoid robot startup raises funding",
            "seendate": "20260601T120000Z",
            "domain": "site.com",
            "language": "English",
        },
        {
            "url": "https://site.com/older",
            "title": "Automation expands in factories",
            "seendate": "20260530120000",  # no T/Z variant
            "domain": "site.com",
            "language": "English",
        },
        {  # filtered out: non-English
            "url": "https://de.site.com/x",
            "title": "Roboter Nachrichten",
            "seendate": "20260601T100000Z",
            "domain": "de.site.com",
            "language": "German",
        },
        {"title": "missing url", "seendate": "20260601T120000Z", "language": "English"},
    ]
}


def test_parse_seendate_variants() -> None:
    assert _parse_seendate("20260601T120000Z").year == 2026  # type: ignore[union-attr]
    assert _parse_seendate("20260530120000").month == 5  # type: ignore[union-attr]
    assert _parse_seendate("garbage") is None


def test_normalizes_and_filters_language() -> None:
    arts = _articles_from_payload(_PAYLOAD, language="english", top_n=25)
    assert [a.title for a in arts] == [
        "Humanoid robot startup raises funding",
        "Automation expands in factories",
    ]  # German + url-less rows dropped
    assert arts[0].source == "site.com"
    assert arts[0].sentiment is None  # DOC ArtList has no tone


def test_top_n_truncates_preserving_order() -> None:
    arts = _articles_from_payload(_PAYLOAD, language="english", top_n=1)
    assert len(arts) == 1
    assert arts[0].title == "Humanoid robot startup raises funding"


def test_non_list_payload_is_empty() -> None:
    assert _articles_from_payload({"articles": "oops"}, language="english", top_n=5) == []


def _provider(
    tmp_path: Path,
    handler: Any,
    *,
    min_interval: float = 0.0,
    sleep: Any = None,
    clock: Any = None,
) -> GdeltDocProvider:
    # Reset the process-wide throttle stamp so tests don't leak spacing state into one another,
    # and default the interval to 0 (no waiting) for the non-throttle tests.
    GdeltDocProvider._last_request_at = None
    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs: dict[str, Any] = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    if clock is not None:
        kwargs["clock"] = clock
    return GdeltDocProvider(
        settings=Settings(_env_file=None, gdelt_min_interval_seconds=min_interval),
        cache=DiskCache(tmp_path, ttl_seconds=100),
        http=HttpJson("gdelt_doc", client=client),
        **kwargs,
    )


def test_fetch_via_mock_transport(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_PAYLOAD)

    provider = _provider(tmp_path, handler)
    bundle = provider.get_topic_news(
        ["robotics", "humanoid robot"], date(2026, 5, 25), date(2026, 6, 1), top_n=25
    )
    assert len(bundle) == 2
    assert seen["params"]["mode"] == "ArtList"
    assert seen["params"]["sort"] == "DateDesc"
    assert seen["params"]["startdatetime"] == "20260525000000"
    # Provider builds its OWN native query from the keywords (phrases quoted, OR-joined).
    assert seen["params"]["query"] == '(robotics OR "humanoid robot")'


def test_keyless_provider_is_available(tmp_path: Path) -> None:
    provider = GdeltDocProvider(Settings(_env_file=None), DiskCache(tmp_path, 100))
    assert provider.available() is True


# ---- topic tone (ToneChart, #3) ----------------------------------------------
_TONE_PAYLOAD = {
    "tonechart": [{"bin": -10, "count": 2}, {"bin": 0, "count": 6}, {"bin": 5, "count": 2}]
}


def test_tone_from_payload_weighted_mean() -> None:
    tone = _tone_from_payload(_TONE_PAYLOAD)
    assert tone is not None
    assert tone["avg_tone"] == -1.0  # (-10*2 + 0*6 + 5*2) / 10
    assert tone["n_articles"] == 10
    assert tone["label"] == "roughly neutral"


def test_tone_from_payload_empty_is_none() -> None:
    assert _tone_from_payload({"tonechart": []}) is None
    assert _tone_from_payload({"nope": 1}) is None


def test_get_topic_tone_via_mock_transport(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["mode"] = req.url.params.get("mode")
        return httpx.Response(200, json={"tonechart": [{"bin": 4, "count": 5}]})

    provider = _provider(tmp_path, handler)
    tone = provider.get_topic_tone(["robotics"], date(2026, 5, 25), date(2026, 6, 1))
    assert tone is not None
    assert seen["mode"] == "ToneChart"
    assert tone["avg_tone"] == 4.0
    assert tone["label"] == "net positive"


# ---- client-side throttle (avoid GDELT's ~1 req/5s IP ban) --------------------------------------


def test_throttle_spaces_successive_requests(tmp_path: Path) -> None:
    """A second DOC call within the interval sleeps out the remaining time (fake clock/sleep)."""
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(d: float) -> None:
        sleeps.append(d)
        clock.advance(d)  # simulate time passing so the next check sees the wait

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PAYLOAD)

    provider = _provider(
        tmp_path, handler, min_interval=5.0, sleep=fake_sleep, clock=clock
    )
    # First call: no prior request -> no wait; stamps t=0.
    provider.get_topic_news(["robotics"], date(2026, 5, 25), date(2026, 6, 1))
    assert sleeps == []
    # 1s later, a DIFFERENT query (cache miss) must wait out the remaining 4s of the interval.
    clock.advance(1.0)
    provider.get_topic_news(["semiconductors"], date(2026, 5, 25), date(2026, 6, 1))
    assert sleeps == [pytest.approx(4.0)]


def test_no_wait_when_interval_elapsed(tmp_path: Path) -> None:
    """No sleep when enough time has already passed since the last request."""
    clock = _FakeClock()
    sleeps: list[float] = []

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_PAYLOAD)

    provider = _provider(
        tmp_path, handler, min_interval=5.0, sleep=lambda d: sleeps.append(d), clock=clock
    )
    provider.get_topic_news(["robotics"], date(2026, 5, 25), date(2026, 6, 1))
    clock.advance(6.0)  # past the interval
    provider.get_topic_news(["semiconductors"], date(2026, 5, 25), date(2026, 6, 1))
    assert sleeps == []


def test_429_backoff_retries_once_then_succeeds(tmp_path: Path) -> None:
    """A transient 429 triggers one backoff-retry that then succeeds."""
    calls = {"n": 0}
    sleeps: list[float] = []
    clock = _FakeClock()

    def fake_sleep(d: float) -> None:
        sleeps.append(d)
        clock.advance(d)  # real time passes during the backoff, satisfying the next spacing check

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="Please limit requests to one every 5 seconds")
        return httpx.Response(200, json=_PAYLOAD)

    provider = _provider(tmp_path, handler, min_interval=5.0, sleep=fake_sleep, clock=clock)
    bundle = provider.get_topic_news(["robotics"], date(2026, 5, 25), date(2026, 6, 1))
    assert len(bundle) == 2  # the retry's 200 payload was used
    assert calls["n"] == 2  # exactly one retry
    # A single interval-length backoff; that elapsed time then covers the retry's spacing.
    assert sleeps == [pytest.approx(5.0)]


def test_persistent_429_propagates_for_failover(tmp_path: Path) -> None:
    """A 429 on both the initial call and the retry propagates so the registry can fail over."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="banned")

    provider = _provider(
        tmp_path, handler, min_interval=5.0, sleep=lambda _d: None, clock=_FakeClock()
    )
    with pytest.raises(ProviderRateLimit):
        provider.get_topic_news(["robotics"], date(2026, 5, 25), date(2026, 6, 1))
