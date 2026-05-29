"""Shared HTTP helper for REST providers.

Centralizes the GET + status-code -> typed-error mapping so every REST provider
(Finnhub, Marketaux, Alpha Vantage) reports failures through the same contract.
An ``httpx.Client`` can be injected for tests (e.g. via ``httpx.MockTransport``),
keeping the provider test suite fully offline.
"""

from __future__ import annotations

from typing import Any

import httpx

from stock_agent.providers.base import ProviderRateLimit, ProviderUnavailable


class HttpJson:
    """Minimal JSON-over-HTTP GET client with uniform error mapping."""

    def __init__(
        self,
        provider_name: str,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._name = provider_name
        self._timeout = timeout
        # When injected (tests), we reuse and do not own the client's lifecycle.
        self._client = client

    def get(self, url: str, params: dict[str, Any]) -> Any:
        """GET ``url`` and return parsed JSON, or raise a typed provider error."""
        try:
            if self._client is not None:
                response = self._client.get(url, params=params)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(self._name, f"request error: {exc}") from exc

        code = response.status_code
        if code == 429:
            raise ProviderRateLimit(self._name, "rate limit exceeded (HTTP 429)")
        if code in (401, 403):
            raise ProviderUnavailable(self._name, f"authentication failed (HTTP {code})")
        if code >= 500:
            raise ProviderUnavailable(self._name, f"server error (HTTP {code})")
        if code != 200:
            raise ProviderUnavailable(self._name, f"unexpected status (HTTP {code})")

        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(self._name, f"invalid JSON: {exc}") from exc

    def close(self) -> None:
        """Close an injected client if present (no-op for per-call clients)."""
        if self._client is not None:
            self._client.close()
