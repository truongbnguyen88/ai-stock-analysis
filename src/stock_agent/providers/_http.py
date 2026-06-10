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
    """Minimal HTTP GET client with uniform error mapping (JSON or raw text)."""

    def __init__(
        self,
        provider_name: str,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._name = provider_name
        self._timeout = timeout
        # When injected (tests), we reuse and do not own the client's lifecycle.
        self._client = client
        # Default headers applied to every request (e.g. SEC requires a User-Agent).
        self._headers = headers

    def _fetch(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """GET ``url`` and return the response, mapping non-200 to typed errors."""
        try:
            if self._client is not None:
                response = self._client.get(url, params=params, headers=self._headers)
            else:
                with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
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
        return response

    def get(self, url: str, params: dict[str, Any]) -> Any:
        """GET ``url`` and return parsed JSON, or raise a typed provider error."""
        response = self._fetch(url, params)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderUnavailable(self._name, f"invalid JSON: {exc}") from exc

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """GET ``url`` and return the raw response text (for HTML/TXT documents)."""
        return self._fetch(url, params or {}).text

    def close(self) -> None:
        """Close an injected client if present (no-op for per-call clients)."""
        if self._client is not None:
            self._client.close()
