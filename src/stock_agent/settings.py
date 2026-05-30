"""Application settings — single source of configuration.

Secrets and environment-specific values come from a local ``.env`` file (see
``.env.example``); behavioral defaults live here. Built on pydantic-settings so
every value is typed and validated at load time.

Key-availability policy: API keys are OPTIONAL at construction. A missing key is
only an error when a capability that needs it is actually requested — use
``Settings.require()`` at the call site. This lets the package import, run
``--help``, and run offline tests without any credentials, while still failing
fast (with a clear, actionable message) the moment a real provider call needs a
missing key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingSettingError(RuntimeError):
    """Raised when a required setting is absent for a requested capability."""

    def __init__(self, attr: str, capability: str) -> None:
        super().__init__(
            f"Missing required setting '{attr.upper()}' needed for {capability}. "
            "Add it to your .env (see .env.example)."
        )
        self.attr = attr
        self.capability = capability


class Settings(BaseSettings):
    """Typed application configuration, populated from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    # Default to Sonnet for cost efficiency (news summarization + agent routing
    # do not need Opus-level reasoning). Override via LLM_MODEL in .env.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096

    # ---- Market / news providers ----
    alpha_vantage_api_key: str | None = None
    finnhub_api_key: str | None = None
    marketaux_api_key: str | None = None

    # Provider fallback order, comma-separated in env. Stored as raw strings and
    # parsed via the properties below: pydantic-settings would otherwise try to
    # JSON-decode list-typed fields, which breaks on plain comma syntax.
    #
    # Defaults favor generous/keyless free tiers:
    #   prices -> yfinance (no key, deep history); Alpha Vantage as fallback.
    #   news   -> Finnhub (60 req/min); Marketaux next; Alpha Vantage last
    #             (25 req/day, but uniquely gives sentiment+relevance scores).
    # Finnhub's free tier does NOT include historical OHLCV candles, so it is
    # intentionally absent from the price chain.
    provider_price_priority: str = "yfinance,alpha_vantage"
    provider_news_priority: str = "finnhub,marketaux,alpha_vantage"
    provider_earnings_priority: str = "yfinance"  # earnings dates (keyless, unlimited)

    # ---- Provider behavior ----
    cache_dir: Path = Path(".cache")
    cache_ttl_seconds: int = 86_400
    rate_limit_buffer: bool = True

    # ---- App ----
    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    output_dir: Path = Path("outputs")
    random_seed: int = 42

    @property
    def earnings_priority(self) -> list[str]:
        """Ordered earnings-provider fallback chain (highest priority first)."""
        return _split_csv(self.provider_earnings_priority)

    @property
    def price_priority(self) -> list[str]:
        """Ordered price-provider fallback chain (highest priority first)."""
        return _split_csv(self.provider_price_priority)

    @property
    def news_priority(self) -> list[str]:
        """Ordered news-provider fallback chain (highest priority first)."""
        return _split_csv(self.provider_news_priority)

    def require(self, attr: str, *, capability: str) -> str:
        """Return a required secret/value or raise a clear, actionable error.

        Call at the point a capability is invoked, e.g.::

            settings.require("alpha_vantage_api_key", capability="price data")
        """
        value = getattr(self, attr, None)
        if not value:
            raise MissingSettingError(attr, capability)
        return str(value)


def _split_csv(raw: str) -> list[str]:
    """Parse a comma-separated env value into a clean list (order preserved)."""
    return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance (loaded once)."""
    return Settings()
