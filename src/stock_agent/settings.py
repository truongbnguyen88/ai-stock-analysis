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
    # Per-request timeout (s) and auto-retry count for the Anthropic clients. A
    # shorter timeout with several retries recovers better from transient stalls
    # (e.g. a dropped/slow connection) than one long hang — the SDK retries
    # timeouts/connection errors with exponential backoff.
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 4
    # Self-critique passes for the news summarizer (Role A): after the first
    # draft, the model reviews its own summary for completeness/balance/evidence
    # and revises. 0 disables; 1 is the default (one reflection iteration).
    news_reflection_iterations: int = 1

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
    # Theme/keyword news (Enhancement C). GDELT DOC is keyless and theme-aware so it
    # leads; Marketaux 'search' is the secondary (needs a key, skipped if absent) so
    # the chain fails over when GDELT is unavailable.
    provider_topic_priority: str = "gdelt_doc,marketaux"
    # Expand a FREE-FORM topic phrase into OR-able search keywords via the LLM
    # (curated registry themes are left untouched). Improves recall on ad-hoc
    # themes; best-effort, so it never blocks a search. Off => exact phrase only.
    news_topic_expansion: bool = True

    # ---- Provider behavior ----
    cache_dir: Path = Path(".cache")
    cache_ttl_seconds: int = 86_400
    # News is recency-sensitive: a short TTL (1h) keeps "latest news" pulls fresh intra-day
    # without hammering the news free tiers. Prices/earnings keep the long default above.
    cache_ttl_news_seconds: int = 3_600
    rate_limit_buffer: bool = True

    # ---- App ----
    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    output_dir: Path = Path("outputs")
    random_seed: int = 42
    # Post-hoc calibration of the pooled ML classifiers (per-threshold isotonic on a
    # nested holdout). Validated to cut OOS ECE without moving AUC; 0/False disables
    # it (e.g. to A/B calibrated vs raw in a backtest).
    calibrate_ml: bool = True
    # Apply the offline pooled conformal interval-correction (outputs/models/conformal.json)
    # to served CIs/VaR so the stated coverage is honest. No-op if the artifact is absent.
    conformal_intervals: bool = True

    # ---- Promote gate (scheduled-retrain data-quality guard) ----
    # The CI retrain publishes an artifact only if `verify-models` passes. Beyond the
    # structural checks, require each artifact to have trained on enough of the universe
    # so a degraded data month (e.g. yfinance returns few tickers from the runner IP)
    # can't ship near-empty models that still pass the structural checks. The ticker
    # floor is a fraction of the configured universe (auto-adapts as it grows); the row
    # floor is an absolute backstop against a short-history universe.
    verify_min_ticker_fraction: float = 0.8
    verify_min_rows: int = 80_000

    # ---- Chat history (Streamlit frontend) ----
    # Saved conversation threads (text + charts + resumable agent history). Lives
    # under the gitignored outputs/ tree; threads older than the retention window
    # are pruned when the app loads.
    chat_history_dir: Path = Path("outputs/chat_history")
    chat_history_retention_days: int = 30

    # ---- RAG (SEC-grounded equity research; see docs/RAG_TODO.md) ----
    # Optional embedding-provider keys — only needed for the matching provider.
    openai_api_key: str | None = None  # embedding_provider="openai"
    voyage_api_key: str | None = None  # embedding_provider="voyage"
    # SEC EDGAR fair-access requires a descriptive User-Agent with contact, e.g.
    # "Jane Doe jane@example.com". Required only for LIVE downloads (tests use fixtures).
    sec_user_agent: str | None = None
    # Embeddings: local fastembed/BGE by default (no torch, $0); "openai" or "voyage"
    # swap the backend behind the same Embedder Protocol with no retrieval-code change.
    embedding_provider: Literal["local", "openai", "voyage"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Chunking + retrieval defaults (balance answer quality vs token cost).
    rag_chunk_tokens: int = 900  # target chunk size (section-aware; never crosses a section)
    rag_chunk_overlap: float = 0.15  # fractional overlap between adjacent chunks
    rag_top_k: int = 8  # chunks retrieved per query (deduped before synthesis)
    # Local storage (all under the gitignored data/ tree). Raw is never overwritten.
    documents_dir: Path = Path("data/raw")  # downloaded filings
    processed_dir: Path = Path("data/processed")  # parsed text + chunks
    vector_store_dir: Path = Path("data/vectorstore")  # persistent Chroma store

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

    @property
    def topic_priority(self) -> list[str]:
        """Ordered topic-news provider fallback chain (highest priority first)."""
        return _split_csv(self.provider_topic_priority)

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
