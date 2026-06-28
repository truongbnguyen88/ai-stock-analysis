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
    # Candidate feature groups folded into PRODUCTION training on top of the baseline
    # (CSV; see features/price_features.FEATURE_GROUPS). The artifact records its
    # feature_cols, so inference auto-rebuilds the matching groups — no inference flag.
    # `insider` (Form 4) is enabled because it lifts Brier/calibration on mid/small-caps
    # (validated; nil on mega-caps) and the universe now spans that segment. It requires
    # SEC_USER_AGENT at train time (CI secret); absent it, insider columns train as NaN.
    model_feature_groups: str = "insider"

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
    # Reranking (advanced-RAG A2). "none" (default) = dense retrieval only, byte-identical to A1;
    # "local" = fastembed onnx cross-encoder (no torch, $0); "voyage" = Voyage rerank API (opt-in,
    # reuses voyage_api_key). When on, retrieval over-fetches rerank_fetch_k candidates, the
    # cross-encoder rescores them, and the top rag_top_k survive. rerank_model="" → default.
    rerank_provider: Literal["none", "local", "voyage"] = "none"
    rerank_fetch_k: int = 30  # candidates over-fetched before reranking (must be ≥ rag_top_k)
    rerank_model: str = ""  # override the per-provider default cross-encoder; "" = default
    # Hybrid search (advanced-RAG A3). "hybrid" (default, PROMOTED on a measured eval win — see
    # rag_implementation_notes "Promotion") = fuse dense (semantic) + sparse BM25 (exact terms) by
    # Reciprocal Rank Fusion; "dense" = vector-only (A1/A2 baseline). Each side over-fetches its own
    # k; RRF then ranks by sum_i 1/(rrf_k + rank_i). The sparse index is SQLite FTS5 (stdlib, $0),
    # maintained alongside the vector store at ingest. NB: hybrid needs the sparse index populated —
    # run `documents backfill-sparse` once for a pre-A3 corpus (else it falls back to dense).
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    hybrid_rrf_k: int = 60  # RRF damping constant (larger → flatter rank weighting; 60 is standard)
    hybrid_dense_k: int = 30  # dense candidates over-fetched per query before fusion
    hybrid_sparse_k: int = 30  # sparse (BM25) candidates over-fetched per query before fusion
    # Agentic RAG (advanced-RAG A4) — bounded ReAct loop for multi-hop SEC QA. Each iteration is
    # ONE cheap structured decision call (search-more vs. stop); retrieval between steps is local;
    # the single heavy call is the terminal P7 synthesis. Budget = ≤ agentic_max_steps decision
    # calls + 1 terminal answer. Default 3 ⇒ ≤4 LLM calls — kept tight (up to 3 hops covers the
    # common multi-hop shapes: 2-3-entity compare, before/after, discover-then-follow-up, compound).
    # Bounds cost + context like run_backtest; raise per call (CLI --max-steps) for deeper Qs.
    agentic_max_steps: int = 3  # max decision iterations (+1 terminal synthesis ⇒ ≤4 LLM calls)
    agentic_per_step_k: int = 6  # chunks retrieved per search step (before the union dedup)
    agentic_max_evidence: int = 20  # cap on the deduped union handed to the terminal synthesis
    # Deterministic entity-bridge (A4 bridging fix). After the loop, for a *bridging* question the
    # ReAct model reliably refuses to pivot its search to a discovered related entity (it stays on
    # the question's subject). This makes that pivot structurally: resolve company NAMES in the
    # gathered union to tickers (configs/ticker_aliases.json) and issue this many extra $0 scoped
    # retrievals into the most question-relevant of them (no extra LLM call). 0 disables.
    agentic_bridge_max_entities: int = 2
    # Client-side hard ceiling on embedding tokens per ingest run (RAG_TODO 9a). Independent
    # of the provider dashboard — protects the one-time paid voyage ingest (9c) from surprise
    # over-spend. None = unlimited (the default; local fastembed is free, so no ceiling needed).
    rag_max_embed_tokens: int | None = None
    # Local storage (all under the gitignored data/ tree). Raw is never overwritten.
    documents_dir: Path = Path("data/raw")  # downloaded filings
    processed_dir: Path = Path("data/processed")  # parsed text + chunks
    vector_store_dir: Path = Path("data/vectorstore")  # persistent Chroma store
    sparse_store_dir: Path = Path("data/sparse")  # persistent SQLite FTS5 BM25 index (A3 hybrid)
    graph_store_dir: Path = Path("data/graph")  # persistent SQLite knowledge graph (A5 GraphRAG)

    # GraphRAG (advanced-RAG A5) — a lightweight entity/relationship graph over the corpus,
    # retrieved by traversal (default-OFF; promoted only on a measured bridging win). Extraction is
    # offline, batched, and cost-gated; retrieval (traversal) is $0/local.
    graph_max_extract_calls: int | None = None  # spend ceiling on extraction LLM calls (None = ∞)
    graph_min_confidence: float = 0.5  # drop extracted edges below this self-reported confidence
    # Sections extraction reads (10-K Item 1 Business + Item 1A Risk Factors only — where supplier/
    # competitor/risk relations live). Matched by item-code, tolerant of header wording variants.
    graph_sections: list[str] = ["Item 1. Business", "Item 1A. Risk Factors"]
    graph_hops: int = 1  # GraphRetriever traversal depth (1 = direct neighbors; 2 = their edges)
    graph_max_neighbors: int = 5  # cap neighbor entities expanded into scoped vector searches
    graph_per_neighbor_k: int = 4  # chunks retrieved per neighbor in the scoped "what" search
    # A5.3 PROMOTION (measured win, 2 seeds): route the agentic multi-hop path through the
    # GraphRetriever. Graph-built tickers (configs/graph_universe.txt) get traversal-based bridging;
    # tickers without graph edges degrade to the hybrid base + the (still-ON) alias-bridge fallback.
    # Single-shot/search_filings stays hybrid (single-shot graph regressed easy Qs). Default-ON
    # since additive — non-graph corpora are unaffected (empty traversal).
    graph_multistep_enabled: bool = True

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

    @property
    def feature_groups(self) -> list[str]:
        """Candidate feature groups to include in production training (empty = baseline)."""
        return _split_csv(self.model_feature_groups)

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
