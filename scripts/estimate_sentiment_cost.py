"""Measure the real cost of one Claude batch sentiment call.

Fetches live news, builds the exact prompt the feature pipeline uses, makes ONE
real Anthropic call, and reports token usage + cost. This MAKES A PAID API CALL
(~a few cents) — it is a manual tool, not part of the offline test suite.

Usage:
    PYTHONPATH=src python scripts/estimate_sentiment_cost.py [TICKER] [TOP_N]
Example:
    PYTHONPATH=src python scripts/estimate_sentiment_cost.py NVDA 25
"""

from __future__ import annotations

import sys

from stock_agent.features.news_features import _SENTIMENT_SYSTEM, build_sentiment_user
from stock_agent.news.fetch import NewsFetcher
from stock_agent.providers.registry import build_default_registry
from stock_agent.settings import get_settings

# Claude Sonnet list pricing (USD per million tokens). Update if pricing changes.
PRICE_INPUT_PER_MTOK = 3.00
PRICE_OUTPUT_PER_MTOK = 15.00
PRICE_CACHE_WRITE_PER_MTOK = 3.75  # 1.25x input
PRICE_CACHE_READ_PER_MTOK = 0.30   # 0.1x input


def main() -> int:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY required.")
        return 1

    # Fetch real news and build the production prompt.
    registry = build_default_registry(settings)
    bundle = NewsFetcher(registry).fetch(ticker, lookback_days=14, top_n=top_n)
    registry.close()
    if not bundle.articles:
        print(f"No news found for {ticker}.")
        return 1

    user = build_sentiment_user(bundle.articles, ticker)

    # Direct Anthropic call so we can read usage (complete_json discards it).
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system_block = {"type": "text", "text": _SENTIMENT_SYSTEM, "cache_control": {"type": "ephemeral"}}
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=2048,
        system=[system_block],
        messages=[{"role": "user", "content": user}],
    )

    u = resp.usage
    inp = u.input_tokens
    out = u.output_tokens
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0

    cost = (
        inp / 1e6 * PRICE_INPUT_PER_MTOK
        + out / 1e6 * PRICE_OUTPUT_PER_MTOK
        + cache_write / 1e6 * PRICE_CACHE_WRITE_PER_MTOK
        + cache_read / 1e6 * PRICE_CACHE_READ_PER_MTOK
    )

    print(f"\n{'=' * 56}")
    print(f"  Batch sentiment cost — {ticker} ({settings.llm_model})")
    print(f"{'=' * 56}")
    print(f"  Articles scored      : {len(bundle.articles)}")
    print(f"  Prompt chars         : {len(user):,}")
    print(f"  Input tokens         : {inp:,}")
    print(f"  Output tokens        : {out:,}")
    if cache_write or cache_read:
        print(f"  Cache write / read   : {cache_write:,} / {cache_read:,}")
    print(f"  {'-' * 40}")
    print(f"  TOTAL COST           : ${cost:.4f}")
    print(f"  Cost per article     : ${cost / len(bundle.articles):.5f}")
    print(f"  Est. cost @ 25 arts  : ${cost / len(bundle.articles) * 25:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
