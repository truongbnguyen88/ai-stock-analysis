"""Bounded, resumable warm of the EDGAR Form 4 (insider) disk cache.

Why this exists: fetching Form 4 XML for the whole training universe inline (in the
ablation / backtest) tripped EDGAR fair-access throttling — every request opened a
fresh TLS connection (per-request `httpx.Client`), producing handshake timeouts at
the 114×200-filing scale, and `fetch_insider_activity` *skips* (does not retry)
failed downloads, so they never landed in the cache. This script pre-populates the
SAME disk cache (`settings.cache_dir`, keyed by URL) the ablation reads, using:

  - a POOLED keep-alive `httpx.Client` (one TLS handshake, reused) — the main fix,
  - retry-with-exponential-backoff so transient failures actually get cached,
  - a polite ~5 req/s throttle (under SEC's 10 req/s ceiling),
  - resumability: cached XML is a no-network hit, so re-running only retries misses.

It mirrors the backtest's fetch params (`since = today − lookback`, `limit = 200`)
so the warmed URLs are exactly the ones the insider ablation will request. After a
clean run the ablation reads insider data entirely from cache (no live calls).

Usage:
  python scripts/warm_insider_cache.py            # full universe, defaults
  python scripts/warm_insider_cache.py --rps 4 --limit 200
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import httpx

from stock_agent.logging_config import configure_logging, get_logger
from stock_agent.providers._cache import DiskCache
from stock_agent.providers._http import HttpJson
from stock_agent.providers.base import ProviderError, SymbolNotFound
from stock_agent.providers.sec_edgar import _NAME, SecEdgarProvider
from stock_agent.schemas.insider import InsiderFilingRef
from stock_agent.settings import get_settings

log = get_logger(__name__)

# Matches pipelines/backtest._BACKTEST_LOOKBACK_DAYS (~6y) plus a margin so the
# warmed `since` floor is at-or-before the backtest's min(span); the newest-`limit`
# filing URLs are then identical to what the ablation requests.
_DEFAULT_LOOKBACK_DAYS = 2300


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="configs/universe.txt")
    p.add_argument("--lookback-days", type=int, default=_DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--limit", type=int, default=200, help="Form 4 filings/ticker (matches backtest)")
    p.add_argument("--rps", type=float, default=5.0, help="max requests/sec (SEC ceiling is 10)")
    p.add_argument("--retries", type=int, default=5, help="retries/request on transient failure")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request HTTP timeout (s)")
    return p.parse_args()


def _build_provider(rps: float, timeout: float) -> SecEdgarProvider:
    """SEC provider backed by a POOLED keep-alive client (the per-request-client fix)."""
    settings = get_settings()
    ua = settings.require("sec_user_agent", capability="insider cache warm")
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    # One reused connection pool → no per-download TLS handshake (the timeout cause).
    client = httpx.Client(
        timeout=timeout,
        headers=headers,
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )
    http = HttpJson(_NAME, client=client, headers=headers)
    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    return SecEdgarProvider(settings, cache, http=http, min_request_interval=1.0 / rps)


def _with_retry(fn, *, what: str, retries: int, base_backoff: float = 1.0):  # type: ignore[no-untyped-def]
    """Call ``fn`` with exponential backoff on transient ProviderError; reraise if exhausted."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except SymbolNotFound:
            raise  # not transient (e.g. an ETF with no CIK) — let caller skip it
        except ProviderError as exc:
            if attempt == retries:
                raise
            sleep = base_backoff * (2**attempt)
            log.warning("warm.retry", what=what, attempt=attempt + 1, sleep=round(sleep, 1),
                        error=str(exc))
            time.sleep(sleep)
    raise RuntimeError("unreachable")


def _load_universe(path: str) -> list[str]:
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip().upper() for s in lines if s.strip() and not s.strip().startswith("#")]


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings)
    provider = _build_provider(args.rps, args.timeout)
    tickers = _load_universe(args.universe)
    since = date.today() - timedelta(days=args.lookback_days)

    log.info("warm.start", tickers=len(tickers), since=str(since), limit=args.limit, rps=args.rps)
    tot_refs = tot_ok = tot_fail = no_cik = with_data = 0
    for i, tkr in enumerate(tickers, 1):
        try:
            refs: list[InsiderFilingRef] = _with_retry(
                lambda t=tkr: provider.list_form4_filings(t, since=since, limit=args.limit),
                what=f"list:{tkr}", retries=args.retries,
            )
        except SymbolNotFound:
            no_cik += 1
            log.info("warm.no_cik", ticker=tkr)  # expected for ETFs / non-filers
            continue
        except ProviderError as exc:
            log.warning("warm.list_failed", ticker=tkr, error=str(exc))
            continue

        ok = fail = 0
        for ref in refs:
            try:
                _with_retry(lambda r=ref: provider.download_form4(r),
                            what=f"dl:{ref.filing_id}", retries=args.retries)
                ok += 1
            except ProviderError as exc:
                fail += 1
                log.warning("warm.dl_failed", filing=ref.filing_id, error=str(exc))
        tot_refs += len(refs)
        tot_ok += ok
        tot_fail += fail
        if ok:
            with_data += 1
        log.info("warm.ticker_done", n=i, total=len(tickers), ticker=tkr,
                 refs=len(refs), ok=ok, fail=fail)

    provider.close()
    log.info(
        "warm.complete",
        tickers=len(tickers), with_data=with_data, no_cik=no_cik,
        filings=tot_refs, cached_ok=tot_ok, failed=tot_fail,
    )
    print(  # noqa: T201 — summary is the deliverable
        f"\nInsider cache warm complete:\n"
        f"  tickers: {len(tickers)} ({with_data} with Form 4 data, {no_cik} no-CIK/ETF)\n"
        f"  filings: {tot_refs} listed | {tot_ok} cached OK | {tot_fail} failed after retries"
    )


if __name__ == "__main__":
    main()
