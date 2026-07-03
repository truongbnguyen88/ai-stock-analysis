#!/usr/bin/env python3
"""Generate ``web/src/tokens.css`` from the Python design tokens (token bridge, Phase 2).

The ``--sa-*`` CSS variables in ``src/stock_agent/ui/theme.py`` are the single source of
truth. This script writes their React-side mirror so the two frontends can never drift
(PHASE2 plan §6). The actual CSS is produced by the pure, gated ``theme.web_tokens_css()``;
this file is only a thin writer. ``tests/unit/test_token_bridge.py`` fails if the committed
file is stale, so run this after any token change.

Usage:
    python scripts/gen_web_tokens.py           # write web/src/tokens.css
    python scripts/gen_web_tokens.py --check    # exit 1 if the file is out of date (no write)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``stock_agent`` importable when run as a plain script (mirrors pytest's pythonpath=src).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_agent.ui.theme import web_tokens_css  # noqa: E402 — after sys.path shim

# Repo-relative target (kept in sync with tests/unit/test_token_bridge.py).
TOKENS_PATH = Path(__file__).resolve().parent.parent / "web" / "src" / "tokens.css"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate web/src/tokens.css from Python tokens.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed file matches (exit 1 if stale); do not write.",
    )
    args = parser.parse_args()

    expected = web_tokens_css()
    if args.check:
        current = TOKENS_PATH.read_text(encoding="utf-8") if TOKENS_PATH.exists() else ""
        if current != expected:
            print(f"✗ {TOKENS_PATH} is out of date — run: python scripts/gen_web_tokens.py")
            return 1
        print(f"✓ {TOKENS_PATH} is up to date")
        return 0

    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.write_text(expected, encoding="utf-8")
    print(f"✓ wrote {TOKENS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
