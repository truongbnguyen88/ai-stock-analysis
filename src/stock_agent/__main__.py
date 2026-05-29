"""Module entry point: enables ``python -m stock_agent``.

The full CLI (analyze / forecast / backtest / chat) is implemented in Phase 4+
(`stock_agent.cli.app`). Until then this provides a minimal, honest status
command so the package is runnable and importable end-to-end.
"""

from __future__ import annotations

from stock_agent import __version__
from stock_agent.logging_config import configure_logging, get_logger
from stock_agent.settings import get_settings


def main() -> int:
    """Entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    log = get_logger(__name__)
    log.info("stock_agent.start", version=__version__, env=settings.env)
    print(
        f"stock-agent v{__version__} — CLI not implemented yet (arrives in Phase 4). "
        "See docs/ROADMAP.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
