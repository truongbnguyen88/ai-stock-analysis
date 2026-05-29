"""Structured logging setup (structlog).

Emits JSON logs in production (machine-parseable; good for experiment/audit
trails) and human-friendly console logs in dev. Call ``configure_logging()``
once at process start (entry points / pipeline boundaries); use ``get_logger()``
everywhere else.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog

from stock_agent.settings import Settings

# Guards against re-configuring structlog when called from multiple entry points.
_configured = False


def configure_logging(settings: Settings) -> None:
    """Configure structlog + stdlib logging. Idempotent (safe to call repeatedly)."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    # Shared processor chain: merge context vars -> add level/logger name ->
    # render stack/exception info -> ISO UTC timestamp -> final renderer.
    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    """Return a bound structlog logger (optionally namespaced)."""
    # structlog.get_logger is typed as returning Any; we know the configured
    # wrapper_class is a FilteringBoundLogger, so narrow it explicitly.
    return cast(structlog.typing.FilteringBoundLogger, structlog.get_logger(name))
