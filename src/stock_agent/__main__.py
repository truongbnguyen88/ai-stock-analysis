"""Module entry point: enables ``python -m stock_agent`` and the ``stock-agent`` script.

Dispatches to the Typer CLI app (``stock_agent.cli.app``).
"""

from __future__ import annotations

from stock_agent.cli.app import app


def main() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":
    main()
