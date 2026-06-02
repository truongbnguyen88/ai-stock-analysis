"""Universe-file parsing for pooled training (pure, network-free)."""

from __future__ import annotations

from pathlib import Path

from stock_agent.forecasting.train_pooled import load_universe


def test_load_universe_parses_comments_blanks_and_case(tmp_path: Path) -> None:
    f = tmp_path / "universe.txt"
    f.write_text(
        "\n".join(
            [
                "# leading comment",
                "nvda",
                "  msft  ",  # surrounding whitespace
                "",  # blank line
                "aapl  # inline comment",
                "   # whitespace-then-comment",
                "spy",
            ]
        ),
        encoding="utf-8",
    )
    assert load_universe(f) == ["NVDA", "MSFT", "AAPL", "SPY"]


def test_load_universe_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    assert load_universe(f) == []
