"""Price-series data-quality checks.

These are NON-FATAL: they return a list of issues rather than raising, so the
pipeline can proceed and surface concerns in the report's Uncertainty Notes
(e.g. sparse history, stale data, gaps). Hard structural guarantees (ascending,
unique dates, OHLC sanity) are already enforced by the ``PriceSeries`` schema.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel

from stock_agent.schemas.market import PriceSeries

Severity = Literal["info", "warning", "error"]


class DataIssue(BaseModel):
    """A single data-quality finding."""

    code: str  # machine-readable, e.g. "short_history", "date_gap"
    severity: Severity
    message: str


def validate_price_series(
    series: PriceSeries,
    *,
    min_bars: int = 20,
    max_gap_days: int = 5,
    stale_after_days: int = 5,
    today: Date | None = None,
) -> list[DataIssue]:
    """Return data-quality issues for ``series`` (empty list == clean).

    Args:
        min_bars: below this bar count, history is flagged as sparse.
        max_gap_days: a calendar gap larger than this between consecutive bars is
            flagged (covers long weekends/holidays; tune per use).
        stale_after_days: if the latest bar is older than this vs ``today``,
            the series is flagged as stale.
        today: reference "now" (injectable for deterministic tests).
    """
    today = today or Date.today()
    issues: list[DataIssue] = []

    n = len(series)
    if n == 0:
        issues.append(DataIssue(code="empty", severity="error", message="no price bars"))
        return issues

    if n < min_bars:
        issues.append(
            DataIssue(
                code="short_history",
                severity="warning",
                message=f"only {n} bars (< {min_bars}); indicators/forecasts may be unreliable",
            )
        )

    # Gaps between consecutive trading days. strict=False: pairwise over dates.
    dates = series.dates
    for prev, curr in zip(dates, dates[1:], strict=False):
        gap = (curr - prev).days
        if gap > max_gap_days:
            issues.append(
                DataIssue(
                    code="date_gap",
                    severity="info",
                    message=f"{gap}-day gap between {prev} and {curr}",
                )
            )

    last = dates[-1]
    if last > today:
        issues.append(
            DataIssue(
                code="future_date", severity="error", message=f"last bar {last} is in the future"
            )
        )
    elif (today - last).days > stale_after_days:
        issues.append(
            DataIssue(
                code="stale_data",
                severity="warning",
                message=f"latest bar {last} is {(today - last).days} days old",
            )
        )

    return issues


def has_errors(issues: list[DataIssue]) -> bool:
    """True if any issue is of ``error`` severity."""
    return any(i.severity == "error" for i in issues)
