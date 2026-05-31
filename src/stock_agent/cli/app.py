"""Command-line interface (Typer).

Thin dispatch layer: parse arguments, run a pipeline, render output.
Commands: analyze (Phase 4), forecast (Phase 5), chat (Phase 4.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from stock_agent.schemas.backtest import BacktestResult

from stock_agent.logging_config import configure_logging
from stock_agent.pipelines.analyze import run_analyze
from stock_agent.pipelines.forecast import MODEL_NAMES, run_forecast
from stock_agent.reports.render_md import render_markdown
from stock_agent.settings import get_settings

app = typer.Typer(
    add_completion=False,
    help="LLM-powered stock research assistant (research/education only; not financial advice).",
)


@app.callback()
def _root() -> None:
    """Force subcommand mode so commands keep their name (analyze; forecast/backtest later)."""


@app.command()
def analyze(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Ticker symbol, e.g. NVDA")],
    days: Annotated[int, typer.Option("--days", "-d", help="News lookback window in days")] = 30,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write Markdown to this file")
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Skip the LLM news summary (offline / no cost)")
    ] = False,
    company: Annotated[
        str | None, typer.Option("--company", help="Company name to improve news relevance")
    ] = None,
) -> None:
    """Generate a research report for a ticker."""
    settings = get_settings()
    configure_logging(settings)

    report = run_analyze(
        ticker,
        days=days,
        settings=settings,
        use_llm=not no_llm,
        company_name=company,
    )
    markdown = render_markdown(report)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote report to {output}")
    else:
        typer.echo(markdown)


@app.command()
def forecast(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Ticker symbol")],
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forecast horizon in trading days")
    ] = 20,
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"Forecaster: {MODEL_NAMES}")
    ] = "historical_sim",
    all_models: Annotated[
        bool, typer.Option("--all-models", help="Run and compare all available models")
    ] = False,
) -> None:
    """Run a probabilistic scenario forecast for a ticker."""
    settings = get_settings()
    configure_logging(settings)
    models = MODEL_NAMES if all_models else [model]
    for m in models:
        try:
            fc = run_forecast(ticker, horizon, model_name=m, settings=settings)
        except ValueError as exc:
            typer.echo(f"[{m}] {exc}")
            continue
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"  {fc.ticker} — {fc.horizon_days}d forecast ({fc.model_name})")
        typer.echo(f"{'=' * 60}")
        typer.echo(f"  Expected return : {fc.expected_return:+.2%}")
        typer.echo(f"  P(up)           : {fc.upside_prob:.0%}")
        typer.echo(f"  P(down)         : {fc.downside_prob:.0%}")
        if fc.var_95 is not None:
            typer.echo(f"  VaR 95%         : {fc.var_95:.2%}")
        if fc.ci_low is not None and fc.ci_high is not None:
            typer.echo(f"  90% CI          : [{fc.ci_low:.2%}, {fc.ci_high:.2%}]")
        typer.echo(f"  Calibration     : {fc.calibration_status}")
        typer.echo("")
        typer.echo("  Scenario buckets:")
        for b in fc.buckets:
            bar = "█" * int(b.probability * 30)
            typer.echo(f"    {b.label:>15s}  {b.probability:5.1%}  {bar}")
        if fc.notes:
            typer.echo(f"\n  ⚠  {fc.notes}")


_ML_MODEL_TYPES = ("logistic", "xgboost", "lightgbm", "random_forest")


@app.command()
def train(
    model: Annotated[
        str, typer.Option("--model", "-m", help=f"ML model: {list(_ML_MODEL_TYPES)}")
    ] = "logistic",  # validated best ML model (see docs/validations_results.md)
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forecast horizon in trading days")
    ] = 20,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file (one ticker per line)")
    ] = Path("configs/universe.txt"),
) -> None:
    """Train a pooled ML model over the universe and persist the artifact."""
    if model not in _ML_MODEL_TYPES:
        typer.echo(f"--model must be one of {list(_ML_MODEL_TYPES)}")
        raise typer.Exit(code=1)
    if not universe.exists():
        typer.echo(f"Universe file not found: {universe}")
        raise typer.Exit(code=1)

    settings = get_settings()
    configure_logging(settings)

    from stock_agent.forecasting.pooled import ModelType
    from stock_agent.forecasting.train_pooled import train_pooled

    model_type: ModelType = model  # type: ignore[assignment]  # validated above
    typer.echo(f"Training pooled {model} (horizon {horizon}) over {universe} …")
    trained, path = train_pooled(universe, settings, model_type=model_type, horizon_days=horizon)
    typer.echo(
        f"Done: {trained.n_tickers} tickers, {trained.n_train_rows:,} rows, "
        f"{len(trained.classifiers)} thresholds → {path}"
    )
    if trained.notes:
        for note in trained.notes:
            typer.echo(f"  note: {note}")


def _print_backtest(result: BacktestResult) -> None:
    """Render a backtest result as a compact, scannable summary."""
    c = result.calibration
    typer.echo(f"\n{'=' * 64}")
    typer.echo(f"  {result.ticker} — {result.horizon_days}d backtest ({result.model_name})")
    typer.echo(f"{'=' * 64}")
    typer.echo(
        f"  OOS forecasts : {result.n_predictions} over {result.n_folds} folds "
        f"({result.as_of_start} → {result.as_of_end})"
    )
    typer.echo(f"  Mean Brier    : {result.mean_brier:.4f}   (lower is better)")
    typer.echo(f"  Mean log loss : {result.mean_log_loss:.4f}")
    post = (
        f"  →  {c.method_post} recal (held-out): {c.ece_pre_holdout:.3f} → {c.ece_post:.3f}"
        if c.ece_post is not None and c.ece_pre_holdout is not None
        else ""
    )
    typer.echo(f"  Calibration   : ECE {c.ece:.3f}  MCE {c.mce:.3f}{post}")
    typer.echo("")
    typer.echo("  Per-threshold (P(r > θ)):")
    typer.echo(f"    {'θ':>7s}  {'base':>6s}  {'Brier':>7s}  {'logloss':>8s}  {'AUC':>5s}")
    for m in result.thresholds:
        auc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "  n/a"
        typer.echo(
            f"    {m.threshold:>+7.0%}  {m.base_rate:>6.0%}  "
            f"{m.brier:>7.4f}  {m.log_loss:>8.4f}  {auc:>5s}"
        )
    if result.notes:
        for note in result.notes:
            typer.echo(f"  ⚠  {note}")


@app.command()
def backtest(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Ticker symbol")],
    horizon: Annotated[
        int, typer.Option("--horizon", help="Forecast horizon in trading days")
    ] = 20,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Single model; omit to compare offline baselines"),
    ] = None,
    test_size: Annotated[int, typer.Option("--test-size", help="Test as-of points per fold")] = 6,
) -> None:
    """Walk-forward backtest a forecaster (OOS metrics + calibration)."""
    from stock_agent.pipelines.backtest import STATELESS_MODELS, run_backtest_pipeline

    settings = get_settings()
    configure_logging(settings)
    models = [model] if model else list(STATELESS_MODELS)
    typer.echo(f"Backtesting {ticker.upper()} (horizon {horizon}d): {', '.join(models)} …")
    try:
        results = run_backtest_pipeline(
            ticker, horizon, model_names=models, settings=settings, test_size=test_size
        )
    except ValueError as exc:
        typer.echo(f"[backtest error] {exc}")
        raise typer.Exit(code=1) from exc
    for result in results.values():
        _print_backtest(result)
    if len(results) > 1:
        typer.echo(f"\n{'=' * 64}")
        typer.echo("  Comparison (mean Brier · ECE — lower is better):")
        ranked = sorted(results.values(), key=lambda r: r.mean_brier)
        for r in ranked:
            typer.echo(
                f"    {r.model_name:>24s}  Brier {r.mean_brier:.4f}  ECE {r.calibration.ece:.3f}"
            )


@app.command()
def chat(
    message: Annotated[
        str | None, typer.Argument(help="One-shot question; omit for an interactive session")
    ] = None,
) -> None:
    """Ask the research agent a question (it answers by calling tools)."""
    settings = get_settings()
    configure_logging(settings)
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is required for chat. Add it to your .env.")
        raise typer.Exit(code=1)

    # Imported lazily so non-chat commands don't pay the agent import cost.
    from stock_agent.agent.runtime import AgentError, AnthropicToolClient, run_agent
    from stock_agent.agent.tools import ToolExecutor
    from stock_agent.llm.client import AnthropicClient

    executor = ToolExecutor(settings, llm=AnthropicClient(settings))
    agent_llm = AnthropicToolClient(settings)

    def answer(question: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one turn; return the updated transcript to carry into the next turn.

        On error the prior history is preserved so the conversation can continue.
        """
        try:
            result = run_agent(question, llm=agent_llm, executor=executor, history=history)
            typer.echo(result.text)
            return result.messages
        except AgentError as exc:
            typer.echo(f"[agent error] {exc}")
            return history

    if message:  # one-shot mode: single turn, no memory needed
        answer(message, [])
        return

    typer.echo(
        "Research agent — ask a question ('exit' to quit, 'reset' to clear context). "
        "Not financial advice."
    )
    # Conversation memory: each turn's transcript feeds the next so follow-ups
    # ("the above", "now forecast it") resolve against prior turns. Numbers are
    # re-grounded per turn, so the agent re-calls tools (cached) to reuse figures.
    history: list[dict[str, Any]] = []
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            history = []
            typer.echo("(context cleared)")
            continue
        if question:
            history = answer(question, history)
