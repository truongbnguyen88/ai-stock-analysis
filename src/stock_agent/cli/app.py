"""Command-line interface (Typer).

Thin dispatch layer: parse arguments, run a pipeline, render output.
Commands: analyze (Phase 4), forecast (Phase 5), chat (Phase 4.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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
    ] = "xgboost",
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

    def answer(question: str) -> None:
        try:
            result = run_agent(question, llm=agent_llm, executor=executor)
            typer.echo(result.text)
        except AgentError as exc:
            typer.echo(f"[agent error] {exc}")

    if message:
        answer(message)
        return

    typer.echo("Research agent — ask a question ('exit' to quit). Not financial advice.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer(question)
