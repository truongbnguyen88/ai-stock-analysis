"""Command-line interface (Typer).

Thin dispatch layer: parse arguments, run a pipeline, render output.
Commands: analyze (Phase 4), forecast (Phase 5), chat (Phase 4.5).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer

if TYPE_CHECKING:
    from stock_agent.schemas.backtest import BacktestResult
    from stock_agent.settings import Settings

from stock_agent.documents.download import DEFAULT_FORMS, download_filings
from stock_agent.documents.ticker_cik import load_universe, normalize_ticker
from stock_agent.llm.client import AnthropicClient
from stock_agent.logging_config import configure_logging
from stock_agent.pipelines.analyze import run_analyze
from stock_agent.pipelines.forecast import MODEL_NAMES, run_forecast
from stock_agent.pipelines.research import ResearchPipelineError, run_research
from stock_agent.providers._cache import DiskCache
from stock_agent.providers.sec_edgar import SecEdgarProvider
from stock_agent.rag.embeddings import build_embedder
from stock_agent.rag.pipeline import ingest_ticker
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import build_vector_store
from stock_agent.reports.render_md import render_markdown
from stock_agent.research.memo import render_memo_markdown
from stock_agent.research.synthesis import answer_question
from stock_agent.schemas.documents import DocumentType
from stock_agent.schemas.retrieval import ChunkFilter
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
def research(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Ticker symbol, e.g. NVDA")],
    days: Annotated[int, typer.Option("--days", "-d", help="News lookback window in days")] = 30,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the memo Markdown to this file")
    ] = None,
    no_news: Annotated[
        bool, typer.Option("--no-news", help="Skip the news summary")
    ] = False,
    company: Annotated[
        str | None, typer.Option("--company", help="Company name to improve news relevance")
    ] = None,
) -> None:
    """Integrated, SEC-grounded research memo (technicals + forecast + news + filings)."""
    settings = get_settings()
    configure_logging(settings)
    try:
        memo = run_research(
            ticker, settings=settings, days=days, use_news=not no_news, company_name=company
        )
    except ResearchPipelineError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    markdown = render_memo_markdown(memo)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote memo to {output}")
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
    ] = "ensemble",
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


_ML_MODEL_TYPES = ("logistic", "lightgbm")
# Production toolkit (see docs/validations_results.md): logistic (stable names) +
# tuned lightgbm (volatile names), at the swing-trade horizons. h5 is dropped.
_PROD_MODELS: tuple[str, ...] = ("logistic", "lightgbm")
_PROD_HORIZONS: tuple[int, ...] = (20, 30, 60)


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
    train_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Train every production artifact (logistic + lightgbm at 20/30/60) and drop h5.",
        ),
    ] = False,
) -> None:
    """Train a pooled ML model over the universe and persist the artifact.

    With ``--all``, retrains the whole production toolkit (logistic + lightgbm at the
    20/30/60-day horizons) and removes the stale h5 artifacts.
    """
    if not universe.exists():
        typer.echo(f"Universe file not found: {universe}")
        raise typer.Exit(code=1)

    settings = get_settings()
    configure_logging(settings)

    if train_all:
        _train_all(universe, settings)
        return

    if model not in _ML_MODEL_TYPES:
        typer.echo(f"--model must be one of {list(_ML_MODEL_TYPES)}")
        raise typer.Exit(code=1)

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


def _train_all(universe: Path, settings: Settings) -> None:
    """Retrain the production toolkit (model × horizon) and drop stale h5 artifacts."""
    from stock_agent.forecasting.pooled import ModelType, default_model_path
    from stock_agent.forecasting.train_pooled import train_pooled

    typer.echo(
        f"Training {list(_PROD_MODELS)} × horizons {list(_PROD_HORIZONS)} over {universe} …\n"
    )
    for model in _PROD_MODELS:
        model_type: ModelType = model  # type: ignore[assignment]
        for h in _PROD_HORIZONS:
            typer.echo(f"→ {model} h{h} …")
            trained, path = train_pooled(universe, settings, model_type=model_type, horizon_days=h)
            typer.echo(
                f"  {trained.n_tickers} tickers, {trained.n_train_rows:,} rows, "
                f"{len(trained.classifiers)} thresholds → {path}"
            )

    # Drop stale h5 artifacts — h5 is too short-term for the swing-trade horizons,
    # so a forecast at h5 should fall back to the baseline, not a stale model.
    models_dir = Path(settings.output_dir) / "models"
    removed = []
    for model in _PROD_MODELS:
        p = default_model_path(models_dir, model, 5)
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if removed:
        typer.echo(f"\nDropped stale h5 artifacts: {', '.join(removed)}")
    typer.echo("\nDone — production toolkit retrained.")


@app.command(name="conformal-calibrate")
def conformal_calibrate(
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file (one ticker per line)")
    ] = Path("configs/universe.txt"),
    basket_size: Annotated[
        int, typer.Option("--basket-size", help="How many universe tickers to pool over")
    ] = 24,
) -> None:
    """Compute pooled conformal interval-corrections → outputs/models/conformal.json.

    For each (model, horizon): build the model as of ~1y ago, forecast the held-out year
    across the basket, pool the (CI, realized) pairs, and fit one distribution-free `q` so
    the served CI/VaR has honest coverage. Slow (offline, one-time). Applied at inference
    when `settings.conformal_intervals` is on.
    """
    if not universe.exists():
        typer.echo(f"Universe file not found: {universe}")
        raise typer.Exit(code=1)
    settings = get_settings()
    configure_logging(settings)
    from stock_agent.forecasting.train_conformal import calibrate, conformal_path
    from stock_agent.providers.registry import build_default_registry

    typer.echo("Calibrating conformal interval-corrections (offline; this takes a while) …")
    art = calibrate(build_default_registry(settings), settings, universe_path=universe,
                    basket_size=basket_size)
    path = conformal_path(settings)
    art.save(path)
    typer.echo(f"\nSaved {path}  (cutoff {art.cal_cutoff}, target {art.ci_level:.0%}):")
    for model, by_h in art.entries.items():
        for h, e in by_h.items():
            typer.echo(
                f"  {model:>22} h{h:<3}  q={e.q:+.3f}  "
                f"coverage {e.coverage_before:.0%} → {e.coverage_after:.0%}  (n={e.n})"
            )


@app.command(name="verify-models")
def verify_models(
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used to size the data-quality floor")
    ] = Path("configs/universe.txt"),
) -> None:
    """Sanity-check the trained artifacts (the scheduled-retrain promote gate).

    Network-free; exits non-zero if any artifact is missing, mis-shaped, can't
    produce valid probabilities, OR was trained on too little of the universe — so
    CI won't publish a degraded model release (e.g. from a yfinance-rate-limited
    data month). The ticker floor is ``verify_min_ticker_fraction`` of the universe.
    """
    import math

    from stock_agent.forecasting.conformal_calibrate import CONFORMAL_FILE
    from stock_agent.forecasting.train_pooled import load_universe
    from stock_agent.forecasting.verify import verify_artifacts, verify_conformal

    settings = get_settings()
    models_dir = Path(settings.output_dir) / "models"

    # Data-quality floor sized from the configured universe (auto-adapts as it grows);
    # falls back to row-only if the universe file is absent.
    min_tickers = 0
    if universe.exists():
        min_tickers = math.ceil(len(load_universe(universe)) * settings.verify_min_ticker_fraction)
    problems = verify_artifacts(
        models_dir,
        models=list(_PROD_MODELS),
        horizons=list(_PROD_HORIZONS),
        min_tickers=min_tickers,
        min_rows=settings.verify_min_rows,
    )
    # Conformal interval-correction: verify if present (it's generated in CI right before
    # this gate, so its absence/breakage there fails the publish); a local run without it
    # just warns — served CIs fall back to un-conformalized with no error.
    conformal_present = (models_dir / CONFORMAL_FILE).exists()
    if conformal_present:
        problems += verify_conformal(
            models_dir,
            required_models=["ensemble", "ml_logistic", "ml_lightgbm"],
            horizons=list(_PROD_HORIZONS),
        )

    if problems:
        typer.echo("✗ artifact verification FAILED:")
        for p in problems:
            typer.echo(f"  - {p}")
        raise typer.Exit(code=1)
    n = len(_PROD_MODELS) * len(_PROD_HORIZONS)
    conf = "+ conformal" if conformal_present else "(no conformal.json — run conformal-calibrate)"
    typer.echo(
        f"✓ verified {n} artifacts {conf} in {models_dir} "
        f"(data-quality floor: >= {min_tickers} tickers, >= {settings.verify_min_rows:,} rows)"
    )


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
    if result.conformal is not None:
        cf = result.conformal
        conf = (
            f" → conformal {cf.conformalized_coverage:.0%}"
            if cf.conformalized_coverage is not None
            else ""
        )
        typer.echo(
            f"  Interval cov  : {cf.empirical_coverage:.0%} of the stated "
            f"{cf.ci_level:.0%} CI held OOS (n={cf.n_eval}){conf}"
        )
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


@app.command(name="ingest-news")
def ingest_news(
    start: Annotated[str, typer.Option("--start", help="Window start (YYYY-MM-DD, inclusive)")],
    end: Annotated[str, typer.Option("--end", help="Window end (YYYY-MM-DD, exclusive)")],
    project: Annotated[
        str | None, typer.Option("--project", help="Google Cloud project for BigQuery billing")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Estimate bytes/cost only — no rows read, no spend"),
    ] = False,
    streams: Annotated[
        str,
        typer.Option("--streams", help="Comma list: per_ticker,market,topics (default both base)"),
    ] = "per_ticker,market",
    no_business_filter: Annotated[
        bool,
        typer.Option("--no-business-filter", help="Disable the per-ticker business-theme filter"),
    ] = False,
    no_topic_names: Annotated[
        bool,
        typer.Option("--no-topic-names", help="Topics: themes only (skip AI/AllNames — cheaper)"),
    ] = False,
) -> None:
    """Pull daily GDELT news-sentiment features into outputs/news_sentiment/ (Task 10).

    Aggregation runs server-side in BigQuery, so only small daily feature rows are
    downloaded — never article text. ALWAYS run --dry-run first to see bytes-to-scan
    (the 1 TiB/month free tier; ~$5/TiB after). Streams write separate CSVs, so you
    can pull --streams topics later without re-pulling the base streams.
    """
    from datetime import date as _Date

    from stock_agent.news.gdelt_ingest import ALL_STREAMS, estimate_bytes, ingest, stream_query

    settings = get_settings()
    configure_logging(settings)
    start_d, end_d = _Date.fromisoformat(start), _Date.fromisoformat(end)
    if end_d <= start_d:
        typer.echo("--end must be after --start")
        raise typer.Exit(code=1)
    selected = [s.strip() for s in streams.split(",") if s.strip()]
    bad = [s for s in selected if s not in ALL_STREAMS]
    if bad:
        typer.echo(f"--streams: unknown {bad} (allowed: {list(ALL_STREAMS)})")
        raise typer.Exit(code=1)
    biz, names = not no_business_filter, not no_topic_names

    if dry_run:
        gib, free = 1024**3, 1024**4
        total = 0
        typer.echo(f"Dry run {start} → {end} (exclusive), streams={selected}:")
        try:
            for s in selected:
                sql = stream_query(
                    s, start_d, end_d, require_business_theme=biz, include_topic_names=names
                )
                b = estimate_bytes(sql, project=project)
                total += b
                typer.echo(f"  {s:>11} stream: {b / gib:8.2f} GiB")
        except RuntimeError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        typer.echo(f"  {'TOTAL':>11}:        {total / gib:8.2f} GiB  (~${total / 1024**4 * 5:.2f})")
        typer.echo(
            "  ✓ within 1 TiB/month free tier"
            if total <= free
            else "  ⚠ exceeds 1 TiB free tier — pull fewer streams / shorter window / next month"
        )
        return

    typer.echo(f"Ingesting GDELT {selected} {start} → {end} (exclusive) …")
    try:
        results = ingest(
            start_d, end_d, project=project, streams=tuple(selected),
            require_business_theme=biz, include_topic_names=names,
        )
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    for name, frame in results.items():
        typer.echo(f"  {name}: {len(frame):,} rows → outputs/news_sentiment/{name}.csv")


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


# ---- documents (RAG corpus acquisition) --------------------------------------
documents_app = typer.Typer(
    add_completion=False, help="Acquire source documents for RAG (SEC filings)."
)
app.add_typer(documents_app, name="documents")

_ALLOWED_FORMS: tuple[DocumentType, ...] = ("10-K", "10-Q", "8-K")


@documents_app.command("download-sec")
def download_sec(
    ticker: Annotated[
        str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")
    ] = None,
    download_all: Annotated[
        bool, typer.Option("--all", help="Download for every ticker in the universe file")
    ] = False,
    forms: Annotated[
        list[str] | None,
        typer.Option("--forms", help="Filing forms (repeatable); default 10-K 10-Q 8-K"),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max filings per form per ticker")
    ] = 4,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Download SEC filings (10-K / 10-Q / 8-K) into the local corpus via EDGAR."""
    settings = get_settings()
    configure_logging(settings)
    if not settings.sec_user_agent:
        typer.echo(
            "SEC_USER_AGENT is not set. SEC fair-access requires a contact User-Agent — "
            "add SEC_USER_AGENT=\"Your Name your@email.com\" to your .env."
        )
        raise typer.Exit(code=1)

    if forms:
        bad = [f for f in forms if f not in _ALLOWED_FORMS]
        if bad:
            typer.echo(f"Unsupported forms {bad}; allowed: {list(_ALLOWED_FORMS)}")
            raise typer.Exit(code=1)
        forms_t = cast("tuple[DocumentType, ...]", tuple(forms))
    else:
        forms_t = DEFAULT_FORMS

    if download_all:
        tickers = load_universe(universe)
    elif ticker:
        tickers = [normalize_ticker(ticker)]
    else:
        typer.echo("Provide --ticker SYMBOL or --all.")
        raise typer.Exit(code=1)

    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    provider = SecEdgarProvider(settings, cache)
    for sym in tickers:
        r = download_filings(
            sym, provider, documents_dir=settings.documents_dir, forms=forms_t, limit=limit
        )
        typer.echo(
            f"{sym}: downloaded {len(r.downloaded)}, skipped {len(r.skipped)}, "
            f"errors {len(r.errors)}"
        )


def _ingest_tickers(download_all: bool, ticker: str | None, universe: Path) -> list[str]:
    """Resolve the ticker list for an ingest run (shared --ticker/--all logic)."""
    if download_all:
        return load_universe(universe)
    if ticker:
        return [normalize_ticker(ticker)]
    typer.echo("Provide --ticker SYMBOL or --all.")
    raise typer.Exit(code=1)


@documents_app.command("ingest")
def ingest(
    ticker: Annotated[
        str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")
    ] = None,
    ingest_all: Annotated[
        bool, typer.Option("--all", help="Ingest every ticker in the universe file")
    ] = False,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Parse → chunk → embed → store downloaded filings into the vector store ([rag] extra)."""
    settings = get_settings()
    configure_logging(settings)
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    for sym in _ingest_tickers(ingest_all, ticker, universe):
        result = ingest_ticker(
            sym,
            documents_dir=settings.documents_dir,
            embedder=embedder,
            store=store,
            chunk_tokens=settings.rag_chunk_tokens,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        typer.echo(f"{sym}: {result.filings} filings → {result.chunks} chunks ingested")


# ---- rag (SEC-grounded retrieval) --------------------------------------------
rag_app = typer.Typer(add_completion=False, help="Query the ingested SEC corpus (RAG).")
app.add_typer(rag_app, name="rag")


@rag_app.command("query")
def rag_query(
    question: Annotated[str, typer.Option("--question", "-q", help="Natural-language question")],
    ticker: Annotated[
        str | None, typer.Option("--ticker", "-t", help="Scope to one ticker")
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Chunks to return (default settings.rag_top_k)")
    ] = None,
    answer: Annotated[
        bool, typer.Option("--answer", help="Synthesize a cited answer (one LLM call) instead")
    ] = False,
) -> None:
    """Retrieve SEC chunks for a question; with --answer, synthesize a cited answer."""
    settings = get_settings()
    configure_logging(settings)
    retriever = Retriever(build_embedder(settings), build_vector_store(settings))
    where = ChunkFilter(ticker=normalize_ticker(ticker)) if ticker else None
    evidence = retriever.retrieve(question, top_k=top_k or settings.rag_top_k, where=where)

    if answer:
        grounded = answer_question(question, evidence, llm=AnthropicClient(settings))
        typer.echo(grounded.answer)
        for cite in grounded.citations:
            typer.echo(f"  [{cite.marker}] {cite.label}")
        return

    if evidence.is_empty:
        typer.echo("No matching evidence found. Ingest first: documents ingest --ticker SYM")
        raise typer.Exit(code=1)
    for i, rc in enumerate(evidence.chunks, start=1):
        typer.echo(f"\n[{i}] {rc.citation_label()}  (score {rc.score:.3f})")
        typer.echo(textwrap.shorten(rc.chunk.text, width=280))
