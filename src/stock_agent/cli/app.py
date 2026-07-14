"""Command-line interface (Typer).

Thin dispatch layer: parse arguments, run a pipeline, render output.
Commands: analyze (Phase 4), forecast (Phase 5), chat (Phase 4.5).
"""

from __future__ import annotations

import json
import random
import textwrap
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer

if TYPE_CHECKING:
    from stock_agent.schemas.backtest import BacktestResult
    from stock_agent.settings import Settings

from stock_agent.documents.download import DEFAULT_FORMS, bulk_download
from stock_agent.documents.ticker_cik import load_universe, normalize_ticker
from stock_agent.graph.extract import (
    GraphExtractBudgetExceeded,
    extract_edges,
    select_extraction_chunks,
)
from stock_agent.graph.store import build_graph_store
from stock_agent.llm.client import AnthropicClient, TextLLM
from stock_agent.logging_config import configure_logging
from stock_agent.pipelines.analyze import run_analyze
from stock_agent.pipelines.forecast import MODEL_NAMES, run_forecast
from stock_agent.pipelines.research import ResearchPipelineError, run_research
from stock_agent.providers._cache import DiskCache
from stock_agent.providers.sec_edgar import SecEdgarProvider
from stock_agent.rag.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    build_embedder,
)
from stock_agent.rag.eval import (
    LabeledQuery,
    SystemReport,
    evaluate_system,
    format_reports_markdown,
    run_ab,
)
from stock_agent.rag.hybrid import SparseRetriever
from stock_agent.rag.pipeline import (
    EmbedBudgetExceeded,
    backfill_sparse,
    build_chunks,
    bulk_ingest,
)
from stock_agent.rag.policy import ARMS
from stock_agent.rag.policy_eval import (
    GatedEvalReport,
    PolicyEvalReport,
    build_dataset,
    evaluate_gated,
    evaluate_offline,
    fit_bandit_baseline,
    split_dataset,
)
from stock_agent.rag.policy_retriever import build_arm_system
from stock_agent.rag.read_path import (
    LATTICE_SYSTEMS,
    build_graph_system,
    build_named_system,
    build_retrieval_system,
)
from stock_agent.rag.rl.action import named_action_space
from stock_agent.rag.rl.env import RagRetrievalEnv
from stock_agent.rag.rl.rleval import (
    AlwaysSearchPolicy,
    OneShotArmPolicy,
    ReActBridgePolicy,
    ScriptedPolicy,
    build_baselines,
    evaluate_rl,
    format_report_markdown,
)
from stock_agent.rag.rl.train import TrainConfig, load_checkpoint, save_checkpoint, train_policy
from stock_agent.rag.sparse_store import InMemoryBM25Store, build_sparse_store
from stock_agent.rag.status import corpus_status
from stock_agent.rag.vector_store import InMemoryVectorStore, build_vector_store
from stock_agent.reports.render_md import render_markdown
from stock_agent.research.agentic import answer_multistep
from stock_agent.research.bridge import load_alias_map
from stock_agent.research.memo import render_memo_markdown
from stock_agent.research.multistep_eval import (
    MultiHopQuery,
    evaluate_multihop_set,
    format_multihop_markdown,
)
from stock_agent.research.multistep_gen import generate_multihop, split_multihop
from stock_agent.research.synthesis import answer_question
from stock_agent.schemas.documents import DocumentChunk, DocumentType
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
    no_news: Annotated[bool, typer.Option("--no-news", help="Skip the news summary")] = False,
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
    art = calibrate(
        build_default_registry(settings), settings, universe_path=universe, basket_size=basket_size
    )
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
            start_d,
            end_d,
            project=project,
            streams=tuple(selected),
            require_business_theme=biz,
            include_topic_names=names,
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
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            help=(
                "Deterministic DOMAIN — run a capability area directly, skipping LLM routing: "
                "predictions, news, filings, technicals, brief. Pair with --variant; invalid "
                "values list the choices."
            ),
        ),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            "--variant",
            help=(
                "Variant within --domain (e.g. news: summary/headlines/theme; filings: "
                "single/multi; predictions: forecast/big-move). Omit for the domain default."
            ),
        ),
    ] = None,
    tool: Annotated[
        str | None,
        typer.Option(
            "--tool",
            help=(
                "Advanced: dispatch one EXACT granular route (e.g. forecast, summarize-style news, "
                "filings_multihop), bypassing the domain layer. Invalid values list every route."
            ),
        ),
    ] = None,
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-t", help="Ticker for --domain/--tool routes that need one."),
    ] = None,
    horizon: Annotated[
        int | None, typer.Option("--horizon", help="Horizon in trading days (forecast / big_move).")
    ] = None,
    days: Annotated[
        int | None, typer.Option("--days", help="Lookback window in days (news / price).")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Model override (forecast / big_move).")
    ] = None,
) -> None:
    """Ask the agent (LLM routing), or run a capability directly via --domain (deterministic)."""
    settings = get_settings()
    configure_logging(settings)

    # Imported lazily so non-chat commands don't pay the agent import cost.
    from stock_agent.agent.router import Router, RouterError, resolve_domain
    from stock_agent.agent.runtime import AgentError, AnthropicToolClient, run_agent
    from stock_agent.agent.tools import ToolExecutor
    from stock_agent.llm.client import AnthropicClient

    key = settings.anthropic_api_key
    # The executor's own LLM powers synthesis tools (news / filings); numeric routes need no LLM, so
    # a missing key is fine for those (the tool then returns a clear "no LLM configured" error).
    executor = ToolExecutor(settings, llm=AnthropicClient(settings) if key else None)

    # ---- Deterministic routing: the user named a domain/route -> NO LLM routing call. ----
    # --domain (friendly, resolved via --variant) is the primary surface; --tool is the exact
    # granular escape hatch. --domain wins if both are given.
    if domain is not None or tool is not None:
        router = Router(executor)  # no tool_llm: the deterministic path never routes via an LLM
        try:
            route = resolve_domain(domain, variant) if domain is not None else tool
            result = router.run(
                message or "", route=route, ticker=ticker, horizon=horizon, days=days, model=model
            )
        except RouterError as exc:
            typer.echo(f"[router error] {exc}")
            raise typer.Exit(code=1) from exc
        typer.echo(result.text)
        for cite in (result.structured or {}).get("citations", []):
            typer.echo(f"  [{cite.get('marker')}] {cite.get('label', '')}")
        return

    # ---- LLM routing (the agent tool-use loop): requires the API key. ----
    if not key:
        typer.echo("ANTHROPIC_API_KEY is required for chat. Add it to your .env.")
        raise typer.Exit(code=1)
    agent_llm = AnthropicToolClient(settings)
    # Show the active SEC corpus so it's clear which embeddings filing questions are answered from.
    typer.echo(f"RAG corpus: {corpus_status(settings).one_line}")

    # Optional two-stage routing: a Haiku classifier picks a single deterministic capability for
    # simple questions (Sonnet still does the synthesis) and escalates complex ones to the Sonnet
    # agent. Off unless `use_llm_classifier` is set. When off, we call `run_agent` directly.
    classify_router: Router | None = None
    if settings.use_llm_classifier:
        from stock_agent.agent.classifier import RouteClassifier

        classifier = RouteClassifier(
            AnthropicClient(settings, model=settings.classifier_model),
            min_confidence=settings.classifier_min_confidence,
        )
        classify_router = Router(executor, tool_llm=agent_llm, classifier=classifier)
        typer.echo(f"Routing: two-stage (classifier={settings.classifier_model} -> Sonnet)")

    def answer(question: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one turn; return the updated transcript to carry into the next turn.

        On error the prior history is preserved so the conversation can continue.
        """
        try:
            if classify_router is not None:
                rr = classify_router.run(question, route="classify", history=history)
                typer.echo(rr.text)
                for cite in (rr.structured or {}).get("citations", []):
                    typer.echo(f"  [{cite.get('marker')}] {cite.get('label', '')}")
                # Escalation carries the agent transcript (thread it); a deterministic turn is
                # standalone and returns no messages, so keep the prior history unchanged.
                return rr.messages or history
            result = run_agent(question, llm=agent_llm, executor=executor, history=history)
            typer.echo(result.text)
            return result.messages
        except (AgentError, RouterError) as exc:
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


@app.command(name="eval-routing")
def eval_routing(
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="JSONL of {question, expected}; omit for the built-in set."),
    ] = None,
    min_confidence: Annotated[
        float | None,
        typer.Option("--min-confidence", help="Escalate below this confidence (default: config)."),
    ] = None,
) -> None:
    """Measure the Haiku routing classifier on labeled examples (accuracy / misroute / escalation).

    Live: makes one Haiku call per example. Haiku is used ONLY for routing here — never content.
    """
    settings = get_settings()
    configure_logging(settings)
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is required to run the classifier.")
        raise typer.Exit(code=1)
    from stock_agent.agent.classifier import RouteClassifier
    from stock_agent.agent.routing_eval import DEFAULT_EXAMPLES, evaluate_routing, load_examples

    examples = load_examples(dataset) if dataset is not None else DEFAULT_EXAMPLES
    conf = min_confidence if min_confidence is not None else settings.classifier_min_confidence
    classifier = RouteClassifier(
        AnthropicClient(settings, model=settings.classifier_model), min_confidence=conf
    )
    typer.echo(f"Classifier: {settings.classifier_model} (conf>={conf}) · n={len(examples)}")
    typer.echo(evaluate_routing(classifier, examples).summary())


# ---- documents (RAG corpus acquisition) --------------------------------------
documents_app = typer.Typer(
    add_completion=False, help="Acquire source documents for RAG (SEC filings)."
)
app.add_typer(documents_app, name="documents")

# 20-F = foreign private issuers' annual report (TSMC/ASML/ARM/STM) — pass it explicitly via
# --forms; it is NOT in DEFAULT_FORMS, so domestic `--all` downloads stay 10-K/10-Q/8-K only.
_ALLOWED_FORMS: tuple[DocumentType, ...] = ("10-K", "10-Q", "8-K", "20-F")


@documents_app.command("download-sec")
def download_sec(
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")] = None,
    download_all: Annotated[
        bool, typer.Option("--all", help="Download for every ticker in the universe file")
    ] = False,
    forms: Annotated[
        list[str] | None,
        typer.Option("--forms", help="Filing forms (repeatable); default 10-K 10-Q 8-K"),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max filings per form per ticker (within the window)")
    ] = 4,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Inclusive date floor YYYY-MM-DD (history window)"),
    ] = None,
    years: Annotated[
        int,
        typer.Option("--years", help="History depth in years (floor = today − N yrs; 0 = off)"),
    ] = 0,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Download SEC filings (10-K / 10-Q / 8-K) into the local corpus via EDGAR.

    Bulk history (RAG_TODO 9d): ``--all --years 3 --limit 60`` pulls ~3 years for the universe.
    Idempotent — re-runs resume (already-downloaded filings are skipped) and per-ticker failures
    don't abort the run.
    """
    settings = get_settings()
    configure_logging(settings)
    if not settings.sec_user_agent:
        typer.echo(
            "SEC_USER_AGENT is not set. SEC fair-access requires a contact User-Agent — "
            'add SEC_USER_AGENT="Your Name your@email.com" to your .env.'
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

    # Date floor: explicit --since wins; else --years N -> today − N years; else no floor.
    if since is not None:
        try:
            since_floor: date | None = date.fromisoformat(since)
        except ValueError:
            typer.echo(f"Invalid --since '{since}'; expected YYYY-MM-DD.")
            raise typer.Exit(code=1) from None
    elif years > 0:
        since_floor = date.today() - timedelta(days=365 * years)
    else:
        since_floor = None

    if download_all:
        tickers = load_universe(universe)
    elif ticker:
        tickers = [normalize_ticker(ticker)]
    else:
        typer.echo("Provide --ticker SYMBOL or --all.")
        raise typer.Exit(code=1)

    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    provider = SecEdgarProvider(settings, cache)
    result = bulk_download(
        tickers,
        provider,
        documents_dir=settings.documents_dir,
        forms=forms_t,
        limit=limit,
        since=since_floor,
    )
    for r in result.per_ticker:
        typer.echo(
            f"{r.ticker}: downloaded {len(r.downloaded)}, skipped {len(r.skipped)}, "
            f"errors {len(r.errors)}"
        )
    window = f" since {since_floor.isoformat()}" if since_floor else ""
    typer.echo(
        f"— total: {result.tickers} tickers{window} → downloaded {result.downloaded}, "
        f"skipped {result.skipped}, errors {result.errors}, "
        f"failed tickers {len(result.failed_tickers)}"
    )
    for fail in result.failed_tickers:
        typer.echo(f"  ! {fail}")


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
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")] = None,
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
    # Maintain the BM25 sparse index alongside the vector store ($0, stdlib FTS5) so A3 hybrid is
    # ready without a separate backfill; harmless when retrieval_mode="dense".
    sparse_store = build_sparse_store(settings)
    tickers = _ingest_tickers(ingest_all, ticker, universe)
    try:
        # bulk_ingest isolates + retries per-ticker failures so one transient network blip
        # doesn't abort a multi-hour corpus embed (RAG_TODO 9c-run hardening).
        result = bulk_ingest(
            tickers,
            documents_dir=settings.documents_dir,
            embedder=embedder,
            store=store,
            chunk_tokens=settings.rag_chunk_tokens,
            chunk_overlap=settings.rag_chunk_overlap,
            max_embed_tokens=settings.rag_max_embed_tokens,
            sparse_store=sparse_store,
        )
    except EmbedBudgetExceeded as exc:
        # Spend ceiling hit (RAG_TODO 9a): abort loud — deliberate budget stop, not a transient
        # failure, so it is NOT isolated/retried by bulk_ingest.
        typer.echo(f"Aborting: {exc}")
        raise typer.Exit(code=1) from exc
    for r in result.per_ticker:
        typer.echo(
            f"{r.ticker}: {r.filings} filings → {r.chunks} chunks "
            f"(~{r.embed_tokens:,} embed tokens) ingested"
        )
    typer.echo(
        f"— total: {result.tickers} tickers → {result.chunks:,} chunks, "
        f"~{result.embed_tokens:,} embed tokens, failed {len(result.failed_tickers)}"
    )
    for fail in result.failed_tickers:
        typer.echo(f"  ! {fail}")
    if result.failed_tickers:
        typer.echo(
            f"Re-run to backfill the {len(result.failed_tickers)} failed ticker(s) (idempotent)."
        )


@documents_app.command("backfill-sparse")
def backfill_sparse_cmd(
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")] = None,
    backfill_all: Annotated[
        bool, typer.Option("--all", help="Backfill every ticker in the universe file")
    ] = False,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Populate the BM25 sparse index from already-downloaded filings — NO embedding ($0).

    A corpus dense-ingested before A3 (hybrid) has an empty sparse index, so hybrid silently falls
    back to dense. This indexes the same raw filings into BM25 (parse + chunk + index only; the
    embedder is never touched), making hybrid live. Idempotent/incremental — re-running adds only
    chunks not yet indexed.
    """
    settings = get_settings()
    configure_logging(settings)
    sparse_store = build_sparse_store(settings)
    tickers = _ingest_tickers(backfill_all, ticker, universe)
    indexed = backfill_sparse(
        tickers,
        documents_dir=settings.documents_dir,
        sparse_store=sparse_store,
        chunk_tokens=settings.rag_chunk_tokens,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    typer.echo(
        f"Backfilled BM25 sparse index: {indexed:,} new chunk(s) across {len(tickers)} ticker(s) "
        f"(total indexed now {sparse_store.count():,}). No embedding — $0."
    )


@documents_app.command("extract-graph")
def extract_graph_cmd(
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")] = None,
    extract_all: Annotated[
        bool, typer.Option("--all", help="Extract for every ticker in the universe file")
    ] = False,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Build the A5 knowledge graph from ingested filings (PAID: one LLM call per filing-section).

    Offline, cost-gated extraction (advanced-RAG A5.1): for each ticker it parses + chunks the
    downloaded filings, keeps only the configured graph sections (10-K Item 1 / Item 1A), and makes
    one structured-output call per filing-section to extract typed, **verified, provenance-bearing**
    `(subject, relation, object)` edges into the SQLite graph store. Edges whose object is absent
    from the cited chunk are dropped (the hallucinated-edge guard). Bounded by
    `settings.graph_max_extract_calls` (a run-wide ceiling). Idempotent — re-running upserts.
    """
    settings = get_settings()
    configure_logging(settings)
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is required for extract-graph (extraction makes LLM calls).")
        raise typer.Exit(code=1)
    llm = AnthropicClient(settings)
    alias_map = load_alias_map()
    graph_store = build_graph_store(settings)
    tickers = _ingest_tickers(extract_all, ticker, universe)

    ceiling = settings.graph_max_extract_calls
    used = 0
    n_entities = 0
    n_edges = 0
    for sym in tickers:
        if ceiling is not None and used >= ceiling:
            typer.echo(f"Reached extraction call ceiling ({ceiling}); stopping before {sym}.")
            break
        chunks = select_extraction_chunks(
            build_chunks(
                sym,
                documents_dir=settings.documents_dir,
                chunk_tokens=settings.rag_chunk_tokens,
                chunk_overlap=settings.rag_chunk_overlap,
            ),
            settings.graph_sections,
            alias_map,
        )
        remaining = None if ceiling is None else ceiling - used
        try:
            res = extract_edges(
                sym,
                chunks,
                llm=llm,
                alias_map=alias_map,
                min_confidence=settings.graph_min_confidence,
                max_calls=remaining,
            )
        except GraphExtractBudgetExceeded as exc:
            typer.echo(f"Aborting at {sym}: {exc}")
            break
        graph_store.add_entities(res.entities)
        graph_store.add_edges(res.edges)
        used += res.calls
        n_entities += len(res.entities)
        n_edges += len(res.edges)
        typer.echo(
            f"  {sym}: {len(res.edges)} edge(s), {len(res.entities)} entity(ies) "
            f"({res.calls} LLM call(s))"
        )
    counts = graph_store.count()
    typer.echo(
        f"Graph now has {counts.nodes:,} node(s), {counts.edges:,} edge(s) "
        f"(this run: +{n_edges} edge(s) across {used} LLM call(s))."
    )


@documents_app.command("refresh")
def refresh(
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker, e.g. NVDA")] = None,
    refresh_all: Annotated[
        bool, typer.Option("--all", help="Refresh every ticker in the universe file")
    ] = False,
    months: Annotated[
        int, typer.Option("--months", help="Look-back window for newly-filed documents")
    ] = 6,
    limit: Annotated[
        int, typer.Option("--limit", help="Max filings per form per ticker (within the window)")
    ] = 20,
    forms: Annotated[
        list[str] | None,
        typer.Option("--forms", help="Filing forms (repeatable); default 10-K 10-Q 8-K"),
    ] = None,
    universe: Annotated[
        Path, typer.Option("--universe", help="Universe file used with --all")
    ] = Path("configs/universe.txt"),
) -> None:
    """Pull newly-filed SEC documents + **incrementally** ingest them (RAG_TODO 9e refresh).

    Cheap to repeat on a schedule (e.g. quarterly via launchd; see `make refresh-filings`): only
    filings newer than the last download arrive, and only **new** chunks are embedded (existing
    ones are skipped), so a refresh costs a small fraction of a full rebuild. Embeds with the
    configured production embedder (e.g. voyage-4) under the `rag_max_embed_tokens` ceiling.
    """
    settings = get_settings()
    configure_logging(settings)
    if not settings.sec_user_agent:
        typer.echo(
            "SEC_USER_AGENT is not set. SEC fair-access requires a contact User-Agent — "
            'add SEC_USER_AGENT="Your Name your@email.com" to your .env.'
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

    since_floor = date.today() - timedelta(days=30 * months)
    tickers = _ingest_tickers(refresh_all, ticker, universe)  # reuse --ticker/--all resolver

    # 1) Pull newly-filed documents (idempotent; manifest skips what's already on disk).
    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    provider = SecEdgarProvider(settings, cache)
    dl = bulk_download(
        tickers,
        provider,
        documents_dir=settings.documents_dir,
        forms=forms_t,
        limit=limit,
        since=since_floor,
    )
    changed = [r.ticker for r in dl.per_ticker if r.downloaded]
    typer.echo(
        f"Downloaded {dl.downloaded} new filing(s) across {len(changed)} ticker(s) "
        f"since {since_floor.isoformat()}."
    )
    # 2) Incrementally ingest ONLY the tickers that got new filings (and only their new chunks).
    if changed:
        embedder = build_embedder(settings)
        store = build_vector_store(settings)
        sparse_store = build_sparse_store(settings)  # keep BM25 index in lockstep (A3)
        try:
            ing = bulk_ingest(
                changed,
                documents_dir=settings.documents_dir,
                embedder=embedder,
                store=store,
                chunk_tokens=settings.rag_chunk_tokens,
                chunk_overlap=settings.rag_chunk_overlap,
                max_embed_tokens=settings.rag_max_embed_tokens,
                incremental=True,
                sparse_store=sparse_store,
            )
        except EmbedBudgetExceeded as exc:
            typer.echo(f"Aborting: {exc}")
            raise typer.Exit(code=1) from exc
        typer.echo(
            f"Ingested {ing.chunks:,} new chunk(s) (~{ing.embed_tokens:,} embed tokens; "
            f"{ing.skipped_existing:,} already present), failed {len(ing.failed_tickers)}."
        )
        for fail in ing.failed_tickers:
            typer.echo(f"  ! ingest {fail}")
    else:
        typer.echo("Corpus is up to date — nothing new to ingest.")
    for fail in dl.failed_tickers:
        typer.echo(f"  ! download {fail}")


# ---- rag (SEC-grounded retrieval) --------------------------------------------
rag_app = typer.Typer(add_completion=False, help="Query the ingested SEC corpus (RAG).")
app.add_typer(rag_app, name="rag")


@rag_app.command("status")
def rag_status() -> None:
    """Show the active embedder, vector collection + chunk count, and corpus freshness.

    Confirms which embeddings the chat agent / research pipeline are using (e.g. voyage-4) and how
    current the SEC corpus is. Read-only — no network, no LLM, no embedding calls.
    """
    settings = get_settings()
    configure_logging(settings)
    cs = corpus_status(settings)
    typer.echo(f"Embedder   : {cs.embedder}  (provider={cs.provider})")
    if cs.chunks < 0:
        typer.echo(f"Collection : {cs.collection}  (unavailable)")
    else:
        typer.echo(f"Collection : {cs.collection}  ({cs.chunks:,} chunks)")
        if cs.chunks == 0:
            typer.echo("             (empty — run `documents ingest --all`)")
    if cs.filings:
        typer.echo(f"Downloaded : {cs.filings:,} filings across {cs.tickers} tickers")
        typer.echo(f"Freshness  : {cs.earliest} → {cs.latest} (filing dates)")
    else:
        typer.echo("Downloaded : none (run `documents download-sec --all`)")


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
    # Default-OFF rerank: dense Retriever unless settings.rerank_provider is set (then wrapped).
    retriever = build_retrieval_system(settings)
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


@rag_app.command("graph-query")
def rag_graph_query(
    question: Annotated[str, typer.Option("--question", "-q", help="Natural-language question")],
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-t", help="Seed entity (else from the query)"),
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Chunks to return (default settings.rag_top_k)")
    ] = None,
    answer: Annotated[
        bool, typer.Option("--answer", help="Synthesize a cited answer (one LLM call) instead")
    ] = False,
) -> None:
    """Retrieve via the A5 GraphRetriever (traversal ⊕ vector); with --answer, a cited answer.

    Seeds the traversal from `--ticker` (or company names in the question), folds in neighbor
    companies' chunks + the relationship-stating provenance chunks, and fuses with the vector
    base by RRF. Traversal is $0/local; build the graph first with `documents extract-graph`.
    """
    settings = get_settings()
    configure_logging(settings)
    retriever = build_graph_system(settings)
    where = ChunkFilter(ticker=normalize_ticker(ticker)) if ticker else None
    evidence = retriever.retrieve(question, top_k=top_k or settings.rag_top_k, where=where)

    if answer:
        grounded = answer_question(question, evidence, llm=AnthropicClient(settings))
        typer.echo(grounded.answer)
        for cite in grounded.citations:
            typer.echo(f"  [{cite.marker}] {cite.label}")
        return

    if evidence.is_empty:
        typer.echo("No matching evidence found. Ingest + extract-graph first.")
        raise typer.Exit(code=1)
    for i, rc in enumerate(evidence.chunks, start=1):
        typer.echo(f"\n[{i}] {rc.citation_label()}  (score {rc.score:.3f})")
        typer.echo(textwrap.shorten(rc.chunk.text, width=280))


@rag_app.command("ask")
def rag_ask(
    question: Annotated[str, typer.Option("--question", "-q", help="Multi-hop question")],
    multi: Annotated[
        bool,
        typer.Option(
            "--multi/--single",
            help="Multi-hop ReAct loop (default) vs. a single-shot retrieval+answer.",
        ),
    ] = True,
    max_steps: Annotated[
        int | None, typer.Option("--max-steps", help="Cap decision steps (default settings)")
    ] = None,
    graph: Annotated[
        bool, typer.Option("--graph", help="Retrieve via the A5 GraphRetriever (traversal+vector)")
    ] = False,
) -> None:
    """Answer a question with the A4 multi-hop ReAct loop (`--single` for one-shot).

    Bounded agentic retrieval: each step reasons over evidence-so-far and either retrieves more or
    stops, then the reused P7 guarded synthesis answers over the union. ``--single`` falls back to
    one retrieval + one ``answer_question`` (the plain `rag query --answer` path) for comparison.
    ``--graph`` swaps in the A5 graph retriever (build the graph first: `documents extract-graph`).
    """
    settings = get_settings()
    configure_logging(settings)
    llm = AnthropicClient(settings)
    # The retrieval substrate (A5 graph or the default vector/hybrid stack); both satisfy the same
    # RetrievalSystem contract, so the loop / single-shot path are agnostic to the choice.
    retriever = build_graph_system(settings) if graph else build_retrieval_system(settings)

    if not multi:
        evidence = retriever.retrieve(question, top_k=settings.rag_top_k)
        grounded = answer_question(question, evidence, llm=llm)
        typer.echo(grounded.answer)
        for cite in grounded.citations:
            typer.echo(f"  [{cite.marker}] {cite.label}")
        return

    result = answer_multistep(
        question, settings=settings, llm=llm, retriever=retriever, max_steps=max_steps
    )
    # Trace first (what the loop did), then the cited answer — transparency for an agentic run.
    for i, st in enumerate(result.trace, start=1):
        scope = f" [{st.ticker}]" if st.ticker else ""
        typer.echo(f"  step {i}: {st.query!r}{scope} → {st.n_retrieved} chunks")
    typer.echo(f"  ({result.n_steps} steps, {result.n_evidence} evidence chunks)\n")
    typer.echo(result.answer.answer)
    for cite in result.answer.citations:
        typer.echo(f"  [{cite.marker}] {cite.label}")


def _named_embedder(name: str, settings: Settings) -> Embedder:
    """Map an A/B ``--compare`` name to a constructed embedder (RAG_TODO 9b).

    Independent of ``embedding_provider`` so a single run can compare several. ``voyage*`` /
    ``openai`` need their API key (lazy — errors only when actually embedded).
    """
    key = name.strip().lower()
    if key in ("local", "fastembed"):
        return FastEmbedEmbedder(settings.embedding_model)
    if key in ("voyage", "voyage-4"):
        return VoyageEmbedder(settings)
    if key in ("voyage-finance", "voyage-finance-2"):
        return VoyageEmbedder(settings, model="voyage-finance-2")
    if key == "openai":
        return OpenAIEmbedder(settings)
    raise typer.BadParameter(
        f"unknown embedder '{name}' (use: local, voyage, voyage-finance, openai)"
    )


def _eval_corpus(settings: Settings, labeled: list[LabeledQuery]) -> list[DocumentChunk]:
    """Build the chunk corpus for the tickers referenced by the labeled queries (chunking fixed)."""
    corpus: list[DocumentChunk] = []
    for sym in sorted({q.ticker for q in labeled if q.ticker}):
        corpus.extend(
            build_chunks(
                sym,
                documents_dir=settings.documents_dir,
                chunk_tokens=settings.rag_chunk_tokens,
                chunk_overlap=settings.rag_chunk_overlap,
            )
        )
    return corpus


def _eval_lattice(
    settings: Settings,
    labeled: list[LabeledQuery],
    corpus: list[DocumentChunk],
    *,
    system_names: list[str],
    rerank_provider: str,
    diagnostic: bool,
    top_k: int,
) -> list[SystemReport]:
    """Score the named retrieval lattice on one corpus (embedder + chunking held constant).

    Embeds + BM25-indexes the corpus ONCE, then builds each named config (`dense` / `reranked` /
    `hybrid` / `hybrid+rerank`) via `build_named_system` and scores it with `evaluate_system`, so
    the systems differ only in the retrieval *stage*. `--diagnostic` appends a sparse-only (BM25)
    baseline — a reference point, not a deployable mode. In-memory stores; offline except the
    configured embedder / (for rerank-bearing systems) the local reranker model.
    """
    emb = build_embedder(settings)
    store = InMemoryVectorStore()
    store.add(corpus, emb.embed_documents([c.text for c in corpus]))
    sparse = InMemoryBM25Store()
    sparse.add(corpus)
    reports: list[SystemReport] = []
    for name in system_names:
        system = build_named_system(
            name, settings, store=store, sparse_store=sparse, rerank_provider=rerank_provider
        )
        reports.append(evaluate_system(system, labeled, top_k=top_k, corpus_chunks=corpus))
    if diagnostic:
        reports.append(
            evaluate_system(SparseRetriever(sparse), labeled, top_k=top_k, corpus_chunks=corpus)
        )
    return reports


@rag_app.command("eval")
def rag_eval(
    queries: Annotated[
        Path, typer.Option("--queries", "-q", help="JSON file: list of labeled queries")
    ],
    compare: Annotated[
        str,
        typer.Option("--compare", help="Embedder A/B: comma list local,voyage,openai"),
    ] = "local",
    systems: Annotated[
        str | None,
        typer.Option(
            "--systems",
            help="Lattice mode: comma list of " + ",".join(LATTICE_SYSTEMS),
        ),
    ] = None,
    rerank: Annotated[
        str, typer.Option("--rerank", help="Reranker for rerank-bearing systems: local|voyage")
    ] = "local",
    diagnostic: Annotated[
        bool,
        typer.Option("--diagnostic", help="Add a sparse-only BM25 baseline (not deployable)"),
    ] = False,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Chunks per query (default settings.rag_top_k)")
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Also write the reports as JSON to this path"),
    ] = None,
) -> None:
    """Score retrieval quality on a labeled query set (A1 metrics: hit@k / MRR / nDCG@k / P / R).

    Two modes over the same corpus (ingested filings for the query tickers, chunking fixed):
    - **Embedder A/B** (default, `--compare`): compare embedders, each as a dense retriever (P9b).
    - **Lattice** (`--systems`): compare retrieval *stages* — dense / reranked / hybrid /
      hybrid+rerank — holding the embedder fixed, to decide which stage to promote. `--diagnostic`
      adds a sparse-only baseline.

    `--report` dumps the reports as JSON. Run via `make rag-eval` for the real local corpus.
    """
    settings = get_settings()
    configure_logging(settings)
    labeled = [LabeledQuery.model_validate(obj) for obj in json.loads(queries.read_text())]
    corpus = _eval_corpus(settings, labeled)
    if not corpus:
        typer.echo("No downloaded filings for the query tickers. Run download-sec + ingest first.")
        raise typer.Exit(code=1)
    k = top_k or settings.rag_top_k
    reports: Sequence[SystemReport]
    if systems:
        names = [s.strip() for s in systems.split(",") if s.strip()]
        unknown = [n for n in names if n not in LATTICE_SYSTEMS]
        if unknown:
            raise typer.BadParameter(
                f"unknown system(s) {unknown}; choose from {list(LATTICE_SYSTEMS)}"
            )
        reports = _eval_lattice(
            settings,
            labeled,
            corpus,
            system_names=names,
            rerank_provider=rerank,
            diagnostic=diagnostic,
            top_k=k,
        )
    else:
        embedders = {n: _named_embedder(n, settings) for n in compare.split(",") if n.strip()}
        reports = run_ab(corpus, labeled, embedders, top_k=k)
    typer.echo(format_reports_markdown(reports))
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8"
        )
        typer.echo(f"Wrote {len(reports)} report(s) to {report}")


@rag_app.command("eval-multistep")
def rag_eval_multistep(
    queries: Annotated[
        Path, typer.Option("--queries", "-q", help="JSON file: list of labeled multi-hop queries")
    ],
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", help="Single-shot baseline top-k (default settings.rag_top_k)"),
    ] = None,
    report: Annotated[
        Path | None, typer.Option("--report", help="Also write reports + summary JSON to this path")
    ] = None,
    graph: Annotated[
        bool, typer.Option("--graph", help="Run the loop over the A5 GraphRetriever (A5.3 measure)")
    ] = False,
) -> None:
    """Measure the A4 agentic loop's multi-hop coverage vs. a single-shot baseline (PAID: real LLM).

    Loads labeled multi-hop questions — each with `aspects` (answer-bearing spans per hop) — then
    for each runs the ReAct loop AND one single retrieval over the configured corpus, and reports
    **coverage**, the **coverage gain** (multi-step − single-shot), citation accuracy, and loop
    behaviour. Needs ANTHROPIC_API_KEY and the query tickers' filings ingested. See
    `configs/rag_eval_multistep.example.json`; run via `make rag-eval-multistep`. ``--graph`` runs
    both arms over the A5 graph retriever (the A5.3 bridging-benchmark comparison vs. hybrid).
    """
    settings = get_settings()
    configure_logging(settings)
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is required for eval-multistep (the loop makes LLM calls).")
        raise typer.Exit(code=1)
    from stock_agent.llm.client import AnthropicClient

    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(queries.read_text())]
    retriever = build_graph_system(settings) if graph else None  # None → default hybrid read path
    reports, summary = evaluate_multihop_set(
        labeled, settings=settings, llm=AnthropicClient(settings), retriever=retriever, top_k=top_k
    )
    typer.echo(format_multihop_markdown(reports, summary))
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "summary": summary.model_dump(mode="json"),
                    "reports": [r.model_dump(mode="json") for r in reports],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"Wrote {len(reports)} report(s) to {report}")


@rag_app.command("gen-multistep")
def rag_gen_multistep(
    universe: Annotated[
        Path, typer.Option("--universe", help="Graph universe file (one ticker per line)")
    ] = Path("configs/graph_universe.txt"),
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the generated benchmark JSON")
    ] = Path("configs/rag_eval_multistep_generated.json"),
    target_count: Annotated[
        int | None,
        typer.Option("--target-count", help="Cap on emitted questions (a CAP, never a filler)"),
    ] = None,
    per_seed_relation_cap: Annotated[
        int | None,
        typer.Option("--per-seed-cap", help="Max questions per (seed, relation) — diversity guard"),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="RNG seed (target-sampling reproducibility)")
    ] = 0,
    supply_report: Annotated[
        Path, typer.Option("--supply-report", help="Where to write the D1 supply report JSON")
    ] = Path("outputs/rag_eval/multistep_supply.json"),
    split_test_frac: Annotated[
        float | None,
        typer.Option("--split-test-frac", help="Also write group-wise train/test splits (D2)"),
    ] = None,
) -> None:
    """Generate the A6.0 multi-hop benchmark from the A5 graph + ingested corpus (offline, $0).

    Mines graph-verified bridge pairs × topics into stratified, corpus-checked `MultiHopQuery` rows
    (the A6 episode catalog). Needs the graph (`documents extract-graph`) and the sparse corpus
    ingested; no LLM, no network. Prints the **D1 supply report** (count is an output = clean
    supply; `--target-count` caps, never dilutes). `--split-test-frac` also writes the **D2** group
    train/test splits so a pair's variants never straddle the fold.
    """
    settings = get_settings()
    configure_logging(settings)
    if not universe.exists():
        typer.echo(f"Universe file not found: {universe}")
        raise typer.Exit(code=1)
    graph = build_graph_store(settings)
    sparse = build_sparse_store(settings)
    queries, report = generate_multihop(
        graph,
        sparse,
        load_alias_map(),
        seeds=load_universe(universe),
        seed=seed,
        target_count=target_count,
        per_seed_relation_cap=per_seed_relation_cap,
    )
    typer.echo(report.render())

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([q.model_dump(mode="json") for q in queries], indent=2), encoding="utf-8"
    )
    supply_report.parent.mkdir(parents=True, exist_ok=True)
    supply_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"\nWrote {len(queries)} questions to {out}; supply report to {supply_report}")

    if split_test_frac is not None:
        train, test = split_multihop(queries, test_frac=split_test_frac, seed=seed)
        for name, rows in (("train", train), ("test", test)):
            path = out.with_suffix(f".{name}.json")
            path.write_text(
                json.dumps([q.model_dump(mode="json") for q in rows], indent=2), encoding="utf-8"
            )
        typer.echo(f"Group-wise split (D2): {len(train)} train / {len(test)} test written.")


def _format_policy_eval(report: PolicyEvalReport) -> str:
    """Human-readable verdict table: per-policy value + the promote/reject decision."""
    lines = [
        f"Policy eval — {report.n_train} train / {report.n_test} test "
        f"(seed={report.seed}, λ_c={report.lambda_cost:g}, arms={list(report.arms)})",
        "",
        f"  {'policy':<24}{'DR':>8}{'  95% CI':>18}{'true':>8}{'ESS':>8}",
    ]
    for v in sorted(report.values, key=lambda x: x.dr, reverse=True):
        ci = f"[{v.dr_ci[0]:+.3f},{v.dr_ci[1]:+.3f}]"
        lines.append(f"  {v.name:<24}{v.dr:>8.3f}{ci:>18}{v.true_value:>8.3f}{v.ess:>8.1f}")
    lines += [
        "",
        f"  best fixed : {report.best_fixed}",
        f"  bandit     : {report.bandit}",
        f"  Δ DR       : {report.delta_dr:+.4f}  CI [{report.delta_ci[0]:+.4f}, "
        f"{report.delta_ci[1]:+.4f}]  P(Δ>0)={report.delta_p_positive:.1%}",
        "  per-stratum (bandit − best fixed):",
    ]
    for s in report.per_stratum:
        lines.append(
            f"    {s.stratum:<5} n={s.n:<4} bandit={s.dr_bandit:+.3f} "
            f"fixed={s.dr_fixed:+.3f} Δ={s.delta:+.4f}"
        )
    lines += ["", f"  VERDICT: {report.rationale}"]
    return "\n".join(lines)


@rag_app.command("policy-eval")
def rag_policy_eval(
    queries: Annotated[
        Path, typer.Option("--queries", "-q", help="Benchmark JSON (list of MultiHopQuery)")
    ] = Path("configs/rag_eval_multistep_generated.json"),
    policy: Annotated[
        str, typer.Option("--policy", help="Learned bandit to judge: linucb | epsilon_greedy")
    ] = "linucb",
    test_frac: Annotated[
        float, typer.Option("--test-frac", help="Group-wise held-out fraction (leakage-safe)")
    ] = 0.3,
    seed: Annotated[
        int, typer.Option("--seed", help="RNG seed (split + mu-logs + bootstrap)")
    ] = 42,
    alpha: Annotated[float, typer.Option("--alpha", help="LinUCB exploration weight")] = 1.0,
    epsilon: Annotated[
        float, typer.Option("--epsilon", help="epsilon-greedy exploration rate")
    ] = 0.1,
    lambda_cost: Annotated[
        float | None,
        typer.Option("--lambda-cost", help="Cost weight λ_c (default settings.reward_lambda_cost)"),
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Retrieval depth (default settings.rag_top_k)")
    ] = None,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    n_boot: Annotated[
        int, typer.Option("--n-boot", help="Group-bootstrap resamples for the CIs")
    ] = 1000,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Verdict JSON path (default timestamped)")
    ] = None,
) -> None:
    """Train the retrieval bandit offline and judge it vs the best fixed arm (A6.1f verdict, local).

    Builds the 5-arm reward matrix over the benchmark ($0 retrieval — needs the ingested corpus +
    the A5 graph, like `rag eval-multistep`), group-splits it, fits the learned policy on train, and
    off-policy-evaluates (IPS/SNIPS/DR + group bootstrap CI) on the held-out test fold. Applies the
    pre-registered rule: promote `adaptive_retrieval` iff `DR(bandit) − DR(best fixed)` clears the
    95% CI and does not regress the CTRL stratum. Writes the verdict JSON for the notes docs.
    """
    from datetime import UTC, datetime

    settings = get_settings()
    configure_logging(settings)
    if policy not in ("linucb", "epsilon_greedy"):
        typer.echo("--policy must be 'linucb' or 'epsilon_greedy' (a learned bandit).")
        raise typer.Exit(code=1)
    if not queries.exists():
        typer.echo(f"Benchmark not found: {queries} (generate via `rag gen-multistep`).")
        raise typer.Exit(code=1)

    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(queries.read_text())]
    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    # ARMS order == reward-matrix column order == policy action order (non-negotiable alignment).
    systems = {name: build_arm_system(name, settings) for name in ARMS}
    dataset = build_dataset(
        labeled,
        systems,
        settings=settings,
        alias_map=alias_map,
        graph_universe=universe,
        lambda_cost=lambda_cost,
        top_k=top_k,
    )
    train, test = split_dataset(dataset, labeled, test_frac=test_frac, seed=seed)
    lam = lambda_cost if lambda_cost is not None else settings.reward_lambda_cost
    report = evaluate_offline(
        train,
        test,
        bandit=policy,
        alpha=alpha,
        epsilon=epsilon,
        lambda_cost=lam,
        n_boot=n_boot,
        seed=seed,
    )
    typer.echo(_format_policy_eval(report))

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out or Path(f"outputs/rag_eval/policy_eval_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    typer.echo(f"\nWrote verdict to {out}")


def _format_gated_eval(report: GatedEvalReport) -> str:
    """Human-readable gated-router verdict: value table + the two pre-registered decisions."""
    lines = [
        f"Gated-router eval — {report.n_train} train / {report.n_test} test "
        f"(seed={report.seed}, λ_c={report.lambda_cost:g}, gate={report.gate_feature}, "
        f"easy={report.easy_arm}, arms={list(report.arms)})",
        "",
        f"  {'policy':<32}{'DR':>8}{'  95% CI':>18}{'true':>8}{'ESS':>8}",
    ]
    for v in sorted(report.values, key=lambda x: x.dr, reverse=True):
        ci = f"[{v.dr_ci[0]:+.3f},{v.dr_ci[1]:+.3f}]"
        lines.append(f"  {v.name:<32}{v.dr:>8.3f}{ci:>18}{v.true_value:>8.3f}{v.ess:>8.1f}")
    # Verdict 1 — promote the gated router vs the best fixed arm.
    lines += [
        "",
        "  [1] PROMOTE gated router?",
        f"    gated      : {report.gated_policy}",
        f"    best fixed : {report.best_fixed}",
        f"    Δ DR       : {report.delta_dr:+.4f}  CI [{report.delta_ci[0]:+.4f}, "
        f"{report.delta_ci[1]:+.4f}]  P(Δ>0)={report.delta_p_positive:.1%}",
        "    per-stratum (gated − best fixed):",
    ]
    for s in report.per_stratum:
        lines.append(
            f"      {s.stratum:<5} n={s.n:<4} gated={s.dr_bandit:+.3f} "
            f"fixed={s.dr_fixed:+.3f} Δ={s.delta:+.4f}"
        )
    lines.append(f"    VERDICT: {report.rationale}")
    # Verdict 2 — does the bandit EARN the hard branch (vs the fixed graph default) on HARD+MED?
    lines += [
        "",
        f"  [2] Does the bandit EARN the hard branch?  (n={report.hard_n} HARD+MED rows)",
        f"    {report.hard_bandit} vs {report.hard_fixed}",
        f"    Δ DR       : {report.hard_delta_dr:+.4f}  CI [{report.hard_delta_ci[0]:+.4f}, "
        f"{report.hard_delta_ci[1]:+.4f}]  P(Δ>0)={report.hard_delta_p_positive:.1%}",
        f"    VERDICT: {report.hard_rationale}",
    ]
    return "\n".join(lines)


@rag_app.command("gated-eval")
def rag_gated_eval(
    queries: Annotated[
        Path, typer.Option("--queries", "-q", help="Benchmark JSON (list of MultiHopQuery)")
    ] = Path("configs/rag_eval_multistep_generated.json"),
    easy_arm: Annotated[
        str, typer.Option("--easy-arm", help="Cheap arm the gate routes easy queries to")
    ] = "dense",
    gate_feature: Annotated[
        str, typer.Option("--gate-feature", help="Label-free deploy-time gate signal")
    ] = "is_bridging",
    hard_fixed_arm: Annotated[
        str, typer.Option("--hard-fixed-arm", help="Fixed default the bandit must beat on HARD+MED")
    ] = "graph",
    test_frac: Annotated[
        float, typer.Option("--test-frac", help="Group-wise held-out fraction (leakage-safe)")
    ] = 0.3,
    seed: Annotated[
        int, typer.Option("--seed", help="RNG seed (split + mu-logs + bootstrap)")
    ] = 42,
    alpha: Annotated[float, typer.Option("--alpha", help="LinUCB exploration weight")] = 1.0,
    epsilon: Annotated[
        float, typer.Option("--epsilon", help="epsilon-greedy exploration rate")
    ] = 0.1,
    lambda_cost: Annotated[
        float | None,
        typer.Option("--lambda-cost", help="Cost weight λ_c (default settings.reward_lambda_cost)"),
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", help="Retrieval depth (default settings.rag_top_k)")
    ] = None,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    n_boot: Annotated[
        int, typer.Option("--n-boot", help="Group-bootstrap resamples for the CIs")
    ] = 1000,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Verdict JSON path (default timestamped)")
    ] = None,
) -> None:
    """Evaluate the gated router: easy→cheap arm, hard→bandit — the A6.1-verdict follow-up (local).

    Same reward matrix + group split as `rag policy-eval`, but judges `gated(easy_arm | linucb)`
    instead of the bare bandit, and answers TWO pre-registered questions: (1) does the gated router
    beat the best fixed arm (paired group-bootstrap CI, no CTRL regression)? and (2) does the bandit
    *earn* the hard branch — beat `fixed(hard_fixed_arm)` on the HARD+MED rows only, rather than
    being assumed onto it? Writes the verdict JSON (with the nested `hard_branch` block) for docs.
    """
    from datetime import UTC, datetime

    settings = get_settings()
    configure_logging(settings)
    if not queries.exists():
        typer.echo(f"Benchmark not found: {queries} (generate via `rag gen-multistep`).")
        raise typer.Exit(code=1)
    if easy_arm not in ARMS or hard_fixed_arm not in ARMS:
        typer.echo(f"--easy-arm/--hard-fixed-arm must be one of {list(ARMS)}.")
        raise typer.Exit(code=1)

    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(queries.read_text())]
    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    # ARMS order == reward-matrix column order == policy action order (non-negotiable alignment).
    systems = {name: build_arm_system(name, settings) for name in ARMS}
    dataset = build_dataset(
        labeled,
        systems,
        settings=settings,
        alias_map=alias_map,
        graph_universe=universe,
        lambda_cost=lambda_cost,
        top_k=top_k,
    )
    train, test = split_dataset(dataset, labeled, test_frac=test_frac, seed=seed)
    lam = lambda_cost if lambda_cost is not None else settings.reward_lambda_cost
    report = evaluate_gated(
        train,
        test,
        easy_arm=easy_arm,
        gate_feature=gate_feature,
        hard_fixed_arm=hard_fixed_arm,
        alpha=alpha,
        epsilon=epsilon,
        lambda_cost=lam,
        n_boot=n_boot,
        seed=seed,
    )
    typer.echo(_format_gated_eval(report))

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out or Path(f"outputs/rag_eval/gated_eval_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    typer.echo(f"\nWrote verdict to {out}")


@rag_app.command("rl-train")
def rag_rl_train(
    queries: Annotated[
        Path,
        typer.Option("--queries", "-q", help="Train-fold benchmark JSON (list of MultiHopQuery)"),
    ] = Path("configs/rag_eval_multistep_generated.train.json"),
    algo: Annotated[
        str, typer.Option("--algo", help="Learner: bc | reinforce | ppo (ppo needs the [rl] extra)")
    ] = "reinforce",
    action_space: Annotated[
        str, typer.Option("--action-space", help="Action space preset: pruned (7) | full (16)")
    ] = "pruned",
    seed: Annotated[int, typer.Option("--seed", help="Master RNG seed (init + rollouts)")] = 0,
    iterations: Annotated[
        int, typer.Option("--iterations", help="REINFORCE/PPO iterations (bc uses its own epochs)")
    ] = 200,
    lr: Annotated[
        float | None,
        typer.Option("--lr", help="Learning rate override (defaults per algo: reinforce/bc/ppo)"),
    ] = None,
    gamma: Annotated[float, typer.Option("--gamma", help="Discount γ (1.0 = exact cover)")] = 1.0,
    lambda_cost: Annotated[
        float | None,
        typer.Option("--lambda-cost", help="Cost weight λ_c (default settings.reward_lambda_cost)"),
    ] = None,
    test_frac: Annotated[
        float,
        typer.Option("--test-frac", help="If >0, group-split and train on the train side (safety)"),
    ] = 0.0,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Checkpoint dir (default outputs/experiments/<run_id>)"),
    ] = None,
) -> None:
    """Train a retrieval-RL policy on the group-split train fold and freeze it (A6.2f, local, $0).

    Builds the deterministic retrieval MDP over the labeled episodes (real $0 retrieval — needs the
    ingested corpus + A5 graph, like `rag policy-eval`), fits a feature standardizer, trains
    `--algo`, and writes a self-describing `policy.json` (weights + STATE_FEATURE_NAMES order +
    action-space name + standardizer + config + seed) the A6.2g eval / a deploy can reload. No LLM.
    """
    from datetime import UTC, datetime

    settings = get_settings()
    configure_logging(settings)
    if algo not in ("bc", "reinforce", "ppo"):
        typer.echo("--algo must be one of bc | reinforce | ppo.")
        raise typer.Exit(code=1)
    if action_space not in ("pruned", "full"):
        typer.echo("--action-space must be 'pruned' or 'full'.")
        raise typer.Exit(code=1)
    if not queries.exists():
        typer.echo(f"Benchmark not found: {queries} (generate via `rag gen-multistep`).")
        raise typer.Exit(code=1)

    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(queries.read_text())]
    if test_frac > 0.0:  # optional internal group-split guard when given the FULL catalog
        labeled, _ = split_multihop(labeled, test_frac=test_frac, seed=seed)
    if not labeled:
        typer.echo("No training episodes after loading/splitting.")
        raise typer.Exit(code=1)

    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    env = RagRetrievalEnv(
        labeled,
        settings=settings,
        action_space=named_action_space(action_space),
        alias_map=alias_map,
        graph_universe=universe,
        gamma=gamma,
        lambda_cost=lambda_cost,
    )

    # Route a single --lr override to the algo-appropriate field (each optimizer has its own scale).
    lr_field = {"reinforce": "lr", "bc": "bc_lr", "ppo": "ppo_lr"}[algo]
    overrides: dict[str, float] = {lr_field: lr} if lr is not None else {}
    config = TrainConfig(
        algo=algo,
        action_space=action_space,
        seed=seed,
        gamma=gamma,
        lambda_cost=lambda_cost,
        iterations=iterations,
        queries_path=str(queries),
        test_frac=test_frac,
        **overrides,  # type: ignore[arg-type]
    )
    trained = train_policy(env, list(range(len(labeled))), config)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_rl-{algo}-{action_space}-s{seed}"
    checkpoint = (out_dir or Path("outputs/experiments") / run_id) / "policy.json"
    save_checkpoint(trained, checkpoint)

    final = trained.history[-1] if trained.history else float("nan")
    typer.echo(
        # `trained.config`, not `config` — train_policy fills n_train in on its own copy.
        f"Trained {algo} ({action_space}, {trained.n_actions} actions) on "
        f"{trained.config.n_train} episodes × {iterations if algo != 'bc' else config.bc_epochs} "
        f"{'epochs' if algo == 'bc' else 'iterations'}; "
        f"final {trained.history_metric}={final:.4f}\nWrote checkpoint to {checkpoint}"
    )


@rag_app.command("rl-eval")
def rag_rl_eval(
    policy_path: Annotated[
        Path, typer.Option("--policy", "-p", help="Frozen checkpoint from `rag rl-train`")
    ],
    train_queries: Annotated[
        Path,
        typer.Option("--train-queries", help="TRAIN fold (the one the policy was trained on)"),
    ] = Path("configs/rag_eval_multistep_generated.train.json"),
    test_queries: Annotated[
        Path, typer.Option("--test-queries", help="Held-out TEST-fold benchmark (group-disjoint)")
    ] = Path("configs/rag_eval_multistep_generated.test.json"),
    seeds: Annotated[
        str, typer.Option("--seeds", help="Comma-separated rollout seeds (sampled rows averaged)")
    ] = "0,1,2",
    react_arms: Annotated[
        str, typer.Option("--react-arms", help="Production arms for the A4-loop baseline")
    ] = "hybrid,graph",
    bandit: Annotated[
        bool,
        typer.Option("--bandit/--no-bandit", help="Fit the A6.1 LinUCB baseline (5-arm matrix)"),
    ] = True,
    lambda_cost: Annotated[
        float | None,
        typer.Option("--lambda-cost", help="Cost weight λ_c (default: the checkpoint's own)"),
    ] = None,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    n_boot: Annotated[
        int, typer.Option("--n-boot", help="Group-bootstrap resamples for the paired CI")
    ] = 1000,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed")] = 42,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Verdict JSON (default: next to the policy)")
    ] = None,
) -> None:
    """Judge a frozen RL policy on the held-out fold vs the 4 baselines (A6.2g verdict, local, $0).

    Replays the checkpoint greedily (deploy semantics) on the group-wise held-out episodes of the
    same deterministic MDP it trained in, against: the fixed pipelines (one per arm), the A4 ReAct
    loop + A5 bridge, the A6.1 LinUCB bandit, and a random policy — plus a reward-hacking sentinel.
    Reports mean return / coverage / retrievals / cost, the train-vs-held-out gap, and the paired
    group-bootstrap CI, then applies the pre-registered promote rule. No LLM (templated queries):
    the sim-to-real gap is measured separately.
    """
    settings = get_settings()
    configure_logging(settings)
    for path in (policy_path, train_queries, test_queries):
        if not path.exists():
            typer.echo(f"Not found: {path}")
            raise typer.Exit(code=1)

    loaded = load_checkpoint(policy_path)
    train_labeled = [
        MultiHopQuery.model_validate(obj) for obj in json.loads(train_queries.read_text())
    ]
    test_labeled = [
        MultiHopQuery.model_validate(obj) for obj in json.loads(test_queries.read_text())
    ]
    if not train_labeled or not test_labeled:
        typer.echo("Both folds must be non-empty.")
        raise typer.Exit(code=1)

    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    cfg = loaded.config
    # One env over BOTH folds ⇒ one shared TransitionCache: the baselines' retrievals ARE the
    # learned policy's cached ones, so every candidate sees byte-identical transitions.
    env = RagRetrievalEnv(
        train_labeled + test_labeled,
        settings=settings,
        action_space=named_action_space(
            loaded.action_space, n_discovered_slots=loaded.n_discovered_slots
        ),
        alias_map=alias_map,
        graph_universe=universe,
        gamma=float(cfg.get("gamma", 1.0)),  # the checkpoint's own reward definition, not a new one
        lambda_cost=lambda_cost if lambda_cost is not None else cfg.get("lambda_cost"),
    )
    train_idx = list(range(len(train_labeled)))
    test_idx = list(range(len(train_labeled), len(train_labeled) + len(test_labeled)))

    scorer, ctx_mean, ctx_std = None, None, None
    if bandit:
        typer.echo(f"Fitting the A6.1 LinUCB baseline on {len(train_labeled)} train episodes …")
        # ARMS order == reward-matrix column order == bandit action order (load-bearing alignment).
        systems = {name: build_arm_system(name, settings) for name in ARMS}
        train_ds = build_dataset(
            train_labeled,
            systems,
            settings=settings,
            alias_map=alias_map,
            graph_universe=universe,
            lambda_cost=lambda_cost,
        )
        scorer, ctx_mean, ctx_std = fit_bandit_baseline(train_ds, seed=seed)

    baselines = build_baselines(
        env,
        react_arms=tuple(a.strip() for a in react_arms.split(",") if a.strip()),
        bandit=scorer,
        bandit_arm_names=ARMS,
        context_mean=ctx_mean,
        context_std=ctx_std,
    )
    report = evaluate_rl(
        env,
        train_indices=train_idx,
        test_indices=test_idx,
        learned=loaded.policy,
        baselines=baselines,
        sentinel=AlwaysSearchPolicy(action_space=env.action_space, arm_costs=env.arm_costs),
        learned_name=f"rl-{loaded.algo}",
        algo=loaded.algo,
        action_space=loaded.action_space,
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
        n_boot=n_boot,
        seed=seed,
    )
    typer.echo("\n" + format_report_markdown(report))
    destination = out or policy_path.parent / "rl_eval.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    typer.echo(f"\nWrote verdict to {destination}")


@rag_app.command("rl-simreal")
def rag_rl_simreal(
    policy_path: Annotated[
        Path, typer.Option("--policy", "-p", help="Frozen checkpoint from `rag rl-train`")
    ],
    test_queries: Annotated[
        Path, typer.Option("--test-queries", help="Held-out benchmark (the fold rl-eval judged)")
    ] = Path("configs/rag_eval_multistep_generated.test.json"),
    n: Annotated[
        int, typer.Option("--n", help="Episodes to run (sampled deterministically from the strata)")
    ] = 24,
    strata: Annotated[
        str, typer.Option("--strata", help="Strata to draw from (bridges are the sim's weak spot)")
    ] = "HARD,MED",
    seed: Annotated[int, typer.Option("--seed", help="Episode-subset seed (reproducible)")] = 0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Use a $0 stand-in LLM to test the harness"),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm the paid LLM calls (required unless --dry-run)")
    ] = False,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Report JSON (default: next to the policy)")
    ] = None,
) -> None:
    """Measure the sim-to-real gap of a frozen RL policy (A6.2g — **PAID**, ~$2-3 for 24 episodes).

    Runs the same policy over the same held-out episodes twice: once with the $0 simulator's
    TEMPLATED queries (what `rag rl-eval` judged), once with the query text written by the LLM at
    each hop (what production does). The policy still owns which arm, where to point it and when to
    stop — the LLM only phrases the search — so the difference in coverage IS the gap. The
    accumulated union then goes through the real guarded synthesis (citation + number guards), the
    one place the faithfulness floor can fire. Prints the call bound and refuses to spend without
    `--yes`.
    """
    from stock_agent.research.rl_simreal import (
        DryRunLLM,
        LLMQueryWriter,
        estimate_llm_calls,
        format_simreal_markdown,
        run_sim_to_real,
        write_report,
    )

    settings = get_settings()
    configure_logging(settings)
    for path in (policy_path, test_queries):
        if not path.exists():
            typer.echo(f"Not found: {path}")
            raise typer.Exit(code=1)

    loaded = load_checkpoint(policy_path)
    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(test_queries.read_text())]
    wanted = {s.strip().upper() for s in strata.split(",") if s.strip()}
    pool = [q for q in labeled if (q.stratum or "NA").upper() in wanted]
    if not pool:
        typer.echo(f"No held-out episodes in strata {sorted(wanted)}.")
        raise typer.Exit(code=1)
    # Deterministic subset: a seeded shuffle of the filtered pool (reproducible) — never "the first
    # n", which would bias the sample toward whichever seed tickers sort first in the benchmark.
    episodes = list(pool)
    random.Random(seed).shuffle(episodes)
    episodes = episodes[: min(n, len(episodes))]

    bound = estimate_llm_calls(len(episodes), settings.agentic_max_steps)
    typer.echo(
        f"Sim-to-real: {len(episodes)} episodes from {sorted(wanted)} "
        f"(policy={loaded.algo}/{loaded.action_space}). "
        f"Worst case {bound} calls ({settings.agentic_max_steps} query hops + 1 synthesis each)."
    )
    if not dry_run and not yes:
        typer.echo("This spends API budget. Re-run with --yes (or --dry-run for a $0 check).")
        raise typer.Exit(code=1)

    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    cfg = loaded.config
    space = named_action_space(
        loaded.action_space, n_discovered_slots=loaded.n_discovered_slots
    )

    def _build(writer: object | None) -> RagRetrievalEnv:
        # Both envs are identical apart from the query writer — that is what isolates the gap.
        return RagRetrievalEnv(
            episodes,
            settings=settings,
            action_space=space,
            alias_map=alias_map,
            graph_universe=universe,
            gamma=float(cfg.get("gamma", 1.0)),
            lambda_cost=cfg.get("lambda_cost"),
            query_writer=writer,  # type: ignore[arg-type]
        )

    # --dry-run swaps the whole LLM (writer + synthesis) for a $0 stand-in: the writer returns "",
    # the env keeps the template, so the gap must come out exactly 0 — the harness self-test.
    llm: TextLLM = DryRunLLM() if dry_run else AnthropicClient(settings)
    writer = LLMQueryWriter(llm, alias_map=alias_map)
    report = run_sim_to_real(
        loaded.policy,
        episodes,
        sim_env=_build(None),
        real_env=_build(writer),
        writer=writer,
        llm=llm,
        seed=seed,
    )
    typer.echo("\n" + format_simreal_markdown(report))
    destination = out or policy_path.parent / "rl_simreal.json"
    write_report(report, destination)
    typer.echo(f"\nWrote sim-to-real report to {destination}")


@rag_app.command("rl-h2h")
def rag_rl_h2h(
    policy_path: Annotated[
        Path, typer.Option("--policy", "-p", help="Frozen checkpoint from `rag rl-train`")
    ],
    baseline: Annotated[
        str,
        typer.Option("--baseline", help="Opponent: `react:ARM` (A4) or `fixed:ARM` (A5.3)"),
    ] = "react:hybrid",
    simreal: Annotated[
        Path | None,
        typer.Option("--simreal", help="Paid rl-simreal report to pair against"),
    ] = None,
    test_queries: Annotated[
        Path, typer.Option("--test-queries", help="Held-out benchmark (must match rl-simreal)")
    ] = Path("configs/rag_eval_multistep_generated.test.json"),
    n: Annotated[int, typer.Option("--n", help="Episodes (must match the rl-simreal run)")] = 24,
    strata: Annotated[
        str, typer.Option("--strata", help="Strata (must match the rl-simreal run)")
    ] = "HARD,MED",
    seed: Annotated[
        int, typer.Option("--seed", help="Subset seed (must match the rl-simreal run)")
    ] = 0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="$0 stand-in LLM: the delta must be exact"),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm the paid LLM calls (required unless --dry-run)")
    ] = False,
    graph_universe: Annotated[
        Path, typer.Option("--graph-universe", help="Graph universe file (in_graph_universe flag)")
    ] = Path("configs/graph_universe.txt"),
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Report JSON (default: next to the policy)")
    ] = None,
) -> None:
    """Re-run the promote comparison with BOTH policies on real (LLM-written) queries — **PAID**.

    `rag rl-eval` judged the learned policy against the baselines under the sim's TEMPLATED queries,
    and `rag rl-simreal` then showed the sim understates the learned policy. The obvious objection
    to a REJECT is that the baseline never got real query text either — so this gives it the same
    environment and re-measures the paired Δ. It costs roughly half of `rl-simreal`: the return is
    computed by the env (coverage oracle + cost table, both `$0`), so only the baseline's query hops
    bill, and the learned policy's real returns are read back from the already-paid report.
    """
    from stock_agent.research.rl_simreal import (
        DryRunLLM,
        HeadToHeadReport,
        LLMQueryWriter,
        RealRow,
        format_h2h_markdown,
        head_to_head,
        run_real_policy,
    )

    settings = get_settings()
    configure_logging(settings)
    saved_path = simreal or policy_path.parent / "rl_simreal.json"
    for path in (policy_path, test_queries, saved_path):
        if not path.exists():
            typer.echo(f"Not found: {path}  (run `rag rl-simreal` first)")
            raise typer.Exit(code=1)

    loaded = load_checkpoint(policy_path)
    labeled = [MultiHopQuery.model_validate(obj) for obj in json.loads(test_queries.read_text())]
    wanted = {s.strip().upper() for s in strata.split(",") if s.strip()}
    pool = [q for q in labeled if (q.stratum or "NA").upper() in wanted]
    # Reproduce the rl-simreal subset EXACTLY: same filter, same seeded shuffle, same truncation.
    episodes = list(pool)
    random.Random(seed).shuffle(episodes)
    episodes = episodes[: min(n, len(episodes))]

    saved = json.loads(saved_path.read_text())
    # BOTH sides of the pair must face the SAME environment or the delta is meaningless. --dry-run
    # gives the baseline a $0 stand-in writer that returns "" (⇒ the env keeps the template), so it
    # must be paired against the learned policy's TEMPLATED (sim) returns; the paid run pairs
    # against its LLM-written (real) ones. Mixing the two would hand the learned policy real query
    # text and the baseline a template — exactly the bias this command exists to remove.
    side = "sim" if dry_run else "real"
    rows_a = [
        RealRow(
            question=e["question"],
            stratum=e["stratum"],
            group_id="",  # filled from the reconstructed episodes below (the JSON omits it)
            coverage=e[f"coverage_{side}"],
            total_return=e[f"return_{side}"],
            actions=tuple(e[f"actions_{side}"]),
            queries=tuple(e["queries"]) if not dry_run else (),
            n_llm_calls=e["n_llm_calls"] if not dry_run else 0,
        )
        for e in saved["episodes"]
    ]
    # The pairing is only valid if BOTH runs cover the same episodes in the same order. Check that
    # here rather than trusting the flags were retyped identically — a silent misalignment would
    # look exactly like a result.
    if len(rows_a) != len(episodes) or any(
        a.question != e.question for a, e in zip(rows_a, episodes, strict=False)
    ):
        typer.echo(
            f"Episode set does not match {saved_path}: rerun with the SAME "
            f"--n/--strata/--seed/--test-queries used for `rag rl-simreal` "
            f"({len(rows_a)} saved vs {len(episodes)} here)."
        )
        raise typer.Exit(code=1)
    rows_a = [
        replace(a, group_id=e.group_id or e.question)
        for a, e in zip(rows_a, episodes, strict=True)
    ]

    space = named_action_space(loaded.action_space, n_discovered_slots=loaded.n_discovered_slots)
    kind, _, arm = baseline.partition(":")
    opponent: ScriptedPolicy
    if kind == "react":
        opponent = ReActBridgePolicy(arm, action_space=space)
    elif kind == "fixed":
        opponent = OneShotArmPolicy(arm, action_space=space)
    else:
        typer.echo(f"Unknown --baseline {baseline!r} (expected `react:ARM` or `fixed:ARM`)")
        raise typer.Exit(code=1)

    bound = len(episodes) * settings.agentic_max_steps  # no synthesis on this side ⇒ hops only
    typer.echo(
        f"Real-env head-to-head: rl({loaded.algo}) vs {baseline} on {len(episodes)} episodes. "
        f"Worst case {bound} calls (query hops only; the rl side is already paid for)."
    )
    if not dry_run and not yes:
        typer.echo("This spends API budget. Re-run with --yes (or --dry-run for a $0 check).")
        raise typer.Exit(code=1)

    alias_map = load_alias_map()
    universe = load_universe(graph_universe) if graph_universe.exists() else None
    cfg = loaded.config
    llm: TextLLM = DryRunLLM() if dry_run else AnthropicClient(settings)
    writer = LLMQueryWriter(llm, alias_map=alias_map)
    real_env = RagRetrievalEnv(
        episodes,
        settings=settings,
        action_space=space,
        alias_map=alias_map,
        graph_universe=universe,
        gamma=float(cfg.get("gamma", 1.0)),
        lambda_cost=cfg.get("lambda_cost"),
        query_writer=writer,
    )
    rows_b = run_real_policy(opponent, episodes, real_env=real_env, writer=writer, seed=seed)

    suffix = " [templated]" if dry_run else ""
    report: HeadToHeadReport = head_to_head(
        rows_a, rows_b, name_a=f"rl({loaded.algo}){suffix}", name_b=f"{baseline}{suffix}"
    )
    typer.echo("\n" + format_h2h_markdown(report))
    if dry_run:
        typer.echo(
            "\nDRY RUN: both sides ran TEMPLATED queries ($0 stand-in writer), so this reproduces "
            "the `rag rl-eval` comparison on this subset — it is NOT the sim-to-real head-to-head."
        )
    destination = out or policy_path.parent / "rl_h2h.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    typer.echo(f"\nWrote head-to-head report to {destination}")
