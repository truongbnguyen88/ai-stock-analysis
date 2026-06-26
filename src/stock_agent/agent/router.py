"""Hybrid routing — deterministic fast-path + LLM-routing fallback (the supervisor front-door).

Two ways to get from a request to a capability:

- **Deterministic routing** — the caller NAMES the route (the user already knows which capability
  they want). We dispatch straight to that one tool with structured params and make **no routing
  LLM call** at all. Fast, cheap, unambiguous. (A *synthesis* tool — news / filings — may still make
  its own internal guarded LLM call; that is the tool's business, not a routing decision.)
- **LLM routing** — no route given (or ``"auto"``): hand the request to the existing tool-use agent
  loop (``run_agent``), which IS the LLM "supervisor" — it selects and composes one or many tools
  and grounds the numbers. We deliberately do NOT re-implement that here.

Pure orchestration: depends only on the existing ``ToolExecutor`` (deterministic dispatch) and
``run_agent`` (LLM dispatch). No new tool, no new guard, no routing LLM call of its own. Dependency
direction stays downward (``cli`` -> ``agent.router`` -> ``agent.tools`` / ``agent.runtime``).

The route registry is the deterministic action set — and a natural future home for an A6-style
learned retrieval/capability selector (a policy choosing a route per request).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from stock_agent.agent.runtime import ToolLLM, run_agent
from stock_agent.agent.tools import ToolExecutor
from stock_agent.logging_config import get_logger

log = get_logger(__name__)

AUTO = "auto"  # sentinel route name meaning "let the LLM decide" (equivalent to route=None)


class RouterError(ValueError):
    """A deterministic route was misused: unknown route, or a required parameter is missing."""


@dataclass(frozen=True)
class RouteInput:
    """Structured inputs a deterministic route may draw on — never NL-parsed (that keeps it
    deterministic). ``request`` is the user's text, used as the question/topic where a route needs
    one; the rest are explicit params (from CLI flags / a UI form)."""

    request: str = ""
    ticker: str | None = None
    horizon: int | None = None  # trading days (forecast / large-move)
    days: int | None = None  # lookback window (news / price)
    model: str | None = None  # forecast / large-move model override


@dataclass(frozen=True)
class Route:
    """One deterministic route: a user-selectable capability mapped to a single tool.

    ``needs`` lists the required input facets — ``"ticker"`` (``RouteInput.ticker`` set) and/or
    ``"question"``/``"topic"`` (``RouteInput.request`` non-blank). ``build`` is a *pure* function
    turning a validated ``RouteInput`` into the tool's argument dict.
    """

    name: str
    tool: str
    needs: tuple[str, ...]
    build: Callable[[RouteInput], dict[str, Any]]
    blurb: str


def _with_optional(args: dict[str, Any], **maybe: Any) -> dict[str, Any]:
    """Add each keyword to ``args`` only when not None (so the tool applies its own defaults)."""
    for key, value in maybe.items():
        if value is not None:
            args[key] = value
    return args


# The deterministic action set. Each route targets exactly one existing tool; ``build`` mirrors that
# tool's input schema. Numeric routes need no LLM at all; synthesis routes (news / filings /
# research_brief) reuse the tool's own internal guarded call.
_ROUTES: tuple[Route, ...] = (
    Route(
        "technicals", "compute_indicators", ("ticker",),
        lambda i: {"ticker": i.ticker},
        "Technical indicators — MAs, RSI, MACD, ATR, drawdown, trend.",
    ),
    Route(
        "price", "get_price_summary", ("ticker",),
        lambda i: _with_optional({"ticker": i.ticker}, days=i.days),
        "Recent price statistics over a lookback window.",
    ),
    Route(
        "forecast", "run_forecast", ("ticker",),
        lambda i: _with_optional(
            {"ticker": i.ticker, "horizon_days": i.horizon if i.horizon is not None else 20},
            model=i.model,
        ),
        "Probabilistic return forecast for a horizon (default 20d, ensemble).",
    ),
    Route(
        "big_move", "get_large_move", ("ticker",),
        lambda i: _with_optional({"ticker": i.ticker}, horizon_days=i.horizon, model=i.model),
        "Probability of a large up/down move over the horizon.",
    ),
    Route(
        "headlines", "get_news", ("ticker",),
        lambda i: _with_optional({"ticker": i.ticker}, days=i.days),
        "Recent news headlines, newest-first.",
    ),
    Route(
        "news", "summarize_news", ("ticker",),
        lambda i: _with_optional({"ticker": i.ticker}, days=i.days),
        "LLM news synthesis — themes, risks, catalysts (cited).",
    ),
    Route(
        "theme_news", "analyze_topic_news", ("topic",),
        lambda i: _with_optional({"topic": i.request.strip()}, days=i.days),
        "Synthesis of a sector/theme's news (the request text is the theme).",
    ),
    Route(
        "filings", "search_filings", ("ticker", "question"),
        lambda i: {"ticker": i.ticker, "question": i.request.strip()},
        "Single-shot SEC-filing question, cited (the request text is the question).",
    ),
    Route(
        "filings_multihop", "research_multistep", ("question",),
        lambda i: {"question": i.request.strip()},
        "Multi-hop SEC-filing research, cited (the request text is the question).",
    ),
    Route(
        "research_brief", "research_summary", ("ticker",),
        lambda i: _with_optional({"ticker": i.ticker}, days=i.days),
        "Integrated executive brief — filings + news + forecast, cited.",
    ),
)

ROUTES: dict[str, Route] = {r.name: r for r in _ROUTES}
ROUTE_NAMES: tuple[str, ...] = tuple(ROUTES)


# ---------------------------------------------------------------------------
# Domain layer — the friendly ~5-way selector over the granular routes.
# ---------------------------------------------------------------------------
# The granular ``ROUTES`` above are the precise engine action set (one route = one tool = the future
# A6 action). Users, though, think in DOMAINS ("news", "predictions", "filings") — so a domain
# groups a few sibling routes and may "do a few tasks" by picking a VARIANT. The selector (CLI
# --domain/--variant, and the planned UI dropdown) reads this layer; dispatch still resolves down to
# one granular route, so determinism + precision are unchanged. The domains form a complete,
# non-overlapping cover of ``ROUTES`` (asserted in tests) — every tool stays reachable.


@dataclass(frozen=True)
class Domain:
    """A user-facing capability area grouping sibling granular routes.

    ``variants`` maps a short variant label -> a granular route name; ``default`` is the variant
    chosen when the caller names only the domain. A single-variant domain (e.g. ``brief``) is just a
    friendly alias for one route.
    """

    name: str
    blurb: str
    variants: dict[str, str]
    default: str


_DOMAINS: tuple[Domain, ...] = (
    Domain(
        "predictions", "ML/quant forecasts and big-move odds",
        {"forecast": "forecast", "big-move": "big_move"}, "forecast",
    ),
    Domain(
        "news", "Company or sector news — synthesized, raw, or by theme",
        {"summary": "news", "headlines": "headlines", "theme": "theme_news"}, "summary",
    ),
    Domain(
        "filings", "SEC-filing research — single-shot or multi-hop",
        {"single": "filings", "multi": "filings_multihop"}, "single",
    ),
    Domain(
        "technicals", "Price statistics and technical indicators",
        {"indicators": "technicals", "price": "price"}, "indicators",
    ),
    Domain(
        "brief", "Integrated executive brief (filings + news + forecast)",
        {"brief": "research_brief"}, "brief",
    ),
)

DOMAINS: dict[str, Domain] = {d.name: d for d in _DOMAINS}
DOMAIN_NAMES: tuple[str, ...] = tuple(DOMAINS)


def resolve_domain(domain: str, variant: str | None = None) -> str:
    """Resolve a (domain, variant) selection to a granular route name.

    ``variant=None`` uses the domain's default. Raises ``RouterError`` on an unknown domain or an
    unknown variant (listing the valid choices) so the selector fails loudly, never mis-routing.
    """
    spec = DOMAINS.get(domain)
    if spec is None:
        raise RouterError(f"unknown domain '{domain}'; choose from: {', '.join(DOMAIN_NAMES)}")
    label = variant if variant is not None else spec.default
    route = spec.variants.get(label)
    if route is None:
        raise RouterError(
            f"domain '{domain}' has no variant '{label}'; choose from: {', '.join(spec.variants)}"
        )
    return route


@dataclass
class RouterResult:
    """Outcome of one routed request.

    ``mode`` is ``"deterministic"`` (a single named tool was dispatched, no routing LLM call) or
    ``"llm"`` (delegated to the agent loop). ``structured`` is the raw tool result on the
    deterministic path (numbers straight from the tool); ``messages`` is the agent transcript on the
    LLM path (carry it as ``history`` for the next turn).
    """

    mode: str
    text: str
    route: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


class Router:
    """Hybrid-routing front door: deterministic dispatch, or delegate to the LLM agent loop."""

    def __init__(self, executor: ToolExecutor, *, tool_llm: ToolLLM | None = None) -> None:
        self._executor = executor
        self._tool_llm = tool_llm  # required only for the LLM-routing path

    def run(
        self,
        request: str,
        *,
        route: str | None = None,
        ticker: str | None = None,
        horizon: int | None = None,
        days: int | None = None,
        model: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> RouterResult:
        """Route ``request``: deterministically when ``route`` names a capability, else via the LLM.

        ``route=None`` or ``route="auto"`` -> LLM routing (the agent loop). A named ``route`` ->
        deterministic dispatch with the given structured params (``ticker``/``horizon``/``days``/
        ``model``; ``request`` supplies the question/topic where the route needs one). Raises
        ``RouterError`` on an unknown route or a missing required parameter.
        """
        if route is None or route == AUTO:
            return self._run_llm(request, history=history)
        inp = RouteInput(
            request=request,
            ticker=_normalize_ticker(ticker),
            horizon=horizon,
            days=days,
            model=model,
        )
        return self._run_deterministic(route, inp)

    def _run_deterministic(self, route: str, inp: RouteInput) -> RouterResult:
        spec = ROUTES.get(route)
        if spec is None:
            raise RouterError(f"unknown route '{route}'; choose from: {', '.join(ROUTE_NAMES)}")
        _validate(spec, inp)
        args = spec.build(inp)
        log.info("router.deterministic", route=route, tool=spec.tool)
        result = self._executor.execute(spec.tool, args)  # no routing LLM call
        return RouterResult(
            mode="deterministic",
            route=route,
            tool_calls=[spec.tool],
            structured=result,
            text=_render(result),
        )

    def _run_llm(
        self, request: str, *, history: list[dict[str, Any]] | None
    ) -> RouterResult:
        if self._tool_llm is None:
            raise RouterError("LLM routing requires a tool LLM (none configured)")
        log.info("router.llm")
        res = run_agent(request, llm=self._tool_llm, executor=self._executor, history=history)
        return RouterResult(
            mode="llm",
            route=None,
            text=res.text,
            tool_calls=res.tool_calls,
            messages=res.messages,
        )


def _normalize_ticker(ticker: str | None) -> str | None:
    """Uppercase + strip a ticker (None stays None) so deterministic args are canonical."""
    return ticker.upper().strip() if ticker else None


def _validate(spec: Route, inp: RouteInput) -> None:
    """Raise ``RouterError`` if a required facet of the route is missing."""
    for need in spec.needs:
        if need == "ticker" and not inp.ticker:
            raise RouterError(f"route '{spec.name}' requires a ticker (e.g. --ticker NVDA)")
        if need in ("question", "topic") and not inp.request.strip():
            raise RouterError(
                f"route '{spec.name}' requires a {need} — pass it as the request text"
            )


def _render(result: dict[str, Any]) -> str:
    """Best-effort LLM-free text for a tool result: the synthesized answer, else the raw structure.

    Synthesis tools (news / filings) already carry a prose ``answer``; numeric tools return numbers,
    which we surface verbatim as pretty JSON (the deterministic path adds no narration of its own).
    """
    if "error" in result:
        return f"Error: {result['error']}"
    answer = result.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    return json.dumps(result, indent=2, default=str)
