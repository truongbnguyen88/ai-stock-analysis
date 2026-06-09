"""Topic/theme registry for theme-scoped news (Enhancement C).

Config-driven mapping from a theme name (e.g. "robotics") to a query spec
(keywords + optional GDELT theme codes + language) used to pull theme-scoped news
from the GDELT DOC API. Curated for PRECISION; an unlisted theme falls back to a
free-form phrase query so the agent can still route any sector/theme the user names.

Pure data + pure functions (no I/O) — the provider composes the final API query
from ``gdelt_query_expression``; transparency is preserved by surfacing the
resolved keywords back to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopicQuery:
    """A theme's query spec: phrases to OR together (+ optional GDELT theme codes)."""

    keywords: tuple[str, ...]
    theme_codes: tuple[str, ...] = ()  # optional GDELT GKG theme codes (precision boost)
    language: str = "english"  # GDELT sourcelang filter


@dataclass(frozen=True)
class ResolvedTopic:
    """A topic resolved to a concrete query (from the registry or free-form)."""

    topic: str  # normalized slug (registry) or the raw phrase (free-form)
    label: str  # human-facing label for display/citation
    keywords: tuple[str, ...]
    theme_codes: tuple[str, ...] = field(default=())
    language: str = "english"
    known: bool = False  # True => came from the curated registry


# Curated registry (slug -> query spec). Keep keyword sets tight for precision;
# add theme codes where a clean GDELT code exists. Extend freely.
TOPIC_REGISTRY: dict[str, TopicQuery] = {
    "ai": TopicQuery(("artificial intelligence", "generative AI", "AI model")),
    "ai_infra": TopicQuery(("AI data center", "GPU cluster", "AI accelerator")),
    "ai_energy": TopicQuery(("AI power demand", "data center energy", "AI electricity")),
    "ai_memory": TopicQuery(("HBM", "high bandwidth memory", "AI memory")),
    "ev": TopicQuery(("electric vehicle", "EV sales", "EV battery")),
    "robotics": TopicQuery(("robotics", "humanoid robot", "industrial automation")),
    "semiconductors": TopicQuery(("semiconductor", "chipmaker", "chip foundry")),
}

# Human-friendly aliases -> canonical registry slug (lower-cased match).
_ALIASES: dict[str, str] = {
    "artificial intelligence": "ai",
    "ai infrastructure": "ai_infra",
    "ai infra": "ai_infra",
    "data centers": "ai_infra",
    "ai energy": "ai_energy",
    "ai power": "ai_energy",
    "ai memory": "ai_memory",
    "hbm": "ai_memory",
    "evs": "ev",
    "electric vehicles": "ev",
    "electric vehicle": "ev",
    "robots": "robotics",
    "humanoid robots": "robotics",
    "semis": "semiconductors",
    "semiconductor": "semiconductors",
    "chips": "semiconductors",
}


def _norm(name: str) -> str:
    """Normalize a topic name for lookup (lower, trim, collapse spaces/underscores)."""
    return "_".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


def list_topics() -> tuple[str, ...]:
    """Known registry slugs (for the agent prompt / showcase). Stable order."""
    return tuple(TOPIC_REGISTRY)


def resolve_topic(name: str) -> ResolvedTopic:
    """Resolve a theme name to a concrete query.

    Registry hit (direct slug or alias) -> the curated keyword set (``known=True``).
    Otherwise -> a free-form phrase query of the raw name (``known=False``), so any
    sector/theme the user names is still searchable (lower precision).
    """
    raw = name.strip()
    slug = _norm(raw)
    # Alias match first (alias keys use the space form), then direct slug.
    alias_key = " ".join(slug.split("_"))
    canonical = _ALIASES.get(alias_key) or (slug if slug in TOPIC_REGISTRY else None)
    if canonical is not None:
        spec = TOPIC_REGISTRY[canonical]
        return ResolvedTopic(
            topic=canonical,
            label=canonical.replace("_", " "),
            keywords=spec.keywords,
            theme_codes=spec.theme_codes,
            language=spec.language,
            known=True,
        )
    # Free-form fallback: the phrase itself is the single keyword.
    return ResolvedTopic(topic=raw, label=raw, keywords=(raw,), known=False)


def _quote(keyword: str) -> str:
    """Quote multi-word phrases for the GDELT query; leave single tokens bare."""
    return f'"{keyword}"' if " " in keyword else keyword


def gdelt_query_expression(resolved: ResolvedTopic) -> str:
    """Build the GDELT DOC keyword expression: ``(kw1 OR "kw two" OR theme:CODE)``.

    OR-joins the keywords and any theme codes; wraps in parentheses when there is
    more than one term so it composes cleanly with provider-added filters.
    """
    terms = [_quote(k) for k in resolved.keywords]
    terms += [f"theme:{code}" for code in resolved.theme_codes]
    expr = " OR ".join(terms)
    return f"({expr})" if len(terms) > 1 else expr
