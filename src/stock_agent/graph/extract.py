"""Graph extraction (advanced-RAG A5.1) — filings → typed, verified, provenance-bearing edges.

Offline, batched, cost-gated construction of the A5 knowledge graph. Per ``(filing, section)`` it
makes ONE structured-output call (after a cheap candidate pre-filter), parses typed triples, maps
object company names to tickers via the SHARED alias map (the same resolver the A4 query-time bridge
uses — but applied once, offline), and **verifies** each edge before keeping it:

1. ``confidence >= min_confidence`` — drop low-confidence guesses.
2. **hallucinated-edge guard** — for a company→company edge the object's surface name (or a resolved
   alias) must literally appear in the cited provenance chunk; a fabricated supplier won't be in the
   text. (The subject is the filing's own company, referred to as "we/our", so it is implicit,
   not required to appear.) Risk/topic objects are paraphrased, so they pass on confidence + a
   valid provenance mapping rather than a substring match.

This is the edge analogue of the P7 citation guard: an edge that survives is both a typed fact and a
pointer back to the exact chunk that licenses it. No edge is ever produced without provenance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from stock_agent.graph.prompts import GRAPH_EXTRACT_SYSTEM, build_extract_user
from stock_agent.graph.store import _edge_key
from stock_agent.llm.client import TextLLM
from stock_agent.logging_config import get_logger
from stock_agent.research._shared import loads_lenient
from stock_agent.research.bridge import mentioned_tickers
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.graph import RELATION_OBJECT_TYPE, Edge, Entity

log = get_logger(__name__)

# Item 1A (Risk Factors) sections are long and yield many triples; a tight cap truncates the JSON
# mid-object (observed: "Expecting ',' delimiter") and the whole section's edges are then dropped.
# 4000 covers ~50 triples; if a section still truncates it degrades to "skip section" (not a crash).
_EXTRACT_MAX_TOKENS = 4000

# Cheap candidate pre-filter: a section group is sent to the LLM only if its text mentions a known
# company alias OR any relationship/risk cue. (Item 1A "Risk Factors" almost always contains "risk",
# so this mainly skips irrelevant Item 1 chunks — bounding the paid-call set.)
_RELATION_CUES: tuple[str, ...] = (
    "risk", "depend", "supplier", "customer", "compete", "competitor", "regulat", "tariff",
    "export", "sanction", "foundry", "manufactur", "license",
)

# Item-code matcher. The negative lookahead (?!\.\d) rejects the 8-K decimal sub-item form
# ("Item 1.01") so it is NOT collapsed to the 10-K code "1" — only the 10-K "Item 1." / "Item 1A."
# style is captured. (8-K/10-Q are excluded outright by the document-type gate below; this is a
# second guard so a stray decimal label can never masquerade as a 10-K business/risk section.)
_ITEM_CODE = re.compile(r"item\s+(\d{1,2}[a-z]?)(?!\.\d)", re.IGNORECASE)

# Precise section matching applies to 10-K ONLY: the graph_sections (Item 1 Business / Item 1A Risk
# Factors) are 10-K items. A 10-Q "Item 1" is Financial Statements, an 8-K "Item 1.01" is a material
# agreement, and a 20-F "Item 1" is *Directors* — all share the code "1" but mean different things,
# so the section gate stays 10-K-only (else they pollute the graph + multiply the call count).
_GRAPH_DOC_TYPE = "10-K"
# Doc types eligible for extraction overall. 20-F (foreign annual: TSMC/ASML/ARM/STM) has business +
# risk factors but a different item structure than 10-K, so it never matches the 10-K section codes
# and is always routed through the cue-bearing whole-document fallback (like an INTC 10-K whose
# headers didn't parse). Domestic 10-K still prefers precise Item 1/1A matching.
_GRAPH_DOC_TYPES: tuple[str, ...] = ("10-K", "20-F")


class GraphExtractBudgetExceeded(RuntimeError):
    """Extraction refused: the LLM-call ceiling was reached (mirrors ``EmbedBudgetExceeded``).

    A client-side spend stop (``settings.graph_max_extract_calls``): once that many extraction calls
    have been made this run, the next ``(filing, section)`` aborts rather than spending more.
    """

    def __init__(self, ticker: str, calls: int, ceiling: int) -> None:
        self.ticker = ticker
        self.calls = calls
        self.ceiling = ceiling
        super().__init__(
            f"{ticker}: graph extraction hit the {ceiling}-call ceiling "
            f"(raise settings.graph_max_extract_calls to proceed)"
        )


@dataclass
class ExtractResult:
    """What one extraction run produced: deduped entities + edges, and the paid-call count."""

    entities: list[Entity] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    calls: int = 0  # number of extraction LLM calls actually made (spend proxy)


def _section_code(label: str | None) -> str | None:
    """The EDGAR item code of a section label, lowercased: 'Item 1A. Risk Factors' -> '1a'.

    Matching on the *code* (not the full label) is tolerant of header-wording variants while still
    distinguishing 'Item 1' from 'Item 10'/'Item 11' (equality on the captured code, not a prefix).
    """
    m = _ITEM_CODE.search(label or "")
    return m.group(1).lower() if m else None


def filter_graph_sections(
    chunks: Sequence[DocumentChunk], sections: Sequence[str]
) -> list[DocumentChunk]:
    """Keep only **10-K** chunks whose section item-code matches one of the configured ``sections``.

    ``sections`` is the human-readable config (e.g. ['Item 1. Business', 'Item 1A. Risk Factors']);
    each is reduced to its item-code and a 10-K chunk qualifies on code equality. The document-type
    gate is essential: 10-Q "Item 1" (Financial Statements) and 8-K "Item 1.01" share the code "1"
    with 10-K "Item 1. Business" but mean different things — without it they pollute the graph and
    multiply the call count. Empty ``sections`` (or no resolvable codes) returns the 10-K chunks.
    """
    tenk = [ch for ch in chunks if ch.document_type == _GRAPH_DOC_TYPE]
    codes = {c for s in sections if (c := _section_code(s))}
    if not codes:
        return tenk
    return [ch for ch in tenk if _section_code(ch.section) in codes]


# A 10-K with at least this many Item 1/1A chunks is treated as cleanly section-detected. Below it,
# the filer's headers likely didn't parse (e.g. INTC renders Item headers only in its TOC, so
# the body collapses into 'Preamble'), and we fall back to the whole 10-K.
_MIN_SECTION_CHUNKS = 5
# Cap the whole-10-K fallback per filing so a section-detection failure doesn't feed the entire
# document (incl. financial statements) to the LLM — bounds prompt size, cost, and truncation.
_FALLBACK_MAX_CHUNKS_PER_DOC = 40


def _chunk_has_cue(chunk: DocumentChunk, alias_map: Mapping[str, Sequence[str]]) -> bool:
    """Whether a chunk plausibly states a relationship (a relation/risk cue or a company name).

    The per-chunk analogue of ``_has_candidate``; used to focus the whole-10-K fallback on
    relationship-bearing prose and skip pure financial-table chunks.
    """
    low = chunk.text.lower()
    if any(cue in low for cue in _RELATION_CUES):
        return True
    return bool(mentioned_tickers(chunk.text, alias_map))


def select_extraction_chunks(
    chunks: Sequence[DocumentChunk],
    sections: Sequence[str],
    alias_map: Mapping[str, Sequence[str]],
) -> list[DocumentChunk]:
    """Chunks to extract a ticker's graph from: prefer 10-K Item 1/1A, fall back to the whole 10-K.

    Most 10-K filers split cleanly into ``Item 1`` / ``Item 1A`` (``filter_graph_sections``). Two
    cases fall back to the **cue-bearing** whole-document path instead: (1) a 10-K (e.g. INTC) that
    renders Item headers only in its TOC, so section detection collapses into ``Preamble``; and (2)
    any **20-F** (foreign annual — TSMC/ASML/ARM/STM), whose item structure never matches the 10-K
    codes. The fallback keeps cue-bearing chunks (company/risk terms), capped per filing, so
    extraction runs on the real business/risk prose without feeding entire financial statements to
    the model. Quality stays gated by the same verification + confidence checks; only the candidate
    set widens. The eligible-doc pool is ``_GRAPH_DOC_TYPES`` (10-K + 20-F).
    """
    matched = filter_graph_sections(chunks, sections)  # precise 10-K Item 1/1A only
    pool = [ch for ch in chunks if ch.document_type in _GRAPH_DOC_TYPES]
    if len(matched) >= _MIN_SECTION_CHUNKS or len(pool) < _MIN_SECTION_CHUNKS:
        return matched  # clean 10-K section detection (or nothing extractable to fall back to)

    per_doc: dict[str, list[DocumentChunk]] = {}
    for ch in pool:
        bucket = per_doc.setdefault(ch.document_id, [])
        if len(bucket) < _FALLBACK_MAX_CHUNKS_PER_DOC and _chunk_has_cue(ch, alias_map):
            bucket.append(ch)
    fallback = [ch for bucket in per_doc.values() for ch in bucket]
    log.info(
        "graph.section_fallback", matched=len(matched), pool=len(pool), fallback=len(fallback)
    )
    return fallback or matched


def _normalize_node_id(phrase: str) -> str:
    """Normalized id for a non-company node: lowercase alnum-hyphen, capped ('supply-capacity')."""
    slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")
    return slug[:60]


def resolve_company(name: str, alias_map: Mapping[str, Sequence[str]]) -> str | None:
    """Resolve a company surface name to a ticker via the alias map, or ``None`` if unresolvable.

    Uses the shared word-boundary alias match; an exactly-one hit resolves, an ambiguous (>1) or
    zero hit does not. Also accepts a bare ticker symbol written as the object (e.g. object='MU').
    """
    cands = mentioned_tickers(name, alias_map)
    if len(cands) == 1:
        return next(iter(cands))
    up = name.strip().upper()
    if up in alias_map:  # the object was written as the ticker symbol itself
        return up
    return None


# Corporate legal-form suffixes stripped before name matching: filings name "Hon Hai Precision
# Industry Co., Ltd." but a risk chunk may write just "Hon Hai" / "Hon Hai Precision Industry".
# Matching the *core* name (suffix-stripped, punctuation-normalized) avoids both false drops.
_CORP_SUFFIXES: frozenset[str] = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited",
    "llc", "lp", "plc", "ag", "sa", "nv", "se", "kk", "gmbh", "holdings", "holding", "group",
})


def _normalize_name(text: str) -> str:
    """Lowercase, drop punctuation→space, collapse whitespace (for substring name matching)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _core_name(name: str) -> str:
    """A company's core name: normalized, with trailing corporate-form suffix tokens removed.

    'Samsung Electronics Co., Ltd.' -> 'samsung electronics'; 'SK Hynix Inc.' -> 'sk hynix'.
    Suffixes are peeled iteratively (handles 'Co., Ltd.'); a name that is *only* suffix tokens
    keeps its normalized form (never returns empty).
    """
    tokens = _normalize_name(name).split()
    while len(tokens) > 1 and tokens[-1] in _CORP_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) if tokens else _normalize_name(name)


def _object_in_chunk(
    object_name: str, resolved: str | None, chunk_text: str, alias_map: Mapping[str, Sequence[str]]
) -> bool:
    """Whether the object company's name (or a resolved alias) is named in the provenance chunk.

    The hallucinated-edge guard for company edges: a real supplier/competitor is named in the chunk;
    a fabricated one is not. Matches the **core** (suffix-stripped, punctuation-normalized) name
    as a token-boundary substring of the normalized chunk — robust to legal suffixes and trailing
    punctuation (the old word-boundary regex never matched names ending in '.'), and to the chunk
    abbreviating a full legal name. Also checks every alias of a resolved ticker, so 'Micron' (or
    'MU') verifies an edge resolved to 'MU'. A degenerate 1-2 char core is rejected (too generic).
    """
    surfaces = [object_name]
    if resolved and resolved in alias_map:
        surfaces.extend(alias_map[resolved])
    haystack = f" {_normalize_name(chunk_text)} "
    for s in surfaces:
        core = _core_name(s)
        if len(core) >= 3 and f" {core} " in haystack:
            return True
    return False


def _has_candidate(chunks: Sequence[DocumentChunk], alias_map: Mapping[str, Sequence[str]]) -> bool:
    """Cheap pre-filter: does this section group plausibly contain an extractable relationship?"""
    text = " ".join(c.text for c in chunks).lower()
    if any(cue in text for cue in _RELATION_CUES):
        return True
    return bool(mentioned_tickers(text, alias_map))


def _group_by_document(chunks: Sequence[DocumentChunk]) -> list[list[DocumentChunk]]:
    """Group chunks by ``document_id`` (one filing per group), keeping order within each group."""
    groups: dict[str, list[DocumentChunk]] = {}
    for c in chunks:
        groups.setdefault(c.document_id, []).append(c)
    return list(groups.values())


def extract_edges(
    ticker: str,
    chunks: Sequence[DocumentChunk],
    *,
    llm: TextLLM,
    alias_map: Mapping[str, Sequence[str]],
    min_confidence: float = 0.5,
    max_calls: int | None = None,
) -> ExtractResult:
    """Extract verified, provenance-bearing edges for ``ticker`` from its section ``chunks``.

    One LLM call per filing-section group (after the candidate pre-filter), bounded by ``max_calls``
    (raises ``GraphExtractBudgetExceeded``). Each triple is validated (known relation, valid
    chunk pointer, confidence ≥ ``min_confidence``) and verified (company objects must appear in the
    cited chunk) before becoming an ``Edge``. Returns deduped entities + edges and the call count;
    parsing/LLM errors on one group are logged and skipped (one bad section never aborts the run).
    """
    subj_name = next(iter(alias_map.get(ticker, [])), ticker)  # readable surface for the subject
    subject = Entity(id=ticker, name=subj_name, type="company", ticker=ticker)

    entities: dict[str, Entity] = {}
    edges: dict[tuple[str, str, str, str], Edge] = {}
    calls = 0

    for group in _group_by_document(chunks):
        if not _has_candidate(group, alias_map):
            continue
        if max_calls is not None and calls >= max_calls:
            raise GraphExtractBudgetExceeded(ticker, calls, max_calls)
        raw = llm.complete_json(
            system=GRAPH_EXTRACT_SYSTEM,
            user=build_extract_user(ticker, group),
            max_tokens=_EXTRACT_MAX_TOKENS,
        )
        calls += 1
        try:
            triples = loads_lenient(raw).get("triples") or []
        except ValueError as exc:
            log.warning("graph.extract_parse_failed", ticker=ticker, error=str(exc))
            continue
        if not isinstance(triples, list):
            continue
        for triple in triples:
            edge = _build_edge(triple, ticker, group, alias_map, min_confidence, entities)
            if edge is not None:
                entities[ticker] = subject  # only register the subject once we have ≥1 real edge
                edges[_edge_key(edge)] = edge

    return ExtractResult(
        entities=sorted(entities.values(), key=lambda e: e.id),
        edges=sorted(edges.values(), key=_edge_key),
        calls=calls,
    )


def _build_edge(
    triple: object,
    ticker: str,
    group: Sequence[DocumentChunk],
    alias_map: Mapping[str, Sequence[str]],
    min_confidence: float,
    entities: dict[str, Entity],
) -> Edge | None:
    """Validate + verify one raw triple into an ``Edge`` (registering its object node), or ``None``.

    Returns ``None`` (and logs the drop reason for the hallucination guard) for any triple that
    fails relation/chunk/confidence validation or the object-in-chunk verification. Mutates
    ``entities`` to register the object node only for an edge that survives.
    """
    if not isinstance(triple, dict):
        return None
    relation = str(triple.get("relation", "")).strip()
    if relation not in RELATION_OBJECT_TYPE:
        return None
    rel = relation  # narrowed to Relation by the membership check above
    obj_name = str(triple.get("object", "")).strip()
    if not obj_name:
        return None

    # Map the cited chunk number (1-based, within this group) to its chunk_id; out-of-range -> drop.
    chunk_val = triple.get("chunk")
    if not isinstance(chunk_val, (int, str)):
        return None
    try:
        chunk_no = int(chunk_val)
    except ValueError:
        return None
    if not 1 <= chunk_no <= len(group):
        return None
    src = group[chunk_no - 1]

    conf_val = triple.get("confidence", 1.0)
    try:
        confidence = float(conf_val) if isinstance(conf_val, (int, float, str)) else 1.0
    except ValueError:
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < min_confidence:
        return None

    obj_type = RELATION_OBJECT_TYPE[rel]
    if obj_type == "company":
        resolved = resolve_company(obj_name, alias_map)
        if resolved == ticker:  # a company "depends on itself" is noise
            return None
        if not _object_in_chunk(obj_name, resolved, src.text, alias_map):
            log.info("graph.edge_dropped", ticker=ticker, reason="obj_not_in_chunk", obj=obj_name)
            return None
        # F1: id an unresolved company by its CORE name (suffix-stripped + normalized) so
        # surface variants collapse to one node ("Hon Hai Precision Industry Co." / "… Co., Ltd." /
        # "GlobalFoundries" / "Global Foundries" → one id). Resolved companies keep their ticker id.
        # (Acronym variants like TSMC↔Taiwan Semiconductor still differ — needs an alias entry.)
        obj_id = resolved or _normalize_node_id(_core_name(obj_name))
        entities[obj_id] = Entity(id=obj_id, name=obj_name, type="company", ticker=resolved)
    else:  # risk / regulatory_topic: paraphrased, so no substring guard (confidence + valid chunk)
        obj_id = _normalize_node_id(obj_name)
        if not obj_id:
            return None
        entities[obj_id] = Entity(id=obj_id, name=obj_name, type=obj_type, ticker=None)

    return Edge(
        subject=ticker,
        relation=rel,
        object=obj_id,
        provenance=[src.chunk_id],
        filing_date=src.filing_date,
        source_url=src.source_url,
        confidence=confidence,
    )
