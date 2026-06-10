"""SEC filing parsing: HTML/TXT -> normalized text + section detection (RAG P2).

Turns a downloaded filing (``filing.html`` + ``metadata.json``) into clean text and
splits it on the standardized EDGAR "Item" headers (e.g. ``Item 1A. Risk Factors``,
``Item 7. MD&A``) so each section is a citeable retrieval unit downstream (P3).

Pure functions + fail-soft: malformed HTML degrades to a crude tag-strip rather than
raising, and a document with no recognizable Item headers becomes a single section.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lxml import html as lhtml
from lxml.etree import ParserError

from stock_agent.schemas.documents import Document, DocumentMetadata

# Tags whose text is noise (never part of the readable filing). Includes the inline-XBRL
# METADATA elements (modern filings) — ``ix:hidden`` carries a huge blob of non-displayed
# facts that otherwise leaks into the text; the VISIBLE facts live in ``ix:nonfraction`` /
# ``ix:nonnumeric`` and are intentionally kept.
_SKIP_TAGS = frozenset(
    {
        "script", "style", "head", "title", "noscript",
        "ix:hidden", "ix:header", "ix:references", "ix:resources",
    }
)
# Block-level tags after which we insert a newline, so paragraphs/rows/headers
# land on their own lines (needed for line-anchored section detection below).
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "li", "tr", "table", "section", "article",
        "blockquote", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

_WS = re.compile(r"[ \t\xa0]+")  # spaces, tabs, and non-breaking spaces (EDGAR)
_TAG = re.compile(r"<[^>]+>")
# Modern filings are inline-XBRL XHTML beginning with an XML declaration. lxml.html
# refuses a *str* that carries an encoding declaration, so we strip it before parsing
# (the content is already decoded, so no information is lost).
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)

# An EDGAR "Item" header at the start of a line:
#   10-K / 10-Q -> "Item 1A.", "Item 7." ; 8-K -> "Item 2.02." (decimal numbering).
_ITEM_RE = re.compile(
    r"^Item\s+\d{1,2}(?:\.\d{1,2}|[A-Za-z])?\.?",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    """One labeled span of a filing (the chunking unit in P3)."""

    label: str  # header line, e.g. "Item 1A. Risk Factors" ("" / "Preamble" when none)
    text: str


def _normalize(text: str) -> str:
    """Collapse intra-line whitespace and drop blank lines (keeps one item per line)."""
    lines = (_WS.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _walk(element: object, out: list[str]) -> None:
    """Depth-first text collector that inserts newlines after block elements."""
    tag = getattr(element, "tag", None)
    if not isinstance(tag, str):  # comment / PI: skip (its tail is handled by the parent loop)
        return
    tag = tag.lower()
    if tag in _SKIP_TAGS:
        return
    text = getattr(element, "text", None)
    if text:
        out.append(text)
    for child in element:  # type: ignore[attr-defined]
        _walk(child, out)
        tail = getattr(child, "tail", None)
        if tail:
            out.append(tail)
    if tag in _BLOCK_TAGS:
        out.append("\n")


def html_to_text(content: str) -> str:
    """Extract readable, whitespace-normalized text from a filing (HTML or plain text).

    Handles inline-XBRL XHTML (all modern filings): strips the leading XML declaration so
    lxml parses it as a DOM (otherwise it raises and we lose all block structure), and drops
    the iXBRL hidden-fact metadata. Fail-soft: plain text passes through normalization;
    truly unparseable markup falls back to a regex tag-strip. Decodes entities + nbsp.
    """
    if not content or not content.strip():
        return ""
    if "<" not in content:  # already plain text (older SGML/TXT filings)
        return _normalize(content)
    cleaned = _XML_DECL.sub("", content, count=1)  # let lxml parse the str (iXBRL fix)
    try:
        tree = lhtml.fromstring(cleaned)
    except (ParserError, ValueError):
        return _normalize(_TAG.sub(" ", cleaned))
    parts: list[str] = []
    _walk(tree, parts)
    return _normalize("".join(parts))


def detect_sections(text: str) -> list[Section]:
    """Split normalized filing text on EDGAR "Item" headers into labeled sections.

    Text before the first Item header becomes a ``Preamble`` section; a filing with
    no recognizable Item headers becomes one unlabeled section. Note: a filing's
    table-of-contents repeats the Item headers, so this can emit short duplicate
    sections — tolerated here and handled by dedup/chunking in P3.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [Section(label="", text=text.strip())] if text.strip() else []

    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(label="Preamble", text=preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if not body:
            continue
        label = body.splitlines()[0].strip()[:120]  # the header line, capped
        sections.append(Section(label=label, text=body))
    return sections


def parse_metadata(raw: Mapping[str, object]) -> DocumentMetadata:
    """Build ``DocumentMetadata`` from a ``metadata.json`` sidecar (extra keys ignored)."""
    return DocumentMetadata.model_validate(raw)


@dataclass(frozen=True)
class ParsedFiling:
    """A parsed filing: provenance + normalized text + detected sections."""

    metadata: DocumentMetadata
    text: str
    sections: tuple[Section, ...]

    def to_document(self) -> Document:
        """The full-text ``Document`` (sections are carried separately for chunking)."""
        return Document(metadata=self.metadata, text=self.text)


def load_filing(directory: Path) -> ParsedFiling:
    """Parse a downloaded filing directory (``filing.html`` + ``metadata.json``)."""
    html = (directory / "filing.html").read_text(encoding="utf-8")
    raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata = parse_metadata(raw)
    text = html_to_text(html)
    return ParsedFiling(metadata=metadata, text=text, sections=tuple(detect_sections(text)))
