"""Section-aware chunking (RAG P3) — pure, deterministic."""

from __future__ import annotations

from datetime import UTC, date, datetime

from stock_agent.documents.parsers import ParsedFiling, Section
from stock_agent.rag.chunking import (
    _overlap_words,
    _target_words,
    chunk_filing,
    chunk_sections,
)
from stock_agent.schemas.documents import DocumentMetadata


def _meta() -> DocumentMetadata:
    return DocumentMetadata(
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda-10k.htm",
        filing_date=date(2025, 2, 26),
        document_id="NVDA:10-K:2025-02-26:0001045810-25-000017",
        ingested_at=datetime(2026, 6, 10, tzinfo=UTC),
    )


def _numbered(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# chunk_tokens=200 -> target 150 words; overlap 0.2 -> 30 words; step 120.
_TOK, _OVL = 200, 0.2
_TARGET = _target_words(_TOK)  # 150
_OVERLAP = _overlap_words(_TARGET, _OVL)  # 30


def _chunk(section_text: str, label: str = "Item 1A. Risk Factors"):  # type: ignore[no-untyped-def]
    sections = [Section(label=label, text=section_text)]
    return chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)


def test_no_chunk_exceeds_target() -> None:
    chunks = _chunk(_numbered("w", 400))
    assert chunks  # produced something
    assert all(len(c.text.split()) <= _TARGET for c in chunks)


def test_consecutive_chunks_overlap_exactly() -> None:
    chunks = _chunk(_numbered("w", 400))
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text.split()[-_OVERLAP:] == b.text.split()[:_OVERLAP]


def test_coverage_is_complete_and_ordered() -> None:
    chunks = _chunk(_numbered("w", 400))
    assert chunks[0].text.split()[0] == "w0"  # starts at the beginning
    assert chunks[-1].text.split()[-1] == "w399"  # reaches the end
    # Every original word appears in some chunk (overlap means no gaps).
    seen = {w for c in chunks for w in c.text.split()}
    assert seen == {f"w{i}" for i in range(400)}


def test_short_section_is_one_chunk() -> None:
    chunks = _chunk(_numbered("w", 40))  # < target (150), >= min_chunk_words
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 40


def test_overlap_zero_means_no_shared_words() -> None:
    sections = [Section(label="Item 7. MD&A", text=_numbered("w", 400))]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=0.0)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.text.split()) & set(b.text.split()) == set()  # disjoint


def test_chunks_never_cross_section_boundaries() -> None:
    sections = [
        Section(label="Item 1A. Risk Factors", text=_numbered("a", 300)),
        Section(label="Item 7. MD&A", text=_numbered("b", 300)),
    ]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)
    for c in chunks:
        words = c.text.split()
        prefixes = {w[0] for w in words}
        assert prefixes in ({"a"}, {"b"})  # no chunk mixes the two sections
        assert c.section == ("Item 1A. Risk Factors" if prefixes == {"a"} else "Item 7. MD&A")


def test_toc_and_trivial_sections_are_dropped() -> None:
    sections = [
        Section(label="Item 1A.", text="Item 1A."),  # TOC header-only -> dropped
        Section(label="Preamble", text="NVIDIA CORP Annual Report"),  # < 15 words -> dropped
        Section(label="Item 1A. Risk Factors", text="Item 1A. Risk Factors " + _numbered("r", 60)),
    ]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)
    assert chunks and all(c.section == "Item 1A. Risk Factors" for c in chunks)


def test_toc_stub_bare_header_short_body_dropped() -> None:
    # Bare item header ("Item 5.") + short body (title + page number) = a table-of-contents
    # line that survives the 15-word floor (17 words) but is real noise -> dropped.
    toc = (
        "Item 5.\nMarket for Registrant's Common Equity Related Stockholder Matters "
        "and Issuer Purchases of Equity Securities\n33"
    )
    sections = [
        Section(label="Item 5.", text=toc),
        Section(label="Item 1A. Risk Factors", text="Item 1A. Risk Factors " + _numbered("r", 60)),
    ]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)
    assert chunks and all(c.section == "Item 1A. Risk Factors" for c in chunks)


def test_real_bare_header_section_with_prose_is_kept() -> None:
    # 8-K items render the number on its own line (bare header) but carry real prose; the
    # word guard keeps them — only short bare-header stubs are TOC noise.
    body = "Item 8.01\nOther Events.\n" + _numbered("w", 40)  # 44 words >= toc-stub threshold
    sections = [Section(label="Item 8.01", text=body)]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)
    assert chunks and chunks[0].section == "Item 8.01"


def test_titled_short_section_is_kept() -> None:
    # Real short section: the label carries a title (not bare), so it is never a TOC stub.
    body = (
        "Item 3. Legal Proceedings Please see Note 12 of the Notes to the "
        "Consolidated Financial Statements included in this report."
    )
    sections = [Section(label="Item 3. Legal Proceedings", text=body)]
    chunks = chunk_sections(sections, _meta(), chunk_tokens=_TOK, chunk_overlap=_OVL)
    assert chunks and chunks[0].section == "Item 3. Legal Proceedings"


def test_no_item_fallback_section_has_none_section() -> None:
    chunks = _chunk(_numbered("w", 60), label="")  # detect_sections' no-item fallback
    assert all(c.section is None for c in chunks)


def test_metadata_and_chunk_ids() -> None:
    chunks = _chunk(_numbered("w", 400))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # unique
    for i, c in enumerate(chunks):
        assert c.chunk_index == i  # sequential, document-global
        assert c.chunk_id == f"{_meta().document_id}:{i}"
        assert c.ticker == "NVDA" and c.document_type == "10-K"
        assert c.filing_date == date(2025, 2, 26)
        assert c.document_id == _meta().document_id


def test_deterministic() -> None:
    assert [c.model_dump() for c in _chunk(_numbered("w", 500))] == [
        c.model_dump() for c in _chunk(_numbered("w", 500))
    ]


def test_chunk_filing_end_to_end() -> None:
    parsed = ParsedFiling(
        metadata=_meta(),
        text="(full text unused here)",
        sections=(
            Section(label="Item 1A. Risk Factors", text=_numbered("r", 300)),
            Section(label="Item 7. MD&A", text=_numbered("m", 200)),
        ),
    )
    chunks = chunk_filing(parsed, chunk_tokens=_TOK, chunk_overlap=_OVL)
    assert {c.section for c in chunks} == {"Item 1A. Risk Factors", "Item 7. MD&A"}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_sections_produce_no_chunks() -> None:
    assert chunk_sections([], _meta()) == []
    assert chunk_sections([Section(label="", text="   ")], _meta()) == []
