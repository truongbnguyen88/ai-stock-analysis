"""Chat-summary export: valid PDF/DOCX/MD bytes with the non-advisory header."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stock_agent.reports.export import DISCLAIMER, EXPORT_META, export_summary

_SUMMARY = """## Executive summary — NVDA

**Snapshot.** Last close $120.50, RSI 58.

- The historical_sim 20-day model puts P(up) at 76%.
- Big-move probability P(|r|>10%) is 18% → lean up.

**Risks.** Earnings inside the horizon; the price-only model can't see them.
"""

_GEN = datetime(2026, 5, 31, 14, 30, tzinfo=UTC)


def test_markdown_export_keeps_text_and_header() -> None:
    out = export_summary(_SUMMARY, "md", title="NVDA Summary", generated_at=_GEN).decode("utf-8")
    assert out.startswith("# NVDA Summary")
    assert DISCLAIMER in out
    assert "P(up) at 76%" in out  # original markdown preserved verbatim
    assert "2026-05-31 14:30 UTC" in out


def test_pdf_export_is_valid_pdf() -> None:
    data = export_summary(_SUMMARY, "pdf", generated_at=_GEN)
    assert data[:5] == b"%PDF-"  # PDF magic
    assert len(data) > 800  # non-trivial document


def test_docx_export_is_valid_zip() -> None:
    data = export_summary(_SUMMARY, "docx", generated_at=_GEN)
    assert data[:2] == b"PK"  # docx is a zip (OOXML)
    # readable as a zip with the document part
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "word/document.xml" in zf.namelist()


def test_pdf_handles_unicode_without_error() -> None:
    # Arrows, checkmarks, emoji, math symbols must not crash the latin-1 PDF font.
    tricky = "Result → up ✅ (≈ 0.92, ≤ 1.0) — 中文 emoji 🚀"
    data = export_summary(tricky, "pdf", generated_at=_GEN)
    assert data[:5] == b"%PDF-"


_WITH_TABLE = """## Model comparison — NVDA

| model | P(up) | expected |
|---|---|---|
| historical_sim | 76% | +24.14% |
| lightgbm | 67% | +5.06% |

**Note.** Figures from the models.
"""


def test_markdown_table_renders_in_docx() -> None:
    import io
    import zipfile

    data = export_summary(_WITH_TABLE, "docx")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
    assert "<w:tbl>" in doc_xml  # a real Word table, not a paragraph of pipes
    assert "historical_sim" in doc_xml and "+24.14%" in doc_xml
    assert "|" not in doc_xml.split("</w:tbl>")[0].split("<w:tbl>")[-1]  # no raw pipes in the table


def test_markdown_table_renders_in_pdf() -> None:
    data = export_summary(_WITH_TABLE, "pdf")
    assert data[:5] == b"%PDF-"
    assert len(data) > 1000  # table laid out, not a one-liner


# A minimal valid 1x1 PNG (so the embed path is exercised without a real chart render).
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00"
    b"\x00\x00IEND\xaeB`\x82"
)


def test_pdf_embeds_figures() -> None:
    no_img = export_summary(_SUMMARY, "pdf", generated_at=_GEN)
    with_img = export_summary(_SUMMARY, "pdf", generated_at=_GEN, images=[_PNG_1x1, _PNG_1x1])
    assert with_img[:5] == b"%PDF-"
    assert len(with_img) > len(no_img)  # figures added bytes


def test_docx_embeds_figures() -> None:
    import io
    import zipfile

    data = export_summary(_SUMMARY, "docx", generated_at=_GEN, images=[_PNG_1x1])
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    assert any(n.startswith("word/media/") for n in names)  # image part embedded


def test_markdown_ignores_images() -> None:
    a = export_summary(_SUMMARY, "md", generated_at=_GEN)
    b = export_summary(_SUMMARY, "md", generated_at=_GEN, images=[_PNG_1x1])
    assert a == b  # text-only format unaffected by images


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        export_summary(_SUMMARY, "rtf")


def test_export_meta_covers_all_formats() -> None:
    assert set(EXPORT_META) == {"pdf", "docx", "md"}
    for mime, ext in EXPORT_META.values():
        assert mime and ext
