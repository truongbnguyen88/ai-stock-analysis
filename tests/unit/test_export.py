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


def test_unknown_format_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        export_summary(_SUMMARY, "rtf")


def test_export_meta_covers_all_formats() -> None:
    assert set(EXPORT_META) == {"pdf", "docx", "md"}
    for mime, ext in EXPORT_META.values():
        assert mime and ext
