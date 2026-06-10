"""SEC filing parsing (RAG P2): html_to_text + section detection (pure, offline)."""

from __future__ import annotations

import json
from pathlib import Path

from stock_agent.documents.parsers import (
    Section,
    detect_sections,
    html_to_text,
    load_filing,
    parse_metadata,
)

_10K_HTML = """
<html><head><title>NVDA 10-K</title><style>.x{color:red}</style></head>
<body>
  <p>NVIDIA CORPORATION&nbsp;&mdash; Annual Report (Form 10-K)</p>
  <table><tr><td>Item 1A.</td></tr><tr><td>Item 7.</td></tr></table>
  <script>var tracking = 1;</script>
  <div>Item 1A. Risk Factors</div>
  <p>Our business faces&nbsp;substantial competition and supply-chain risk.</p>
  <div>Item 7. Management&rsquo;s Discussion and Analysis</div>
  <p>Revenue grew on accelerated-computing demand during fiscal 2025.</p>
</body></html>
"""


def _section_with(sections: list[Section], needle: str) -> Section:
    match = [s for s in sections if needle.lower() in s.label.lower()]
    assert match, f"no section labelled like {needle!r} in {[s.label for s in sections]}"
    return match[0]


# ---- html_to_text ------------------------------------------------------------
def test_html_to_text_strips_noise_and_decodes_entities() -> None:
    text = html_to_text(_10K_HTML)
    assert "tracking" not in text  # <script> dropped
    assert "color:red" not in text  # <style> dropped
    assert "NVDA 10-K" not in text  # <title> dropped
    assert "NVIDIA CORPORATION" in text
    assert " " not in text  # nbsp collapsed to a normal space
    assert "Management’s" in text  # &rsquo; decoded


def test_html_to_text_inserts_block_breaks() -> None:
    # Block elements become separate lines (so Item headers anchor at line start).
    lines = html_to_text("<div>Item 1A. Risk Factors</div><p>Body text here.</p>").splitlines()
    assert lines == ["Item 1A. Risk Factors", "Body text here."]


def test_html_to_text_plain_text_passthrough() -> None:
    assert html_to_text("Item 1A.  Risk\tFactors") == "Item 1A. Risk Factors"


def test_html_to_text_malformed_falls_back_not_raises() -> None:
    out = html_to_text("<div>unclosed <b>bold text")  # tolerated by lxml / fallback
    assert "bold text" in out


def test_html_to_text_empty() -> None:
    assert html_to_text("") == ""
    assert html_to_text("   \n  ") == ""


# Inline-XBRL XHTML (all modern filings): an XML declaration + an ix:hidden fact blob +
# Item headers wrapped in <div>s. Regression for the bug where lxml rejected the str
# (encoding declaration), the crude fallback collapsed everything to one line + leaked the
# hidden facts, and section detection found nothing.
_IXBRL_10K = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>NVDA 10-K</title></head>'
    "<body>"
    '<ix:header><ix:hidden>'
    '<ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>'
    "JUNKFACT-0001045810-2026-FY-false-362-460-P1Y-P2Y</ix:hidden></ix:header>"
    "<div>NVIDIA CORPORATION</div>"
    "<div>Item 1A. Risk Factors</div>"
    "<div>Our business faces "
    '<ix:nonNumeric name="x">substantial</ix:nonNumeric> competition and supply-chain risk.</div>'
    "<div>Item 7. Management&rsquo;s Discussion and Analysis</div>"
    '<div>Revenue was <ix:nonFraction name="us-gaap:Revenues">130,497</ix:nonFraction> m.</div>'
    "</body></html>"
)


def test_html_to_text_handles_inline_xbrl() -> None:
    text = html_to_text(_IXBRL_10K)
    assert text.count("\n") >= 3  # block structure preserved (NOT one collapsed line)
    assert "JUNKFACT" not in text  # ix:hidden metadata dropped
    assert "substantial" in text and "130,497" in text  # VISIBLE ix facts kept
    assert "NVDA 10-K" not in text  # <title> still dropped


def test_detect_sections_on_inline_xbrl() -> None:
    sections = detect_sections(html_to_text(_IXBRL_10K))
    risk = _section_with(sections, "Risk Factors")
    assert "competition and supply-chain risk" in risk.text
    mdna = _section_with(sections, "Management")
    assert "130,497" in mdna.text


# ---- detect_sections ---------------------------------------------------------
def test_detect_sections_finds_risk_and_mdna() -> None:
    sections = detect_sections(html_to_text(_10K_HTML))
    risk = _section_with(sections, "Risk Factors")
    assert "substantial competition" in risk.text
    mdna = _section_with(sections, "Management")
    assert "accelerated-computing demand" in mdna.text
    assert sections[0].label == "Preamble"  # cover text before the first Item header


def test_detect_sections_no_items_is_single_section() -> None:
    sections = detect_sections("Just some prose with no item headers at all.")
    assert len(sections) == 1
    assert sections[0].label == "" and "prose" in sections[0].text


def test_detect_sections_8k_decimal_items() -> None:
    text = "Item 2.02. Results of Operations\nWe reported revenue.\nItem 9.01. Exhibits\nEX-99.1"
    labels = [s.label for s in detect_sections(text)]
    assert "Item 2.02. Results of Operations" in labels
    assert "Item 9.01. Exhibits" in labels


def test_detect_sections_empty_text() -> None:
    assert detect_sections("") == []


# ---- metadata + load_filing --------------------------------------------------
_META = {
    "ticker": "NVDA",
    "document_type": "10-K",
    "source": "SEC",
    "source_url": "https://www.sec.gov/Archives/edgar/data/1045810/x/nvda-10k.htm",
    "filing_date": "2025-02-26",
    "document_id": "NVDA:10-K:2025-02-26:0001045810-25-000017",
    "accession_number": "0001045810-25-000017",  # extra key (ignored)
    "primary_document": "nvda-10k.htm",  # extra key (ignored)
    "section": None,
    "ingested_at": "2026-06-10T00:00:00+00:00",
}


def test_parse_metadata_ignores_extra_keys() -> None:
    meta = parse_metadata(_META)
    assert meta.ticker == "NVDA" and meta.document_type == "10-K"
    assert meta.filing_date.isoformat() == "2025-02-26"
    assert meta.section is None  # null preserved


def test_load_filing_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "filing.html").write_text(_10K_HTML, encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps(_META), encoding="utf-8")
    parsed = load_filing(tmp_path)
    assert parsed.metadata.ticker == "NVDA"
    assert "accelerated-computing demand" in parsed.text
    assert any("Risk Factors" in s.label for s in parsed.sections)
    doc = parsed.to_document()
    assert doc.metadata.document_id == _META["document_id"] and doc.text == parsed.text
