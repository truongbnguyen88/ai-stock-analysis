"""Export a chat summary (markdown text) to PDF / DOCX / Markdown bytes.

Used by the Streamlit chat to let a user save the agent's executive summary. The
summary text already contains the model figures (the agent is text-only); this
module just renders that text into a document with a non-advisory header — it adds
no numbers of its own. Pure (returns bytes); the heavy libs are lazy-imported.

PDF uses fpdf2's core latin-1 font, so unicode punctuation is transliterated to
ASCII for PDF only; DOCX and Markdown keep the text verbatim.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from io import BytesIO

DISCLAIMER = (
    "Research and education only - NOT financial advice. Figures come from statistical "
    "models and cited sources; the assistant provides no recommendations."
)

# fmt -> (mime type, file extension) for the download button.
EXPORT_META: dict[str, tuple[str, str]] = {
    "pdf": ("application/pdf", "pdf"),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
    ),
    "md": ("text/markdown", "md"),
}

_UNICODE_MAP = {
    "→": "->", "←": "<-", "•": "-", "–": "-", "—": "--",
    "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
    "−": "-", "×": "x", "✓": "[ok]", "✅": "[ok]", "❌": "[x]",
    "⚠": "[!]", "≈": "~", "≤": "<=", "≥": ">=", "±": "+/-",
}


def _ascii(s: str) -> str:
    """Transliterate common unicode to ASCII, then drop anything still non-latin-1."""
    for k, v in _UNICODE_MAP.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1")


def _blocks(text: str) -> list[tuple[str, int, str]]:
    """Classify each line as (kind, level, content): heading / bullet / paragraph / blank."""
    out: list[tuple[str, int, str]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out.append(("blank", 0, ""))
            continue
        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            out.append(("h", len(h.group(1)), h.group(2).strip()))
            continue
        b = re.match(r"^\s*[-*+]\s+(.*)", line)
        if b:
            out.append(("bullet", 0, b.group(1).strip()))
            continue
        out.append(("p", 0, line.strip()))
    return out


def export_summary(
    text: str,
    fmt: str,
    *,
    title: str = "Stock Research Summary",
    generated_at: datetime | None = None,
) -> bytes:
    """Render ``text`` (markdown) to ``fmt`` ('pdf' | 'docx' | 'md') as bytes."""
    gen = generated_at or datetime.now(UTC)
    if fmt == "md":
        return _to_md(text, title, gen)
    if fmt == "docx":
        return _to_docx(text, title, gen)
    if fmt == "pdf":
        return _to_pdf(text, title, gen)
    raise ValueError(f"unsupported export format: {fmt!r} (use pdf/docx/md)")


def _stamp(gen: datetime) -> str:
    return f"Generated {gen:%Y-%m-%d %H:%M UTC}"


def _to_md(text: str, title: str, gen: datetime) -> bytes:
    header = f"# {title}\n\n*{_stamp(gen)}*\n\n> {DISCLAIMER}\n\n---\n\n"
    return (header + text.strip() + "\n").encode("utf-8")


def _add_runs(paragraph: object, content: str) -> None:
    """Add text to a docx paragraph, honoring **bold** spans."""
    for part in re.split(r"(\*\*[^*]+\*\*)", content):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2])  # type: ignore[attr-defined]
            run.bold = True
        else:
            paragraph.add_run(part)  # type: ignore[attr-defined]


def _to_docx(text: str, title: str, gen: datetime) -> bytes:
    from docx import Document  # lazy

    doc = Document()
    doc.add_heading(title, level=0)
    doc.add_paragraph(_stamp(gen)).runs[0].italic = True
    doc.add_paragraph(DISCLAIMER)
    for kind, level, content in _blocks(text):
        if kind == "h":
            doc.add_heading(content.replace("**", ""), level=min(max(level, 1), 4))
        elif kind == "bullet":
            _add_runs(doc.add_paragraph(style="List Bullet"), content)
        elif kind == "p":
            _add_runs(doc.add_paragraph(), content)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _to_pdf(text: str, title: str, gen: datetime) -> bytes:
    from fpdf import FPDF  # lazy

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _ascii(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, _ascii(_stamp(gen)), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 5, _ascii(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for kind, level, content in _blocks(text):
        if kind == "blank":
            pdf.ln(2)
            continue
        if kind == "h":
            pdf.set_font("Helvetica", "B", max(11, 15 - level))
            pdf.multi_cell(0, 7, _ascii(content.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 11)
            body = ("- " + content) if kind == "bullet" else content
            pdf.multi_cell(0, 6, _ascii(body), markdown=True, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
