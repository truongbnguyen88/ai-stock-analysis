"""Document export endpoint: ``POST /export`` (Phase 2, P2.5).

Render the assistant's final answer (markdown) plus its chart figures to PDF / Word / Markdown bytes
for download. Thin — the pure exporter (``reports.export.export_summary``) formats; the figures come
from the same ``ChartSpec`` dicts the client holds, rasterized via ``viz.render`` (the one chart
source of truth). No numbers are computed here: the answer text already carries the model figures;
the exporter only adds the non-advisory header (same as the Streamlit export).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from stock_agent.api.schemas import ExportRequest
from stock_agent.reports.export import EXPORT_META, export_summary
from stock_agent.viz.charts import ChartSpec
from stock_agent.viz.render import to_png

router = APIRouter()


def _slug(title: str, *, fallback: str = "summary") -> str:
    """A filesystem-safe download stem: lowercase alnum runs joined by single hyphens."""
    out: list[str] = []
    for ch in title.strip().lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or fallback


@router.post("/export")
def post_export(req: ExportRequest) -> Response:
    """Render the summary text (+ optional chart figures) to ``fmt`` bytes as a file download.

    Returns the document with the format's MIME type and a ``Content-Disposition: attachment`` name
    so the browser saves it. Unknown format or empty text → 422 (never a 500). Figures are embedded
    only for pdf/docx (markdown is text-only), rendered from the posted ChartSpec dicts.
    """
    if req.fmt not in EXPORT_META:
        raise HTTPException(status_code=422, detail=f"bad format: {req.fmt!r} (use pdf/docx/md)")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="nothing to export (empty text)")

    # pdf/docx embed the same charts the client rendered; markdown ignores images.
    images: list[bytes] | None = None
    if req.fmt != "md" and req.charts:
        images = [to_png(ChartSpec.from_dict(c)) for c in req.charts]

    data = export_summary(req.text, req.fmt, title=req.title, images=images)
    mime, ext = EXPORT_META[req.fmt]
    filename = f"{_slug(req.title)}.{ext}"
    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
