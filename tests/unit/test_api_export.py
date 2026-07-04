"""Contract tests for the P2.5 export endpoint (``POST /export``).

Deterministic (no network/LLM): the exporter is pure and the chart figures render via vl-convert
(same path the ``viz.render`` tests exercise). Asserts the download contract the React ExportMenu
relies on — right MIME + attachment filename per format, real bytes, and 422 (not 500) on bad input.
"""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from stock_agent.api.app import create_app
from stock_agent.reports.export import EXPORT_META
from stock_agent.viz.charts import ChartSpec

TEXT = "# Summary\n\nOver **20 days** the ensemble assigns 58% to a positive return.\n"


def _client() -> TestClient:
    return TestClient(create_app())


def _chart_dict() -> dict[str, object]:
    spec = ChartSpec(
        title="Scenario probabilities",
        kind="bar",
        data=pd.DataFrame({"bucket": ["down", "up"], "probability": [0.42, 0.58]}),
        x="bucket",
        y="probability",
        y_is_percent=True,
    )
    return spec.to_dict()


def test_export_md_returns_markdown_bytes_with_attachment_name() -> None:
    r = _client().post("/export", json={"text": TEXT, "fmt": "md", "title": "NVDA Research"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(EXPORT_META["md"][0])  # text/markdown
    assert r.headers["content-disposition"] == 'attachment; filename="nvda-research.md"'
    body = r.content.decode("utf-8")
    assert "NVDA Research" in body  # title header
    assert "NOT financial advice" in body  # non-advisory disclaimer always present
    assert "positive return" in body  # the answer text carried through verbatim


def test_export_pdf_and_docx_return_real_document_bytes() -> None:
    client = _client()
    pdf = client.post("/export", json={"text": TEXT, "fmt": "pdf"})
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"
    assert pdf.headers["content-type"].startswith("application/pdf")

    docx = client.post("/export", json={"text": TEXT, "fmt": "docx"})
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"  # docx is a zip (Office Open XML)
    assert "wordprocessingml" in docx.headers["content-type"]


def test_export_embeds_chart_figures_in_pdf() -> None:
    # A pdf WITH a figure must be larger than the same text-only pdf (the PNG got embedded).
    client = _client()
    text_only = client.post("/export", json={"text": TEXT, "fmt": "pdf"}).content
    with_chart = client.post(
        "/export", json={"text": TEXT, "fmt": "pdf", "charts": [_chart_dict()]}
    ).content
    assert len(with_chart) > len(text_only)


def test_export_markdown_ignores_charts() -> None:
    # Markdown is text-only; posting charts must not change or break the output.
    client = _client()
    plain = client.post("/export", json={"text": TEXT, "fmt": "md"}).content
    payload = {"text": TEXT, "fmt": "md", "charts": [_chart_dict()]}
    with_charts = client.post("/export", json=payload).content
    assert plain == with_charts


def test_export_unknown_format_is_422_not_500() -> None:
    r = _client().post("/export", json={"text": TEXT, "fmt": "xls"})
    assert r.status_code == 422


def test_export_empty_text_is_422() -> None:
    r = _client().post("/export", json={"text": "   ", "fmt": "md"})
    assert r.status_code == 422
