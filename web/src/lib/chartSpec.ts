// ChartSpec → Vega-Lite translation (Phase 2, P2.3).
//
// The `chart` SSE frame carries the render-agnostic ChartSpec dict (viz.charts.ChartSpec.to_dict),
// NOT a Vega-Lite spec — the Python UI turns it into an Altair chart in viz/render.to_altair. This
// module is the React twin of to_altair + ui.chart_theme.altair_config: it must produce the same
// marks/encodings/theme so the two frontends render identical charts (cross-stack parity, plan §6).
//
// Palette parity: Python fixes the mark palette to the *dark-theme* token hexes for BOTH themes
// (ui.chart_theme comment: one bar fill must read on the dark app AND the white export page), so we
// hardcode the same dark hexes here rather than reading the theme-flipping CSS vars — matching
// Python exactly. Text/label colors are left to Vega's defaults (Python leaves them to Streamlit's
// adaptive theme); a future slice can theme labels via the active --sa-* tokens if needed.

import type { VisualizationSpec } from "vega-embed";

/** JSON form of viz.charts.ChartSpec (column-oriented `data`, per DataFrame.to_dict("list")). */
export interface ChartSpecDict {
  title: string;
  /** "bar" | "grouped_bar" | "reliability". */
  kind: string;
  x: string;
  y: string;
  caption: string;
  color: string | null;
  x_sort: string[] | null;
  y_is_percent: boolean;
  /** column name -> column values (row i is data[col][i] across all columns). */
  data: Record<string, unknown[]>;
}

// --- palette constants (ui.chart_theme, dark-token hexes) ---------------------------------------
const MARK_COLOR = "#E8A13A"; // --sa-accent (single-series bars)
const POINT_COLOR = "#4FA8E8"; // --sa-sky (reliability points)
const REFERENCE_COLOR = "rgba(138,147,163,0.55)"; // y=x guide (muted, higher alpha than grid)
const GRID = "rgba(138,147,163,0.14)"; // --sa-grid
// CATEGORY_PALETTE order (sky, rose, indigo, teal, violet) — colorblind-separable first pair.
const CATEGORY_PALETTE = ["#4FA8E8", "#E5709B", "#7C82F0", "#35B0A7", "#B38BEA"];
const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace';

/** The Vega-Lite `config` block — twin of ui.chart_theme.altair_config (theme-independent look). */
function vlConfig(): Record<string, unknown> {
  return {
    view: { stroke: null },
    axis: {
      grid: true,
      gridColor: GRID,
      gridWidth: 1,
      domainColor: GRID,
      tickColor: GRID,
      labelFont: MONO,
      titleFont: MONO,
      labelFontSize: 11,
      titleFontSize: 12,
    },
    legend: { labelFont: MONO, titleFont: MONO, labelFontSize: 11, titleFontSize: 11 },
    title: { font: MONO, fontSize: 14, fontWeight: 600, anchor: "start" },
    range: { category: CATEGORY_PALETTE },
  };
}

/** Column-oriented data (ChartSpec.to_dict) -> Vega-Lite row records. */
export function toRecords(data: Record<string, unknown[]>): Array<Record<string, unknown>> {
  const cols = Object.keys(data);
  const n = cols.length ? Math.max(...cols.map((c) => data[c].length)) : 0;
  const rows: Array<Record<string, unknown>> = [];
  for (let i = 0; i < n; i++) {
    const row: Record<string, unknown> = {};
    for (const c of cols) row[c] = data[c][i];
    rows.push(row);
  }
  return rows;
}

/** Build the themed Vega-Lite spec for a ChartSpec dict (twin of viz.render.to_altair). */
export function chartSpecToVegaLite(spec: ChartSpecDict): VisualizationSpec {
  const rows = toRecords(spec.data);
  const cols = Object.keys(spec.data);
  const yAxis = spec.y_is_percent ? { format: "%" } : {};
  const xSort = spec.x_sort ?? "ascending";
  const base = {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    title: spec.title,
    config: vlConfig(),
  };

  if (spec.kind === "reliability") {
    // Predicted vs realized points + the y=x ideal as a dashed reference line (layered).
    return {
      ...base,
      layer: [
        {
          data: { values: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
          mark: { type: "line", strokeDash: [5, 5], color: REFERENCE_COLOR },
          encoding: {
            x: { field: "x", type: "quantitative" },
            y: { field: "y", type: "quantitative" },
          },
        },
        {
          data: { values: rows },
          mark: { type: "circle", size: 90, color: POINT_COLOR },
          encoding: {
            x: { field: "predicted", type: "quantitative", scale: { domain: [0, 1] }, title: "Predicted" },
            y: { field: "realized", type: "quantitative", scale: { domain: [0, 1] }, title: "Realized" },
          },
        },
      ],
    } as VisualizationSpec;
  }

  if (spec.kind === "grouped_bar" && spec.color) {
    // Series colored by the semantic secondary palette; xOffset separates the grouped bars.
    return {
      ...base,
      data: { values: rows },
      mark: "bar",
      encoding: {
        x: { field: spec.x, type: "nominal", sort: xSort, title: null },
        y: { field: spec.y, type: "quantitative", axis: yAxis, title: null },
        color: { field: spec.color, type: "nominal", title: null, scale: { range: CATEGORY_PALETTE } },
        xOffset: { field: spec.color, type: "nominal" },
        tooltip: cols.map((c) => ({ field: c })),
      },
    } as VisualizationSpec;
  }

  // Default: single-series brass bars.
  return {
    ...base,
    data: { values: rows },
    mark: { type: "bar", color: MARK_COLOR },
    encoding: {
      x: { field: spec.x, type: "nominal", sort: xSort, title: null },
      y: { field: spec.y, type: "quantitative", axis: yAxis, title: null },
      tooltip: cols.map((c) => ({ field: c })),
    },
  } as VisualizationSpec;
}
