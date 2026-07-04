import { useMemo } from "react";
import { VegaEmbed } from "react-vega";
import { chartSpecToVegaLite, type ChartSpecDict } from "@/lib/chartSpec";

/**
 * Render one chart via react-vega from the render-agnostic ChartSpec dict the `chart` frame carries.
 * The dict → Vega-Lite translation (chartSpecToVegaLite) is the twin of viz/render.to_altair, so the
 * React chart matches Streamlit's exactly (cross-stack parity, plan §6) — including the in-chart
 * title, so the card doesn't duplicate it in a header. SVG renderer (no canvas dep); actions off.
 */
export function ChartCard({ spec }: { spec: ChartSpecDict }) {
  const vlSpec = useMemo(() => chartSpecToVegaLite(spec), [spec]);
  return (
    <div className="chartcard">
      <div className="chart">
        <VegaEmbed spec={vlSpec} options={{ actions: false, renderer: "svg" }} />
      </div>
      {spec.caption && <div className="caption">{spec.caption}</div>}
    </div>
  );
}
