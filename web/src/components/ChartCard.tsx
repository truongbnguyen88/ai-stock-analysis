import { useMemo } from "react";
import { VegaEmbed } from "react-vega";
import { chartSpecToVegaLite, type ChartSpecDict } from "@/lib/chartSpec";

/**
 * Render one chart via react-vega from the render-agnostic ChartSpec dict the `chart` frame carries.
 * The dict → Vega-Lite translation (chartSpecToVegaLite) is the twin of viz/render.to_altair, so the
 * React chart matches Streamlit's exactly (cross-stack parity, plan §6). SVG renderer (no canvas
 * native dep); Vega's action menu is hidden to match the app chrome.
 */
export function ChartCard({ spec }: { spec: ChartSpecDict }) {
  const vlSpec = useMemo(() => chartSpecToVegaLite(spec), [spec]);
  return (
    <figure className="overflow-x-auto rounded-sa-sm border border-border bg-surface-2 p-3 shadow-sa">
      <VegaEmbed spec={vlSpec} options={{ actions: false, renderer: "svg" }} />
      {spec.caption && (
        <figcaption className="mt-1 text-[11px] text-muted">{spec.caption}</figcaption>
      )}
    </figure>
  );
}
