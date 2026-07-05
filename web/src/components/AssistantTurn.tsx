import type { Turn } from "@/store/conversation";
import { TileRow } from "@/components/TileRow";
import { ChartCard } from "@/components/ChartCard";
import { TraceRow } from "@/components/TraceRow";
import { Sources } from "@/components/Sources";
import { ExportMenu } from "@/components/ExportMenu";
import { Markdown } from "@/components/Markdown";

/**
 * The assistant side of a turn, composed summary-before-detail (plan §5): tiles → prose →
 * figures(charts) → trace → sources. Charts sit AFTER the prose as a "Figures" group rather than
 * stacked above the first line: the narrative is organized into LLM-chosen sections we can't map a
 * given chart to (chart origin is the tool, not a section), so grouping the plots below the text
 * reads better than a wall of plots before any words. Tiles stay on top as the KPI summary strip.
 * Prose is PROVISIONAL while streaming (brass caret); on `error` the reducer has already cleared it
 * and we show an error surface instead (the client never persists unguarded prose, §5).
 */
export function AssistantTurn({ turn }: { turn: Turn }) {
  const streaming = turn.status === "streaming";
  return (
    <div data-testid="assistant-turn" data-status={turn.status}>
      <TileRow tiles={turn.tiles} />

      {turn.status === "error" ? (
        <div className="errbox">
          <span className="ek">{turn.error?.code ?? "error"}</span>
          <p className="em">{turn.error?.message}</p>
        </div>
      ) : (
        turn.answer &&
        // While streaming, render raw text + caret — partial Markdown (a half-built table, an
        // unclosed **bold) would render broken and reflow on every token. On `final` we switch to
        // the formatted Markdown render (headings, tables, lists). One reflow at completion is
        // expected and cheap.
        (streaming ? (
          <p style={{ whiteSpace: "pre-wrap" }}>
            {turn.answer}
            <span className="caret">▍</span>
          </p>
        ) : (
          <Markdown>{turn.answer}</Markdown>
        ))
      )}

      {turn.charts.length > 0 && (
        <div className="figures">
          {/* Label only when there's prose above to divide from (a chart-only turn needs no header). */}
          {turn.answer && <div className="figures-label">Figures</div>}
          {turn.charts.map((spec, i) => (
            <ChartCard key={`${spec.title}-${i}`} spec={spec} />
          ))}
        </div>
      )}

      {streaming && !turn.answer && turn.trace.length === 0 && (
        <p className="route-note">thinking…</p>
      )}

      <TraceRow trace={turn.trace} />
      <Sources sources={turn.sources} />

      {/* Export the finished, grounded answer (mockup's ↓ Export ▾). Only once final and non-empty. */}
      {turn.status === "final" && turn.answer && (
        <ExportMenu
          text={turn.answer}
          charts={turn.charts}
          title={turn.ticker ? `${turn.ticker} — research summary` : undefined}
        />
      )}
    </div>
  );
}
