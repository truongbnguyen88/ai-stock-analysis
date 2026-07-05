import type { Turn } from "@/store/conversation";
import { TileRow } from "@/components/TileRow";
import { ChartCard } from "@/components/ChartCard";
import { TraceRow } from "@/components/TraceRow";
import { Sources } from "@/components/Sources";
import { ExportMenu } from "@/components/ExportMenu";
import { Markdown } from "@/components/Markdown";

/**
 * The assistant side of a turn, composed in the mockup's summary-before-detail order (plan §5):
 * tiles → chart(s) → prose → trace → sources. Prose is PROVISIONAL while streaming (brass caret);
 * on `error` the provisional prose is already cleared by the reducer and we show an error surface
 * instead (the client never persists unguarded prose, §5).
 */
export function AssistantTurn({ turn }: { turn: Turn }) {
  const streaming = turn.status === "streaming";
  return (
    <div data-testid="assistant-turn" data-status={turn.status}>
      <TileRow tiles={turn.tiles} />

      {turn.charts.map((spec, i) => (
        <ChartCard key={`${spec.title}-${i}`} spec={spec} />
      ))}

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
