import type { Turn } from "@/store/conversation";
import { TileRow } from "@/components/TileRow";
import { ChartCard } from "@/components/ChartCard";
import { TraceRow } from "@/components/TraceRow";
import { Sources } from "@/components/Sources";

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
        turn.answer && (
          <p style={{ whiteSpace: "pre-wrap" }}>
            {turn.answer}
            {streaming && <span className="caret">▍</span>}
          </p>
        )
      )}

      {streaming && !turn.answer && turn.trace.length === 0 && (
        <p className="route-note">thinking…</p>
      )}

      <TraceRow trace={turn.trace} />
      <Sources sources={turn.sources} />
    </div>
  );
}
