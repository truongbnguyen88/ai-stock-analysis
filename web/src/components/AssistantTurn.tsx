import type { Turn } from "@/store/conversation";
import { TileRow } from "@/components/TileRow";
import { ChartCard } from "@/components/ChartCard";
import { TraceRow } from "@/components/TraceRow";
import { Sources } from "@/components/Sources";

/**
 * The assistant side of a turn, composed in the mockup's summary-before-detail order (plan §5):
 * tiles → chart(s) → prose → trace → sources. Prose is PROVISIONAL while streaming (dim + caret);
 * on `error` the provisional prose is already cleared by the reducer and we show an error surface
 * instead (the client never persists unguarded prose, §5).
 */
export function AssistantTurn({ turn }: { turn: Turn }) {
  const streaming = turn.status === "streaming";
  return (
    <div className="flex flex-col gap-3" data-testid="assistant-turn" data-status={turn.status}>
      <TileRow tiles={turn.tiles} />

      {turn.charts.map((spec, i) => (
        <ChartCard key={`${spec.title}-${i}`} spec={spec} />
      ))}

      {turn.status === "error" ? (
        <div className="rounded-sa-sm border border-down bg-surface-2 p-3 text-sm text-down">
          <span className="font-mono text-[11px] uppercase tracking-wider">
            {turn.error?.code ?? "error"}
          </span>
          <p className="mt-1 text-muted">{turn.error?.message}</p>
        </div>
      ) : (
        turn.answer && (
          <div className="max-w-prose whitespace-pre-wrap text-sm leading-relaxed text-text">
            {turn.answer}
            {streaming && <span className="ml-0.5 inline-block animate-pulse text-accent">▍</span>}
          </div>
        )
      )}

      {streaming && !turn.answer && turn.trace.length === 0 && (
        <p className="font-mono text-xs text-faint">thinking…</p>
      )}

      <TraceRow trace={turn.trace} />
      <Sources sources={turn.sources} />
    </div>
  );
}
