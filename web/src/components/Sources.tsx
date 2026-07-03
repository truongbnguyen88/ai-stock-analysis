import type { Citation } from "@/lib/events";

/**
 * SEC filing citations for the turn. Sources come from the tool OUTPUT (resolved against the
 * retrieved set server-side), never the LLM — so rendering them here cannot introduce a fabricated
 * source (numbers/citations-from-tools invariant). `url` is not carried yet (P2.5 export path).
 */
export function Sources({ sources }: { sources: Citation[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-mono text-[10px] uppercase tracking-wider text-faint">sources</span>
      {sources.map((c, i) => (
        <span
          key={`${c.marker ?? i}-${c.label}`}
          className="inline-flex items-center gap-1 rounded-sa-sm border border-border-strong bg-surface-2 px-2 py-1 font-mono text-[11px] text-muted"
        >
          {c.marker != null && <span className="text-accent">[{c.marker}]</span>}
          {c.label}
        </span>
      ))}
    </div>
  );
}
