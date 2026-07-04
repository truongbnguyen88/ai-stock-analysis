import type { Citation } from "@/lib/events";

/**
 * Collapsible SEC filing citations for the turn. Sources come from the tool OUTPUT (resolved against
 * the retrieved set server-side), never the LLM — so rendering them here cannot introduce a
 * fabricated source (numbers/citations-from-tools invariant). Mockup: a <details> panel with
 * [n] marker badges + labels. `url` is not carried yet (P2.5 export path).
 */
export function Sources({ sources }: { sources: Citation[] }) {
  if (sources.length === 0) return null;
  return (
    <details className="sources" open>
      <summary>▸ Filing sources ({sources.length})</summary>
      {sources.map((c, i) => (
        <div className="cite" key={`${c.marker ?? i}-${c.label}`}>
          {c.marker != null && <span className="mk">[{c.marker}]</span>}
          <span className="cl">{c.label}</span>
        </div>
      ))}
    </details>
  );
}
