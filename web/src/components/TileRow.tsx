import type { CSSProperties } from "react";
import type { Tile } from "@/lib/events";
import { hueVar } from "@/lib/hues";

/**
 * Headline stat tiles above the answer ("summary-before-detail"). Mirrors ui.html.stat_tile: a
 * colored category stripe (the tone hue) + mono uppercase label + big value + optional sub. Values
 * are pre-formatted by the server (numbers-from-tools) — this is presentation only.
 */
export function TileRow({ tiles }: { tiles: Tile[] }) {
  if (tiles.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-3">
      {tiles.map((t) => (
        <div
          key={t.label}
          style={{ "--tile-hue": hueVar(t.tone) } as CSSProperties}
          className="min-w-[8.5rem] flex-1 rounded-sa-sm border border-border bg-surface-2 p-3 shadow-sa"
        >
          <div
            className="mb-2 h-1 w-8 rounded-full"
            style={{ background: "var(--tile-hue)" }}
            aria-hidden
          />
          <div className="font-mono text-[10px] uppercase tracking-wider text-faint">{t.label}</div>
          <div className="font-mono text-xl font-semibold text-text">{t.value}</div>
          {t.sub && <div className="mt-0.5 text-[11px] text-muted">{t.sub}</div>}
        </div>
      ))}
    </div>
  );
}
