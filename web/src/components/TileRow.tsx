import type { CSSProperties } from "react";
import type { Tile } from "@/lib/events";
import { hueVar } from "@/lib/hues";

/** Map the tool-driven direction to the value-color class (mockup .t-v.up / .t-v.down). */
const dirClass = (d: Tile["direction"]): string => (d === "up" ? " up" : d === "down" ? " down" : "");

/**
 * Headline stat tiles above the answer ("summary-before-detail"). Mockup: a left category stripe
 * (the tone hue, via the --tc custom prop) + mono uppercase label + big value + optional sub. The
 * value is tinted green/red by `direction` — a deterministic tool-driven sign, never the LLM
 * (absent → neutral). Values are pre-formatted by the server (numbers-from-tools); this is
 * presentation only.
 */
export function TileRow({ tiles }: { tiles: Tile[] }) {
  if (tiles.length === 0) return null;
  return (
    <div className="tiles">
      {tiles.map((t) => (
        <div key={t.label} className="tile" style={{ "--tc": hueVar(t.tone) } as CSSProperties}>
          <div className="t-k">{t.label}</div>
          <div className={`t-v${dirClass(t.direction)}`}>{t.value}</div>
          {t.sub && <div className="t-sub">{t.sub}</div>}
        </div>
      ))}
    </div>
  );
}
