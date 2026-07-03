import type { CSSProperties } from "react";
import { hueName } from "@/lib/events";
import { hueVar } from "@/lib/hues";
import type { TraceItem } from "@/store/conversation";

/**
 * Live per-tool trace — the Streamlit ceiling item (plan §1). Each chip appears on `tool_start`
 * (pulsing "running" dot) and resolves on `tool_finish` (✓ ok / ✕ error). Tinted by the tool's
 * stable hue (hueName(hue_key) reproduces ui.html.tool_hue), so a tool is the same color every turn.
 */
export function TraceRow({ trace }: { trace: TraceItem[] }) {
  if (trace.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-mono text-[10px] uppercase tracking-wider text-faint">trace</span>
      {trace.map((item, i) => {
        const running = item.status === "running";
        const marker = running ? "" : item.ok ? "✓" : "✕";
        return (
          <span
            key={`${item.tool}-${i}`}
            style={{ "--tc-hue": hueVar(hueName(item.hueKey)) } as CSSProperties}
            title={item.inputSummary || item.tool}
            className="inline-flex items-center gap-1.5 rounded-full border border-border-strong px-2 py-1 font-mono text-[11px]"
            data-status={item.status}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${running ? "animate-pulse" : ""}`}
              style={{ background: "var(--tc-hue)" }}
              aria-hidden
            />
            <span style={{ color: "var(--tc-hue)" }}>{item.tool}</span>
            {marker && <span className={item.ok ? "text-up" : "text-down"}>{marker}</span>}
            {item.elapsedMs != null && (
              <span className="text-faint">{Math.round(item.elapsedMs)}ms</span>
            )}
          </span>
        );
      })}
    </div>
  );
}
