import type { CSSProperties } from "react";
import { hueName } from "@/lib/events";
import { hueVar } from "@/lib/hues";
import type { TraceItem } from "@/store/conversation";

/**
 * Live per-tool trace — the Streamlit ceiling item (plan §1). Each chip appears on `tool_start`
 * (pulsing hue "running" dot) and resolves on `tool_finish` (✓ ok / ✕ error). Tinted by the tool's
 * stable hue (hueName(hue_key) reproduces ui.html.tool_hue), so a tool is the same color every turn.
 */
export function TraceRow({ trace }: { trace: TraceItem[] }) {
  if (trace.length === 0) return null;
  return (
    <div className="metabar">
      <span className="label" style={{ letterSpacing: "0.05em" }}>
        trace
      </span>
      {trace.map((item, i) => {
        const running = item.status === "running";
        const marker = running ? "" : item.ok ? "✓" : "✕";
        return (
          <span
            key={`${item.tool}-${i}`}
            className="trace"
            title={item.inputSummary || item.tool}
            data-status={item.status}
            style={{ "--tc-hue": hueVar(hueName(item.hueKey)) } as CSSProperties}
          >
            <span
              className={`dot${running ? " running" : ""}`}
              style={{ background: "var(--tc-hue)" }}
              aria-hidden
            />
            <span className="tool" style={{ color: "var(--tc-hue)" }}>
              {item.tool}
            </span>
            {marker && <span className={item.ok ? "ok" : "err"}>{marker}</span>}
            {item.elapsedMs != null && (
              <span style={{ color: "var(--sa-faint)" }}>{Math.round(item.elapsedMs)}ms</span>
            )}
          </span>
        );
      })}
    </div>
  );
}
