// Recorded AgentEvent stream fixtures (P2.3 tests) — the same wire frames the Python adapter emits
// (api/streaming.adapt_events), in the §5 order. Used to drive the reducer, the SSE parser, and the
// component tests with NO live backend (repo test rule). Kept as plain data so a schema change on
// either side breaks a test loudly.

import type { AgentEvent } from "@/lib/events";
import type { ChartSpecDict } from "@/lib/chartSpec";

export const FORECAST_CHART: ChartSpecDict = {
  title: "Scenario probabilities",
  kind: "bar",
  x: "bucket",
  y: "prob",
  caption: "20-day GBM",
  color: null,
  x_sort: ["down", "flat", "up"],
  y_is_percent: true,
  data: { bucket: ["down", "flat", "up"], prob: [0.3, 0.12, 0.58] },
};

/** A deterministic forecast turn: tiles + a chart, no sources (mirrors the Python §5 test). */
export const FORECAST_TURN: AgentEvent[] = [
  { type: "turn_start", thread_id: "th", turn_id: "tn", route: "forecast", ticker: "NVDA" },
  { type: "route_decided", mode: "deterministic", route_name: "forecast", note: "ML/quant forecast" },
  { type: "tool_start", tool: "run_forecast", input_summary: "NVDA · 20d", hue_key: 12345 },
  { type: "tool_finish", tool: "run_forecast", ok: true, elapsed_ms: 42.3 },
  {
    type: "tiles",
    tiles: [
      { label: "P(up)", value: "58%", sub: "20d · gbm", tone: "indigo" },
      { label: "Exp. return", value: "+2.3%", sub: "20d · gbm", tone: "violet" },
    ],
  },
  { type: "chart", spec: FORECAST_CHART },
  { type: "token", text: "Over the next 20 trading days the model leans mildly up." },
  { type: "final", turn_id: "tn", tool_calls: ["run_forecast"], iterations: 1, grounded: true },
];

/** An LLM turn that trips the grounding guard: tokens stream, then `error` (no `final`, §5). */
export const GROUNDING_ERROR_TURN: AgentEvent[] = [
  { type: "turn_start", thread_id: "th", turn_id: "tn2", route: "auto", ticker: null },
  { type: "route_decided", mode: "auto", route_name: "auto", note: "" },
  { type: "tool_start", tool: "get_price_summary", input_summary: "NVDA", hue_key: 999 },
  { type: "tool_finish", tool: "get_price_summary", ok: true, elapsed_ms: 10.0 },
  { type: "token", text: "There is a 92.5% chance it moons." },
  { type: "error", code: "grounding", message: "ungrounded figure(s): 92.5%" },
];

/** Serialize a fixture to an SSE body (`data: <json>\n\n` per frame) — twin of api.streaming.sse_frame. */
export function toSSEBody(events: AgentEvent[]): string {
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
}
