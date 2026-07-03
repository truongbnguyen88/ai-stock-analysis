// AgentEvent — the SSE wire schema the React client consumes (Phase 2, P2.3).
//
// One-to-one mirror of the Python `to_wire()` dicts in src/stock_agent/agent/events.py: the
// discriminant is the `type` string, so a frame parsed off the SSE stream narrows to the right
// variant. Keeping this in lock-step with the Python union is the frontend half of the numbers-
// from-tools contract — the client only renders what the server sends, it computes nothing.
//
// Field shapes follow the *actual builders*, not just plan §4: `sources` citations carry
// {marker, label} (ui.state.sources_from_tool_results emits no url yet); `tiles` are display-ready
// {label,value,sub,tone} bags (ui.tiles); `chart.spec` is the render-agnostic ChartSpec dict
// (viz.charts.ChartSpec.to_dict — translated to Vega-Lite client-side, see lib/chartSpec.ts).

import type { ChartSpecDict } from "@/lib/chartSpec";

/** A headline stat tile (ui.tiles.stat_tiles_from_tool_results). Values are pre-formatted. */
export interface Tile {
  label: string;
  value: string;
  sub: string;
  /** Semantic hue token name: teal|sky|indigo|violet|rose|accent (never up/down). */
  tone: string;
}

/** A SEC filing citation (ui.state.sources_from_tool_results). */
export interface Citation {
  marker: string | number | null;
  label: string;
}

export interface TurnStartEvent {
  type: "turn_start";
  thread_id: string;
  turn_id: string;
  route: string;
  ticker: string | null;
}

export interface RouteDecidedEvent {
  type: "route_decided";
  /** "deterministic" | "auto". */
  mode: string;
  route_name: string;
  note: string;
}

export interface ToolStartEvent {
  type: "tool_start";
  tool: string;
  input_summary: string;
  /** crc32(tool) — map to a palette slot via `hue_key % N` (see lib/hues). */
  hue_key: number;
}

export interface ToolFinishEvent {
  type: "tool_finish";
  tool: string;
  ok: boolean;
  elapsed_ms: number;
}

export interface TilesEvent {
  type: "tiles";
  tiles: Tile[];
}

export interface ChartEvent {
  type: "chart";
  spec: ChartSpecDict;
}

export interface TokenEvent {
  type: "token";
  text: string;
}

export interface SourcesEvent {
  type: "sources";
  citations: Citation[];
}

export interface FinalEvent {
  type: "final";
  turn_id: string | null;
  tool_calls: string[];
  iterations: number;
  grounded: boolean;
}

export interface ErrorEvent {
  type: "error";
  /** stable machine tag: grounding | max_iterations | agent | router. */
  code: string;
  message: string;
}

/** Discriminated union (the `type` field is the discriminant), ordered per the §5 stream contract. */
export type AgentEvent =
  | TurnStartEvent
  | RouteDecidedEvent
  | ToolStartEvent
  | ToolFinishEvent
  | TilesEvent
  | ChartEvent
  | TokenEvent
  | SourcesEvent
  | FinalEvent
  | ErrorEvent;

// The 5-hue secondary cycle, in the SAME order as ui.html._CAP_HUES so `hue_key % 5` reproduces
// ui.html.tool_hue exactly (the tool gets the identical trace-chip color as in Streamlit).
export const TRACE_HUES = ["teal", "sky", "indigo", "violet", "rose"] as const;

/** Map a `tool_start.hue_key` (raw crc32) to a hue token name (mirrors ui.html.tool_hue). */
export function hueName(hueKey: number): (typeof TRACE_HUES)[number] {
  // crc32 is unsigned in Python; JS bitwise ops are signed, but the wire int is already >= 0.
  return TRACE_HUES[hueKey % TRACE_HUES.length];
}
