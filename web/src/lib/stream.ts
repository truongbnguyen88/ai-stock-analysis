// SSE client for POST /chat/stream (Phase 2, P2.3).
//
// We POST the query body (so a plain EventSource, which is GET-only, won't do — plan §6) and read
// the response body as a ReadableStream, parsing the `data: <json>\n\n` frames the FastAPI adapter
// emits (api/streaming.sse_frame). Each parsed AgentEvent is handed to `onEvent`; the reducer in the
// store folds them into turn state. Deterministic + framework-free so it is unit-testable by mocking
// `fetch` with a canned ReadableStream (no live backend, repo test rule).

import { API_BASE } from "@/lib/api";
import type { AgentEvent } from "@/lib/events";

/** Request body for POST /chat/stream (mirrors api.schemas.ChatStreamRequest). */
export interface ChatStreamRequest {
  message: string;
  route?: string | null;
  ticker?: string | null;
  horizon?: number | null;
  days?: number | null;
  model?: string | null;
  thread_id?: string;
  turn_id?: string;
  history?: Array<Record<string, unknown>>;
}

export interface StreamHandlers {
  onEvent: (event: AgentEvent) => void;
  /** Optional: called once the stream closes cleanly (server sent its terminal frame). */
  onDone?: () => void;
  signal?: AbortSignal;
}

/**
 * Parse a growing SSE buffer, emitting each COMPLETE `data:` frame's JSON payload.
 *
 * Frames are separated by a blank line (`\n\n`); a partial trailing frame stays in the buffer for
 * the next chunk. Non-`data:` lines (comments/`event:`) are ignored — our adapter only emits `data:`.
 * Returns the unconsumed remainder so the caller can carry it across chunk boundaries.
 */
export function parseSSEBuffer(buffer: string, onEvent: (event: AgentEvent) => void): string {
  let rest = buffer;
  let sep = rest.indexOf("\n\n");
  while (sep !== -1) {
    const block = rest.slice(0, sep);
    rest = rest.slice(sep + 2);
    for (const line of block.split("\n")) {
      const trimmed = line.trimStart();
      if (trimmed.startsWith("data:")) {
        const payload = trimmed.slice("data:".length).trim();
        if (payload) onEvent(JSON.parse(payload) as AgentEvent);
      }
    }
    sep = rest.indexOf("\n\n");
  }
  return rest;
}

/**
 * Run one turn: POST the request and stream its AgentEvents to `onEvent`.
 *
 * Rejects on a non-OK status or a network error (the caller surfaces it as an `error` turn). The
 * server converts in-band failures (grounding rejection, bad route) into terminal `error` *frames*
 * (still HTTP 200), so those arrive via `onEvent`, not as a throw.
 */
export async function streamChat(
  req: ChatStreamRequest,
  { onEvent, onDone, signal }: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`POST /chat/stream failed: ${res.status} ${res.statusText}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSSEBuffer(buffer, onEvent);
  }
  // Flush any final frame that wasn't newline-terminated (defensive; our adapter always terminates
  // frames with `\n\n`, but this guards against a truncated tail so no event is silently dropped).
  buffer += decoder.decode();
  if (buffer.trim()) parseSSEBuffer(buffer + "\n\n", onEvent);
  onDone?.();
}
