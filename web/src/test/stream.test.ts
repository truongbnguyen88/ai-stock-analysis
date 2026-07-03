// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseSSEBuffer, streamChat } from "@/lib/stream";
import type { AgentEvent } from "@/lib/events";
import { FORECAST_TURN, toSSEBody } from "./fixtures";

/** A Response-like whose body streams `body` as byte chunks split at `cuts` (test chunk boundaries). */
function streamedResponse(body: string, cuts: number[] = []): Response {
  const enc = new TextEncoder();
  const bounds = [0, ...cuts, body.length];
  const chunks = bounds.slice(0, -1).map((start, i) => enc.encode(body.slice(start, bounds[i + 1])));
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
      controller.close();
    },
  });
  return { ok: true, status: 200, statusText: "OK", body: stream } as unknown as Response;
}

afterEach(() => vi.restoreAllMocks());

describe("parseSSEBuffer", () => {
  it("emits complete frames and returns the partial trailing frame as remainder", () => {
    const events: AgentEvent[] = [];
    const rest = parseSSEBuffer(
      'data: {"type":"token","text":"hi"}\n\ndata: {"type":"toke',
      (e) => events.push(e),
    );
    expect(events).toEqual([{ type: "token", text: "hi" }]);
    expect(rest).toBe('data: {"type":"toke'); // incomplete frame held back
  });
});

describe("streamChat", () => {
  it("parses a full turn even when frames are split across byte chunks", async () => {
    const body = toSSEBody(FORECAST_TURN);
    // Cut mid-frame at a few awkward offsets to exercise cross-chunk buffering.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(streamedResponse(body, [10, 55, 130, body.length - 5])),
    );

    const events: AgentEvent[] = [];
    await streamChat({ message: "forecast", route: "forecast", ticker: "NVDA" }, {
      onEvent: (e) => events.push(e),
    });

    expect(events.map((e) => e.type)).toEqual([
      "turn_start",
      "route_decided",
      "tool_start",
      "tool_finish",
      "tiles",
      "chart",
      "token",
      "final",
    ]);
    expect(events).toEqual(FORECAST_TURN); // byte-for-byte round-trip through the SSE wire
  });

  it("POSTs the request body to /chat/stream with an SSE Accept header", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamedResponse(toSSEBody(FORECAST_TURN)));
    vi.stubGlobal("fetch", fetchMock);

    await streamChat({ message: "forecast", route: "forecast", ticker: "NVDA" }, {
      onEvent: () => {},
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/chat\/stream$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject({ message: "forecast", route: "forecast" });
    expect(init.headers.Accept).toBe("text/event-stream");
  });

  it("throws on a non-OK response (caller turns it into an error turn)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "err", body: null }),
    );
    await expect(
      streamChat({ message: "x", route: "auto", ticker: null }, { onEvent: () => {} }),
    ).rejects.toThrow(/500/);
  });
});
