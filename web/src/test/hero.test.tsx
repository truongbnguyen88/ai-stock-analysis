import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { initialTypeState, stepType } from "@/components/Hero";
import type { ConfigResponse, CorpusResponse } from "@/lib/api";
import { fetchConfig, fetchCorpus } from "@/lib/api";
import { useConversation } from "@/store/conversation";
import { FORECAST_TURN, toSSEBody } from "./fixtures";
import type { AgentEvent } from "@/lib/events";

// FORECAST_TURN carries a chart → VegaEmbed; jsdom can't rasterize, so stub it (see conversation.test).
vi.mock("react-vega", () => ({
  VegaEmbed: ({ spec }: { spec: { title?: string } }) => (
    <div data-testid="vega" data-title={spec?.title} />
  ),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, fetchConfig: vi.fn(), fetchCorpus: vi.fn() };
});

const CONFIG: ConfigResponse = {
  default_ticker: "NVDA",
  auto_mode: "🤖 Auto (LLM router)",
  domains: ["predictions", "news", "filings"],
  keys: [{ label: "Anthropic", present: true, required: true }],
};
const CORPUS: CorpusResponse = {
  provider: "voyage",
  embedder: "voyage-voyage-4",
  collection: "c",
  chunks: 96576,
  filings: 4476,
  tickers: 108,
  earliest: "2022-08-31",
  latest: "2026-06-11",
  one_line: "voyage-voyage-4 · 96,576 chunks · fresh to 2026-06-11",
};

/** Stub global fetch to stream `events` as an SSE response (the /chat/stream body). */
function stubStream(events: AgentEvent[]): void {
  const enc = new TextEncoder();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      body: new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(enc.encode(toSSEBody(events)));
          controller.close();
        },
      }),
    }),
  );
}

beforeEach(() => {
  vi.mocked(fetchConfig).mockResolvedValue(CONFIG);
  vi.mocked(fetchCorpus).mockResolvedValue(CORPUS);
  useConversation.getState().reset();
});
afterEach(() => vi.restoreAllMocks());

describe("empty-state hero", () => {
  it("shows the hero + capability cards on an empty conversation, with NEW CHAT active", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    // Eyebrow tagline + two collision-free capability titles (sidebar starters use other labels).
    expect(screen.getByText("SEC-grounded")).toBeInTheDocument();
    expect(screen.getByText("Probabilistic forecasts")).toBeInTheDocument();
    expect(screen.getByText("Chance of a big move")).toBeInTheDocument();
    // The hero IS the "new chat" view → NEW CHAT is the pressed segment.
    expect(screen.getByRole("button", { name: "NEW CHAT" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "CONVERSATION" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("a capability card dispatches its canned prompt and switches to the conversation", async () => {
    stubStream(FORECAST_TURN);
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    // "Chance of a big move" has no sidebar-starter namesake → unambiguous button match.
    await userEvent.click(screen.getByRole("button", { name: /Chance of a big move/ }));

    // The card's prompt becomes the user bubble; the hero gives way to the folded turn.
    await waitFor(() =>
      expect(screen.getByText(/chance of a big move in NVDA/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText("SEC-grounded")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "CONVERSATION" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("NEW CHAT returns to the hero and hides the prior turn", async () => {
    stubStream(FORECAST_TURN);
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    await userEvent.type(screen.getByLabelText("Message"), "forecast NVDA");
    await userEvent.click(screen.getByRole("button", { name: "send" }));
    await waitFor(() =>
      expect(screen.getByTestId("assistant-turn")).toHaveAttribute("data-status", "final"),
    );

    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));

    // Back on the hero; the conversation view (and its turn) is unmounted.
    expect(screen.getByText("SEC-grounded")).toBeInTheDocument();
    expect(screen.queryByTestId("assistant-turn")).not.toBeInTheDocument();
  });
});

describe("typewriter state machine (pure)", () => {
  it("advances past the first phrase instead of sticking (mockup off-by-one regression)", () => {
    const P = ["ab", "cd"];
    let st = initialTypeState(P);
    const seen = new Set<number>([st.pi]);
    // Bounded loop: a correct stepper deletes "ab" then advances to phrase 1 within a handful of steps.
    for (let i = 0; i < 50 && !seen.has(1); i++) {
      st = stepType(st, P).state;
      seen.add(st.pi);
    }
    expect(seen.has(1)).toBe(true);
  });

  it("first step erases one char from the initially-full phrase", () => {
    const P = ["abc"];
    const r = stepType(initialTypeState(P), P);
    expect(r.text).toBe("ab"); // deleting phase: "abc" → "ab"
    expect(r.state.del).toBe(true);
  });

  it("never emits a negative-length or over-length slice across a full cycle", () => {
    const P = ["probabilistic forecasts", "the SEC filings"];
    let st = initialTypeState(P);
    for (let i = 0; i < 300; i++) {
      const r = stepType(st, P);
      expect(r.text.length).toBeGreaterThanOrEqual(0);
      expect(r.text.length).toBeLessThanOrEqual(P[r.state.pi].length);
      expect(P[r.state.pi].startsWith(r.text)).toBe(true); // always a real prefix of the active phrase
      st = r.state;
    }
  });
});
