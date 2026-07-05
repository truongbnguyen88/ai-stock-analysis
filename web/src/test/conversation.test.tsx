import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import type { ConfigResponse, CorpusResponse } from "@/lib/api";
import { fetchConfig, fetchCorpus } from "@/lib/api";
import { useConversation } from "@/store/conversation";
import { FORECAST_TURN, GROUNDING_ERROR_TURN, toSSEBody } from "./fixtures";
import type { AgentEvent } from "@/lib/events";

// Vega can't render in jsdom — stub the chart to a marker carrying the spec title (the translation
// itself is covered by chartSpec.test.ts).
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

describe("conversation flow", () => {
  it("renders a streamed forecast turn: user bubble, tiles, prose, figures(chart), trace", async () => {
    stubStream(FORECAST_TURN);
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    await userEvent.type(screen.getByLabelText("Message"), "forecast NVDA");
    await userEvent.click(screen.getByRole("button", { name: "send" }));

    // The folded turn renders in order: tiles, prose, figures(chart), resolved trace.
    await waitFor(() => expect(screen.getByText("mildly up", { exact: false })).toBeInTheDocument());
    expect(screen.getByText("forecast NVDA")).toBeInTheDocument(); // user bubble
    expect(screen.getByText("P(up)")).toBeInTheDocument();
    expect(screen.getByText("58%")).toBeInTheDocument();
    const chart = screen.getByTestId("vega");
    expect(chart).toHaveAttribute("data-title", "Scenario probabilities");
    // Charts now render as a "Figures" group AFTER the prose (not stacked above the first section):
    // the prose element must precede the chart in document order, under a "Figures" header.
    const prose = screen.getByText("mildly up", { exact: false });
    expect(prose.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText("Figures")).toBeInTheDocument();
    // Trace chip resolved (✓) for the forecast tool.
    expect(screen.getByText("run_forecast")).toBeInTheDocument();
    const turn = screen.getByTestId("assistant-turn");
    expect(turn).toHaveAttribute("data-status", "final");
  });

  it("discards provisional prose on an error frame and shows the error surface", async () => {
    stubStream(GROUNDING_ERROR_TURN);
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    await userEvent.type(screen.getByLabelText("Message"), "will it moon");
    await userEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() =>
      expect(screen.getByTestId("assistant-turn")).toHaveAttribute("data-status", "error"),
    );
    expect(screen.getByText(/ungrounded figure/)).toBeInTheDocument();
    // The provisional token prose ("…chance it moons") was never persisted (§5) — only the
    // error surface remains (its message may legitimately quote the offending figure).
    expect(screen.queryByText(/chance it moons/)).not.toBeInTheDocument();
  });
});
