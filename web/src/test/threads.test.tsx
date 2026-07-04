import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import type { ConfigResponse, CorpusResponse, ThreadMeta } from "@/lib/api";
import { deleteThread, fetchConfig, fetchCorpus, fetchThread, fetchThreads, saveThread } from "@/lib/api";
import { useConversation, type Turn } from "@/store/conversation";
import { useThreads } from "@/store/threads";
import { FORECAST_TURN, toSSEBody } from "./fixtures";
import type { AgentEvent } from "@/lib/events";

// Vega can't render in jsdom — stub the chart (FORECAST_TURN carries one).
vi.mock("react-vega", () => ({
  VegaEmbed: ({ spec }: { spec: { title?: string } }) => (
    <div data-testid="vega" data-title={spec?.title} />
  ),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchConfig: vi.fn(),
    fetchCorpus: vi.fn(),
    fetchThreads: vi.fn(),
    fetchThread: vi.fn(),
    saveThread: vi.fn(),
    deleteThread: vi.fn(),
  };
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

/** Stub global fetch to stream `events` as the /chat/stream SSE body. */
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

const SAVED = {
  id: "TH1",
  title: "forecast NVDA",
  created_at: "2026-07-01T00:00:00+00:00",
  updated_at: "2026-07-01T00:00:00+00:00",
  display_messages: [],
  agent_history: [],
};

beforeEach(() => {
  vi.mocked(fetchConfig).mockResolvedValue(CONFIG);
  vi.mocked(fetchCorpus).mockResolvedValue(CORPUS);
  vi.mocked(fetchThreads).mockResolvedValue([]);
  vi.mocked(saveThread).mockResolvedValue(SAVED);
  useConversation.getState().reset();
  useThreads.setState({ items: [] });
});
afterEach(() => vi.restoreAllMocks());

describe("saved chat threads (P2.5b)", () => {
  it("persists the transcript after a turn finalizes (display-level)", async () => {
    stubStream(FORECAST_TURN);
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    await userEvent.type(screen.getByLabelText("Message"), "forecast NVDA");
    await userEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() =>
      expect(screen.getByTestId("assistant-turn")).toHaveAttribute("data-status", "final"),
    );
    await waitFor(() => expect(saveThread).toHaveBeenCalled());
    const body = vi.mocked(saveThread).mock.calls[0][0];
    expect(body.title).toContain("forecast NVDA"); // title derived from the first question
    expect(body.id).toBe(""); // first save mints a new id server-side
    const turns = body.display_messages as Turn[];
    expect(turns[0].question).toBe("forecast NVDA"); // the SPA stores its own Turn[] shape
    expect(turns[0].status).toBe("final");
  });

  it("lists saved chats, reopens one on click, and deletes on trash", async () => {
    const meta: ThreadMeta = {
      id: "TH1",
      title: "forecast NVDA",
      created_at: "2026-07-01T00:00:00+00:00",
      updated_at: "2026-07-01T00:00:00+00:00",
    };
    const loaded: Turn[] = [
      {
        id: "t1",
        question: "forecast NVDA",
        route: "auto",
        ticker: "NVDA",
        routeMode: "deterministic",
        routeName: "forecast",
        trace: [],
        tiles: [],
        charts: [],
        answer: "Reloaded transcript.",
        sources: [],
        status: "final",
      },
    ];
    vi.mocked(fetchThreads).mockResolvedValue([meta]);
    vi.mocked(fetchThread).mockResolvedValue({ ...meta, display_messages: loaded, agent_history: [] });
    vi.mocked(deleteThread).mockResolvedValue(undefined);

    render(<App />);
    // The chat row appears from the mount refresh.
    await waitFor(() => expect(screen.getByRole("button", { name: /Delete forecast NVDA/i })).toBeInTheDocument());

    // Click the row title -> reopens the thread (fetchThread + display-level load).
    await userEvent.click(screen.getByText("forecast NVDA"));
    await waitFor(() => expect(fetchThread).toHaveBeenCalledWith("TH1"));
    expect(await screen.findByText("Reloaded transcript.")).toBeInTheDocument();

    // Trash deletes it (and does NOT re-open — stopPropagation).
    await userEvent.click(screen.getByRole("button", { name: /Delete forecast NVDA/i }));
    expect(deleteThread).toHaveBeenCalledWith("TH1");
  });
});
