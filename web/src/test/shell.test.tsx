import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import type { ConfigResponse, CorpusResponse } from "@/lib/api";
import { fetchConfig, fetchCorpus } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, fetchConfig: vi.fn(), fetchCorpus: vi.fn() };
});

const CONFIG: ConfigResponse = {
  default_ticker: "NVDA",
  auto_mode: "🤖 Auto (LLM router)",
  domains: ["predictions", "news", "filings"],
  keys: [
    { label: "Anthropic", present: true, required: true },
    { label: "Marketaux", present: false, required: false },
  ],
};

const CORPUS: CorpusResponse = {
  provider: "voyage",
  embedder: "voyage-voyage-4",
  collection: "filings-voyage-voyage-4",
  chunks: 96576,
  filings: 4476,
  tickers: 108,
  earliest: "2022-08-31",
  latest: "2026-06-11",
  one_line: "voyage-voyage-4 · 96,576 chunks · fresh to 2026-06-11",
};

beforeEach(() => {
  vi.mocked(fetchConfig).mockResolvedValue(CONFIG);
  vi.mocked(fetchCorpus).mockResolvedValue(CORPUS);
});

describe("App shell", () => {
  it("renders sidebar + top bar from live /config and /corpus", async () => {
    render(<App />);

    // From /config: default ticker seeded into the field, routing options populated.
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));
    expect(screen.getByRole("option", { name: "predictions" })).toBeInTheDocument();
    // Key chips reflect present/required state.
    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText("Marketaux")).toBeInTheDocument();
    // From /corpus: the status card shows the embedder + coverage.
    expect(screen.getByText("voyage-voyage-4")).toBeInTheDocument();
    expect(screen.getByText(/96,576 chunks/)).toBeInTheDocument();
  });

  it("theme toggle flips data-theme instantly (the Streamlit ceiling item)", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("Ticker symbol")).toHaveValue("NVDA"));

    expect(document.documentElement.dataset.theme ?? "dark").toBe("dark");
    await userEvent.click(screen.getByRole("button", { name: /switch to light theme/i }));
    expect(document.documentElement.dataset.theme).toBe("light");
    await userEvent.click(screen.getByRole("button", { name: /switch to dark theme/i }));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("surfaces an API error when /config fails", async () => {
    vi.mocked(fetchConfig).mockRejectedValueOnce(new Error("Failed to fetch"));
    render(<App />);
    await waitFor(() => expect(screen.getByText(/API unavailable/)).toBeInTheDocument());
  });
});
