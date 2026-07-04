import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExportMenu } from "@/components/ExportMenu";
import { exportSummary } from "@/lib/api";

// Mock the API module so the component test never hits fetch; the real helper is covered separately.
vi.mock("@/lib/api", () => ({ exportSummary: vi.fn() }));

afterEach(() => vi.clearAllMocks());

describe("ExportMenu", () => {
  it("opens the popover and exports the picked format with text + charts", async () => {
    vi.mocked(exportSummary).mockResolvedValue(undefined);
    const charts = [{ title: "c", kind: "bar" }];
    render(<ExportMenu text="the answer" charts={charts} title="NVDA — research summary" />);

    // Formats are hidden until the popover opens.
    expect(screen.queryByRole("button", { name: "Markdown" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Export" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Markdown" })).toBeInTheDocument(),
    );

    await userEvent.click(screen.getByRole("button", { name: "Markdown" }));
    expect(exportSummary).toHaveBeenCalledWith("md", "the answer", {
      charts,
      title: "NVDA — research summary",
    });
  });

  it("surfaces a failure inline instead of throwing", async () => {
    vi.mocked(exportSummary).mockRejectedValue(new Error("boom"));
    render(<ExportMenu text="the answer" />);

    await userEvent.click(screen.getByRole("button", { name: "Export" }));
    await userEvent.click(await screen.findByRole("button", { name: "PDF" }));

    expect(await screen.findByText(/Export failed/)).toBeInTheDocument();
  });
});
