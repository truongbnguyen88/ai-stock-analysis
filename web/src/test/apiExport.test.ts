import { afterEach, describe, expect, it, vi } from "vitest";
import { exportSummary } from "@/lib/api";

afterEach(() => vi.restoreAllMocks());

describe("exportSummary", () => {
  it("POSTs the fmt/text/title/charts and downloads the returned blob", async () => {
    const blob = new Blob(["x"], { type: "text/markdown" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      blob: () => Promise.resolve(blob),
      headers: new Headers({ "content-disposition": 'attachment; filename="nvda.md"' }),
    });
    vi.stubGlobal("fetch", fetchMock);
    // triggerDownload clicks a transient anchor — spy so no real navigation happens.
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    await exportSummary("md", "hello", { title: "NVDA", charts: [{ kind: "bar" }] });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/export"),
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse((fetchMock.mock.calls[0][1] as { body: string }).body);
    expect(body).toEqual({ fmt: "md", text: "hello", title: "NVDA", charts: [{ kind: "bar" }] });
    expect(clickSpy).toHaveBeenCalledTimes(1);
  });

  it("defaults title/charts when omitted", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["x"])),
      headers: new Headers(),
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    await exportSummary("pdf", "hi");

    const body = JSON.parse((fetchMock.mock.calls[0][1] as { body: string }).body);
    expect(body).toEqual({ fmt: "pdf", text: "hi", title: "Stock Research Summary", charts: [] });
  });

  it("throws on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Server Error" }),
    );
    await expect(exportSummary("pdf", "hi")).rejects.toThrow(/500/);
  });
});
