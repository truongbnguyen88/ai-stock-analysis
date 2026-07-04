import { afterEach, describe, expect, it, vi } from "vitest";
import { deleteThread, fetchThread, fetchThreads, saveThread } from "@/lib/api";

afterEach(() => vi.restoreAllMocks());

/** A fetch mock returning one JSON value with an ok status. */
function stubJSON(value: unknown): ReturnType<typeof vi.fn> {
  const fn = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: "OK",
    json: () => Promise.resolve(value),
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("threads API client", () => {
  it("fetchThreads GETs /threads and returns the list", async () => {
    const items = [{ id: "a", title: "A", created_at: "t", updated_at: "t" }];
    const fetchMock = stubJSON(items);
    const got = await fetchThreads();
    expect(got).toEqual(items);
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/threads"));
  });

  it("fetchThread GETs /threads/{id} url-encoded", async () => {
    const fetchMock = stubJSON({ id: "x y", title: "T", created_at: "t", updated_at: "t", display_messages: [], agent_history: [] });
    await fetchThread("x y");
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/threads/x%20y"));
  });

  it("saveThread POSTs the full body (defaults filled) and returns the saved thread", async () => {
    const saved = { id: "TH1", title: "hi", created_at: "t", updated_at: "t", display_messages: [], agent_history: [] };
    const fetchMock = stubJSON(saved);
    const got = await saveThread({ title: "hi", display_messages: [{ id: "t1" }] });
    expect(got).toEqual(saved);
    const [url, opts] = fetchMock.mock.calls[0] as [string, { method: string; body: string }];
    expect(url).toContain("/threads");
    expect(opts.method).toBe("POST");
    expect(JSON.parse(opts.body)).toEqual({
      id: "",
      title: "hi",
      display_messages: [{ id: "t1" }],
      agent_history: [],
    });
  });

  it("deleteThread DELETEs /threads/{id}", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204, statusText: "No Content" });
    vi.stubGlobal("fetch", fetchMock);
    await deleteThread("TH1");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/threads/TH1"),
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("throws on a non-2xx save", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Err" }));
    await expect(saveThread({ title: "x" })).rejects.toThrow(/500/);
  });
});
