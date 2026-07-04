import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ThreadMeta } from "@/lib/api";
import { deleteThread, fetchThreads } from "@/lib/api";
import { useThreads } from "@/store/threads";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, fetchThreads: vi.fn(), deleteThread: vi.fn() };
});

const META: ThreadMeta[] = [{ id: "a", title: "A", created_at: "t", updated_at: "t" }];

beforeEach(() => useThreads.setState({ items: [] }));
afterEach(() => vi.restoreAllMocks());

describe("threads store", () => {
  it("refresh populates items from GET /threads", async () => {
    vi.mocked(fetchThreads).mockResolvedValue(META);
    await useThreads.getState().refresh();
    expect(useThreads.getState().items).toEqual(META);
  });

  it("refresh swallows errors and keeps the last-known list (best-effort)", async () => {
    useThreads.setState({ items: META });
    vi.mocked(fetchThreads).mockRejectedValue(new Error("down"));
    await useThreads.getState().refresh(); // must not throw
    expect(useThreads.getState().items).toEqual(META); // unchanged
  });

  it("remove deletes then refreshes the list", async () => {
    vi.mocked(deleteThread).mockResolvedValue(undefined);
    vi.mocked(fetchThreads).mockResolvedValue([]); // list after deletion
    await useThreads.getState().remove("a");
    expect(deleteThread).toHaveBeenCalledWith("a");
    expect(useThreads.getState().items).toEqual([]);
  });

  it("remove still refreshes even if the delete call fails", async () => {
    vi.mocked(deleteThread).mockRejectedValue(new Error("boom"));
    vi.mocked(fetchThreads).mockResolvedValue(META);
    await useThreads.getState().remove("a"); // must not throw
    expect(fetchThreads).toHaveBeenCalled();
  });
});
