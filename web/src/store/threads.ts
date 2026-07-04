// Saved-thread list store (Phase 2, P2.5b).
//
// A tiny Zustand store backing the sidebar's chat list. It mirrors the server's /threads index
// (display-level persistence — see store/conversation.ts for how a turn transcript is saved on
// finalize). All network access is BEST-EFFORT: a failed refresh/remove keeps the last-known list
// and never throws into React, so the chat keeps working even when the API is down.

import { create } from "zustand";
import { deleteThread, fetchThreads, type ThreadMeta } from "@/lib/api";

interface ThreadsState {
  items: ThreadMeta[];
  /** Reload the list from GET /threads (server returns it most-recent first). Swallows errors. */
  refresh: () => Promise<void>;
  /** Delete a thread then reload the list. Swallows errors (refresh reconciles with server truth). */
  remove: (id: string) => Promise<void>;
}

export const useThreads = create<ThreadsState>((set, get) => ({
  items: [],
  refresh: async () => {
    try {
      set({ items: await fetchThreads() });
    } catch {
      /* best-effort: keep the last-known list if the API is unreachable */
    }
  },
  remove: async (id) => {
    try {
      await deleteThread(id);
    } catch {
      /* ignore — the refresh below reconciles the list with server truth */
    }
    await get().refresh();
  },
}));
