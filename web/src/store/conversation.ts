// Conversation store (Phase 2, P2.3).
//
// A Zustand store holding the active thread's turns and the streaming flag, plus the PURE reducer
// `applyEvent(turn, event)` that folds one AgentEvent into a Turn. The reducer is exported and
// framework-free so it is unit-tested directly (trace fills in order; `error` discards provisional
// tokens; `final` commits) without React or the network. `send()` opens the SSE stream and threads
// each event through the reducer, replacing the active turn immutably so React re-renders.
//
// Provisional→final (plan §5): `token`s accumulate into `answer` while `status === "streaming"`; the
// client treats that prose as PROVISIONAL until `final`. An `error` frame (grounding rejection after
// retry, or a tool/LLM failure) clears `answer` — the client never persists unguarded prose. Full
// thread CRUD (multiple threads, /threads) lands in P2.5; here a single in-memory thread suffices.

import { create } from "zustand";
import type { AgentEvent, Citation, Tile } from "@/lib/events";
import type { ChartSpecDict } from "@/lib/chartSpec";
import { streamChat, type ChatStreamRequest } from "@/lib/stream";
import { saveThread } from "@/lib/api";

/** One trace chip: a tool execution, running then done (drives TraceRow's animation). */
export interface TraceItem {
  tool: string;
  inputSummary: string;
  hueKey: number;
  status: "running" | "done";
  ok?: boolean;
  elapsedMs?: number;
}

export type TurnStatus = "streaming" | "final" | "error";

/** A conversation turn: the user's question + the assistant's folded, streamed response. */
export interface Turn {
  id: string;
  question: string;
  route: string | null;
  ticker: string | null;
  routeMode: string | null; // "deterministic" | "auto"
  routeName: string | null;
  trace: TraceItem[];
  tiles: Tile[];
  charts: ChartSpecDict[];
  /** Accumulated token text — PROVISIONAL until status === "final" (§5). */
  answer: string;
  sources: Citation[];
  status: TurnStatus;
  grounded?: boolean;
  error?: { code: string; message: string };
}

/** A fresh assistant turn awaiting the stream (question already known from the composer). */
export function newTurn(id: string, question: string, route: string | null, ticker: string | null): Turn {
  return {
    id,
    question,
    route,
    ticker,
    routeMode: null,
    routeName: null,
    trace: [],
    tiles: [],
    charts: [],
    answer: "",
    sources: [],
    status: "streaming",
  };
}

/**
 * Fold one AgentEvent into a turn, returning a NEW turn (immutable — safe for React state).
 *
 * Ownership mirrors the server: `tool_start` pushes a running chip, `tool_finish` resolves the
 * matching one; `tiles`/`chart`/`sources` are the adapter's pure-builder output rendered verbatim;
 * `token` appends provisional prose; `final` commits (grounded); `error` discards provisional prose.
 */
export function applyEvent(turn: Turn, ev: AgentEvent): Turn {
  switch (ev.type) {
    case "turn_start":
      return { ...turn, route: ev.route, ticker: ev.ticker };
    case "route_decided":
      return { ...turn, routeMode: ev.mode, routeName: ev.route_name };
    case "tool_start":
      return {
        ...turn,
        trace: [
          ...turn.trace,
          { tool: ev.tool, inputSummary: ev.input_summary, hueKey: ev.hue_key, status: "running" },
        ],
      };
    case "tool_finish": {
      // Resolve the LAST still-running chip for this tool (handles the same tool used twice).
      let idx = -1;
      for (let i = turn.trace.length - 1; i >= 0; i--) {
        if (turn.trace[i].tool === ev.tool && turn.trace[i].status === "running") {
          idx = i;
          break;
        }
      }
      if (idx === -1) return turn; // no matching start (shouldn't happen); ignore defensively
      const trace = turn.trace.slice();
      trace[idx] = { ...trace[idx], status: "done", ok: ev.ok, elapsedMs: ev.elapsed_ms };
      return { ...turn, trace };
    }
    case "tiles":
      return { ...turn, tiles: ev.tiles };
    case "chart":
      return { ...turn, charts: [...turn.charts, ev.spec] };
    case "token":
      return { ...turn, answer: turn.answer + ev.text };
    case "sources":
      return { ...turn, sources: ev.citations };
    case "final":
      return { ...turn, status: "final", grounded: ev.grounded };
    case "error":
      // Discard provisional prose — never persist an unguarded answer (§5).
      return { ...turn, status: "error", answer: "", error: { code: ev.code, message: ev.message } };
    default:
      return turn;
  }
}

interface ConversationState {
  turns: Turn[];
  streaming: boolean;
  /** The saved-thread id this conversation persists to; null until the first turn is saved (P2.5b). */
  activeThreadId: string | null;
  /** Run one turn: append it, open the SSE stream, fold events, then persist the transcript. */
  send: (args: { message: string; route: string | null; ticker: string | null }) => Promise<void>;
  /** Replace the conversation with a reopened thread's turns (display-level resume, P2.5b). */
  loadTurns: (turns: Turn[], threadId: string) => void;
  reset: () => void;
}

let _seq = 0;
const nextId = (): string => `t${++_seq}`;

/** Advance the id counter past any loaded turn ids so a subsequent send never reuses one — a reused
 *  id would make `fold` update two turns at once (the reopened one and the new one). */
function bumpSeqPast(turns: Turn[]): void {
  for (const t of turns) {
    const m = /^t(\d+)$/.exec(t.id);
    if (m) _seq = Math.max(_seq, Number(m[1]));
  }
}

/** Thread title from the first non-empty question (ChatGPT-style), trimmed to 60 chars. Mirrors the
 *  server's derive_title fallback; sent explicitly because display_messages carry no role/content. */
function threadTitle(turns: Turn[]): string {
  const q = turns.find((t) => t.question.trim())?.question.trim() ?? "";
  if (!q) return "New chat";
  return q.length <= 60 ? q : q.slice(0, 59) + "…";
}

export const useConversation = create<ConversationState>((set, get) => ({
  turns: [],
  streaming: false,
  activeThreadId: null,
  reset: () => set({ turns: [], streaming: false, activeThreadId: null }),
  loadTurns: (turns, threadId) => {
    bumpSeqPast(turns);
    set({ turns, streaming: false, activeThreadId: threadId });
  },
  send: async ({ message, route, ticker }) => {
    const turnId = nextId();
    const turn = newTurn(turnId, message, route, ticker);
    set((s) => ({ turns: [...s.turns, turn], streaming: true }));

    // Fold each event into THIS turn (found by id — other turns are untouched).
    const fold = (ev: AgentEvent): void =>
      set((s) => ({
        turns: s.turns.map((t) => (t.id === turnId ? applyEvent(t, ev) : t)),
      }));

    const req: ChatStreamRequest = {
      message,
      route,
      ticker,
      turn_id: turnId,
    };
    try {
      try {
        await streamChat(req, { onEvent: fold });
      } catch (e) {
        fold({ type: "error", code: "network", message: e instanceof Error ? e.message : String(e) });
      }
      // Persist the settled transcript (display-level) BEFORE clearing `streaming`, so the sidebar's
      // refresh-on-idle sees the saved thread. Only committed (final, grounded) turns are saved — an
      // errored turn has its provisional prose discarded (§5), so we skip it. Best-effort: a save
      // failure must never surface into the chat (the API may be down).
      const settled = get();
      const last = settled.turns[settled.turns.length - 1];
      if (last && last.status === "final") {
        try {
          const saved = await saveThread({
            id: settled.activeThreadId ?? "",
            title: threadTitle(settled.turns),
            display_messages: settled.turns,
          });
          set({ activeThreadId: saved.id });
        } catch {
          /* best-effort persistence — keep the chat working even if the API is down */
        }
      }
    } finally {
      set({ streaming: false });
    }
  },
}));
