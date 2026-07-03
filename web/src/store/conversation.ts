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
  /** Run one turn: append it, open the SSE stream, and fold events as they arrive. */
  send: (args: { message: string; route: string | null; ticker: string | null }) => Promise<void>;
  reset: () => void;
}

let _seq = 0;
const nextId = (): string => `t${++_seq}`;

export const useConversation = create<ConversationState>((set) => ({
  turns: [],
  streaming: false,
  reset: () => set({ turns: [], streaming: false }),
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
      await streamChat(req, { onEvent: fold });
    } catch (e) {
      fold({ type: "error", code: "network", message: e instanceof Error ? e.message : String(e) });
    } finally {
      set({ streaming: false });
    }
  },
}));
