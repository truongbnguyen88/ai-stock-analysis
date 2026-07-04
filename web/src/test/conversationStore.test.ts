import { beforeEach, describe, expect, it } from "vitest";
import { useConversation, type Turn } from "@/store/conversation";

/** A minimal committed (final) turn with a given id/question for state-transition tests. */
function finalTurn(id: string, question = "q"): Turn {
  return {
    id,
    question,
    route: null,
    ticker: null,
    routeMode: null,
    routeName: null,
    trace: [],
    tiles: [],
    charts: [],
    answer: "a",
    sources: [],
    status: "final",
  };
}

beforeEach(() => useConversation.getState().reset());

describe("conversation store — thread state (P2.5b)", () => {
  it("loadTurns replaces the transcript and marks the active thread (display-level resume)", () => {
    useConversation.getState().loadTurns([finalTurn("t1"), finalTurn("t2")], "TH9");
    const s = useConversation.getState();
    expect(s.turns).toHaveLength(2);
    expect(s.activeThreadId).toBe("TH9");
    expect(s.streaming).toBe(false);
  });

  it("reset clears the transcript AND the active thread id (a genuinely new chat)", () => {
    useConversation.getState().loadTurns([finalTurn("t1")], "TH9");
    useConversation.getState().reset();
    const s = useConversation.getState();
    expect(s.turns).toEqual([]);
    expect(s.activeThreadId).toBeNull();
  });

  it("starts unsaved (no active thread) before any turn is persisted", () => {
    expect(useConversation.getState().activeThreadId).toBeNull();
  });
});
