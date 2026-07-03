// @vitest-environment node
import { describe, expect, it } from "vitest";
import { applyEvent, newTurn, type Turn } from "@/store/conversation";
import type { AgentEvent } from "@/lib/events";
import { FORECAST_TURN, GROUNDING_ERROR_TURN } from "./fixtures";

/** Fold a whole event list into a fresh turn (the store does this incrementally). */
function fold(events: AgentEvent[]): Turn {
  return events.reduce(applyEvent, newTurn("t", "forecast NVDA", "forecast", "NVDA"));
}

describe("applyEvent reducer", () => {
  it("folds a forecast turn into tiles + chart + prose + a resolved trace", () => {
    const turn = fold(FORECAST_TURN);

    expect(turn.status).toBe("final");
    expect(turn.grounded).toBe(true);
    expect(turn.routeMode).toBe("deterministic");
    expect(turn.tiles.map((t) => t.label)).toEqual(["P(up)", "Exp. return"]);
    expect(turn.charts).toHaveLength(1);
    expect(turn.charts[0].title).toBe("Scenario probabilities");
    expect(turn.answer).toContain("mildly up");
    expect(turn.sources).toHaveLength(0);
  });

  it("fills the trace in order: a chip is running on tool_start, resolved on tool_finish", () => {
    let turn = newTurn("t", "q", "auto", "NVDA");
    turn = applyEvent(turn, { type: "tool_start", tool: "a", input_summary: "x", hue_key: 1 });
    expect(turn.trace).toHaveLength(1);
    expect(turn.trace[0].status).toBe("running");

    turn = applyEvent(turn, { type: "tool_start", tool: "b", input_summary: "y", hue_key: 2 });
    turn = applyEvent(turn, { type: "tool_finish", tool: "a", ok: true, elapsed_ms: 5 });
    // `a` resolved; `b` still running; ORDER preserved (a before b).
    expect(turn.trace.map((c) => [c.tool, c.status])).toEqual([
      ["a", "done"],
      ["b", "running"],
    ]);
    expect(turn.trace[0].ok).toBe(true);
    expect(turn.trace[0].elapsedMs).toBe(5);
  });

  it("resolves the LAST running chip when the same tool is used twice", () => {
    let turn = newTurn("t", "q", "auto", "NVDA");
    turn = applyEvent(turn, { type: "tool_start", tool: "a", input_summary: "1", hue_key: 1 });
    turn = applyEvent(turn, { type: "tool_start", tool: "a", input_summary: "2", hue_key: 1 });
    turn = applyEvent(turn, { type: "tool_finish", tool: "a", ok: true, elapsed_ms: 5 });
    expect(turn.trace[0].status).toBe("running"); // first still running
    expect(turn.trace[1].status).toBe("done"); // second (last) resolved
  });

  it("discards provisional tokens on `error` and surfaces the error (never persists prose)", () => {
    const turn = fold(GROUNDING_ERROR_TURN);
    expect(turn.status).toBe("error");
    expect(turn.answer).toBe(""); // provisional "92.5% chance" prose discarded (§5)
    expect(turn.error).toEqual({ code: "grounding", message: "ungrounded figure(s): 92.5%" });
  });
});
