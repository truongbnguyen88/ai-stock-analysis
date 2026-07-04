import type { Turn } from "@/store/conversation";
import { AssistantTurn } from "@/components/AssistantTurn";

/**
 * The conversation surface: each turn is the user's question (its own .turn bubble) followed by the
 * streamed assistant .turn (avatar + folded response). Empty state is a lightweight hint here; the
 * full typewriter Hero + capability cards land in P2.6.
 */
export function Stream({ turns }: { turns: Turn[] }) {
  if (turns.length === 0) {
    return (
      <p style={{ color: "var(--sa-muted)", fontSize: 14, maxWidth: "62ch", padding: "6px 2px" }}>
        Ask about a ticker — forecasts, technicals, news, or SEC filings. Numbers come from the
        tools; the assistant summarizes and cites, it never advises.
      </p>
    );
  }
  return (
    <>
      {turns.map((turn) => (
        <div key={turn.id}>
          <div className="turn">
            <div className="who">
              <span className="avatar user">YOU</span>
              <span className="name">You</span>
            </div>
            <div className="msg">
              <div className="user-msg">{turn.question}</div>
            </div>
          </div>
          <div className="turn">
            <div className="who">
              <span className="avatar bot">§</span>
              <span className="name">Research Agent</span>
            </div>
            <div className="msg">
              <AssistantTurn turn={turn} />
            </div>
          </div>
        </div>
      ))}
    </>
  );
}
