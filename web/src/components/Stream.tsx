import type { Turn } from "@/store/conversation";
import { AssistantTurn } from "@/components/AssistantTurn";

/**
 * The conversation surface: each turn is the user's question bubble followed by the streamed
 * assistant response. Empty state is a lightweight hint here; the full typewriter Hero + capability
 * cards land in P2.6.
 */
export function Stream({ turns }: { turns: Turn[] }) {
  if (turns.length === 0) {
    return (
      <p className="max-w-prose text-sm text-muted">
        Ask about a ticker — forecasts, technicals, news, or SEC filings. Numbers come from the tools;
        the assistant summarizes and cites, it never advises.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-8">
      {turns.map((turn) => (
        <div key={turn.id} className="flex flex-col gap-3">
          <UserBubble text={turn.question} />
          <AssistantTurn turn={turn} />
        </div>
      ))}
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="self-end max-w-[80%] rounded-sa border border-accent-line bg-accent-weak px-3 py-2 text-sm text-text">
      {text}
    </div>
  );
}
