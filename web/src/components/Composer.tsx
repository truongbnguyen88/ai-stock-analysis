import { useState, type FormEvent, type KeyboardEvent } from "react";

interface ComposerProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  /** Context chips (e.g. ticker · route) shown above the input — what pressing enter will do. */
  contextChips?: string[];
}

/**
 * Message composer (mockup input bar): "On enter:" context chips + a growing input + brass send.
 * Enter sends, Shift+Enter newlines; disabled while a turn streams (one in-flight turn, P2.3). The
 * send button keeps the accessible name "send" (the conversation test asserts it) behind the ↑ glyph.
 */
export function Composer({ onSend, disabled, contextChips = [] }: ComposerProps) {
  const [value, setValue] = useState("");

  const submit = (): void => {
    const msg = value.trim();
    if (!msg || disabled) return;
    onSend(msg);
    setValue("");
  };

  const onSubmit = (e: FormEvent): void => {
    e.preventDefault();
    submit();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={onSubmit}>
      {contextChips.length > 0 && (
        <div className="context-chips">
          <span className="lbl">On enter:</span>
          {contextChips.map((c, i) => (
            <span key={c} className={`chip${i === 0 ? " accent" : ""}`}>
              {i === 0 && <span className="dot" />}
              {c}
            </span>
          ))}
        </div>
      )}
      <div className="composer">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          aria-label="Message"
          placeholder="Ask anything about a stock…"
        />
        <button
          type="submit"
          className="send"
          aria-label="send"
          disabled={disabled || !value.trim()}
        >
          ↑
        </button>
      </div>
    </form>
  );
}
