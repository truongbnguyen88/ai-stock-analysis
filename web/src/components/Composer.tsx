import { useState, type FormEvent, type KeyboardEvent } from "react";

interface ComposerProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  /** Context chips (e.g. ticker · route) shown above the input — what pressing enter will do. */
  contextChips?: string[];
}

/**
 * The message composer: context chips + a growing input + send. Enter sends, Shift+Enter newlines.
 * Disabled while a turn streams (one in-flight turn at a time in P2.3). `focus-within` brass accent
 * mirrors the mockup.
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
    <form onSubmit={onSubmit} className="mt-4">
      {contextChips.length > 0 && (
        <div className="mb-2 flex items-center gap-2 text-[11px]">
          <span className="font-mono uppercase tracking-wider text-faint">on enter</span>
          {contextChips.map((c) => (
            <span
              key={c}
              className="rounded-full border border-border-strong px-2 py-0.5 font-mono text-muted"
            >
              {c}
            </span>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2 rounded-sa border border-border-strong bg-surface-2 p-2 focus-within:border-accent-line">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          aria-label="Message"
          placeholder="Ask about a ticker…"
          className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1 text-sm text-text outline-none placeholder:text-faint"
        />
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="rounded-sa-sm border border-accent-line px-3 py-1.5 font-mono text-xs text-accent transition-colors hover:bg-accent-weak disabled:opacity-40"
        >
          send
        </button>
      </div>
    </form>
  );
}
