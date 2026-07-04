import type { Theme } from "@/lib/useTheme";
import type { ConfigResponse } from "@/lib/api";

/** Which main view is showing: the folded conversation, or the empty-state hero ("new chat"). */
export type View = "chat" | "hero";

interface TopBarProps {
  config: ConfigResponse | null;
  ticker: string;
  mode: string;
  theme: Theme;
  onToggleTheme: () => void;
  view: View;
  onView: (v: View) => void;
}

/**
 * Sticky blurred top bar (mockup §6): brass glyph brand, non-advisory disclaimer, active
 * ticker/mode context chips, the CONVERSATION / NEW CHAT segmented view control, and the instant
 * theme toggle. The theme button keeps a dynamic "Switch to … theme" accessible name (the shell
 * test asserts it) while showing the mockup's ◐ glyph.
 */
export function TopBar({ config, ticker, mode, theme, onToggleTheme, view, onView }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="glyph">§</span>
        STOCK&nbsp;RESEARCH <span className="dim">/ AGENT</span>
      </div>
      <div className="disclaimer">RESEARCH &amp; EDUCATION — NOT FINANCIAL ADVICE</div>
      <div className="spacer" />
      {config && (
        <>
          <span className="chip accent">
            <span className="dot" /> {ticker || "no ticker"}
          </span>
          <span className="chip">
            MODE <b>{mode}</b>
          </span>
        </>
      )}
      <div className="seg" role="group" aria-label="View">
        <button type="button" aria-pressed={view === "chat"} onClick={() => onView("chat")}>
          CONVERSATION
        </button>
        <button type="button" aria-pressed={view === "hero"} onClick={() => onView("hero")}>
          NEW CHAT
        </button>
      </div>
      <button
        type="button"
        className="iconbtn"
        onClick={onToggleTheme}
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        title="Toggle light / dark"
      >
        ◐
      </button>
    </header>
  );
}
