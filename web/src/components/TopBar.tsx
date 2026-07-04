import type { Theme } from "@/lib/useTheme";
import type { ConfigResponse } from "@/lib/api";

interface TopBarProps {
  config: ConfigResponse | null;
  ticker: string;
  mode: string;
  theme: Theme;
  onToggleTheme: () => void;
}

/**
 * Sticky blurred top bar (mockup §6): brass glyph brand, non-advisory disclaimer, active
 * ticker/mode context chips, and the instant theme toggle. The segmented view control lands with
 * the empty-state Hero (P2.6). The theme button keeps a dynamic "Switch to … theme" accessible
 * name (the shell test asserts it) while showing the mockup's ◐ glyph.
 */
export function TopBar({ config, ticker, mode, theme, onToggleTheme }: TopBarProps) {
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
