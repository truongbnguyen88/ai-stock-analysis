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
 * Sticky top bar: brand mark, persistent non-advisory chip, active ticker/mode context chips,
 * and the instant theme toggle. Mirrors the mockup's blurred header (PHASE2 §6); the full
 * segmented view control lands with the conversation slice.
 */
export function TopBar({ config, ticker, mode, theme, onToggleTheme }: TopBarProps) {
  return (
    <header className="sticky top-0 z-10 flex items-center gap-4 border-b border-border bg-surface/80 px-5 py-3 backdrop-blur">
      <div className="flex items-center gap-2">
        <span className="text-lg leading-none">◆</span>
        <span className="font-mono text-sm font-semibold tracking-tight text-text">stock-agent</span>
      </div>

      <span className="rounded-full border border-border-strong px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-faint">
        research only · not advice
      </span>

      <div className="ml-auto flex items-center gap-2">
        {config && (
          <>
            <ContextChip label={ticker || "no ticker"} tone="accent" />
            <ContextChip label={mode} tone="muted" />
          </>
        )}
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          className="rounded-sa-sm border border-border-strong px-2 py-1 font-mono text-xs text-muted transition-colors hover:border-accent-line hover:text-accent"
        >
          {theme === "dark" ? "◐ dark" : "◑ light"}
        </button>
      </div>
    </header>
  );
}

function ContextChip({ label, tone }: { label: string; tone: "accent" | "muted" }) {
  const toneClass = tone === "accent" ? "border-accent-line text-accent" : "border-border-strong text-muted";
  return (
    <span className={`rounded-full border px-2 py-1 font-mono text-[11px] ${toneClass}`}>{label}</span>
  );
}
