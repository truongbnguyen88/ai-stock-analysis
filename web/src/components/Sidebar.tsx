import type { ReactNode } from "react";
import type { ConfigResponse, CorpusResponse, KeyStatus } from "@/lib/api";

interface SidebarProps {
  config: ConfigResponse | null;
  corpus: CorpusResponse | null;
  ticker: string;
  onTicker: (t: string) => void;
  mode: string;
  onMode: (m: string) => void;
}

/**
 * Left rail: corpus status card, ticker field, routing select, and API-key chips — all driven
 * by live /config + /corpus (the P2.0 DoD). Quick-starters + chat list land in later slices.
 */
export function Sidebar({ config, corpus, ticker, onTicker, mode, onMode }: SidebarProps) {
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-5 border-r border-border bg-surface px-4 py-5">
      <StatusCard corpus={corpus} />

      <section>
        <Eyebrow>Ticker</Eyebrow>
        <input
          value={ticker}
          onChange={(e) => onTicker(e.target.value.toUpperCase())}
          maxLength={10}
          aria-label="Ticker symbol"
          className="w-full rounded-sa-sm border border-border-strong bg-surface-2 px-3 py-2 font-mono text-sm text-text outline-none focus:border-accent-line"
        />
      </section>

      <section>
        <Eyebrow>Routing</Eyebrow>
        <select
          value={mode}
          onChange={(e) => onMode(e.target.value)}
          aria-label="Routing mode"
          className="w-full rounded-sa-sm border border-border-strong bg-surface-2 px-3 py-2 font-mono text-sm text-text outline-none focus:border-accent-line"
        >
          {config && <option value={config.auto_mode}>{config.auto_mode}</option>}
          {config?.domains.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </section>

      <section>
        <Eyebrow>API keys</Eyebrow>
        <div className="flex flex-wrap gap-2">
          {config?.keys.map((k) => (
            <KeyChip key={k.label} status={k} />
          ))}
        </div>
      </section>
    </aside>
  );
}

function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div className="mb-2 font-mono text-[11px] uppercase tracking-wider text-muted">{children}</div>
  );
}

function StatusCard({ corpus }: { corpus: CorpusResponse | null }) {
  const unavailable = corpus != null && corpus.chunks < 0;
  const tone = corpus == null ? "bg-faint" : unavailable ? "bg-down" : "bg-up";
  return (
    <div className="rounded-sa-sm border border-border bg-surface-2 p-3 shadow-sa">
      <div className="flex items-start gap-2">
        <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${tone}`} aria-hidden />
        <div className="text-xs leading-relaxed text-muted">
          <div className="font-mono text-[10px] uppercase tracking-wider text-faint">Corpus</div>
          {corpus ? (
            <>
              <div className="font-mono text-text">{corpus.embedder}</div>
              <div className="text-faint">
                {unavailable
                  ? "store unavailable"
                  : `${corpus.chunks.toLocaleString()} chunks · ${corpus.tickers} tickers`}
              </div>
              {corpus.latest && <div className="text-faint">fresh to {corpus.latest}</div>}
            </>
          ) : (
            <div className="text-faint">loading…</div>
          )}
        </div>
      </div>
    </div>
  );
}

function KeyChip({ status }: { status: KeyStatus }) {
  const { label, present, required } = status;
  const cls = present
    ? "border-teal text-teal"
    : required
      ? "border-down text-down"
      : "border-border-strong text-faint";
  const marker = present ? "✓" : required ? "✕" : "·";
  const text = present ? label : required ? `${label} (required)` : label;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 font-mono text-[11px] ${cls}`}>
      <span className="font-bold">{marker}</span>
      {text}
    </span>
  );
}
