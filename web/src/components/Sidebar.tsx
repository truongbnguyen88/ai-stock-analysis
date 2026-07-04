import type { ConfigResponse, CorpusResponse, KeyStatus } from "@/lib/api";

interface SidebarProps {
  config: ConfigResponse | null;
  corpus: CorpusResponse | null;
  ticker: string;
  onTicker: (t: string) => void;
  mode: string;
  onMode: (m: string) => void;
  /** Dispatch a canned quick-starter prompt through the normal send path (optional). */
  onQuickStart?: (prompt: string) => void;
}

interface Starter {
  icon: string;
  hue: string; // --sa-* hue token name for the icon
  label: string;
  prompt: (ticker: string) => string;
}

// Quick starters mirror the mockup's four rows; each dispatches a canned prompt (auto-routed).
const STARTERS: Starter[] = [
  { icon: "◇", hue: "sky", label: "News summary", prompt: (t) => `Summarize the recent news for ${t}.` },
  { icon: "◪", hue: "teal", label: "Technical analysis", prompt: (t) => `Give a technical analysis of ${t}.` },
  { icon: "◔", hue: "accent", label: "Forecast · 20 days", prompt: (t) => `Forecast ${t} 20 days out.` },
  { icon: "◈", hue: "violet", label: "Full analysis", prompt: (t) => `Give a full analysis of ${t}.` },
];

/**
 * Left rail (mockup §6): corpus status card, ticker field, routing select + note, hue-iconed quick
 * starters, and API-key chips — all driven by live /config + /corpus (the P2.0 DoD). The saved-chat
 * list lands with threads (P2.5).
 */
export function Sidebar({ config, corpus, ticker, onTicker, mode, onMode, onQuickStart }: SidebarProps) {
  return (
    <aside className="side">
      <div className="side-sec">
        <StatusCard corpus={corpus} />
      </div>

      <div className="side-sec">
        <div className="label">Ticker</div>
        <div className="ticker-field">
          <span className="label">$</span>
          <input
            value={ticker}
            onChange={(e) => onTicker(e.target.value.toUpperCase())}
            maxLength={10}
            aria-label="Ticker symbol"
          />
        </div>
      </div>

      <div className="side-sec">
        <div className="label">Routing</div>
        <select
          className="select"
          value={mode}
          onChange={(e) => onMode(e.target.value)}
          aria-label="Routing mode"
        >
          {config && <option value={config.auto_mode}>{config.auto_mode}</option>}
          {config?.domains.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        <div className="route-note">Auto lets the model choose &amp; compose tools.</div>
      </div>

      <div className="side-sec">
        <div className="label">Quick starters</div>
        {STARTERS.map((s) => (
          <button
            key={s.label}
            type="button"
            className="btn"
            onClick={() => onQuickStart?.(s.prompt(ticker || "the ticker"))}
          >
            <span className="ico" style={{ color: `var(--sa-${s.hue})` }}>
              {s.icon}
            </span>
            {s.label}
          </button>
        ))}
      </div>

      <div className="side-sec">
        <div className="label">Keys</div>
        <div className="keys">
          {config?.keys.map((k) => (
            <KeyChip key={k.label} status={k} />
          ))}
        </div>
      </div>
    </aside>
  );
}

function StatusCard({ corpus }: { corpus: CorpusResponse | null }) {
  const unavailable = corpus != null && corpus.chunks < 0;
  const dotClass = corpus == null ? "stale" : unavailable ? "down" : "";
  const state = corpus == null ? "loading" : unavailable ? "unavailable" : "fresh";
  return (
    <div className="statuscard">
      <div className="row">
        <span className="label" style={{ letterSpacing: "0.05em" }}>
          Filing search
        </span>
        <span className="live">
          <span className={`dot ${dotClass}`.trim()} aria-hidden />
          <span className="v">{state}</span>
        </span>
      </div>
      <div className="row">
        <span className="k">embedder</span>
        <span className="v">{corpus ? corpus.embedder : "…"}</span>
      </div>
      <div className="row">
        <span className="k">corpus</span>
        <span className="v">
          {corpus == null
            ? "…"
            : unavailable
              ? "unavailable"
              : `${corpus.chunks.toLocaleString()} chunks`}
        </span>
      </div>
      <div className="row">
        <span className="k">coverage</span>
        <span className="v">
          {corpus == null
            ? "…"
            : `${corpus.tickers} tickers${corpus.latest ? ` · ${corpus.latest}` : ""}`}
        </span>
      </div>
    </div>
  );
}

function KeyChip({ status }: { status: KeyStatus }) {
  const { label, present, required } = status;
  const cls = present ? "on" : required ? "required" : "off";
  const marker = present ? "●" : "○";
  return (
    <span className={`keychip ${cls}`}>
      <span className="m">{marker}</span>
      {label}
    </span>
  );
}
