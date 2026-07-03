import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { fetchConfig, fetchCorpus, type ConfigResponse, type CorpusResponse } from "@/lib/api";
import { useTheme } from "@/lib/useTheme";

/**
 * P2.0 static shell: fetch /config + /corpus on mount and render the TopBar + Sidebar from the
 * live responses, with a working data-theme toggle. The conversation surface, composer, and
 * hero are stubbed here and built out in later slices (P2.3+).
 */
export default function App() {
  const { theme, toggle } = useTheme();
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState("");
  const [mode, setMode] = useState("");

  useEffect(() => {
    fetchConfig()
      .then((c) => {
        setConfig(c);
        setTicker(c.default_ticker);
        setMode(c.auto_mode);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    fetchCorpus()
      .then(setCorpus)
      .catch(() => {
        /* corpus card degrades to "loading…"; config error is the surfaced one */
      });
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      <TopBar config={config} ticker={ticker} mode={mode} theme={theme} onToggleTheme={toggle} />
      <div className="flex flex-1">
        <Sidebar
          config={config}
          corpus={corpus}
          ticker={ticker}
          onTicker={setTicker}
          mode={mode}
          onMode={setMode}
        />
        <main className="flex-1 px-6 py-8">
          {error ? (
            <p className="font-mono text-sm text-down">
              API unavailable: {error}. Start it with{" "}
              <code>uvicorn stock_agent.api.app:app --port 8000</code>.
            </p>
          ) : (
            <p className="max-w-prose text-sm text-muted">
              Shell ready. Conversation, composer, and empty-state hero arrive in later Phase-2
              slices (P2.3+). The top bar and sidebar above are driven by the live API.
            </p>
          )}
        </main>
      </div>
    </div>
  );
}
