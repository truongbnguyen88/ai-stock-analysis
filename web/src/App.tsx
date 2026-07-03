import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { Stream } from "@/components/Stream";
import { Composer } from "@/components/Composer";
import { fetchConfig, fetchCorpus, type ConfigResponse, type CorpusResponse } from "@/lib/api";
import { useTheme } from "@/lib/useTheme";
import { useConversation } from "@/store/conversation";

/**
 * P2.3 conversation shell: the TopBar + Sidebar (from live /config + /corpus, P2.0) now sit above a
 * real streamed conversation. The Composer POSTs to /chat/stream and the store folds the AgentEvents
 * into turns (tiles, live trace, chart, provisional→final prose, sources). The empty-state Hero and
 * threads/export land in later slices (P2.5/P2.6).
 */
export default function App() {
  const { theme, toggle } = useTheme();
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState("");
  const [mode, setMode] = useState("");

  const turns = useConversation((s) => s.turns);
  const streaming = useConversation((s) => s.streaming);
  const send = useConversation((s) => s.send);

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

  // mode === auto label → "auto" (LLM path); otherwise it's a domain name the API resolves to a
  // granular route (resolve_domain). Keeps routing logic in Python — the client sends the selection.
  const route = config && mode === config.auto_mode ? "auto" : mode || "auto";
  const contextChips = [ticker || "no ticker", mode].filter(Boolean);

  const onSend = (message: string): void => {
    void send({ message, route, ticker: ticker || null });
  };

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
        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-6 py-8">
          {error ? (
            <p className="font-mono text-sm text-down">
              API unavailable: {error}. Start it with{" "}
              <code>uvicorn stock_agent.api.app:app --port 8000</code>.
            </p>
          ) : (
            <>
              <div className="flex-1">
                <Stream turns={turns} />
              </div>
              <Composer onSend={onSend} disabled={streaming} contextChips={contextChips} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
