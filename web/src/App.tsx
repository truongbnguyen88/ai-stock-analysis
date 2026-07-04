import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { TopBar, type View } from "@/components/TopBar";
import { Stream } from "@/components/Stream";
import { Hero } from "@/components/Hero";
import { Composer } from "@/components/Composer";
import { fetchConfig, fetchCorpus, type ConfigResponse, type CorpusResponse } from "@/lib/api";
import { useTheme } from "@/lib/useTheme";
import { useConversation } from "@/store/conversation";

/**
 * Conversation shell (mockup-exact): the sticky TopBar over a two-column `.shell` (Sidebar + main).
 * Main is the scrolling `.stream` (folded turns from /chat/stream) above the sticky `.inputbar`
 * Composer. Config/corpus come from the live meta endpoints (P2.0); the store folds AgentEvents into
 * turns (tiles, live trace, chart, provisional→final prose, sources). The empty-state Hero and
 * threads/export land in later slices (P2.5/P2.6).
 */
export default function App() {
  const { theme, toggle } = useTheme();
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [corpus, setCorpus] = useState<CorpusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState("");
  const [mode, setMode] = useState("");
  // Empty-state hero until the first send; the TopBar seg toggles it non-destructively (mockup #view).
  const [view, setView] = useState<View>("hero");

  const turns = useConversation((s) => s.turns);
  const streaming = useConversation((s) => s.streaming);
  const send = useConversation((s) => s.send);
  const reset = useConversation((s) => s.reset);

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
        /* corpus card degrades to "…"; the config error is the surfaced one */
      });
  }, []);

  // mode === auto label → "auto" (LLM path); otherwise it's a domain name the API resolves to a
  // granular route (resolve_domain). Keeps routing logic in Python — the client sends the selection.
  const route = config && mode === config.auto_mode ? "auto" : mode || "auto";
  const contextChips = [ticker || "no ticker", mode].filter(Boolean);

  const onSend = (message: string): void => {
    // Composing from the hero while a prior thread exists starts a genuinely new chat (single
    // in-memory thread; multi-thread lands in P2.5). Non-destructive until the user actually sends.
    if (view === "hero" && turns.length > 0) reset();
    void send({ message, route, ticker: ticker || null });
    setView("chat");
  };
  const onQuickStart = (prompt: string): void => {
    if (!streaming) onSend(prompt);
  };

  return (
    <div className="app">
      <TopBar
        config={config}
        ticker={ticker}
        mode={mode}
        theme={theme}
        onToggleTheme={toggle}
        view={view}
        onView={setView}
      />
      <div className="shell">
        <Sidebar
          config={config}
          corpus={corpus}
          ticker={ticker}
          onTicker={setTicker}
          mode={mode}
          onMode={setMode}
          onQuickStart={onQuickStart}
        />
        <main className="main">
          {error ? (
            <div className="stream">
              <div className="measure">
                <div className="errbox">
                  <span className="ek">API unavailable</span>
                  <p className="em">
                    {error}. Start it with{" "}
                    <code>uvicorn stock_agent.api.app:app --port 8000</code>.
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {view === "hero" ? (
                <div className="stream">
                  <Hero ticker={ticker} onQuickStart={onQuickStart} />
                </div>
              ) : (
                <div className="stream">
                  <div className="measure">
                    <Stream turns={turns} />
                  </div>
                </div>
              )}
              <div className="inputbar">
                <div className="measure">
                  <Composer onSend={onSend} disabled={streaming} contextChips={contextChips} />
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
