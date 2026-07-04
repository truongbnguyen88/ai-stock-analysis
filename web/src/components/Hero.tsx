import { useEffect, useState, type CSSProperties } from "react";
import { hueVar } from "@/lib/hues";

// Typewriter phrases — one per capability card (same order as `CAPS`), so the hero headline advertises
// EVERY tool the agent can compose, not just a subset. Each reads naturally after "You can ask about".
const PHRASES = [
  "technical analysis",
  "the SEC filings",
  "multi-hop filing research",
  "probabilistic forecasts",
  "forecast calibration",
  "a big-move probability",
  "the latest news",
  "news themes & catalysts",
  "an executive research brief",
  "a multi-ticker comparison",
  "a theme's news",
];

interface Capability {
  glyph: string;
  hue: string; // --sa-* hue token name — drives the badge tint via --bc (like TileRow's --tc)
  title: string;
  body: string;
  prompt: (ticker: string) => string;
}

// The full capability set — one card per agent tool family, kept at PARITY with the Streamlit empty
// state (`src/stock_agent/ui/capabilities.py`, 11 cards) so both frontends advertise every tool the
// router can compose. Each dispatches a canned, auto-routed prompt on the ACTIVE ticker through the
// normal send path — phrased so the LLM router composes the matching tools. Numbers still come only
// from those tools, never this prose (invariant §1); these are prompts, not results. Geometric badge
// glyphs + `--sa-*` hue tokens (mockup aesthetic, not emoji); with 6 hues for 11 cards the palette
// repeats in a balanced 2-per-hue spread.
const CAPS: Capability[] = [
  {
    glyph: "◪",
    hue: "teal",
    title: "Technical analysis",
    body: "Trend, momentum, volatility — MAs, RSI, MACD, ATR, drawdown",
    prompt: (t) => `Give a technical analysis of ${t}.`,
  },
  {
    glyph: "▤",
    hue: "sky",
    title: "Ask the SEC filings",
    body: "Risk factors, MD&A, business drivers — from the 10-K/10-Q, cited",
    prompt: (t) => `What do ${t}'s SEC filings say about its risk factors and key business drivers?`,
  },
  {
    glyph: "⊹",
    hue: "indigo",
    title: "Multi-hop filing research",
    body: "Compare filings, track changes, follow the trail across documents",
    prompt: (t) =>
      `Do multi-hop filing research on ${t}: compare the latest 10-Q with the prior 10-K and highlight what changed.`,
  },
  {
    glyph: "◔",
    hue: "accent",
    title: "Probabilistic forecasts",
    body: "Calibrated up/down odds, expected move, VaR & CI — 5–60 days",
    prompt: (t) => `Forecast ${t} 20 days out — calibrated up/down odds, expected move, and VaR.`,
  },
  {
    glyph: "◎",
    hue: "violet",
    title: "Is this forecast trustworthy?",
    body: "Out-of-sample calibration (ECE) and walk-forward track record",
    prompt: (t) =>
      `Is the forecast for ${t} trustworthy? Show the out-of-sample calibration (ECE) and walk-forward track record.`,
  },
  {
    glyph: "⚡",
    hue: "rose",
    title: "Chance of a big move",
    body: "P(|return| > k) split into up- and down-tails — the ML edge",
    prompt: (t) => `What's the chance of a big move in ${t}? Split the probability into up- and down-tails.`,
  },
  {
    glyph: "▦",
    hue: "sky",
    title: "Latest news, newest-first",
    body: "Recent headlines with real sources and links",
    prompt: (t) => `Show me the latest news for ${t}.`,
  },
  {
    glyph: "◧",
    hue: "teal",
    title: "News synthesis",
    body: "Themes, risks, and catalysts from the news — cited",
    prompt: (t) => `Summarize ${t} news and pull out the key themes, risks, and catalysts.`,
  },
  {
    glyph: "❖",
    hue: "accent",
    title: "Executive research brief",
    body: "Filings + news + forecast fused into one cited summary",
    prompt: (t) => `Give me the full executive research brief on ${t}.`,
  },
  {
    glyph: "⧉",
    hue: "indigo",
    title: "Compare multiple tickers",
    body: "Side-by-side forecasts and news sentiment across names",
    prompt: (t) => `Compare the 20-day forecasts and news sentiment for ${t}, MSFT, and AMD.`,
  },
  {
    glyph: "⬢",
    hue: "violet",
    title: "Analyze a theme's news",
    // Theme-driven, ticker-independent — mirrors the Streamlit example.
    body: "News by topic — robotics, EVs, AI memory, semis…",
    prompt: () => `Pull and analyze recent news about robotics.`,
  },
];

/** Typewriter position: phrase index `pi`, characters shown `ci`, and whether we're deleting. */
export interface TwState {
  pi: number;
  ci: number;
  del: boolean;
}

/** Initial state: the first phrase fully shown, poised to delete — so the array actually cycles. */
export function initialTypeState(phrases: string[]): TwState {
  return { pi: 0, ci: phrases[0].length, del: true };
}

/**
 * One typewriter step (pure): grow/shrink the visible slice by one char; hold at a full phrase, then
 * delete; advance to the next phrase after a full erase. Returns the next state, the text to show,
 * and the delay (ms) before the following step. Extracted so the cycle is unit-tested without timers
 * or the DOM. Uses `>=`/`<=` bounds (not `==`) so an overshoot can't strand it on one phrase — the
 * off-by-one that froze the mockup's version on phrase 0. Timings mirror the mockup.
 */
export function stepType(
  s: TwState,
  phrases: string[],
): { state: TwState; text: string; delayMs: number } {
  const full = phrases[s.pi];
  const ci = s.ci + (s.del ? -1 : 1);
  const text = full.slice(0, Math.max(0, ci));
  let { pi, del } = s;
  let delayMs = s.del ? 34 : 66; // delete faster than type
  if (!s.del && ci >= full.length) {
    delayMs = 1400; // hold at the full phrase before erasing
    del = true;
  } else if (s.del && ci <= 0) {
    del = false;
    pi = (s.pi + 1) % phrases.length; // next phrase after a full erase
    delayMs = 260; // brief gap before typing the next one
  }
  return { state: { pi, ci, del }, text, delayMs };
}

/**
 * Typewriter effect for the hero headline: drives `stepType` on a self-rescheduling timer. Honors
 * `prefers-reduced-motion` (and jsdom, where `matchMedia` is undefined) by leaving the first phrase
 * static.
 */
function useTypewriter(): string {
  const [text, setText] = useState(PHRASES[0]);
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches) return;
    let st = initialTypeState(PHRASES);
    let timer = window.setTimeout(run, 1400);
    function run() {
      const r = stepType(st, PHRASES);
      st = r.state;
      setText(r.text);
      timer = window.setTimeout(run, r.delayMs);
    }
    return () => window.clearTimeout(timer);
  }, []);
  return text;
}

interface HeroProps {
  ticker: string;
  onQuickStart: (prompt: string) => void;
}

/**
 * Empty-state hero (mockup #view-empty): eyebrow tagline, a typewriter cycling through every
 * capability phrase, a subtitle, and a responsive grid of all capability cards (compact, `auto-fill`
 * so they flow to fewer columns on narrow widths). Each card dispatches a canned prompt on the active
 * ticker through the same `onQuickStart` the sidebar starters use (normal send path, auto-routed).
 * Presentation only — no numbers originate here.
 */
export function Hero({ ticker, onQuickStart }: HeroProps) {
  const tw = useTypewriter();
  const t = ticker || "the ticker";
  return (
    <div className="hero">
      <div className="eyebrow label">
        <span style={{ color: "var(--sa-sky)" }}>SEC-grounded</span>
        {" · "}
        <span style={{ color: "var(--sa-teal)" }}>probabilistic</span>
        {" · "}
        <span style={{ color: "var(--sa-rose)" }}>non-advisory</span>
      </div>
      <div className="typewriter">
        <span className="pre">You can ask about&nbsp;</span>
        <span className="tw">{tw}</span>
        <span className="caret">▌</span>
      </div>
      <p className="sub">
        Pick a capability to try it on <b>{ticker || "a ticker"}</b> — or just ask in the box below.
      </p>
      <div className="capgrid">
        {CAPS.map((c) => (
          <button
            key={c.title}
            type="button"
            className="capcard"
            onClick={() => onQuickStart(c.prompt(t))}
          >
            <div className="badge" style={{ "--bc": hueVar(c.hue) } as CSSProperties}>
              {c.glyph}
            </div>
            <div>
              <div className="ct">{c.title}</div>
              <div className="cb">{c.body}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
