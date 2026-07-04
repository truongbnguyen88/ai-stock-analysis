import { useEffect, useState, type CSSProperties } from "react";
import { hueVar } from "@/lib/hues";

// Typewriter phrases — verbatim from the mockup's hero cycle (#view-empty).
const PHRASES = [
  "probabilistic forecasts",
  "the SEC filings",
  "a big-move probability",
  "multi-hop filing research",
  "forecast calibration",
  "technical analysis",
];

interface Capability {
  glyph: string;
  hue: string; // --sa-* hue token name — drives the badge tint via --bc (like TileRow's --tc)
  title: string;
  body: string;
  prompt: (ticker: string) => string;
}

// The six capability cards (mockup order + copy). Each dispatches a canned, auto-routed prompt on the
// ACTIVE ticker through the normal send path — phrased so the LLM router composes the matching tools.
// Numbers still come only from those tools, never this prose (invariant §1); these are prompts, not
// results. Badge glyphs/hues are the mockup's.
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
 * Empty-state hero (mockup #view-empty): eyebrow tagline, a typewriter cycling capability phrases, a
 * subtitle, and a 2-column grid of six capability cards. Each card dispatches a canned prompt on the
 * active ticker through the same `onQuickStart` the sidebar starters use (normal send path, auto-
 * routed). Presentation only — no numbers originate here.
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
