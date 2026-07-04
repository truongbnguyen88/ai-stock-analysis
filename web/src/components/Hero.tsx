import { useEffect, useState, type CSSProperties } from "react";
import { hueVar } from "@/lib/hues";

// Typewriter phrases — one per underlying capability, so the hero headline advertises EVERY tool the
// agent can compose, not just the five category cards. Each reads naturally after "You can ask about".
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

/** One runnable prompt inside a category (revealed when the category is expanded). */
interface SubPrompt {
  title: string;
  body: string;
  prompt: (ticker: string) => string;
}

/**
 * A top-level capability card. Exactly one of `prompt` / `children` is set:
 *  - `prompt`  → a direct-dispatch leaf (clicking fires the prompt immediately, e.g. Technical analysis).
 *  - `children`→ an expandable category (clicking toggles an inline accordion of `SubPrompt`s).
 */
interface Capability {
  glyph: string;
  hue: string; // --sa-* hue token name — drives the badge tint via --bc (like TileRow's --tc)
  title: string;
  body: string;
  prompt?: (ticker: string) => string;
  children?: SubPrompt[];
}

// Capability taxonomy — grouped so the empty state stays compact (5 cards) while still advertising
// every tool the router can compose. Technical analysis is a direct-dispatch leaf; the other four are
// accordion categories. Each prompt routes through the normal auto-routed send path; numbers come only
// from the tools, never this prose (invariant §1) — these are prompts, not results. Geometric badge
// glyphs + `--sa-*` hue tokens (mockup aesthetic, not emoji). Parity intent mirrors the Streamlit
// empty state (`src/stock_agent/ui/capabilities.py`), reorganized into categories here.
const CAPS: Capability[] = [
  {
    glyph: "◪",
    hue: "teal",
    title: "Technical analysis",
    body: "Trend, momentum, volatility — MAs, RSI, MACD, ATR, drawdown",
    prompt: (t) => `Give a technical analysis of ${t}: trend, momentum, and volatility.`,
  },
  {
    glyph: "▤",
    hue: "sky",
    title: "SEC filings",
    body: "Grounded, cited answers from the 10-K/10-Q",
    children: [
      {
        title: "Risk factors & drivers",
        body: "What management flags — risks, demand, margins",
        prompt: (t) =>
          `What do ${t}'s SEC filings say about its risk factors and key business drivers?`,
      },
      {
        title: "Multi-hop filing research",
        body: "Compare filings, track what changed across documents",
        prompt: (t) =>
          `Compare ${t}'s latest 10-Q with the prior 10-K and highlight what changed — cite the filings.`,
      },
    ],
  },
  {
    glyph: "◔",
    hue: "accent",
    title: "Forecasts & odds",
    body: "Calibrated probabilities, expected move, tail risk",
    children: [
      {
        title: "Probabilistic forecast · 20-day",
        body: "Up/down odds, expected move, VaR & CI",
        prompt: (t) => `Forecast ${t} 20 days out — calibrated up/down odds, expected move, and VaR.`,
      },
      {
        title: "Chance of a big move",
        body: "P(|return| > k), split into up- and down-tails",
        prompt: (t) =>
          `What's the chance of a big move in ${t}? Split the probability into up- and down-tails.`,
      },
      {
        title: "Is the forecast trustworthy?",
        body: "Out-of-sample calibration (ECE) + walk-forward record",
        prompt: (t) =>
          `Is the forecast for ${t} trustworthy? Show the out-of-sample calibration (ECE) and walk-forward track record.`,
      },
    ],
  },
  {
    glyph: "▦",
    hue: "rose",
    title: "News & sentiment",
    body: "Headlines, synthesis, and theme scans — cited",
    children: [
      {
        title: "Latest news, newest-first",
        body: "Recent headlines with real sources and links",
        prompt: (t) => `Show me the latest news for ${t}.`,
      },
      {
        title: "News synthesis",
        body: "Themes, risks, and catalysts — cited, qualitative",
        prompt: (t) => `Summarize ${t} news and pull out the key themes, risks, and catalysts.`,
      },
      {
        title: "A theme's news",
        // Theme-driven, ticker-independent — mirrors the Streamlit example.
        body: "News by topic — robotics, EVs, AI memory, semis…",
        prompt: () => `Pull and analyze recent news about robotics.`,
      },
    ],
  },
  {
    glyph: "❖",
    hue: "violet",
    title: "Research briefs",
    body: "Multi-signal deep dives, one name or many",
    children: [
      {
        title: "Executive research brief",
        body: "Filings + news + forecast fused into one cited summary",
        prompt: (t) => `Give me the full executive research brief on ${t}.`,
      },
      {
        title: "Compare multiple tickers",
        body: "Side-by-side forecasts and news sentiment across names",
        prompt: (t) => `Compare the 20-day forecasts and news sentiment for ${t}, MSFT, and AMD.`,
      },
    ],
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
 * Empty-state hero (mockup #view-empty): eyebrow tagline, a typewriter cycling through every capability
 * phrase, a subtitle, and a compact grid of five capability cards. Technical analysis dispatches its
 * prompt immediately; the other four are single-open accordion categories that reveal their child
 * prompts inline (the clicked cell grows, pushing rows below it down — the grid is `align-items: start`
 * so children render *inside* the cell, no full-width span needed). Every prompt routes through the same
 * `onQuickStart` the sidebar starters use (normal send path, auto-routed). Presentation only — no
 * numbers originate here.
 */
export function Hero({ ticker, onQuickStart }: HeroProps) {
  const tw = useTypewriter();
  const t = ticker || "the ticker";
  // Single-open accordion: index of the expanded category, or null. Local state — the hero unmounts
  // when a prompt is dispatched (view switches to the conversation), so it resets naturally.
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <div className="hero" data-testid="hero">
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
        {CAPS.map((c, i) => {
          const isCategory = !!c.children;
          const open = openIndex === i;
          const panelId = `cap-panel-${i}`;
          return (
            <div
              key={c.title}
              className="capcard-wrap"
              style={{ "--bc": hueVar(c.hue) } as CSSProperties}
            >
              <button
                type="button"
                className="capcard"
                aria-expanded={isCategory ? open : undefined}
                aria-controls={isCategory && open ? panelId : undefined}
                onClick={() =>
                  isCategory
                    ? setOpenIndex(open ? null : i) // toggle; opening one closes any other (single-open)
                    : onQuickStart(c.prompt!(t)) // leaf → dispatch immediately
                }
              >
                <div className="badge">{c.glyph}</div>
                <div className="cap-txt">
                  <div className="ct">{c.title}</div>
                  <div className="cb">{c.body}</div>
                </div>
                {isCategory && (
                  <span className="chev" data-open={open} aria-hidden="true">
                    ›
                  </span>
                )}
              </button>
              {isCategory && open && (
                <div className="subcards" id={panelId}>
                  {c.children!.map((s) => (
                    <button
                      key={s.title}
                      type="button"
                      className="subcard"
                      onClick={() => onQuickStart(s.prompt(t))}
                    >
                      <div className="sct">{s.title}</div>
                      <div className="scb">{s.body}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
