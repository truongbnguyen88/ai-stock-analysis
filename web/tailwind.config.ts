import type { Config } from "tailwindcss";

// Token bridge (PHASE2 plan §6): every color maps to a --sa-* CSS variable defined in
// src/tokens.css (generated from src/stock_agent/ui/theme.py). Tailwind classes therefore
// resolve through the same variables the theme toggle flips — no hardcoded hex on this side,
// so the React palette can never drift from the Python source of truth.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--sa-bg)",
        surface: "var(--sa-surface)",
        "surface-2": "var(--sa-surface-2)",
        "surface-3": "var(--sa-surface-3)",
        border: "var(--sa-border)",
        "border-strong": "var(--sa-border-strong)",
        text: "var(--sa-text)",
        muted: "var(--sa-muted)",
        faint: "var(--sa-faint)",
        accent: "var(--sa-accent)",
        "accent-weak": "var(--sa-accent-weak)",
        "accent-line": "var(--sa-accent-line)",
        teal: "var(--sa-teal)",
        sky: "var(--sa-sky)",
        indigo: "var(--sa-indigo)",
        violet: "var(--sa-violet)",
        rose: "var(--sa-rose)",
        up: "var(--sa-up)",
        down: "var(--sa-down)",
      },
      fontFamily: {
        sans: "var(--sa-font-sans)",
        mono: "var(--sa-font-mono)",
      },
      borderRadius: {
        sa: "var(--sa-r)",
        "sa-sm": "var(--sa-r-sm)",
      },
      boxShadow: {
        sa: "var(--sa-shadow)",
      },
    },
  },
  plugins: [],
} satisfies Config;
