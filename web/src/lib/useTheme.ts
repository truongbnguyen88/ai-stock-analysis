import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "sa-theme";

/** Read the persisted theme, defaulting to dark (the design's primary; PHASE2 §6). */
function initialTheme(): Theme {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved === "light" ? "light" : "dark";
}

/**
 * Instant, client-side theme toggle — the ceiling item Streamlit R6.B could only approximate.
 * Flips `document.documentElement.dataset.theme`, which switches the `--sa-*` token set
 * (see tokens.css `:root` vs `:root[data-theme="light"]`) with zero reload. Persisted so a
 * refresh keeps the choice.
 */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}
