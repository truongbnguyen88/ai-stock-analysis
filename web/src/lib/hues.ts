// Hue token → CSS variable, for tiles (tone) and trace chips (hueName). Keeps the palette in the
// shared --sa-* token set (theme-aware, no hardcoded hex on the React side). Unknown tones fall
// back to brass accent — mirrors ui.html._hue_token so a typo can't blank a chip.

const HUE_TOKENS = new Set(["teal", "sky", "indigo", "violet", "rose", "accent"]);

/** Return `var(--sa-<hue>)` for a valid hue name, else the brass `var(--sa-accent)`. */
export function hueVar(name: string): string {
  return `var(--sa-${HUE_TOKENS.has(name) ? name : "accent"})`;
}
