// @vitest-environment node
// (file-read only, no DOM — node env keeps import.meta.url a file:// URL for fileURLToPath)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Web-side guard on the token bridge (the Python side asserts file == generator in
// tests/unit/test_token_bridge.py). Here we pin the structural contract the SPA relies on:
// both theme blocks exist and the tokens Tailwind references are actually defined.
const css = readFileSync(fileURLToPath(new URL("../tokens.css", import.meta.url)), "utf-8");

// Tokens referenced by tailwind.config.ts — if the generator drops one, Tailwind classes
// would resolve to an undefined var. Keep in sync with tailwind.config.ts colors.
const REQUIRED_TOKENS = [
  "--sa-bg",
  "--sa-surface",
  "--sa-border",
  "--sa-text",
  "--sa-muted",
  "--sa-accent",
  "--sa-teal",
  "--sa-sky",
  "--sa-up",
  "--sa-down",
  "--sa-font-sans",
  "--sa-font-mono",
];

describe("token bridge (web side)", () => {
  it("is auto-generated (carries the do-not-edit header)", () => {
    expect(css).toContain("AUTO-GENERATED");
  });

  it("ships both theme sets for the real toggle", () => {
    expect(css).toContain(":root {");
    expect(css).toContain(':root[data-theme="light"] {');
  });

  it("defines every token Tailwind references", () => {
    for (const token of REQUIRED_TOKENS) {
      expect(css, `${token} missing from tokens.css`).toContain(`${token}:`);
    }
  });

  it("uses system font stacks only (no webfont)", () => {
    expect(css).not.toContain("url(");
    expect(css).not.toContain("@import");
  });
});
