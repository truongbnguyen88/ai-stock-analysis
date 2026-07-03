import "@testing-library/jest-dom/vitest";
import { afterEach, beforeAll } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom (in this vitest build) doesn't expose window.localStorage, and Node's built-in is
// gated behind --localstorage-file. The theme toggle persists the choice via localStorage,
// so install a minimal in-memory Storage shim for tests (deterministic, no disk).
class MemoryStorage implements Storage {
  private store = new Map<string, string>();
  get length(): number {
    return this.store.size;
  }
  clear(): void {
    this.store.clear();
  }
  getItem(key: string): string | null {
    return this.store.has(key) ? this.store.get(key)! : null;
  }
  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null;
  }
  removeItem(key: string): void {
    this.store.delete(key);
  }
  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }
}

// jsdom has no canvas backend; importing vega (via react-vega) probes HTMLCanvasElement.getContext
// and jsdom throws "Not implemented". No test renders a real chart (charts are mocked / the spec
// translation is tested purely), so stub getContext to null. Done at module top-level (not in
// beforeAll) so it lands before the test files import vega, silencing the import-time probe.
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
}

beforeAll(() => {
  const storage = new MemoryStorage();
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true });
  // window only exists in the jsdom environment (node-env tests, e.g. tokens.test.ts, skip this).
  if (typeof window !== "undefined") {
    Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  }
});

afterEach(() => {
  cleanup();
  globalThis.localStorage?.clear();
  if (typeof document !== "undefined") {
    document.documentElement.removeAttribute("data-theme");
  }
});
