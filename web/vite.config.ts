/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// Local-dev only (PHASE2 plan §7/§11): the SPA talks to the FastAPI app on VITE_API_BASE
// (default localhost:8000). No proxy indirection so CORS is exercised exactly as in prod-like use.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: { port: 5173 },
  test: {
    globals: true,
    environment: "jsdom",
    // A concrete origin (not the default about:blank opaque origin) so window.localStorage —
    // used by the theme toggle's persistence — is available in tests.
    environmentOptions: { jsdom: { url: "http://localhost/" } },
    setupFiles: ["./src/test/setup.ts"],
    css: true,
  },
});
