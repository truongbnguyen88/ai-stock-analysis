// Typed client for the FastAPI meta endpoints (P2.0). Types mirror src/stock_agent/api/schemas.py;
// they are the SPA's view of the same pure contracts the Streamlit sidebar reads, so the two
// frontends cannot drift on what they display.

/** One API-key row (mirrors api.schemas.KeyStatus). */
export interface KeyStatus {
  label: string;
  present: boolean;
  required: boolean;
}

/** Static app config from GET /config (mirrors api.schemas.ConfigResponse). */
export interface ConfigResponse {
  default_ticker: string;
  auto_mode: string;
  domains: string[];
  keys: KeyStatus[];
}

/** Corpus status from GET /corpus (mirrors api.schemas.CorpusResponse). */
export interface CorpusResponse {
  provider: string;
  embedder: string;
  collection: string;
  chunks: number;
  filings: number;
  tickers: number;
  earliest: string | null;
  latest: string | null;
  one_line: string;
}

/** Base URL of the FastAPI dev server. Configurable via VITE_API_BASE (local-dev default). */
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const fetchConfig = (): Promise<ConfigResponse> => getJSON<ConfigResponse>("/config");
export const fetchCorpus = (): Promise<CorpusResponse> => getJSON<CorpusResponse>("/corpus");

/** Document formats POST /export accepts (mirrors reports.export.EXPORT_META keys). */
export type ExportFormat = "pdf" | "docx" | "md";

/** Parse the download filename out of a Content-Disposition header (attachment; filename="x.pdf"). */
function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const m = /filename="?([^";]+)"?/.exec(header);
  return m ? m[1] : null;
}

/** Save a Blob to disk by clicking a transient object-URL anchor (browser-only side effect). */
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Render the assistant's answer (+ its chart specs) to `fmt` via POST /export and download the file.
 * The server owns the formatting and the non-advisory header (reports.export); the client only ships
 * the text + the same ChartSpec dicts it rendered, and saves the returned bytes. Throws on non-2xx.
 */
export async function exportSummary(
  fmt: ExportFormat,
  text: string,
  opts: { title?: string; charts?: unknown[] } = {},
): Promise<void> {
  const res = await fetch(`${API_BASE}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fmt,
      text,
      title: opts.title ?? "Stock Research Summary",
      charts: opts.charts ?? [],
    }),
  });
  if (!res.ok) {
    throw new Error(`POST /export failed: ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  triggerDownload(blob, filenameFromDisposition(res.headers.get("content-disposition")) ?? `summary.${fmt}`);
}
