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

// --- Saved chat threads (P2.5b) -------------------------------------------------------------------
// Display-level persistence: the SPA stores its own `Turn[]` transcript in `display_messages` (opaque
// to the server). The store is nested under a `web/` subdir server-side, so it never sees the
// Streamlit `{role, content}` shape. See src/stock_agent/api/routes/threads.py.

/** One saved-thread descriptor from GET /threads (mirrors api.schemas.ThreadMetaResponse). */
export interface ThreadMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

/** A full thread from GET /threads/{id} or POST /threads (mirrors api.schemas.ThreadResponse).
 *  `display_messages` is opaque to the server — the SPA round-trips its own Turn[] display shape. */
export interface ThreadFull extends ThreadMeta {
  display_messages: unknown[];
  agent_history: unknown[];
}

/** Body for POST /threads — create (empty/absent id) or update one thread (ThreadSaveRequest). */
export interface ThreadSave {
  id?: string;
  title?: string;
  display_messages?: unknown[];
  agent_history?: unknown[];
}

export const fetchThreads = (): Promise<ThreadMeta[]> => getJSON<ThreadMeta[]>("/threads");
export const fetchThread = (id: string): Promise<ThreadFull> =>
  getJSON<ThreadFull>(`/threads/${encodeURIComponent(id)}`);

/** Create or update a thread; returns the saved thread (the server mints the id when empty). */
export async function saveThread(body: ThreadSave): Promise<ThreadFull> {
  const res = await fetch(`${API_BASE}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: body.id ?? "",
      title: body.title ?? "",
      display_messages: body.display_messages ?? [],
      agent_history: body.agent_history ?? [],
    }),
  });
  if (!res.ok) throw new Error(`POST /threads failed: ${res.status} ${res.statusText}`);
  return (await res.json()) as ThreadFull;
}

/** Delete a thread. Idempotent server-side (204 even if absent). Throws on non-2xx. */
export async function deleteThread(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/threads/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE /threads/${id} failed: ${res.status} ${res.statusText}`);
}
