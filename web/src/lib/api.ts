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
