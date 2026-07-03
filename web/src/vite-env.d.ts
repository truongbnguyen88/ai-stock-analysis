/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI dev server (default http://localhost:8000). */
  readonly VITE_API_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
