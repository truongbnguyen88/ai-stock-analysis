import { useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { exportSummary, type ExportFormat } from "@/lib/api";

// Mockup's export affordance: "↓ Export ▾" → a small popover of the three formats. Labels match the
// mockup ("PDF · Word · Markdown"); values are the /export fmt keys.
const FORMATS: ReadonlyArray<readonly [ExportFormat, string]> = [
  ["pdf", "PDF"],
  ["docx", "Word"],
  ["md", "Markdown"],
];

interface ExportMenuProps {
  /** The assistant's final answer (markdown) — the document body. */
  text: string;
  /** ChartSpec dicts to embed as figures (pdf/docx); passed straight to the server renderer. */
  charts?: unknown[];
  /** Optional download title / filename stem. */
  title?: string;
}

/**
 * Export the current turn to PDF / Word / Markdown (P2.5). A Radix Popover (the app's first use)
 * anchored on the mockup's `.export-btn`; picking a format POSTs to /export and downloads the file.
 * The server does all formatting — this only chooses the format and surfaces failures inline.
 */
export function ExportMenu({ text, charts, title }: ExportMenuProps) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [failed, setFailed] = useState(false);

  const pick = async (fmt: ExportFormat): Promise<void> => {
    setBusy(fmt);
    setFailed(false);
    try {
      await exportSummary(fmt, text, { charts, title });
    } catch {
      setFailed(true);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className="export-btn" aria-label="Export">
          ↓ Export ▾
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="export-menu" align="start" sideOffset={6}>
          {FORMATS.map(([fmt, label]) => (
            <button
              key={fmt}
              type="button"
              className="export-item"
              disabled={busy !== null}
              onClick={() => void pick(fmt)}
            >
              {label}
              {busy === fmt ? " …" : ""}
            </button>
          ))}
          {failed && <div className="export-err">Export failed — is the API running?</div>}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
