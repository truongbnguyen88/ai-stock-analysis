import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Render an assistant answer as GitHub-flavored Markdown.
 *
 * The agent's grounded prose is Markdown (## sections, **bold**, `|`-pipe tables, lists). The old
 * raw `pre-wrap` render leaked the syntax (literal `##`) and could not align tables. react-markdown
 * + remark-gfm parses it to semantic HTML; all visual styling lives in app.css scoped under `.md`
 * so this component stays presentation-light.
 *
 * Safety: raw HTML in the source is NOT rendered (react-markdown default — no rehype-raw), so LLM
 * text can never inject markup. This is consistent with the repo's grounding / hallucination-control
 * posture (the guard governs *numbers*; this governs *markup*).
 *
 * Only two element overrides are needed; everything else is styled by CSS:
 *  - `a`     → open in a new tab (sources / SEC-filing links) with `noopener`.
 *  - `table` → wrapped in an overflow-x scroller so a wide table scrolls *inside* the 760px
 *    measure column instead of forcing the whole page to scroll horizontally.
 *
 * (`node` is destructured only to strip it from the DOM spread — passing react-markdown's mdast
 * node onto a real `<a>`/`<table>` would trigger an "unknown prop" warning. The paired `...rest`
 * is why TS's noUnusedLocals does not flag it.)
 */
const COMPONENTS: Components = {
  a: ({ node, ...rest }) => <a {...rest} target="_blank" rel="noopener noreferrer" />,
  table: ({ node, ...rest }) => (
    <div className="md-tablewrap">
      <table {...rest} />
    </div>
  ),
};

export function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
