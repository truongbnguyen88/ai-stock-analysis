import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "@/components/Markdown";

// The assistant answer is Markdown; before this component it rendered as raw pre-wrap text (literal
// "##", unaligned tables). These lock the parse: headers become <h2> (no "##"), pipe tables become
// real <table>s, and links open in a new tab. Styling itself lives in app.css and is not asserted.
describe("Markdown", () => {
  it("renders a section header as <h2> without the literal '##' markers", () => {
    render(<Markdown>{"## 📅 5. Earnings Context"}</Markdown>);
    const h2 = screen.getByRole("heading", { level: 2 });
    expect(h2).toHaveTextContent("📅 5. Earnings Context");
    expect(h2.textContent).not.toContain("#");
  });

  it("renders a GFM pipe table as an aligned <table> (thead + tbody cells)", () => {
    const md = ["| Metric | Value |", "| --- | ---: |", "| P(up) | 58% |", "| Exp. return | +2.3% |"].join(
      "\n",
    );
    render(<Markdown>{md}</Markdown>);
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "Metric" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "58%" })).toBeInTheDocument();
    // GFM right-alignment on the Value column arrives as an inline style on the cell.
    const valueCell = within(table).getByRole("cell", { name: "58%" });
    expect(valueCell).toHaveStyle({ textAlign: "right" });
  });

  it("emphasizes **bold** as <strong> and *italic* as <em>", () => {
    render(<Markdown>{"A **firm** and *soft* word."}</Markdown>);
    expect(screen.getByText("firm").tagName).toBe("STRONG");
    expect(screen.getByText("soft").tagName).toBe("EM");
  });

  it("renders inline code and opens links in a new tab", () => {
    render(<Markdown>{"See `NVDA` at [SEC](https://sec.gov/x)."}</Markdown>);
    expect(screen.getByText("NVDA").tagName).toBe("CODE");
    const link = screen.getByRole("link", { name: "SEC" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(link).toHaveAttribute("href", "https://sec.gov/x");
  });

  it("does NOT render raw HTML embedded in the source (injection guard)", () => {
    render(<Markdown>{'before <img src=x onerror="alert(1)"> after'}</Markdown>);
    // react-markdown (no rehype-raw) escapes raw HTML — no <img> element is created.
    expect(document.querySelector("img")).toBeNull();
  });
});
