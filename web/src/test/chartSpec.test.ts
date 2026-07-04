// @vitest-environment node
import { describe, expect, it } from "vitest";
import { chartSpecToVegaLite, toRecords, type ChartSpecDict } from "@/lib/chartSpec";
import { hueName } from "@/lib/events";
import { FORECAST_CHART } from "./fixtures";

// Minimal cast helper: the returned VisualizationSpec is a union; tests inspect concrete fields.
type Rec = Record<string, unknown>;
const asRec = (v: unknown): Rec => v as Rec;

describe("toRecords", () => {
  it("transposes column-oriented data to row records", () => {
    expect(toRecords({ a: [1, 2], b: ["x", "y"] })).toEqual([
      { a: 1, b: "x" },
      { a: 2, b: "y" },
    ]);
  });
});

describe("chartSpecToVegaLite", () => {
  it("builds a single-series brass bar with percent axis + explicit x sort", () => {
    const vl = asRec(chartSpecToVegaLite(FORECAST_CHART));
    expect(asRec(vl.mark).type).toBe("bar");
    expect(asRec(vl.mark).color).toBe("#E8A13A"); // MARK_COLOR (--sa-accent dark)
    const enc = asRec(vl.encoding);
    expect(asRec(enc.x).sort).toEqual(["down", "flat", "up"]);
    expect(asRec(asRec(enc.y).axis).format).toBe("%");
    expect(asRec(vl.data).values).toHaveLength(3);
  });

  it("colors a bar by tool-driven direction (up=green, down=red, neutral=brass)", () => {
    const spec: ChartSpecDict = {
      title: "Large-move breakdown",
      kind: "bar",
      x: "outcome",
      y: "prob",
      caption: "",
      color: null,
      x_sort: ["Big up", "No big move", "Big down"],
      y_is_percent: true,
      direction: "direction",
      data: {
        outcome: ["Big up", "No big move", "Big down"],
        prob: [0.2, 0.65, 0.15],
        direction: ["up", "neutral", "down"],
      },
    };
    const vl = asRec(chartSpecToVegaLite(spec));
    // Direction drives the fill via a color encoding (not a flat mark color) — twin of to_altair.
    expect(vl.mark).toBe("bar");
    const enc = asRec(vl.encoding);
    expect(asRec(enc.color).field).toBe("direction");
    const scale = asRec(asRec(enc.color).scale);
    expect(scale.domain).toEqual(["up", "down", "neutral"]);
    expect(scale.range).toEqual(["#3FB950", "#E5534B", "#E8A13A"]); // up / down / brass
  });

  it("builds a grouped bar with color scale + xOffset when a color field is set", () => {
    const spec: ChartSpecDict = {
      title: "By model",
      kind: "grouped_bar",
      x: "bucket",
      y: "prob",
      caption: "",
      color: "model",
      x_sort: null,
      y_is_percent: true,
      data: { bucket: ["up", "up"], prob: [0.6, 0.5], model: ["gbm", "ewma"] },
    };
    const vl = asRec(chartSpecToVegaLite(spec));
    const enc = asRec(vl.encoding);
    expect(asRec(enc.color).field).toBe("model");
    expect(asRec(asRec(enc.color).scale).range).toContain("#4FA8E8"); // CATEGORY_PALETTE sky
    expect(asRec(enc.xOffset).field).toBe("model");
  });

  it("builds a layered reliability diagram (dashed y=x ideal + points)", () => {
    const spec: ChartSpecDict = {
      title: "Calibration",
      kind: "reliability",
      x: "predicted",
      y: "realized",
      caption: "",
      color: null,
      x_sort: null,
      y_is_percent: false,
      data: { predicted: [0.2, 0.8], realized: [0.3, 0.7] },
    };
    const vl = asRec(chartSpecToVegaLite(spec));
    const layers = vl.layer as Rec[];
    expect(layers).toHaveLength(2);
    expect(asRec(layers[0].mark).strokeDash).toEqual([5, 5]); // y=x reference
    expect(asRec(layers[1].mark).type).toBe("circle"); // points
  });
});

describe("hueName parity with ui.html.tool_hue", () => {
  // crc32(tool) values computed from Python (zlib.crc32); hueName must pick the same hue as tool_hue.
  it.each([
    ["run_forecast", 296368892, "indigo"],
    ["get_price_summary", 2386928240, "teal"],
    ["get_news_sentiment", 3105863143, "violet"],
  ])("%s → %s", (_tool, crc, expected) => {
    expect(hueName(crc as number)).toBe(expected);
  });
});
