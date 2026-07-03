# APP_REDESIGN.md — Chat Agent UI Redesign Plan

**Status:** Approved direction — ready to implement, phase by phase.
**Owner:** truongbnguyen88
**Scope:** Visual + UX redesign of the Streamlit chat frontend (`ui/chat_app.py`). **View layer only** — no changes to `agent/`, `router`, `runtime`, tools, forecasting, or RAG.
**Design reference:** interactive mockup Artifact `colorful-brass-mockup` (dark/light + conversation/empty-state toggles), published 2026-07-02. That mockup is the visual target this plan implements.
**Last updated:** 2026-07-02

This plan carries **two co-equal pillars**, weighted the same at every phase:
- **LOOK** — the confirmed *brass-on-ink, colorful-accent* visual identity (§2, §5).
- **FRAMEWORK/ENGINEERING** — how it's built: framework choice, component architecture, preservation of content, tests (§3, §4, §6).

Every phase in §8 has a **LOOK** deliverable *and* an **ENGINEERING + TESTS** deliverable, and is not done until both are satisfied and `make check` is green.

---

## 1. Objective

Make the chat agent look and feel like a polished, modern **quant-research instrument** — without changing what it does or removing any content. Concretely:

- Ship the confirmed **design system** (tokens, typography, color, motion) — §2, §5.
- **Refactor** the 576-line monolith into a maintainable `ui/` component package — §4.
- Polish every **surface** (sidebar, hero, chat, trace, charts, sources, export, input) — §7.
- Preserve **100% of current content and behavior** — §6 is the hard contract.

Non-goals this cycle: new agent capabilities; a recommendation field (forbidden by invariant); framework rewrite (React is documented as optional Phase 2, §9).

---

## 2. LOOK — confirmed visual identity

**Thesis:** a *quiet quant-research instrument*, not a consumer-fintech app. Precision, density, calm. The subject's world — SEC filings, probabilistic forecasts, quantitative rigor — drives every choice.

**Palette — "brass on ink":** cool near-black slate ground, hairline borders, one restrained **brass/amber primary accent** for active state and signals. A **secondary hue family** (teal / sky / indigo / violet / rose) is applied **semantically** — one hue per capability, per metric-tile category, per tool in the trace — so the UI reads livelier without becoming a rainbow. Chart **up/down green-red is reserved for marks only**, never UI chrome (a correctness-of-signaling rule).

**Typography — mono as structure:** system-sans for prose; **monospace for every label, ticker, metric, and eyebrow** with `tabular-nums`. Mono-as-label is the terminal vernacular and does the characterful work — **no webfont** (Artifact/Streamlit CSP + silent-fallback risk; system mono/sans stacks only).

**Motion:** minimal and purposeful — typewriter on the empty state, subtle hover elevation on cards, focus-within accent on the composer. All gated by `prefers-reduced-motion`.

**Anti-generic guardrails:** avoid the AI-default looks — cream+serif+terracotta, lone acid-green pop, purple→blue gradient hero, Inter/Space-Grotesk-as-safe-default, emoji section markers, rounded-lg everywhere. Spend boldness in the brass accent + semantic hues; keep everything else quiet.

**Design principles (apply to every phase):**
1. Content first, chrome second — chrome makes the analysis legible, never competes with it.
2. One design language — all color/space/radius/type from the token set; no ad-hoc hex in views.
3. Theme-aware — dark primary, first-class light; both AA-contrast.
4. Color carries meaning — semantic hue assignment, not decoration.
5. Accessible — AA contrast both themes, visible focus, `prefers-reduced-motion`.
6. Graceful degradation — if custom CSS fails (version drift), the app stays fully usable.
7. Leave it more maintainable than found — small modules, one CSS source of truth.

---

## 3. FRAMEWORK decision (co-equal with LOOK)

**Decision: build the redesign in Streamlit now (Phase 1); keep React+FastAPI as an optional Phase 2.**

| | **Restyle Streamlit (chosen, Phase 1)** | React + FastAPI (Phase 2, optional) |
|---|---|---|
| Effort | ~R0–R6, incremental | Multi-week rewrite |
| Look ceiling | Reaches the mockup's ~85% (proven achievable) | Unbounded |
| Risk to content | Low (all plumbing preserved) | High (re-implement state/persistence/streaming/export) |
| Fits global rules | Yes (incremental, no big rewrite) | Conflicts unless genuinely needed |

**Reachability (honest):** the mockup is hand-built HTML showing the target. In Streamlit + CSS we can reach the palette, mono typography, colored capability cards, metric tiles, chip rows, styled sidebar & chat bubbles, sources/export. The last ~15% — top-bar context chips, `Export ▾` popover, live per-tool trace during a run — uses `st.popover` (≥1.31) / `st.status`, or is approximated. Each such item is flagged in the phase that touches it. **If, after R0–R6, you want polish beyond that ceiling (bespoke animation, token streaming), that's the trigger for Phase 2** — and the §4 refactor is exactly what makes that port incremental rather than a rewrite.

**Assumptions:** `streamlit>=1.35` pinned; theming + `st.markdown(unsafe_allow_html=True)` + `components.html` are the styling levers; CSS-against-internal-DOM is version-fragile, so it's centralized in one module with graceful degradation. All quantitative content still comes from tools; invariants unchanged.

---

## 4. ENGINEERING — component architecture

Split the monolith into a thin entrypoint + a `ui/` component package. **Behavior-preserving refactor** (move code, don't rewrite logic). This is the maintainability unlock *and* the thing that de-risks a future React port (the data contracts below become the API a React app would consume).

**Split by purity** — the harness only type-checks/tests `src` (mypy `files=["src","tests"]`, pytest `pythonpath=["src"]`), so anything worth a gate lives under `src/stock_agent/ui/`; Streamlit-coupled view code lives in repo-root `ui/` (same ungated status as `chat_app.py` today, verified by `py_compile` + import-check).

```
src/stock_agent/ui/          # PURE — typed, strict-checked, unit-tested (no Streamlit import)
  capabilities.py            # (existing) showcase data
  theme.py                   # R0: design tokens + base CSS (theme_style_tag)
  state.py                   # R1: serialize/deserialize + sources_from_tool_results
  routing.py                 # R1: RoutingChoice dataclass + chat_input_placeholder
  # R5: chart Altair-theme helper (pure spec transform) lands here too

ui/                          # VIEW — Streamlit-coupled, import-checked (not gate-covered)
  chat_app.py                # thin: page config, load_resources, wire components, turn loop
  session.py                 # R1: session-state init, thread save/open/new/clear
  components/
    __init__.py
    sidebar.py               # R1 render_sidebar(*, settings, store) -> RoutingChoice
    hero.py                  # R1 typewriter + capability cards (R3 restyles)
    message.py               # R1 chart / sources / export (R4 adds tiles + trace)
    inputs.py                # R5: composer + context chips
```

- Pure data stays put: `viz/charts.py`, `chat/history.py`, `reports/export.py` — unchanged.
- View modules import from `stock_agent.*` (+ sibling `session`); **dependency direction unchanged** (UI → package, never inverted); component graph is a tree (no sibling-component imports).
- **Stable data contracts** (the React-port API surface): `RoutingChoice`, `ChartSpec` (already JSON-serializable), the sources list, the export bytes function — all pure, in `src`.

---

## 5. Design tokens & helpers — `ui/theme.py`

Single source of truth. Emits one `<style>` block injected once at app top; light overrides under a `prefers-color-scheme` / Streamlit-dark-class block. Values lifted from the approved mockup.

### 5.1 `.streamlit/config.toml` (new)
`[theme]` base=dark, `primaryColor="#E8A13A"`, `backgroundColor="#0E1116"`, `secondaryBackgroundColor="#161B22"`, `textColor="#E6EAF0"`, mono/sans font pair. Document per-key minimum Streamlit version; keep to widely-supported keys.

### 5.2 Color tokens

| Role | Token | Dark | Light |
|---|---|---|---|
| Ground | `--sa-bg` | `#0E1116` | `#F5F6F8` |
| Surface | `--sa-surface` | `#161B22` | `#FFFFFF` |
| Raised | `--sa-surface-2/3` | `#1C222B` / `#222A35` | `#FBFBFC` / `#F1F3F6` |
| Border | `--sa-border` / `-strong` | `#262D38` / `#323B48` | `#E4E7EC` / `#D3D8DF` |
| Text | `--sa-text` | `#E6EAF0` | `#1A1F27` |
| Muted / faint | `--sa-muted` / `--sa-faint` | `#8A93A3` / `#616B7A` | `#5B6472` / `#8A93A3` |
| **Accent (brass)** | `--sa-accent` (+`-weak`,`-line`) | `#E8A13A` | `#B7791F` |
| Secondary hues | `--sa-teal / sky / indigo / violet / rose` | `#35B0A7 / #4FA8E8 / #7C82F0 / #B38BEA / #E5709B` | `#0E9384 / #2E7FC4 / #5B60D6 / #8B54C9 / #C6417A` |
| Chart up / down | `--sa-up` / `--sa-down` | `#3FB950` / `#E5534B` | `#1A7F37` / `#C4392F` |
| Grid | `--sa-grid` | `rgba(138,147,163,.14)` | `rgba(91,100,114,.14)` |

**Color-usage rules (enforced in review):** brass = primary/active only; secondary hues carry meaning (capability identity, tile category, per-tool trace); up/down = chart marks only; every hue passes AA on its surface in both themes.

### 5.3 Type / spacing / shape
- **Type:** sans stack (`-apple-system, "Segoe UI", Inter, system-ui`) for prose; mono stack (`ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas`) for labels/data. Uppercase mono labels at `letter-spacing:.09em`. Scale: 11/12/13/15/19/26; body 15, line-height 1.55; headings `text-wrap: balance`.
- **Spacing:** `--sa-space-1..6` = 4/8/12/16/24/32.
- **Shape:** `--sa-r`=10px, `--sa-r-sm`=7px; one soft shadow token.

### 5.4 Helper functions (return HTML strings so views compose consistently)
`inject_theme()`, `chip(text, tone)`, `status_dot(state)`, `stat_tile(label, value, tone, sub)`, `cap_card(cap, hue)`, `trace_bar(tools)`. Plus an **Altair chart theme** applied in the render path (mono font, faint grid, semantic up/down, percent-axis honoring `y_is_percent`, colorblind-aware categorical palette per the `dataviz` skill) — **no change to `ChartSpec`**.

---

## 6. Content & behavior preservation checklist (hard contract)

Must work identically after **every** phase. Existing tests (`test_capabilities`, `test_chat_history`, `test_viz_charts`, `test_viz_render`, `test_router`) stay green. **Walked + verified at R6 (2026-07-03)** — evidence cited per item; R1 was a behavior-preserving move-only refactor, so the named tests remain the contract.

- [x] Sidebar: corpus/embedder status badge (`corpus_status`) + freshness. — `ui/components/sidebar.py` + `rag.status.corpus_status`; `test_ui_html::status_card`, `corpus_freshness`.
- [x] Sidebar: 4 quick-starter buttons inject `pending_prompt`. — `sidebar.py`; verified in `chat_app.py` pending-prompt plumbing (unchanged since R1).
- [x] Sidebar: ticker input feeds quick-starters, hero, deterministic routes. — single `ticker` session key read by `hero.render_capability_hero` + routing.
- [x] Sidebar: routing mode = Auto + all `DOMAIN_NAMES`; variant selector; horizon/lookback params only for domains that use them. — `test_ui_routing` (`context_chips` param surfacing); `router.ROUTES` needs-map.
- [x] Sidebar: chat-thread list, active highlight, open, delete, `New chat`, retention caption. — `chat.history.ChatStore`; `test_chat_history` (round-trip/delete).
- [x] Sidebar: API-key status (Anthropic required + optional providers). — `html.keys_row`; `test_ui_html`.
- [x] Hero: typewriter (honors `prefers-reduced-motion`); 11 capability cards; card click → `pending_prompt` → submit. — `hero.py` iframe guards `prefers-reduced-motion`; `test_capabilities`; theme reduced-motion rule (`test_ui_theme::reduced_motion_and_responsive`).
- [x] Chat: user/assistant markdown; per-turn **Altair charts** re-render after reruns (persisted via `ChartSpec.to_dict`). — `test_ui_state` (charts round-trip), `test_viz_render`.
- [x] Chat: **SEC filing sources** expander (citations from tool output only). — `message.render_sources`; `test_ui_state` (sources round-trip).
- [x] Chat: **Export** PDF / Word / Markdown per answer (with chart PNGs; cached). — `reports.export.export_summary`; `test_export` (3-format round-trip).
- [x] Chat: "tools used" trace (Auto) + "deterministic route" caption (domain). — `html.trace_bar`/`tool_hue`; `test_ui_html`.
- [x] Auto path (`run_agent`) + deterministic path (`router.run`) intact, incl. `AgentGroundingError` / `AgentError` / `RouterError` messages. — `test_router`, `test_agent_runtime` (untouched by the view-layer redesign).
- [x] Bounce-to-Auto fallback (`fallback_prompt` → `force_auto_prompt`). — `chat_app.py` fallback path (unchanged since R1).
- [x] Thread persistence across restarts (`ChatStore`; serialize/deserialize incl. charts + sources). — `test_chat_history` + `test_ui_state` serialize/deserialize.
- [x] Non-advisory disclaimer present; **no recommendation field** added. — schema carries no recommendation field (invariant); disclaimer rendered in `chat_app.py`.
- [x] Invariants intact: numbers-from-tools, citations-from-tool-output, dependency direction downward. — tiles/charts derive from tool results (`test_ui_tiles`, `test_viz_charts`); `ui/` pure layer has no Streamlit import (import-gated).

---

## 7. Per-surface spec (LOOK detail)

- **Global shell:** theme injected once; slim top bar with product mark, persistent non-advisory chip, context chips (active ticker + mode), theme toggle. Main content max-width for readability.
- **Sidebar:** brand block; **status card** with `status_dot` (fresh/stale/unavailable from `corpus_status`); grouped sections (routing / quick starters / chats / keys); chat list with accent active-bar, hover, truncation, trailing delete; **keys as chip row**; quick-starter icons carry semantic hues.
- **Hero:** theme-aware typewriter (brass accent word); **colored capability cards** (one hue each, icon badge, equal heights, hover elevation, responsive 2→1 col); colored eyebrow.
- **Chat message:** user/assistant avatars + bubbles; **stat tiles first** (summary-before-detail, colored category stripe); chart card; **tool-trace chip row** (per-tool color) / direct-route chip; **styled sources** (marker badge + label); **`Export ▾`** single affordance (same 3 formats); Auto-turn loading via `st.status` step list.
- **Charts:** shared Altair theme (semantic up/down, faint grid, percent axes, mono labels); consistent height; legible in both themes.
- **Input:** styled composer with focus-within accent; **context chips** ("on enter: NVDA · Auto") above it surfacing the existing placeholder logic.

---

## 8. Phased task breakdown — each phase = LOOK + ENGINEERING + TESTS

Independently mergeable; app works after each. Run `make check`; keep §6 green.

### Phase R0 — Foundation: theme + tokens — ✅ DONE (2026-07-02, branch `feat/ui-redesign`)
- **LOOK:** app adopts brass-on-ink palette + mono typography; nothing else changes yet.
- **ENGINEERING:** add `.streamlit/config.toml` (§5.1); create `src/stock_agent/ui/theme.py` with the full token block + `theme_style_tag()` (dark + light); inject it at the top of `chat_app.py` via `st.markdown(..., unsafe_allow_html=True)`.
- **TESTS:** `tests/unit/test_ui_theme.py` (8 tests) — CSS builder returns a wrapped `<style>` block containing all `--sa-*` tokens in **both** dark and light themes (parity), balanced braces, deterministic/pure output, brass accent in both themes, system-only font stacks (no webfont). `make check` green (ruff, mypy strict/278 files, check-math, 862 passed / 3 pre-existing skips).
- **Scope note:** theme lives in `src/stock_agent/ui/theme.py` (not top-level `ui/`) so it is importable, strict-typed, and unit-tested (tests use `pythonpath=["src"]`); it holds **no Streamlit import** (purity). HTML helper builders (`chip`/`status_dot`/…) deferred to the phases that consume them (R2+) to avoid dead code.

### Phase R1 — Component refactor (behavior-preserving) — ✅ DONE (2026-07-02, branch `feat/ui-redesign`)
- **LOOK:** none (identical render) — this is the architecture phase.
- **ENGINEERING:** split by **purity** (harness reality: mypy + pytest cover only `src`, so pure logic must live there to be gate-covered):
  - Pure → `src/stock_agent/ui/`: `state.py` (`serialize_messages`, `deserialize_messages`, `sources_from_tool_results`) + `routing.py` (`RoutingChoice` dataclass + `chat_input_placeholder`). Typed, strict, unit-tested.
  - Streamlit view → repo-root `ui/` (verbatim moves; import-checked, not gate-covered — same status as `chat_app.py`): `session.py` (state + thread persistence), `components/{message,hero,sidebar}.py`. `sidebar.render_sidebar(*, settings, store) -> RoutingChoice`.
  - `chat_app.py` slimmed **585 → 243 lines** (orchestration + turn loop only).
  - **Deferred:** `inputs.py` → R5 (the input bar is one `st.chat_input`; its placeholder logic is already in the tested `routing.py`).
- **TESTS:** `tests/unit/test_ui_state.py` (5) + `tests/unit/test_ui_routing.py` (5) — serialize/deserialize round-trip rebuilds `ChartSpec`, empty charts/sources omitted, citation dedup+order, `is_auto` + all placeholder branches. View layer verified via `py_compile` + import-check in the app's `sys.path`. `make check` green (ruff, mypy strict/282 files, check-math, **872 passed / 3 skips**). **Zero behavior diff** (moved verbatim; §6 contract intact).

### Phase R2 — Sidebar — ✅ DONE (2026-07-02)
- **LOOK:** brand block (mark + mono name + non-advisory tagline); corpus **status card** with `status_dot` (green fresh / brass stale / red unavailable + soft halo); mono uppercase **section eyebrows** replace headings/dividers; chat list with **brass active-chat** (native `type="primary"`, no fragile per-key CSS) + hover nudge + trailing delete; **keys chip row** (✓ present / ✕ required-missing / · optional); subtle sidebar-button hover elevation.
- **ENGINEERING:** new **pure** module `src/stock_agent/ui/html.py` (typed, tested, no Streamlit) — `brand_block`, `eyebrow`, `status_dot`, `corpus_freshness` (impurity isolated via injected `today`; 120-day = quarterly window), `status_card`, `chip`, `keys_row`; component CSS added to `theme.py` (`_COMPONENTS`, appended in `theme_style_tag()`, one source of truth for the classes). `components/sidebar.py` rewritten to **compose** those builders via `st.markdown(unsafe_allow_html=True)` for static parts; **all interactive widgets + handlers unchanged** (`pending_prompt`, ticker, routing, open/delete/new). §6 sidebar items all preserved (disclaimer now in brand tagline; embedder/chunks/tickers/freshness in the status card; same 4 keys).
- **TESTS:** `tests/unit/test_ui_html.py` (14) — freshness boundary/unavailable/bad-date, dot states + fallback, card chunk-format & coverage, chip tone/marker/escaping fallback, keys-row state encoding, brand/eyebrow structure. Theme tests still green (component block keeps braces balanced, no `url(`). `make check` green (ruff, mypy strict/284, check-math, **887 passed / 3 skips**). **View render (classed HTML picking up the injected `<style>`) needs a browser eyeball** — degrades gracefully to unstyled-but-legible (principle #6).

### Phase R3 — Empty-state hero — ✅ DONE (2026-07-02)
- **LOOK:** typewriter recolored **from tokens** (no hardcoded hex) with a **brass accent** typed word + caret, dark/light via the iframe's own `prefers-color-scheme`; each capability is a **hue-coded card** (`cap_card` — hue top-rail + `color-mix` tinted icon badge + title + blurb, one of 5 semantic hues cycled by index) with hover lift; **brass "colored eyebrow"** on the section.
- **ENGINEERING:** pure additions in `src/stock_agent/ui/html.py` — `cap_card(*, icon, title, blurb, hue)`, `cap_hue(index)` (cycles teal→sky→indigo→violet→rose, brass fallback), `eyebrow(text, *, tone)` (accent variant); `theme.iframe_colors()` exposes the dark/light hex for text/muted/accent so the **sandboxed iframe** (can't inherit `--sa-*`) stays synced to one source; card + accent-eyebrow CSS in `_COMPONENTS`. `components/hero.py` restyled: typewriter substitutes token hex; cards render `cap_card` HTML **+ a sibling `st.button`** (injected HTML can't fire a Streamlit callback), click → `pending_prompt` + rerun **unchanged** from R1.
- **TESTS:** `test_ui_html.py` +6 (hue cycle+wrap, card parts/hue-var/accent-fallback/escaping, every-capability well-formed, eyebrow accent tone) and `test_ui_theme.py` +1 (`iframe_colors` both-theme parity + values are real token hex, no drift). Verified no placeholder survives the iframe substitution. `make check` green (ruff, mypy strict/284, check-math, **894 passed / 3 skips**).
- **Known limits (→ R6):** Streamlit `st.columns` don't reflow 2→1 on narrow viewports (cards fill their column + min-height for now); classed-HTML render (hue rail/badge picking up the injected CSS) needs a **browser eyeball** — degrades to unstyled-but-legible.

### Phase R4 — Chat message: tiles + trace + sources + export — ✅ DONE (2026-07-03)
- **LOOK:** **stat tiles first** (summary-before-detail): headline metric tiles with a colored category stripe (`stat_tile` — mono uppercase label + `tabular-nums` value + faint sub), rendered above the answer in a 3-up grid. **Per-tool-color trace chip row** on Auto turns (`trace_bar` — one hue-tinted pill per tool, hue stable per tool name via `crc32`); deterministic turns get a single brass **direct-route chip** ("no LLM routing call"). **Styled sources** (mono marker badge + label). Single **`Export ▾`** popover affordance (same 3 formats). Auto turns run inside an **`st.status`** step surface (done/error summary; a live per-tool trace needs an event-emitting `run_agent` → Phase 2).
- **ENGINEERING:** new **pure** module `src/stock_agent/ui/tiles.py` (typed, tested, no Streamlit) — `stat_tiles_from_tool_results` dispatches by tool **name** (like `charts_for`), recognizes `get_price_summary` / `run_forecast` / `get_large_move` / `get_news_sentiment`, formats numbers + assigns a semantic hue, skips error results, dedups by label, caps at 6. Tiles are display-ready plain dicts → **persisted with the turn** (`ui.state` serialize/deserialize round-trips `tiles`, empty-omitted, back-compat) so history re-renders them after a rerun/restart, exactly like charts. Pure additions in `ui/html.py`: `stat_tile`, `tool_hue`, `trace_bar` (+ `_hue_token` shared with `cap_card`); tile/trace/source CSS in `theme._COMPONENTS`. `components/message.py` gains `render_stat_tiles` / `render_trace`, restyled `render_sources`, and `render_export` via `st.popover` (**graceful fallback to the pre-R4 expander** when `st.popover` is absent). `chat_app.py` computes tiles + tool names on both paths, wires them into the history loop and live turn, and swaps the Auto spinner for `st.status`. **All numbers still come from tool output** (numbers-vs-narrative invariant); the trace chip row is live-turn-only (matches the pre-R4 ephemeral "tools used" caption).
- **TESTS:** `tests/unit/test_ui_tiles.py` (11, **new**) — per-tool golden shapes (currency/percent/signed formatting), error-skip, unknown-tool → none, missing-scalar → none, dedup first-seen, cap, empty. `test_ui_html.py` +8 (`stat_tile` parts/hue/empty-sub/accent-fallback/escaping; `tool_hue` determinism+family; `trace_bar` dedup/order/empty/escaping). `test_ui_state.py` +1 (tiles round-trip, empty-omitted, missing-key tolerant). Existing `test_export.py` covers the 3-format export round-trip; theme/router/viz tests still green. `make check` green (ruff, mypy strict/286, check-math, **914 passed / 3 skips**). View render (classed tile/trace HTML picking up the injected `<style>`) needs a browser eyeball — degrades to unstyled-but-legible (principle #6).

### Phase R5 — Charts + input bar — ✅ DONE (2026-07-03)
- **LOOK:** shared **brass-on-ink Altair theme** — single-series bars in the brass accent, grouped series in the semantic secondary palette (colorblind-ordered sky→rose→… so the common 2-series case avoids a red-green pair), **faint gridlines**, **mono** axis/legend/title labels, percent axes preserved; reliability points in sky with a muted y=x reference line. **Composer context chips** ("on enter: NVDA · Auto") above the input surfacing the active ticker + routing mode + horizon/lookback, plus a **focus-within brass accent** on the chat input.
- **ENGINEERING:** new **pure** module `src/stock_agent/ui/chart_theme.py` (typed, tested, **no Altair import**) — returns a Vega-Lite `config` dict + color constants **derived from the design tokens** (`theme.dark_token` / `theme.mono_font_stack`, two small accessors added) so the chart palette can never drift from the CSS (§5.2). `viz/render.py` applies it via `chart.configure(**altair_config())` and sets mark colors at the **mark/encoding level** (so they win Streamlit's Vega-theme merge in-app; the config's category range carries the palette to the PNG export too) — **`ChartSpec` unchanged**. Text *colors* are deliberately left unset so Streamlit's adaptive theme (in-app, AA both themes) / Vega's default (white export page) supply them. Chip logic added to the pure layer: `routing.context_chips` (ticker + mode + params → `(label, tone)` pairs) + `html.context_row` (markup); new thin view `ui/components/inputs.py` (`render_chat_input`) renders the chip row then `st.chat_input` — `chat_app.py` calls it. **Chips sit at the end of the content flow just above the bottom-docked composer** (the acknowledged Streamlit approximation of "attached" chips — §3 "last ~15%"); focus-within accent is CSS in `theme._COMPONENTS` (graceful-degrade enhancement). Up/down green-red stays reserved for direction marks (current specs carry no direction flag, by design — the LLM/heuristic must not assign financial semantics).
- **TESTS:** `tests/unit/test_ui_chart_theme.py` (7, **new**) — config sections/determinism, mono fonts (no webfont), text-colors-unset (adaptive), **palette-derives-from-tokens drift guard**, palette well-formed/distinct, no red-green in the first two series. `test_viz_render.py` +4 (brass single bar, percent axis survives theme, config applied, grouped semantic palette). `test_ui_routing.py` +5 (`context_chips` across Auto/named-domain/variant/params/missing-ticker + all tones renderable by `html.chip`). `test_ui_html.py` +3 (`context_row` lead+pills, empty→"", escaping). View layer verified via import-check (`components.inputs`) + `chat_app.py` compile. `make check` green (ruff, mypy strict/288, check-math, **933 passed / 3 skips**). Chart look in-app (Streamlit Vega-theme merge) + focus-within accent need a **browser eyeball**; degrade to legible/native (principle #6).

### Phase R6 — Polish, a11y, docs — ✅ DONE (dark-only) (2026-07-03); light-mode items → Phase 2
Scoped to **dark-only** after the light-mode decision (see R6.B below): the "both themes" audit and before/after screenshots are inherently blocked on light mode, which now lands in Phase 2 (React), so they move there rather than being built against a compromised Streamlit approximation.
- **LOOK:** ✅ **responsive** — column *count* delegated to Streamlit `st.columns` (native reflow) + every custom row is `flex-wrap: wrap`; a narrow-viewport guard shrinks the mono tile value so it can't overflow if 3 tiles stay side-by-side on a phone (`@media (max-width: 640px)`, no fragile column-container selector per §10). ✅ **reduced-motion** — typewriter iframe + `.sa-cap` hover honor `prefers-reduced-motion: reduce`. ⏭️ **contrast/focus audit _both_ themes** → Phase 2 (needs light mode; dark-only contrast held to AA by the token palette in §5.2).
- **ENGINEERING:** ✅ final §6 walkthrough (all 16 boxes ticked with per-item evidence above). ✅ `README.md` launch note added (app is dark-only until Phase 2). ✅ `make check` green. ⏭️ before/after screenshots (light + dark) → Phase 2 with light mode.
- **TESTS:** ✅ full suite green; new `test_ui_theme::test_reduced_motion_and_responsive_rules_present` pins the a11y/responsive CSS. a11y checklist recorded below.

**a11y checklist (dark-only, 2026-07-03):**
- **Motion:** all transitions/animations opt-out via `prefers-reduced-motion` (typewriter falls back to a static first line; card hover nudges disabled). ✅
- **Semantics:** headings/markdown from Streamlit native elements; sources use `<details>`/native expander; buttons are real `st.button`. ✅
- **Labels:** icon-only affordances carry text (ticker `aria-label`, chat input placeholder, key chips labeled). ✅
- **Contrast (dark):** text/muted/faint + semantic hues chosen to meet AA on the ink ground (§5.2 tokens); chart up/down red-green reserved for marks only, never as sole signal (§2). ✅
- **Focus:** relies on native Streamlit focus rings (not suppressed); composer adds a *supplementary* focus-within accent, not a replacement. ✅
- **Deferred to Phase 2:** AA re-audit in **light** mode; keyboard-only walkthrough of the thread list + export popover with a screen reader; visible focus-ring verification in both themes.

#### R6 render-check findings (2026-07-03, headless Chrome vs. live app on Streamlit 1.58)
The redesign CSS **is** applying (all `data-testid`/`sa-*` selectors resolve; injected `<style>` present; mono fonts computed; dark tokens active). The "look barely changes" perception is because the empty state was already card-heavy — the redesign's payload (stat tiles, per-tool trace, styled sources, themed charts, context chips) lives on the **conversation surface**, which the empty state never renders. One real bug found + fixed, one deferred:

- **R6.A — theme-signal mismatch — ✅ DONE (2026-07-03).** Our `--sa-*` tokens switched on the browser/OS `prefers-color-scheme`, but Streamlit's chrome is pinned dark by `config.toml base=dark`, so a **light-preference viewer got dark chrome + light component tokens** (white cards on the dark ground; dark-on-dark brand text — confirmed via painted colors: app bg `rgb(14,17,22)` vs. card `rgb(255,255,255)`). Fix: `theme_style_tag()` emits **dark-only** (drop the `prefers-color-scheme: light` override); the hero typewriter iframe likewise goes dark-only. `_LIGHT` / `iframe_colors()`'s light half retained as the R6.B source. Regression guard: `test_ui_theme.test_dark_only_no_light_media`. `make check` green (933 passed / 3 skips).
- **R6.B — first-class light mode → moved to Phase 2 (React), 2026-07-03.** Investigated in Streamlit and **deliberately not built there**: Streamlit **cannot switch its own chrome theme from Python** (`config.toml` is static; the only runtime switch is the user's native Settings-menu choice), and the sole read signal `st.context.theme.type` is **laggy at the switch moment** (inferred from background color; "may be incorrect during a change in theme" — Streamlit issue #11920). So a Streamlit light mode is necessarily a Settings-menu-driven, one-rerun-laggy *approximation* that re-introduces a transient version of the R6.A desync — and Phase 2 React does it properly (instant, lag-free, real in-app `◐` toggle). Rather than build throwaway work Phase 2 obsoletes, **the app stays intentionally dark-only** and real light mode is a Phase-2 deliverable (see [PHASE2_REACT_FASTAPI_PLAN.md](PHASE2_REACT_FASTAPI_PLAN.md) §6 token bridge; the retained `_LIGHT` / `iframe_colors()` light palette is its source). The "both themes" contrast audit + before/after screenshots (R6 LOOK/ENGINEERING) move with it.

---

## 9. Optional Phase 2 (planned): React + FastAPI

**Plan of record → [PHASE2_REACT_FASTAPI_PLAN.md](PHASE2_REACT_FASTAPI_PLAN.md)** (ordered slices P2.0–P2.6, streaming event schema, API surface, tests). Summary below; that doc governs.

Trigger: the four interactions above Streamlit's ceiling after R0–R6 — live per-tool trace, token streaming, instant client-side theme toggle, `Export ▾` popover + top-bar chips.
- **Decisions (locked):** coexist with Streamlit (React primary, Streamlit as reference/fallback until parity); full streaming (live trace + tokens); local-dev only (no auth/containers); Vite + React + TS + Tailwind + shadcn/ui, Vega-Lite via `react-vega`.
- **Backend:** new top-level `api/` (FastAPI); SSE streams `AgentEvent`s from a new event-emitting `run_agent_events` generator (`run_agent` becomes its drain — behavior-preserving). Reuse `ChatStore`, `export_summary`, `charts_for`, `tiles_for`, `corpus_status` as-is. Grounding guard still runs before the terminal `final` event.
- **Why cheap:** the §4 contracts (`RoutingChoice`, `ChartSpec`, sources, tiles, export) are already the API; the only real new code is the streaming emission + the React app.
- **Cost:** streaming refactor + adapter, React app, thread/export wiring, token bridge, two-stack maintenance until Streamlit is retired.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Streamlit CSS selectors break on version bump | Centralize CSS in `ui/theme.py`; token/variable approach + stable containers; graceful degradation; pin streamlit minor. |
| Refactor silently changes behavior | R1 is move-only; §6 + existing tests are the contract; one phase at a time. |
| `components.html` iframe theming drifts from app theme | Single source for the two hex pairs; keep iframe `prefers-color-scheme` synced to tokens. |
| Color misread (semantic hues / red-green) | AA-checked hues; up/down only on marks; dataviz colorblind-aware categorical palette. |
| Scope creep into agent logic | Hard rule: view layer only; no edits under `src/stock_agent/agent/`, `router`, `runtime`, tools. |
| Export/persistence regressions | Keep cached fns + serialize/deserialize untouched in R1; test thread round-trip. |

---

## 11. Definition of done (per phase and overall)

- Both pillars satisfied: **LOOK** deliverable matches §2/§5/§7 *and* **ENGINEERING+TESTS** deliverable complete.
- All §6 items verified in light + dark.
- `make check` green; existing UI/router tests pass; new phase tests pass.
- No new runtime dependency for Phase 1 beyond Streamlit + CSS (justify here first if considered).
- Invariants intact: non-advisory, numbers-from-tools, citations-from-tool-output, dependency direction downward.
- This doc's phase checkboxes updated as each lands; before/after screenshots attached at R6.
