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

Must work identically after **every** phase. Existing tests (`test_capabilities`, `test_chat_history`, `test_viz_charts`, `test_viz_render`, `test_router`) stay green.

- [ ] Sidebar: corpus/embedder status badge (`corpus_status`) + freshness.
- [ ] Sidebar: 4 quick-starter buttons inject `pending_prompt`.
- [ ] Sidebar: ticker input feeds quick-starters, hero, deterministic routes.
- [ ] Sidebar: routing mode = Auto + all `DOMAIN_NAMES`; variant selector; horizon/lookback params only for domains that use them.
- [ ] Sidebar: chat-thread list, active highlight, open, delete, `New chat`, retention caption.
- [ ] Sidebar: API-key status (Anthropic required + optional providers).
- [ ] Hero: typewriter (honors `prefers-reduced-motion`); 11 capability cards; card click → `pending_prompt` → submit.
- [ ] Chat: user/assistant markdown; per-turn **Altair charts** re-render after reruns (persisted via `ChartSpec.to_dict`).
- [ ] Chat: **SEC filing sources** expander (citations from tool output only).
- [ ] Chat: **Export** PDF / Word / Markdown per answer (with chart PNGs; cached).
- [ ] Chat: "tools used" trace (Auto) + "deterministic route" caption (domain).
- [ ] Auto path (`run_agent`) + deterministic path (`router.run`) intact, incl. `AgentGroundingError` / `AgentError` / `RouterError` messages.
- [ ] Bounce-to-Auto fallback (`fallback_prompt` → `force_auto_prompt`).
- [ ] Thread persistence across restarts (`ChatStore`; serialize/deserialize incl. charts + sources).
- [ ] Non-advisory disclaimer present; **no recommendation field** added.
- [ ] Invariants intact: numbers-from-tools, citations-from-tool-output, dependency direction downward.

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

### Phase R2 — Sidebar
- **LOOK:** brand block; status card + `status_dot`; grouped sections; chat list with accent active-bar + hover + trailing delete; keys chip row; hued quick-starter icons.
- **ENGINEERING:** implement in `components/sidebar.py` using theme helpers; all handlers (`pending_prompt`, ticker, routing, open/delete/new) unchanged.
- **TESTS:** helpers used render valid HTML; §6 sidebar items verified.

### Phase R3 — Empty-state hero
- **LOOK:** theme-aware typewriter (remove hardcoded hex; brass accent word); colored capability card component; colored eyebrow; responsive grid.
- **ENGINEERING:** `components/hero.py`; `cap_card(...)` helper; click behavior (`pending_prompt` + rerun) unchanged; keep `components.html` iframe colors synced to tokens.
- **TESTS:** `cap_card` renders each capability with its hue; reduced-motion path present.

### Phase R4 — Chat message: tiles + trace + sources + export
- **LOOK:** user/assistant bubbles + avatars; stat tiles (colored stripes) first; per-tool-color trace chip row / direct-route chip; styled sources; `Export ▾` affordance; `st.status` step list on Auto turns.
- **ENGINEERING:** `components/message.py`; `stat_tile`/`trace_bar` helpers; export uses existing cached bytes + `st.popover` (fallback to expander if <1.31); sources still from tool output only.
- **TESTS:** `stat_tile`/`trace_bar` HTML; sources dedup unchanged; export round-trips all 3 formats.

### Phase R5 — Charts + input bar
- **LOOK:** shared Altair theme (semantic palette, faint grid, percent axes, mono labels); composer with focus-within accent + context chips.
- **ENGINEERING:** register/apply Altair theme in the render path (no `ChartSpec` change); `components/inputs.py` surfaces existing placeholder logic as chips.
- **TESTS:** themed Altair spec still valid + percent formatting honored; input placeholder logic unchanged across modes.

### Phase R6 — Polish, a11y, docs
- **LOOK:** contrast/focus audit both themes; responsive (narrow → 1-col cards, sidebar behavior); reduced-motion verified.
- **ENGINEERING:** before/after screenshots; update `README.md` launch notes if needed; final §6 walkthrough; `make check` green.
- **TESTS:** full suite green; a11y checklist recorded here.

---

## 9. Optional Phase 2 (documented, not now): React + FastAPI

Trigger only if Streamlit's ceiling blocks a look/interaction you want after R0–R6.
- **Backend:** expose the agent via FastAPI; SSE for streaming tool events + tokens (needs an event-emitting `run_agent` variant). Reuse `ChatStore`, `export`, `charts_for`, `corpus_status` as-is.
- **Frontend:** Vite + React + Tailwind + a component kit; Vega-Lite in-browser for the same `ChartSpec` JSON; `/export` endpoint for document bytes.
- **Why cheap by then:** the §4 contracts (`RoutingChoice`, `ChartSpec`, sources, export) are already the API; the port is incremental, not a rewrite.
- **Cost:** re-implement session/thread state, streaming, persistence wiring, export UX; two-stack maintenance.

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
