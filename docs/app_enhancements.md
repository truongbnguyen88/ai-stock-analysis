# App Enhancements — Implementation Plan

> **Living implementation guide.** Each step is independently shippable and must pass
> `make check` (ruff + mypy + pytest) before the next. Pick up by saying e.g. *"implement
> Enhancement A, step 2"*.
>
> **Decisions locked (2026-06-06):**
> 1. Centerpiece animation = **typewriter**.
> 2. Showcase architecture = **hybrid C** (animated centerpiece + native interactive cards).
> 3. Include a **multi-ticker** capability (news analysis + model predictions for several
>    tickers at once) — Enhancement B; the showcase (A) advertises it.

Two enhancements, related: **B** adds the multi-ticker capability; **A** is the launch
screen that makes it (and everything else) discoverable. Recommended order: **A.1–A.2**
(useful, low-risk) → **B** (the new feature) → **A.3–A.4** (animation polish) so the
showcase can feature a capability that actually works when clicked.

---

## Enhancement A — Launch-screen capability showcase (Streamlit)

### Objective
On launch (empty chat), show a **dynamic, cyclic, interactive** display in the center of
the screen conveying what the agent can do — not a static tool list. Disappears once a
conversation starts.

### Target surface
The Streamlit web UI: [ui/chat_app.py](../ui/chat_app.py) (`make ui`). Not the terminal CLI.
Streamlit **1.58** is installed (supports `components.html` and `st.fragment(run_every=)`).

### Reuse hooks (already in the code — do not rebuild)
- **Empty-state slot:** render the hero only when `not st.session_state.messages`
  (fresh launch / new chat). The message-render loop ([chat_app.py:224](../ui/chat_app.py#L224))
  takes over once a turn exists, so the hero auto-hides.
- **Interactivity:** clicking a card sets `st.session_state.pending_prompt` then
  `st.rerun()`. The existing line `prompt = st.chat_input(...) or pending`
  ([chat_app.py:244](../ui/chat_app.py#L244)) submits it — **no new submission plumbing**.
- **Ticker context:** sidebar `ticker_input` (default `NVDA`) — examples use `{ticker}`,
  like the existing sidebar "Quick starters" ([chat_app.py:177](../ui/chat_app.py#L177)).

### Design — hybrid C
- **Animated centerpiece** via `st.components.v1.html`: a **typewriter** that types/erases
  capability headlines in a loop (e.g. `I can … analyze NVDA's trend ▌` → erase → next).
  Pure client-side (CSS/JS), so it never blocks the chat or causes server reruns.
- **Interactive layer** beneath it: a centered grid of capability **cards** (native
  `st.button`, 2–3 columns) → on click set `pending_prompt` + `st.rerun()`.
- Sidesteps the iframe "can't call back to Python" limit: the typewriter is the *visual*,
  the cards are the *action*. Both read from the same capabilities list.

### Capabilities content model (data-driven; NOT a tool dump)
New `ui/capabilities.py` — single source of truth for both the typewriter text and the cards:
```python
@dataclass(frozen=True)
class Capability:
    icon: str        # "🔮"
    title: str       # "Probabilistic forecasts"
    blurb: str       # "Calibrated up/down odds + VaR/CI, 5–60 days"
    example: str     # "Forecast {ticker} 30 days out with the ensemble"  (uses {ticker})
```
Curated ~6–8 (value statements, not tool names):
technical analysis · probabilistic forecasts (ensemble, calibrated) · latest news
(newest-first) · news synthesis (themes/risks/catalysts) · "is this forecast
trustworthy?" (calibration/ECE) · backtest vs baselines · model comparison · **compare
multiple tickers** (← Enhancement B).

### Lifecycle & placement
- Render only in empty state, centered (`st.columns([1,6,1])` or component CSS), below the
  title/caption.
- Keep the sidebar Quick starters (compact); the hero is the discoverable centerpiece.

### Risks / edge cases
- **Theme:** the `components.html` iframe doesn't inherit Streamlit light/dark theme →
  style neutrally or pass the active theme in (avoid a white box in dark mode).
- **Fixed iframe height** → reserve space so typed text never clips.
- **Accessibility:** honor `prefers-reduced-motion` (static fallback); gentle cadence
  (~60–90 ms/char, ~1.5 s hold).
- **No interference:** hero is empty-state only → never competes with a running turn.
- **Testing:** Streamlit isn't unit-testable; unit-test the capabilities data (well-formed;
  every `example` contains `{ticker}`) and keep render logic in small functions. Final
  check = manual `make ui` smoke.

### Build steps
- [ ] **A.1** `ui/capabilities.py` (dataclass + curated list) + `tests/unit/test_capabilities.py`
      (non-empty; each `example` has `{ticker}`; titles/icons present).
- [ ] **A.2** Empty-state **interactive cards**: centered `st.button` grid in
      `if not st.session_state.messages:`; click → `pending_prompt` + `rerun`. (Delivers the
      useful half, zero animation risk.)
- [ ] **A.3** Animated **typewriter centerpiece** (`components.html`) above the cards,
      cycling the capability headlines; theme-aware, fixed height.
- [ ] **A.4** Polish: `prefers-reduced-motion` fallback, responsive wrap, copy pass.
- [ ] **A.5** *(optional)* bidirectional custom component so the typewriter item itself is
      clickable (defer unless wanted).

---

## Enhancement B — Multi-ticker news analysis & model predictions

### Objective
Let the chat agent handle **several tickers at once**: (1) compare **news** across tickers
(sentiment + key themes), and (2) compare **model predictions** across tickers
(probabilities / expected move / VaR), rendered as a side-by-side comparison.

### Current limitation
All agent tools are single-ticker (`get_news`, `summarize_news`, `get_news_sentiment`,
`forecast_scenarios`). The LLM *can* call a tool N times, but that's slow, token-heavy, and
produces an ad-hoc comparison. Dedicated **batch tools** give one clean call + a structured
comparable result the UI can table/chart.

### Design — batch tools (in [agent/tools.py](../src/stock_agent/agent/tools.py) + `TOOL_SCHEMAS`)
- `compare_forecasts(tickers: list[str], horizon_days: int, model?: str)` → per-ticker
  forecast summary (expected_return, upside/downside prob, VaR95, CI, big-move) by looping
  the existing `run_forecast` (default model = `ensemble`). Numbers come from `forecasting/`
  — **invariant: the LLM never computes them**; it only narrates the comparison.
- `compare_news(tickers: list[str], days: int)` → per-ticker numeric sentiment
  (avg sentiment, %pos/%neg) + top newest headlines, via `NewsFetcher`/`get_news_sentiment`
  per ticker. The LLM does the cross-ticker narrative (non-advisory — no "buy X over Y").

### Bounds & constraints
- **Cap the list** (e.g. `MAX_TICKERS = 6`) — N forecasts (ensemble = 5 members each) and N
  news fetches; reject/truncate beyond the cap with a clear message.
- Reuse the **cache** (news short-TTL already in place); forecasts reuse per-call pooled
  artifacts. Bound with a tool timeout like the existing `_BT_TIMEOUT` guard.
- Invariants: numbers from modules only; **no recommendation field**; non-advisory framing.

### Output & UI rendering
- Tool result = a list of per-ticker dicts (consider a small `schemas/comparison.py` model
  for type-safety + conformance tests).
- `ui/charts.py` (`charts_for`): detect a multi-ticker result → render a **comparison**:
  grouped bars (e.g. P(up) or expected move ± VaR per ticker) and/or a table. Numbers are
  tool-produced (never the LLM).

### Agent prompt
- Update [agent/prompts/agent.py](../src/stock_agent/agent/prompts/agent.py): introduce the
  batch tools, when to use them (user names ≥2 tickers / asks to compare), the ticker cap,
  and the non-advisory comparison framing. Bump prompt version.

### Risks
- **Latency/cost:** N×(ensemble forecast) — cap N, reuse cache, timeout. Consider stating
  "comparing N tickers…" progress in the UI.
- **Rate limits:** N news fetches — short-TTL cache + the provider fallback chain mitigate;
  AlphaVantage 25/day is the tightest (Finnhub/Marketaux primary).
- **Grounding guard:** ensure the numeric-grounding guard accepts per-ticker batch numbers
  (they're real tool outputs) — add fixtures.

### Build steps
- [ ] **B.1** `schemas/comparison.py` (per-ticker forecast/news comparison models) + conformance test.
- [ ] **B.2** `compare_forecasts` tool + handler (loop `run_forecast`, cap N) + `TOOL_SCHEMAS`
      entry + unit test (cap enforced; numbers present; schema conforms).
- [ ] **B.3** `compare_news` tool + handler (loop news/sentiment, cap N) + `TOOL_SCHEMAS` + test.
- [ ] **B.4** UI comparison rendering in `ui/charts.py` (grouped bars/table) + chart unit test.
- [ ] **B.5** Agent prompt update (batch tools, cap, non-advisory) + version bump + the
      integration assertions in `tests/integration/test_agent_runtime.py`.
- [ ] **B.6** Add the "compare multiple tickers" entry to `ui/capabilities.py` (ties A↔B).

---

## Combined build order (recommended)
1. **A.1, A.2** — capabilities data + interactive cards (discoverable + useful immediately).
2. **B.1 → B.6** — the multi-ticker feature (so the showcase card works when clicked).
3. **A.3, A.4** — typewriter animation + polish.
4. **A.5 / extras** — optional.

## Notes
- Keep `ui/` thin (presentation only); business logic stays in `agent/`/`pipelines/` per
  ARCHITECTURE §2 (dependency direction downward).
- Every step: `make check` green before moving on; tests in the same change.
