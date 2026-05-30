# The Explorer UI Design Language

> Distilled from three sibling front-ends — `clinique/explorer`, `muon_optimizer/explorer`,
> and `lumina` (rope-visualizer) — and realized in this `explorer/` kit. This is the style
> guide: copy the tokens, the shell, and the contracts to build a new explorer.
>
> This is the **language** doc (what it looks like). For *how to build* a diagram end-to-end see
> [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md); for the *requirements* (R1–R8) and runbooks
> see [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md).

## Two lineages, one aesthetic

The siblings split into two implementation lineages that share a single visual language:

| | Data-explorer lineage | Concept-visualizer lineage |
|---|---|---|
| Examples | `clinique/explorer`, `muon_optimizer/explorer` | `lumina` |
| Styling | plain CSS + ~100 custom properties | Tailwind 4 `@theme` |
| Primitives | bespoke class names | shadcn-style `components/ui/` (Radix + CVA + `cn()`) |
| Data | static JSON in `public/data/` via `lib/fetch.ts` | computed in-browser (`lib/*Math.ts`) |
| Typography | Inter | Fira Code (mono) |
| Signature view | dashboards, tables, recharts | SVG diagrams + a "lens" system |

**This kit adopts the Lumina foundation** (Tailwind 4 + `ui/` primitives) but keeps the
twins' **family-switcher shell**, **indigo palette**, and **static-JSON data contract** — so
both lineages' strengths are available in one app.

The shared aesthetic is **"Observatory"**: a dark void background with a fixed
`.observatory-bg` radial-gradient layer (present verbatim in *both* lineages), glassmorphic
panels (12px radius, `backdrop-filter: blur`), indigo/cyan accents on near-black, monospace
meta text, focus halos, and skip-link accessibility.

## Design tokens

Defined as Tailwind 4 `@theme` custom properties in [`src/index.css`](src/index.css), so each
token also generates utilities (`bg-panel`, `text-cyan`, `border-panelborder`, …).

| Token | Value | Role |
|-------|-------|------|
| `--color-void` | `#0b0f19` | page background |
| `--color-void-deep` | `#070a13` | observatory base |
| `--color-panel` | `rgba(17,24,39,0.75)` | glass panel fill (+ `blur(16px)`) |
| `--color-panel-hover` | `rgba(31,41,55,0.85)` | hover surface |
| `--color-panelborder` | `rgba(255,255,255,0.08)` | hairline border |
| `--color-panelborder-active` | `rgba(99,102,241,0.4)` | active/selected border |
| `--color-ink` / `-soft` / `-muted` | `#f3f4f6` / `#9ca3af` / `#6b7280` | text scale |
| `--color-primary` | `#6366f1` | indigo — primary accent |
| `--color-secondary` | `#8b5cf6` | violet — gradient pair |
| `--color-cyan` | `#38bdf8` | secondary accent / selection |
| `--color-info` / `success` / `warning` / `danger` | `#06b6d4` / `#10b981` / `#f59e0b` / `#ef4444` | status |
| `--radius-badge/btn/panel/card` | `6 / 8 / 12 / 16 px` | radii |
| `--font-sans` | Inter Variable → system | prose & data views |
| `--font-mono` | Fira Code Variable → ui-monospace | code & architecture views |

Theme is **dark-only** (`color-scheme: dark`). Glows/scrollbars use the non-theme vars
`--glow-primary`, `--scrollbar-thumb`.

## The shell — "explorer family switcher"

Canonical structure in [`src/App.tsx`](src/App.tsx):

- App holds a **typed `ExplorerMode[]` registry** (`MODES`) and a top-level `useState`
  for the active mode id. Each entry is `{ id, label, icon, subtitle, View }` per the
  [`ExplorerMode`](src/explorer-kit/mode.ts) contract; the body renders `active.View`,
  so **adding a mode is one array entry — no new conditional**. (Decision: the contract
  documents the shared shape but does *not* hoist data loading into a generic shell —
  each `View` stays its own index→manifest reader.)
- Each mode view receives **`{ navigate(id) }`** (`ExplorerModeProps`). This is the
  general cross-link mechanism (R3): e.g. Component's drawer calls `navigate('architecture')`.
  It replaced the old bespoke `onOpenArchitecture` callback.
- `.observatory-bg` is a fixed `aria-hidden` layer; content sits in `.explorer-container`
  (centered, `max-width: 1600px`, 24px padding, flex-column, 24px gap).
- The header has a `.skip-link`, a back-to-repo link, logo + `<h1>` (gradient text) + the
  active mode's subtitle, and a `.family-switch` nav of `.family-switch-btn` buttons
  (`aria-pressed`, active gradient) mapped from `MODES`.
- Responsive: `.dashboard-grid` is `320px 1fr` desktop, collapsing to one column at 1024px.
- **In-mode controls** (subject switcher + sub-view tabs + param sliders) are laid out with
  Tailwind utility rows, *not* a dedicated `Toolbar` primitive — that was evaluated and
  intentionally left unextracted to avoid forcing a one-size layout on three differently
  shaped control rows. The reusable pieces are `SubjectSwitcher` / `ViewTabs` / `Slider`.

## Data + helpers contract

- [`src/lib/assets.ts`](src/lib/assets.ts) — `dataUrl(path)` resolves against
  `import.meta.env.BASE_URL` (portable to sub-path hosting); `REPO_HOME`, `logoMarkUrl()`,
  and `sourceUrl('file:line')` → a GitHub blob link.
- [`src/lib/fetch.ts`](src/lib/fetch.ts) — `fetchExplorerJson<T>()`, `errorMessage()`,
  `formatMetric()`, `formatCount()`. Each view owns its `loading`/`error` state.
- [`src/lib/utils.ts`](src/lib/utils.ts) — `cn()` = `twMerge(clsx(...))`.
- Static manifests live in `public/data/`, regenerated by `scripts/*.py` (see
  [`PLAN.md`](PLAN.md)). The generators are the source of truth; the SPA is a pure reader.

## UI primitives & interaction

- [`components/ui/`](src/components/ui) — `Button` (CVA variants + Radix `Slot`), `Card`
  family (the `.panel` glass surface), `Slider` (Radix). Compose with `cn()`.
- State is **React hooks only** — no router, no global store. Modes/tabs are local `useState`.
- A11y baseline: `.skip-link`, `:focus-visible` 3px outline, ARIA roles on tablists/groups,
  SVG nodes are `role="button"` + `tabIndex={0}` with Enter/Space handlers.
- **The lens abstraction** (architecture views): one diagram, reframed across
  flow / shapes / compute / memory by swapping the per-node metric — not by switching views.
  See [`src/architecture/ArchitectureExplorer.tsx`](src/architecture/ArchitectureExplorer.tsx).
- **Circuit boxes auto-fit their text (no overflow).** Box widths in the Model Architecture
  circuit are not hard-coded — they are computed per-manifest from monospace text length:
  `width = chars × fontSize × ADV` (ADV ≈ 0.62, Fira Code 600-weight advance in viewBox units).
  `cardWidth(label, kindTag, worstMetric)` measures the worst of the label+tag line and the
  widest lens metric, adds padding and comfort margin, and takes the max over all blocks in the
  manifest. Mainline boxes (`MAIN_W`) and branch panel boxes (`CARD_W`) are computed
  independently; floors are enforced (`MAIN_W_MIN=180`, `CARD_W_MIN=240`). `MAINLINE_X`,
  `CARD_X`, `SHELL_LEFT/RIGHT`, and `VIEW_W` all derive from these measured widths. The invariant
  is enforced in [`src/architecture/ModelCircuit.tsx`](src/architecture/ModelCircuit.tsx) — see
  the top-of-file comment. The shared lens formatter `lensMetric()` lives in `blockTypes.ts` so
  both the explorer panel and the width-measurement code use identical formatting.

  **Measure the worst metric across the *whole* token range, not just `MAX_TOKENS`.** The longest
  metric *string* is not always at 4096 — e.g. the memory lens renders `KV 999.9 KiB` (wider)
  at a mid token count vs `KV 4.5 MiB` at 4096. `computeLayout` therefore samples a geometric
  sweep of token counts over `[8, MAX_TOKENS]` and keeps the longest `lensMetric` per block, so
  boxes fit at **every** slider position, not just the maximum.

- **Hierarchy containment is structural (outline ⊇ panels ⊇ text).** The dashed layer-group
  outline is computed from the *real* branch-panel extents — from the topmost header-pill top
  (`yJunc − PILL_OVER`) down to the bottommost tee dot (`yTee + TEE_OVER`), plus a fixed
  `BRACKET_M` margin — so the outline always fully wraps the transformer blocks it groups.
  Groups are placed by walking a running `cursor` (lowest occupied edge) so adjacent groups and
  their label chips never overlap and keep a `GROUP_GAP`. Combined with the width invariant
  above, the three containment relations (group-outline ⊇ block-panels ⊇ box-text) are all
  enforced by construction. A standalone numeric replay of the layout asserts pills/tees stay
  inside every bracket with margin for all four models.

  **Succinct block text (R1 companion rule).** Block labels must be terse — target ≤ ~22 chars,
  hard ceiling 24. Prefer the short symbol form: `Q down-proj (A)`, `MLA attention`,
  `QKV projection`, `Gate+Up projection`. Full prose belongs in the `desc` (and `note`) fields,
  which appear in the detail drawer — never on the block face. Generators in `scripts/` must
  respect this budget; the no-overflow audit in
  [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md) §5 checks both length and computed
  fit for every block in every manifest.

## Aesthetics and interactability checklists

For the full R1–R8 spec (no overflow, aesthetics, interactability, accessibility, data-driven
contract, portability, state model, generality) see
[`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md).

### Aesthetics (R2) — Observatory glass kit
- Kind-tinted gradient card fills (`url(#grad-<kind>)`) + kind tags (upper-right corner).
- Slim accent bar: 3 px left edge rect in `KIND_COLOR[kind]`.
- Rounded-elbow wiring: `roundedOrthPath` with r=12 on all branch wires.
- Gradient mainline (`rail-grad` indigo→cyan) + animated flow ticks (6 circles, staggered).
- Cyan selection glow: `stroke=#38bdf8`, `strokeWidth=2` on selected node border.
- Tinted branch panels: `fillOpacity=0.05`, accent stroke `strokeOpacity=0.22`.
- Header pills on branch panels: near-black fill, accent stroke.
- **Layer-group outlines are accent-tinted per group + carry a solid label chip.** The accent
  is *derived* from the group's branches (the FFN/MoE branch's color) — no schema field — so a
  dense group reads indigo and a MoE group reads pink (e.g. DeepSeek-V3's dense×3 vs MoE×58 are
  immediately distinct). The chip (group label `× repeat`) is drawn on a top layer so the rail
  and panels never occlude it.
- **Decoration text lives in hover tooltips, not on the canvas.** Small/vague mainline
  annotations — the `+`-junction branch labels, the `residual stream` rail label, and the
  `input_ids` / `output logits` endpoints — are surfaced as SVG `<title>` tooltips (with a wide
  transparent hit-line on the thin rail, and small accent end-caps as hover targets) rather than
  always-on labels that clutter the diagram. The svg keeps its `role="group"` aria-label so the
  input-bottom/output-top orientation stays announced.
- Comfort margins: `COMFORT=16`, `PAD_L=16`, `PAD_R=14`, `PRENORM_GAP=16`.
- Body-text scale: label 12 px mono 600; metric 9 px mono; kind tag 7.5 px.
- `maxWidth`-capped SVG (`style={{ maxWidth: VIEW_W * 1.12 }}`).
- Dark-only: `color-scheme: dark`; no light-theme variants.
- New views reuse the kit — no bespoke style sheets.

### Interactability (R3) — every node operable
- Every node: `role="button"`, `tabIndex={0}`, Enter+Space call `onSelect`.
- Selection opens a sticky detail drawer: `symbol`, `ref` (live GitHub link via `sourceUrl`),
  `desc`, `note`.
- Subject switcher pill tablist (`role="tablist"` / `role="tab"`, `aria-selected`).
- Lens selector + token slider as param controls.
- Cross-links between modes (Architecture ↔ Component ↔ Data).
- Clear hover/focus affordances: surface lift on hover, `:focus-visible` 3 px cyan.

## Building a new explorer mode

1. Add one `ExplorerMode` entry to the `MODES` registry in `App.tsx`
   (`{ id, label, icon, subtitle, View }`).
2. Create `src/<mode>/<Mode>Explorer.tsx` as an `ExplorerModeProps` component
   (`{ navigate }`), loading its `index.json` then per-subject manifest with
   `fetchExplorerJson`. Use `navigate(id)` for any cross-link to another mode.
3. Add a generator under `scripts/` that emits the index + manifests into `public/data/`.
4. Reuse the kit (`AsyncBoundary`, `SubjectSwitcher`, `ViewTabs`, `DetailDrawer`),
   `Card`/`Button`/`Slider`, the `.panel`/`.diagram-*`/`.drawer` classes, and the
   `dashboard-grid` + drawer layout. Keep the dark tokens; don't introduce a light theme.
