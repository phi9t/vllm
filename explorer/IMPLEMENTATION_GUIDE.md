# Building Explorers — A Rigorous Implementation Guide

> How to build an "explorer": a dark-glass, interactive, **data-driven** SVG visualization of a
> complex system. This is the *builder's manual* — it teaches the techniques (so they reproduce
> in any React + SVG project) and anchors each one to the reference implementation in this kit.
>
> It focuses on the two diagram families this kit pioneered:
> **(A) the model circuit** — a residual-mainline computational graph (transformers), and
> **(B) the system-architecture component graph** — a control/data-flow map (the vLLM V1 engine).
>
> **The doc set.** Read this alongside its two siblings — they don't overlap:
> | Doc | Answers |
> |---|---|
> | [`EXPLORER_DESIGN.md`](EXPLORER_DESIGN.md) | *What does it look like?* — the Observatory design language (tokens, shell, primitives). |
> | [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md) | *What rules must it satisfy?* — requirements **R1–R8**, add-subject/add-mode runbooks. |
> | **`IMPLEMENTATION_GUIDE.md`** (this) | *How do I actually build one?* — pipeline, layout algorithms, aesthetics, recipes. |
>
> The data-exploration (dashboards/charts/tokenization) family is **out of scope** here — see
> [`src/data/DataExplorer.tsx`](src/data/DataExplorer.tsx) for that pattern. Everything is
> **dark-only**; there is no light theme.

---

## 0. Mental model

An explorer is a **pure function of static data**. The hard knowledge lives in a generator; the
front-end only renders.

```
  source of truth                generator (Python)         static artifact            React reader
 ─────────────────   resolve_   ──────────────────   JSON   ───────────────   fetch   ─────────────
  repo source code   line()      build_*_manifest.py ────▶  public/data/*.json ────▶   generic <View>
  HF configs         (symbol-                                index.json + manifests     + one SVG
  guide markdown      grep)                                                             diagram component
```

Four consequences, each a rule you should keep:

1. **The generator is the only place that knows the domain.** The SPA never hard-codes a model's
   layer list or an engine's call graph; it renders whatever the manifest says. Add a subject by
   regenerating data, not by editing components.
2. **References are grounded, not transcribed.** A `file:line` in a manifest is resolved by
   grepping a *symbol* at build time, so it survives line drift (§1.2).
3. **The diagram is derived geometry.** Box sizes and positions are *computed* from the data
   (text length, branch counts), never hand-placed — that is what makes "no text overflow" and
   "outline contains its blocks" structural invariants rather than manual tuning (§3).
4. **State is local and disposable.** Hooks only — no router, no global store. The URL never
   changes; refreshing resets to defaults. This keeps an explorer a self-contained widget you can
   drop anywhere (R7).

---

## 1. The pipeline — architecture of an explorer

### 1.1 The data contract: `index → manifest → reader`

Every mode is the same three-part contract:

- **`<mode>/index.json`** — a list of *subjects* (`{ slug, label, … }`). Drives the subject
  switcher.
- **`<mode>/<slug>.json`** — one self-contained *manifest* per subject: everything the view needs
  to render, and nothing it doesn't.
- **One generic reader component** that fetches the index, then the active manifest, and renders.

The reader loads in **two stages** so the UI stays responsive and each failure is localized
(reference: [`src/architecture/ArchitectureExplorer.tsx`](src/architecture/ArchitectureExplorer.tsx)):

```
useEffect(load index.json)         → setIndex(...)        // stage 1: which subjects exist
useEffect([slug], load <slug>.json) → setManifest(...)    // stage 2: the active subject's data
```

Wrap both stages in the shared [`AsyncBoundary`](src/explorer-kit/AsyncBoundary.tsx) — one place
for loading / error / "run `gen-data`" empty states, so no mode re-implements them.

> **Portable principle.** Keep the manifest *complete and declarative*. If the renderer has to
> compute domain facts (param counts, KV bytes), put a registry between the manifest and the view
> (§3.4) — never sprinkle domain math through JSX.

### 1.2 Drift-proof grounding: symbol-grep, not line numbers

A guide or manifest that cites `foo.py:412` rots the moment someone edits `foo.py`. The kit
resolves the *current* line by grepping a stable symbol at generation time
([`scripts/_refs.py`](scripts/_refs.py) → `resolve_line(repo_root, file, symbol, hint)`):

- Search the file for the symbol string (e.g. `self.qkv_proj = QKVParallelLinear`), with a
  word-boundary check so `OutputProcessor` doesn't match `OutputProcessorOutput`.
- Fall back to progressively shorter prefixes, then to a hard-coded `hint` line only if grep fails.

The resulting `"path:line"` is rendered as a live GitHub link by
[`sourceUrl()`](src/lib/assets.ts) (`SourceLink` in the drawer). **Always ground a reference to a
symbol you can grep, and always store the symbol in the generator** so regeneration re-resolves it.

### 1.3 The mode shell

[`src/App.tsx`](src/App.tsx) holds a typed registry, `MODES: ExplorerMode[]` (the contract lives
in [`src/explorer-kit/mode.ts`](src/explorer-kit/mode.ts)). Each entry is
`{ id, label, icon, subtitle, View }`; the shell renders `active.View` and passes every view a
`navigate(id)` prop — the general **cross-link** mechanism (Component's drawer calls
`navigate('architecture')`). Adding a mode is one array entry; there is no `if (mode === …)` chain.

### 1.4 Portability

All data/asset URLs resolve through [`dataUrl(path)`](src/lib/assets.ts) against
`import.meta.env.BASE_URL`, so the app works under a sub-path (`/vllm/` on GitHub Pages). Never
write an absolute `/data/...` path in a `fetch` or `src`.

---

## 2. The Observatory aesthetic system

Aesthetics here are not decoration — they are **legibility under density**. A good explorer shows
a lot of structure at once without becoming noise. The whole language is in
[`EXPLORER_DESIGN.md`](EXPLORER_DESIGN.md); this section is the *reasoning* a builder needs.

### 2.1 Philosophy: instrument panel, not dashboard

Dark, near-black canvas; content floats on faint glass; one or two accent hues carry *all*
emphasis. The viewer's eye should be drawn to **state** (what's selected, what's flowing), not to
chrome. Restraint is the aesthetic.

### 2.2 Color — hue is semantic, not decorative

The token ladder (in [`src/index.css`](src/index.css) `@theme`) is a *value scale*, darkest →
lightest, so depth reads correctly before any hue is involved:

```
--color-void #0b0f19  →  --color-panel rgba(17,24,39,.75)  →  --color-ink #f3f4f6
                         (glass surface, blurred)              (primary text)
 accents: --color-primary #6366f1 (indigo) · --color-cyan #38bdf8 · --color-secondary #8b5cf6
 status:  info #06b6d4 · success #10b981 · warning #f59e0b · danger #ef4444
```

Two hard rules:

- **One meaning per hue.** In the circuit, [`KIND_COLOR`](src/architecture/blockTypes.ts) maps a
  *block category* to a color (`attn #10b981`, `mlp/proj #6366f1`, `moe #f472b6`, `norm #8b5cf6`,
  `rope #38bdf8`, `act #f59e0b`, `latent #22d3ee`, `embed #06b6d4`, `head #ef4444`, `router
  #f59e0b`). A reader learns the legend once. Never assign color to make a node "pop."
- **≤ 2 accents for emphasis.** Indigo is the resting accent; **cyan is reserved for selection**
  (§2.6). Don't introduce a third "look at me" color.

**Derive accents from content when you can.** The per-group circuit outline tints itself by reading
its branches — `groupAccent()` picks the FFN/MoE branch's color — so DeepSeek's dense group reads
indigo and its MoE group reads pink *with no extra schema field*. Derived color stays consistent
automatically.

### 2.3 Depth & material — the glass z-stack

Layered translucency makes a flat dark screen feel three-dimensional without heavy shadows. From
back to front:

```
1. --color-void-deep base               (the page)
2. .observatory-bg                       fixed radial-gradient wash (two faint indigo/violet pools)
3. .panel                                rgba glass + backdrop-filter: blur(16px) + 1px hairline
4. card base rgba(13,18,30,.96)          the diagram node's opaque floor
5. url(#grad-<kind>) fill                kind-tinted vertical gradient (0.2 → 0.035 alpha)
6. 3px accent bar + 1.2px border         identity + edge definition
7. selection glow (drop-shadow, cyan)    state, applied only when active
```

Each layer is *low contrast* against its neighbour; the stack, not any single layer, creates depth.
Reach for translucency and a hairline border before reaching for a shadow.

### 2.4 Typography — two families, one ladder

- **Inter** for prose and data UI; **Fira Code** for code, refs, and *everything inside a diagram*.
- Size ladder: `28` H1 · `18` drawer title · `14` body · `12` block label · `9` metric · `7.5`
  kind tag. Steps are large enough to establish hierarchy at a glance.
- **Why monospace in diagrams:** (1) every glyph has the same advance, so you can *measure* a
  label's pixel width from its character count (§3.3) — the entire no-overflow guarantee depends on
  this; (2) it reads as "technical instrument," matching the domain.

### 2.5 Spacing & rhythm — breathing room is a rule

The container is centered, `max-width: 1600px`, `padding: 24px`, `gap: 24px`. Inside diagrams,
explicit comfort constants (`COMFORT=16`, `PAD_L=16`, `PAD_R=14`, `PRENORM_GAP=16`) keep text off
edges and nodes apart. **Crowding is the most common way an information-dense diagram becomes
unreadable** — budget whitespace deliberately, as named constants, not leftover pixels.

### 2.6 Motion — subtle, purposeful, guarded

Motion should encode meaning, never decorate. The circuit's only animation is six **flow ticks**
drifting up the rail (`@keyframes mainline-flow`, 2.4s, staggered) — it signals "data flows
bottom→top." Every transition is short (0.2–0.25s) with an ease curve. **All motion is wrapped in
`@media (prefers-reduced-motion: reduce)`**, which freezes the ticks and disables transitions. If
you can't name what a motion *communicates*, delete it.

### 2.7 Selection & focus — one cyan language

- **Selected = cyan.** A selected node gets a cyan border (`#38bdf8`, width 2) plus a cyan
  `drop-shadow` glow (`.diagram-node.active`). That is the *only* thing cyan does, so selection is
  unmistakable.
- **Hover = lift.** A softer cyan drop-shadow on `:hover` (`.diagram-node:hover`).
- **Focus = ring.** A 3px `--color-primary` outline on `:focus-visible` for every interactive
  element (keyboard parity, R4).

### 2.8 Anti-patterns

| Don't | Why | Do instead |
|---|---|---|
| Rainbow nodes (color = decoration) | destroys the legend; everything shouts | one hue per semantic category (`KIND_COLOR`) |
| Hard-coded box widths | text overflows at some data/zoom | measure text → width (§3.3) |
| Shrinking text to fit | unreadable; inconsistent scale | grow the box, keep the type size |
| Drop-shadow soup | muddy depth, no focus | translucency stack + hairline; glow only for state |
| Decorative motion | distraction, a11y cost | motion only to encode flow/state; reduced-motion guard |
| Labels scattered on the canvas | clutter near the structure | terse labels on nodes; detail in the drawer; minor labels in hover `<title>` |

---

## 3. Diagram family A — the model circuit (residual mainline)

**When to use:** a topology with a **dominant spine and side-branches that merge back into it** —
a transformer decoder, a residual network, any backbone+branch computation. If your graph is
"mostly a line with detours," this is your family.

Reference renderer: [`src/architecture/ModelCircuit.tsx`](src/architecture/ModelCircuit.tsx).
Manifest types: [`src/architecture/modelArch.ts`](src/architecture/modelArch.ts). Metrics:
[`src/architecture/blockTypes.ts`](src/architecture/blockTypes.ts).

### 3.1 Visual grammar

```
        output logits            ← head blocks (mainline cards), top = output
            │
        ┌── + ──┐  «mlp»          ← branch merges back at a residual-add (+) node
   ┌────│───────│────┐
   │    │   [down]   │            ← branch panel: pre-norm + step cards, tinted by branch accent
   │    │   [silu]   │
   │    │  [gate_up] │
   │    │ [post-norm]│
   └────│───────│────┘  decoder layer × N   ← layer-group bracket (dashed, accent-tinted, label chip)
            │ tee
        ┌── + ──┐  «attn»
        … attn branch …
            │
        input_ids                ← prelude blocks, bottom = input
```

- A **gradient rail** runs bottom (input) → top (output).
- Each **branch** tees off the rail, runs a `preNorm` + ordered `steps` inside a tinted panel with
  a header pill, and merges at a crisp `+` node (`PlusNode`).
- A **layer group** (a repeated stack, e.g. "MoE layer × 58") is wrapped in a dashed bracket with
  a solid **label chip**; the bracket is accent-tinted per group (§2.2).
- A **lens** (`flow` / `shapes` / `compute` / `memory`) reframes every node's metric *without
  changing the diagram* — same geometry, different number.

### 3.2 The Y-geometry (vertical layout)

`ModelCircuit` walks **top→bottom in SVG space** (output at the top), accumulating `y`:

1. **Head** cards at `MAIN_PITCH` spacing from `PAD`.
2. A running **`cursor`** = the lowest occupied edge so far (starts at the bottom of the lowest
   head card).
3. **Each layer group** (processed `layers.reverse()` so later layers sit nearer the output): place
   the topmost branch junction, lay branches downward (`branchHeight(n)` tall, `BRANCH_GAP`
   between), then derive the bracket (§3.2.1) and advance `cursor` past it.
4. **Prelude** cards below the last bracket.
5. `height` falls out of the final `y`; the rail spans `[yTop=PAD, yBot=height−PAD]`.

A single branch's vertical span:

```
branchHeight(n) = PRENORM_GAP + CARD_H/2 + (n−1)·CARD_PITCH + CARD_H/2 + JUNC_GAP   // n = 1 preNorm + steps
```

(constants: `CARD_H=32`, `CARD_PITCH=44`, `PRENORM_GAP=16`, `JUNC_GAP=18`, `MAIN_PITCH=52`,
`PAD=46`).

#### 3.2.1 The containment invariant — outline ⊇ panels ⊇ text

A group outline that doesn't fully enclose its blocks looks broken. Make containment **structural**
by deriving the bracket from the *real* panel extents, not from the junction line:

- A branch panel's visual extent runs from its **header-pill top** at `yJunc − PILL_OVER` down to
  its **tee dot** at `yTee + TEE_OVER` (`PILL_OVER=17`, `TEE_OVER=4`).
- So for a group: `bracketTop = topYJunc − PILL_OVER − BRACKET_M` and
  `bracketBot = bottomYTee + TEE_OVER + BRACKET_M` (`BRACKET_M=14` inner margin).
- Groups are placed along the `cursor` with a `GROUP_GAP=20`, and the topmost junction is set so
  the group's **label chip** (which rises `CHIP_OVER=9` above `bracketTop`) clears the previous
  region by exactly `GROUP_GAP`:
  `topYJunc = cursor + GROUP_GAP + CHIP_OVER + BRACKET_M + PILL_OVER`.

Because every bound is computed from the same extents the panels are drawn with, the relation
*group-outline ⊇ branch-panels* holds for any number of branches/steps, and adjacent groups never
overlap. **Verify it numerically** (don't eyeball): a tiny standalone script can replay this walk
over every manifest and assert `pillTop ≥ bracketTop` and `teeDot ≤ bracketBot` for every branch —
see §6.

### 3.3 The X-geometry — no text overflow *by construction*

This is the signature technique. **Never hard-code a box width.** Because the diagram font is
monospace, a string's pixel width is `chars × fontSize × ADV` with `ADV ≈ 0.62` (Fira Code
600-weight advance, with cushion). `cardWidth()` sizes a card to the wider of its label-line and
its metric-line, plus padding and `COMFORT`:

```
cardWidth(label, kindTag, worstMetric) =
   ceil( max( PAD_L + textW(label) + (TAG_GAP + textW(kindTag))?  + PAD_R,      // label + kind tag
              PAD_L + textW(worstMetric)                          + PAD_R )      // worst metric
         + COMFORT )
```

Then `CARD_W` = max over all branch cards, `MAIN_W` = max over all mainline cards (floors
`CARD_W_MIN=240`, `MAIN_W_MIN=180`), and **every x-coordinate derives from those**: `MAINLINE_X`,
`CARD_X`, `SHELL_LEFT/RIGHT`, `VIEW_W`. Overflow becomes geometrically impossible at any render
scale.

> **The subtle bug worth internalizing:** the *longest metric string* is not always at the maximum
> token count. Memory renders `KV 999.9 KiB` (wider) at a mid token count but `KV 4.5 MiB` at 4096.
> So `computeLayout` measures `worstMetric` across a **geometric sweep of token counts over
> `[8, MAX_TOKENS]`**, taking the longest `lensMetric` per block over `{shapes, compute, memory}`.
> Measure your worst case across the whole *parameter range*, not just its endpoint.

**Companion rule (R1):** keep block labels terse (≤ ~22 chars, hard ceiling 24). Full prose goes in
the drawer (`desc`/`note`), never on the node face.

### 3.4 The metric / lens registry

The renderer stays domain-agnostic by delegating all math to a registry
([`blockTypes.ts`](src/architecture/blockTypes.ts)):

- `BLOCK_METRICS: Record<BlockType, (cfg, T) => BlockMetrics>` — each block type computes its
  `{ shape, flops, params, activeParams?, kvBytes? }` from the model config and token count `T`.
- `lensMetric(metrics, lens)` is the **single formatter** used by *both* the on-node label and the
  width-measurement code — so what you measure is exactly what you draw. (Diverging these is how
  overflow sneaks back in.)
- `summarize()` aggregates model-level totals, handling **total-vs-active params** for MoE
  (`E×3×d×I` total, `k×3×d×I` active) and **KV accounting** for GQA (`2·kv_heads·head_dim·T`) vs
  MLA (compressed latent `(kv_lora_rank + qk_rope)·T`).

To support a new node type you add one `BLOCK_METRICS` entry and (if it shows a new unit) one
`lensMetric` branch — no renderer change.

### 3.5 The manifest & its generator

Manifest shape ([`modelArch.ts`](src/architecture/modelArch.ts)):
`{ model, slug, family, source, config, prelude[], layers: LayerGroup[], head[] }`, where a
`LayerGroup` is `{ repeat, label?, branches: Branch[] }` and a `Branch` is
`{ name, accent, preNorm: Block, steps: Block[] }`.

The generator [`scripts/build_model_arch.py`](scripts/build_model_arch.py):

1. `MODEL_LIST` maps `(slug, hf_id, family, label)`.
2. Loads the real HF config via `transformers.AutoConfig`, falling back to a hard-coded
   `FALLBACKS[slug]` when offline (so the build is deterministic and network-free).
3. Instantiates a per-family **template** (`dense-qknorm`, `moe-qknorm`, `mla-moe`) of `_b(...)`
   block definitions, each carrying a *display symbol* and a *grep symbol*; `resolve_line` turns the
   grep symbol into a real `ref` (§1.2).
4. Emits `<slug>.json` + an `index.json` (`slug/label/family/totalParams`).

Note how `mla-moe` emits **two layer groups** (`dense layer × first k`, `MoE layer × rest`) from
`first_k_dense_replace` — the renderer needs no special case; it just draws two brackets.

### 3.6 Aesthetic specifics (circuit)

Kind-tinted gradient card fills (`url(#grad-<kind>)`, 0.2→0.035 alpha) + a 3px accent bar + kind
tag; rounded-elbow branch wiring via `roundedOrthPath(points, r=12)`; the gradient rail
(`rail-grad` indigo→cyan) with staggered flow ticks; per-group accent tint + solid label chip;
and **decoration text in hover `<title>` tooltips** — the `+`-junction labels, the
`residual stream` rail label, and the `input_ids`/`output logits` end-caps are hover-only so the
canvas stays clean (a wide transparent hit-line makes the thin rail easy to hover).

### 3.7 Recipes

- **Add a model:** add a `MODEL_LIST` row + a `FALLBACKS[slug]`; if the family is new, add a
  template; run `workflow.sh gen-data`; the switcher and diagram pick it up. (No React changes —
  this is the test that your abstraction is right.)
- **Add a block type:** add the `BlockType` to `modelArch.ts`, a `BLOCK_METRICS` entry to
  `blockTypes.ts`, a `KIND_COLOR` if it's a new kind, and use it in a template.
- **Add an architecture family:** add a `TEMPLATES[...]` entry and a branch in
  `build_manifest`'s family switch (e.g. how `mla-moe` builds two groups).

---

## 4. Diagram family B — the system-architecture component graph

**When to use:** a **control/data-flow graph with no dominant spine** — subsystems as boxes,
calls/messages as directed edges. The vLLM **V1 engine** map (`LLM → EngineCore → Scheduler →
Executor → ModelRunner → Sampler → …`) is the worked example.

Reference: [`src/components-deepdive/ArchitectureGraph.tsx`](src/components-deepdive/ArchitectureGraph.tsx),
[`types.ts`](src/components-deepdive/types.ts),
[`ComponentExplorer.tsx`](src/components-deepdive/ComponentExplorer.tsx). Generator:
[`scripts/build_component_manifest.py`](scripts/build_component_manifest.py).

### 4.1 Curated fixed-position layout — and why

For a small, stable graph (~10–20 nodes) a **hand-curated position map beats auto-layout**.
`ArchitectureGraph` keeps a `POS: Record<id, {x,y}>` over a fixed `viewBox` and a uniform node size
(`W=150, H=46`), with a **grid fallback** for any id not in the map (so the graph still renders if
the source guide changes):

```ts
const p = POS[id] ?? { x: 35 + (idx % 3) * 210, y: 16 + Math.floor(idx / 3) * 84 }
```

Why fixed over springy auto-layout: it's **deterministic** (no run-to-run jitter), **legible**
(you place related nodes intentionally — scheduler/KV-manager in a left column, the engine spine in
the center), **reviewable** in diffs, and **dependency-free**. Auto-layout's value (saving manual
placement) doesn't pay off until the graph is large or changes often.

### 4.2 Rendering, selection, and the companion drawer

- **Edges first, nodes second** (so nodes paint over edge ends): an edge is a `.diagram-edge` line
  between node centers; a node is a `<g role="button" tabIndex={0}>` with Enter/Space handlers and
  a cyan border when `active` (identical interaction grammar to the circuit — §5).
- The drawer pairs each node with its grounded `file:line` **and a runnable hack** — the Component
  mode is a live companion to `HACKERS_GUIDE.md` + `hacks/`, so "read the box → open the source →
  run the experiment" is one click each. Cross-link to the circuit via `navigate('architecture')`
  for nodes like `GPUModelRunner`.

### 4.3 The generator

`build_component_manifest.py` parses a **source-of-truth markdown doc** (`HACKERS_GUIDE.md`):
the "30-second architecture" table → core nodes (box / file / symbol), the section headers → one
section node each, the hacks table → runnable experiments. Each node's line is resolved with
`resolve_line` (§1.2). Edges are a small explicit `CORE_EDGES` list over the slugified node ids
(mirroring the guide's mermaid flow). It writes `components.json` + `graphs/index.json`.

> **Portable principle.** Even a "static" diagram should be *generated from a document people
> already maintain*, so the picture can't drift from the prose. Don't draw the graph by hand in
> JSX; parse it from the doc and render the parse.

### 4.4 Aesthetic specifics (graph)

Uniform node size and a single resting border color (`rgba(99,102,241,.5)`); cyan for selection
only; thin edges (`.diagram-edge`, `rgba(148,163,184,.5)`); centered mono labels. Resist the urge
to color nodes by type here — in a flow graph, *position and connectivity* carry the meaning;
color is reserved for state.

### 4.5 Scaling beyond fixed layout

When a graph outgrows hand-placement (≳20 nodes, or frequent churn), keep the **manifest contract**
and change only the *position source*: introduce lanes/ranks, or run a dagre/elk auto-layout pass
(in the generator, or once in the reader) that fills the same `{id → x,y}` map. The renderer,
drawer, and data contract stay identical — only where `POS` comes from changes.

### 4.6 Recipe — add a subsystem graph

Describe the subsystem in the guide (table + section), extend the generator to emit its
node/edge/section list and an `index.json` entry, run `gen-data`, and (when more than one exists)
the `SubjectSwitcher` in `ComponentExplorer` exposes it. See
[`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md) §4.

---

## 5. Interactivity & accessibility (both families)

One interaction grammar, enforced everywhere (R3/R4):

- **Every node is operable by mouse *and* keyboard:** `role="button"`, `tabIndex={0}`,
  `onClick` + `onKeyDown` (Enter/Space) → `onSelect`.
- **The SVG container is `role="group"`, not `role="img"`** — an `img` role tells assistive tech to
  prune the children, which would hide the very nodes that are individually operable. Give the group
  an `aria-label` describing the whole diagram.
- **Selection drives a single shared drawer** ([`DetailDrawer`](src/explorer-kit/DetailDrawer.tsx)):
  eyebrow + title + grounded `SourceLink` + body. The drawer is sticky on scroll.
- **Controls use the canonical pill tablists** — [`SubjectSwitcher`](src/explorer-kit/SubjectSwitcher.tsx)
  (subjects; auto-collapses to a `<select>` past 6) and [`ViewTabs`](src/explorer-kit/ViewTabs.tsx)
  (sub-views / lens). Don't hand-roll pill buttons.
- **Cross-links** between modes go through the `navigate(id)` prop (§1.3).
- **A11y floor:** a `.skip-link` first in the DOM, a 3px `:focus-visible` ring on everything
  focusable, and **`prefers-reduced-motion` guards on all motion**.

---

## 6. Build, verify, ship

```bash
cd explorer
./scripts/workflow.sh gen-data   # regenerate public/data/* via the repo .venv (never bare python3)
./scripts/workflow.sh install    # npm install
./scripts/workflow.sh dev        # vite dev server
./scripts/workflow.sh            # gen-data + install + build
```

**Definition of done** for a diagram change:

1. `npx tsc -b --noEmit` and `npm run build` are clean.
2. **No-overflow audit (R1):** every label ≤ 24 chars and `cardWidth` ≥ measured text for every
   block in every manifest — the inline script in
   [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md) §5.
3. **Containment audit (family A):** a standalone Node replay of the §3.2 walk asserts every
   branch's pill-top ≥ `bracketTop` and tee-dot ≤ `bracketBot`, and that adjacent groups keep
   `GROUP_GAP`, for all models. (Compute the geometry from the manifests; don't eyeball the SVG.)
4. Keyboard-only pass: Tab reaches every node/control; Enter/Space selects; the drawer reflects it.
5. `prefers-reduced-motion: reduce` freezes the flow ticks.
6. Deploy: a push touching `explorer/**` triggers
   [`.github/workflows/explorer-pages.yml`](../.github/workflows/explorer-pages.yml); verify HTTP
   200 and spot-check each subject.

> **Generators run through the repo `.venv`** (`AGENTS.md`): `.venv/bin/python explorer/scripts/…`.
> Never use system `python3` or bare `pip`.

---

## 7. Reference map — the kit, file by file

| File | Role | Copy or extend? |
|---|---|---|
| [`src/index.css`](src/index.css) | `@theme` tokens, `.observatory-bg`, `.panel`, diagram + circuit classes, a11y, motion | **Copy** the tokens + classes; they are the design language. |
| [`src/App.tsx`](src/App.tsx) + [`explorer-kit/mode.ts`](src/explorer-kit/mode.ts) | `MODES` registry + `ExplorerMode`/`navigate` contract | Copy the shell; add `MODES` entries. |
| [`src/explorer-kit/`](src/explorer-kit) | `AsyncBoundary`, `SubjectSwitcher`, `ViewTabs`, `DetailDrawer` | **Copy**; every mode reuses these. |
| [`src/lib/assets.ts`](src/lib/assets.ts) | `dataUrl`/`BASE_URL`, `sourceUrl` | Copy; portability + grounded links. |
| [`src/architecture/ModelCircuit.tsx`](src/architecture/ModelCircuit.tsx) | family-A renderer: the layout algorithm + no-overflow + containment | **The reference for §3.** Adapt geometry; keep the invariants. |
| [`src/architecture/blockTypes.ts`](src/architecture/blockTypes.ts) | metric/lens registry, `KIND_COLOR`, formatters | Extend with your node types. |
| [`src/architecture/modelArch.ts`](src/architecture/modelArch.ts) | manifest types | Adapt to your domain. |
| [`src/components-deepdive/ArchitectureGraph.tsx`](src/components-deepdive/ArchitectureGraph.tsx) | family-B renderer: fixed-position graph | **The reference for §4.** |
| [`scripts/_refs.py`](scripts/_refs.py) | `resolve_line` symbol-grep grounding | Copy verbatim. |
| [`scripts/build_model_arch.py`](scripts/build_model_arch.py) / [`build_component_manifest.py`](scripts/build_component_manifest.py) | the generators (source of truth → JSON) | The reference generators. |

For the design tokens see [`EXPLORER_DESIGN.md`](EXPLORER_DESIGN.md); for the full requirement spec
and add-subject/add-mode runbooks see [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md).
Together the three docs are **language ↔ requirements ↔ implementation** — this one is the
implementation.
