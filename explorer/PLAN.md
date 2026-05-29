# Plan: Distill the "Explorer" UI design & build a unified vLLM Explorer kit

> Saved copy of the approved implementation plan. Source of record for the
> `explorer/` developer-experience tool. Execution is driven by
> [`scripts/workflow.sh`](scripts/workflow.sh).

## Context

Research distilled three sibling front-ends — `~/CodeBase/clinique/explorer`,
`~/CodeBase/lumina` (package `rope-visualizer`, brand "Lumina"), and
`~/CodeBase/muon_optimizer/explorer` — into **one shared design language with two
implementation lineages**:

- **Data-explorer lineage** — `clinique/explorer` and `muon_optimizer/explorer` are
  near-identical twins: plain CSS + ~100 CSS custom properties (no Tailwind), Inter
  font, a top-level **"family switcher"** shell, static JSON in `public/data/` loaded
  via `lib/fetch.ts`/`lib/assets.ts`, `recharts` dashboards, hooks-only state, no router.
- **Concept-visualizer lineage** — `lumina` is the evolved cousin: **Tailwind 4
  `@theme`** tokens, shadcn-style `components/ui/` primitives (Radix + CVA + `cn()`),
  Fira Code monospace, SVG circuit diagrams, and **in-browser math** (`lib/ropeMath.ts`,
  `lib/layerModel.ts`) instead of static JSON.

Both lineages share the **"Observatory" aesthetic**: dark void background with the literal
`.observatory-bg` radial-gradient layer present in *both*, glassmorphic panels (12px radius,
backdrop blur), cyan/indigo accents on near-black, monospace meta text, focus halos, and
skip-link accessibility.

**Outcome:** (1) a written **design-spec** (`EXPLORER_DESIGN.md`) capturing this language,
and (2) a **reusable template scaffold** hosted **inside the vLLM repo** as a unified
explorer with three Qwen3-unified modes, built on the **Tailwind 4 + `ui/` primitive**
foundation, unified by the family-switcher shell.

### The three modes

- **vLLM Component Deep Dive** — interactive companion to [`../HACKERS_GUIDE.md`](../HACKERS_GUIDE.md)
  and [`../hacks/`](../hacks/); highlights every major component of the V1 engine. Renders the
  guide's §2 box→file→symbol table + §2 mermaid (graph edges) + §16 hacks table + §6–§14 deeper
  components as a clickable SVG component graph (Lumina pattern); each node opens a drawer with
  guide prose, `file:line` symbol links, and the paired "▶ Try it" hack — driven by a generated
  JSON manifest (clinique pattern).
- **Data Exploration** — a small subset of [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  showing the dataset format (columns `text`, `id`, `url`, `dump`, `language`, `language_score`,
  `token_count`, `score`, `int_score`) and the **tokenization mechanism**: a schema/format view,
  record browser, recharts stat distributions, and a tokenization view running sample text
  through the **Qwen3 HF tokenizer** rendering byte-level BPE pieces (muon's `TokenPieces`),
  comparing live token count vs the dataset's `token_count`.
- **Model Architecture Deep Dive** — a **Qwen3 dense model's inference forward pass as
  implemented in vLLM** (Lumina SVG diagram + lens), inference only. Draws the 15-block
  decoder-layer sequence grounded in real vLLM symbols (`vllm/model_executor/models/qwen3.py`):
  embeddings → fused-residual RMSNorm → `QKVParallelLinear` → head-wise `q_norm`/`k_norm` →
  `RotaryEmbedding` → `Attention` (paged KV) → `o_proj` → post-attn RMSNorm → SwiGLU MLP
  (`gate_up_proj`/`SiluAndMul`/`down_proj`) → final norm → `lm_head`/`LogitsProcessor`. The
  lens reframes flow / shapes / compute (FLOPs) / memory (KV-cache bytes, GQA), with shapes
  computed in-browser (Lumina `layerModel.ts` pattern) from a real Qwen3 config (e.g.
  Qwen3-0.6B). The `GPUModelRunner` node in Component Deep Dive cross-links here.

> **Scope note (AGENTS.md):** this is `phi9t`'s **research fork** of vLLM. The explorer-kit is
> legitimate local developer-experience work (alongside `devx/` and `hacks/`). The repo's
> strict contribution policy targets **upstream PRs to `vllm-project/vllm`**; Deliverable 0
> reframes `AGENTS.md` so local research work is not treated as upstream.

---

## The distilled design language (reference)

### Design tokens (canonical, reconciled)
Adopt Lumina's Tailwind 4 `@theme` mechanism, seeded with the twins' indigo palette plus
Lumina's semantic accents. Source: `muon_optimizer/explorer/src/index.css:3-31`,
`lumina/src/index.css:3-45`.

| Token | Value | Origin |
|-------|-------|--------|
| `--color-void` / page bg | `#0b0f19` (twins) ~ `#090a0f` (lumina) | dark navy void |
| `--color-panel` / card bg | `rgba(17,24,39,0.75)` + `backdrop-filter: blur(16px)` | glassmorphism |
| `--color-primary` | `#6366f1` indigo (twins) | primary accent |
| `--color-cyan`/info | `#06b6d4`–`#38bdf8` | secondary accent (shared) |
| `--color-secondary` | `#8b5cf6` violet | gradient pair |
| success/warning/danger | `#10b981` / `#f59e0b` / `#ef4444` | status |
| text primary/secondary/muted | `#f3f4f6` / `#9ca3af` / `#6b7280` | type scale |
| border / border-active | `rgba(255,255,255,0.08)` / `rgba(99,102,241,0.4)` | hairline + active |
| radius | 6 / 8 / 12 / 16 px (badge/button/panel/card) | shared |
| fonts | Inter (sans, data views) + Fira Code/mono (architecture views) | dual-lineage |

### Shell pattern (the "explorer family switcher")
Canonical: `muon_optimizer/explorer/src/App.tsx:18-99`. Top-level `useState` holds active
`family`; header renders `.family-switch-btn` buttons (`aria-pressed`), `.skip-link`, a
back-to-repo link, logo + `<h1>` + subtitle; body conditionally renders one explorer per
family. `.observatory-bg` is an `aria-hidden` fixed layer behind a `max-width` centered
container with a 1024px → single-column collapse.

### Data + helpers contract
- `lib/assets.ts` — `dataUrl(path)` = `${import.meta.env.BASE_URL}<path>`, `REPO_HOME`, `logoMarkUrl()`.
- `lib/fetch.ts` — `fetchExplorerJson<T>()`, `errorMessage()`, `formatMetric()`.
- `lib/utils.ts` — `cn()` = `twMerge(clsx(...))`.
- Static JSON in `public/data/`; views own `loading`/`error` state, lazy-load detail records.

### UI primitives & interaction
- `components/ui/` — `Button`, `Card`, `Slider` via CVA + Radix `Slot` + `cn()`.
- State = React hooks only; no router, no global store.
- A11y: skip link, `:focus-visible` 3px outline, ARIA roles, `prefers-reduced-motion`.
- Architecture views use SVG diagrams + a **"lens"** abstraction (reframe one diagram across
  flow/shapes/compute/memory).

---

## Deliverables

### 0. Reframe `AGENTS.md` for the research fork
Per `docs/contributing/editing-agent-instructions.md` (stay < 200 lines; no hardcoded paths;
offset additions). Two scoping edits, no rewrite of dev-workflow/commit sections:
- Header blockquote: reframe as `phi9t`'s research fork; upstream PRs *additionally* satisfy
  the Contribution Policy.
- Retitle `## 1. Contribution Policy (Mandatory)` → `## 1. Upstream Contribution Policy
  (when targeting vllm-project/vllm)` + a lead clarifying local research work (`devx/`,
  `hacks/`, `explorer/`) is not bound by it. Keep subsections intact.

### 1. `explorer/EXPLORER_DESIGN.md` — the distilled spec
The reference section above expanded into a standalone style guide.

### 2. `explorer/` — Tailwind 4 + ui-primitive scaffold

```
explorer/
  index.html
  package.json            # react19, vite8, tailwindcss4 + @tailwindcss/vite, framer-motion,
                          # lucide-react, recharts, radix slot/slider, cva, clsx,
                          # tailwind-merge, fira-code fontsource
  vite.config.ts          # react plugin + base = VITE_BASE_PATH ?? '/'
  tsconfig*.json
  scripts/                # all run via .venv/bin/python per AGENTS.md
    build_component_manifest.py  # parse HACKERS_GUIDE §2 + §16, grep symbols for line numbers
                                 # (drift-proof) -> public/data/components.json
    build_fineweb_sample.py      # fineweb-edu subset (datasets streaming, N rows) + Qwen3
                                 # tokenize -> fineweb_sample/schema/tokenization json
    build_qwen3_config.py        # Qwen3 config (e.g. Qwen3-0.6B) -> qwen3_config.json
    workflow.sh                  # reference workflow orchestrating the whole build
  public/data/
    components.json fineweb_sample.json fineweb_schema.json tokenization.json qwen3_config.json
  src/
    main.tsx App.tsx index.css
    lib/        assets.ts fetch.ts utils.ts
    components/ui/  button.tsx card.tsx slider.tsx
    data/       DataExplorer SchemaView SampleBrowser StatsView TokenizationView types.ts
    components-deepdive/  ComponentExplorer ArchitectureGraph ComponentDrawer types.ts
    architecture/  ArchitectureExplorer qwen3Blocks.ts blockMath.ts
```

**Build sequence:** scaffold + config → `index.css` tokens/shell → `lib/` helpers →
`components/ui/` primitives → `App.tsx` family switcher → 3 Python generators → Data mode →
Component mode → Architecture mode.

---

## Reuse map (lift, don't reinvent)
- Shell JSX/CSS ← `muon_optimizer/explorer/src/App.tsx`, `src/index.css`
- `dataUrl`/`fetchExplorerJson`/`formatMetric` ← `muon_optimizer/explorer/src/lib/*`
- `cn()` + `components/ui/*` ← `lumina/src/lib/utils.ts`, `lumina/src/components/ui/*`
- Recharts/schema/table ← `clinique/explorer/src/prescreen/{StatsView,SchemaView}.tsx`, `cdisc/CdiscExplorer.tsx`
- Byte-level Qwen `TokenPieces` ← `muon_optimizer/explorer/src/qwen/QwenLogitsExplorer.tsx`
- SVG diagram + lens + drawer + math ← `lumina/src/components/{block,dashboard}`, `lumina/src/lib/layerModel.ts`
- Component manifest source ← `HACKERS_GUIDE.md` (§2/§16) + `hacks/*.py`; resolve drift by grepping symbols
- Qwen3 inference diagram source ← `vllm/model_executor/models/qwen3.py`,
  `vllm/model_executor/layers/{layernorm.py,rotary_embedding/,attention/}`; tokenizer ←
  `vllm/tokenizers/registry.py` (`get_tokenizer`, HF `AutoTokenizer`)

---

## Verification
1. Generate data via `.venv/bin/python` (needs network once): `build_component_manifest.py`
   (every §2 box + §16 hack present, all `file:line` resolve), `build_fineweb_sample.py`
   (N rows + Qwen3 pieces/counts), `build_qwen3_config.py` (real config values).
2. `npm install`
3. `npm run dev` — observatory bg + fonts; family switcher (aria/skip/focus); Data mode
   (schema + dists + tokenization vs `token_count`); Component mode (clickable graph, drawer
   with working `file:line` + hack ref); Architecture mode (15-block diagram, lens recompute,
   block drawer, `GPUModelRunner` cross-link); 1024px collapse.
4. `npm run build` clean.
5. `EXPLORER_DESIGN.md` token table matches shipped `src/index.css`.

> Standalone TS/Vite app; independent of vLLM's Python build; not an upstream PR target.
