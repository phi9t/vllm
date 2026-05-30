# vLLM Explorer

> **Live:** **[phi9t.github.io/vllm](https://phi9t.github.io/vllm/)**

An interactive, dark-glass single-page app for exploring vLLM — data, engine internals,
and model architecture — built as a local developer-experience tool of the `phi9t`
research fork (not an upstream artifact). React 19 + TypeScript + Vite + Tailwind 4,
in the **"Observatory"** design language.

## Three modes (one family-switcher shell)

| Mode | What it shows |
|------|---------------|
| **Data Exploration** | A small **FineWeb-Edu** subset — schema/format, sample browser, distribution charts, and a **Qwen3 tokenization** view (byte-level pieces, Qwen3 vs the dataset's GPT-2 `token_count`). |
| **Component Deep Dive** | The **V1 engine** as a clickable graph — an interactive companion to [`../HACKERS_GUIDE.md`](../HACKERS_GUIDE.md) + [`../hacks/`](../hacks/). Each node opens a drawer with its grounded `file:line` source link and the paired runnable hack. |
| **Model Architecture** | The **inference forward pass** as a residual-mainline circuit, switchable across **Qwen3-0.6B**, **Qwen3-8B** (dense), **Qwen3-30B-A3B** (MoE), and **DeepSeek-V3** (MLA + MoE). A **lens** system reframes the same diagram across flow / shapes / compute (FLOPs) / memory (KV-cache), all computed in-browser from each model's real config. |

Every mode is **data-driven and switchable**: an `index.json` lists subjects, each
pointing at a per-subject manifest; the SPA is a pure reader and the generators under
[`scripts/`](scripts/) are the source of truth. Code references are **grounded** to real
`file:line` locations by symbol-grep, so they survive line drift.

## Run it

All steps are wrapped in the reference workflow script. Python runs through the repo's
uv-managed `.venv` (see [`../AGENTS.md`](../AGENTS.md)) — never system `python3`.

```bash
cd explorer
./scripts/workflow.sh gen-data   # regenerate public/data/* manifests (needs the repo .venv)
./scripts/workflow.sh install    # npm install
./scripts/workflow.sh dev        # vite dev server
# or: ./scripts/workflow.sh      # gen-data + install + build
```

Static manifests live in [`public/data/`](public/data) (`datasets/`, `graphs/`,
`models/` indices + their manifests). Deployed to GitHub Pages by
[`../.github/workflows/explorer-pages.yml`](../.github/workflows/explorer-pages.yml) on
every push touching `explorer/**`.

## Design requirements (the pillars)

The explorer is held to a small set of enforced rules — full spec in
[`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md):

- **No text overflow + succinct blocks** — circuit box widths are never hard-coded; they
  are computed from monospace text length so labels/metrics always fit. Block labels stay
  terse; full prose lives in the drawer.
- **Strong aesthetics** — the Observatory glass language (kind-tinted gradient cards,
  rounded-elbow wiring, gradient mainline, selection glow), reused via the shared kit.
- **Strong interactability** — every node is click- and keyboard-operable, selection opens
  a detail drawer, and each mode offers subject/sub-view/param controls with cross-links.
- Plus accessibility, the data-driven manifest contract, portability, hooks-only state, and
  switchable subjects in every mode.

## Layout

```
explorer/
  src/
    App.tsx                 # family-switcher shell
    explorer-kit/           # shared primitives: AsyncBoundary, ViewTabs,
                            #   SubjectSwitcher, DetailDrawer
    data/ components-deepdive/ architecture/   # the three modes
    components/ui/          # shadcn-style Button / Card / Slider
    lib/                    # fetch, assets (dataUrl/sourceUrl), cn
  public/data/              # generated manifests (datasets/ graphs/ models/)
  scripts/                  # Python generators + workflow.sh (run via repo .venv)
  EXPLORER_DESIGN.md        # the "Observatory" design language / style guide
  GENERALIZATION_WORKFLOW.md# requirements (R1–R8) + how to add a subject / mode
```

## Docs

- [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) — **how to build one**: a rigorous
  builder's manual for the two SVG diagram families (model circuit + system-architecture
  component graph), the layout/no-overflow/containment algorithms, and the Observatory aesthetics.
- [`EXPLORER_DESIGN.md`](EXPLORER_DESIGN.md) — the distilled design language (tokens,
  shell, data contract, UI primitives, the no-overflow invariant).
- [`GENERALIZATION_WORKFLOW.md`](GENERALIZATION_WORKFLOW.md) — the requirements spec and the
  step-by-step runbook for adding a new subject (model/dataset/subsystem) or a new mode.
