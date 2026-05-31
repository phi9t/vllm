# Design: Deep-dive the first three Foundations mental models + render diagrams

**Date:** 2026-05-30
**Status:** Approved (brainstorming)
**Area:** `HACKERS_GUIDE.md` §4 + the explorer "Hacker's Guide" page

## Context

`HACKERS_GUIDE.md` (v0.22.0) has a §4 "Foundations" section introducing five mental models in
~one paragraph each. The first three are the conceptual core every later section leans on, and the
user wants them expanded into thorough deep dives:

1. §4.1 Continuous batching & the unified token model
2. §4.2 Paged memory (KV cache as virtual memory)
3. §4.3 Prefill vs decode as compute profiles

The guide is also rendered as a page on the site (the "Hacker's Guide" explorer mode reads
`explorer/public/data/guide.md` and renders it with react-markdown). Today ` ```mermaid ` fences
render as **source code**, not drawn diagrams — so the deep dives' visuals would not display. This
design therefore also enables real mermaid rendering on that page.

**Audience:** ML-engineer / systems-hacker (assume transformers + GPU basics), but cover
fundamentals briefly inline (define a term in a clause, not a paragraph) so each deep dive is
self-contained.

## Grounding (verified against the v0.22.0 tree)

- Unified token model: `scheduler.py:329` NOTE; `num_tokens_with_spec = len(prompt+output) +
  len(spec_draft)` (`vllm/v1/request.py:243`); `num_computed_tokens` starts 0 (`request.py:145`).
- Paged memory: `KVCacheBlock { block_id, ref_cnt }` + doubly-linked free queue
  (`vllm/v1/core/kv_cache_utils.py:116`); `BlockPool` (`vllm/v1/core/block_pool.py:130`);
  `block_size` default 16 (`vllm/config/cache.py:47`).
- Prefill/decode: the kernel sees decode as `max_query_len==1` / `cu_query_lens` with a captured
  fast path (`vllm/v1/attention/backends/flash_attn.py:233,281`).

## Part 1 — Mermaid rendering on the guide page (enabler)

- Add `mermaid` as a dependency, **lazy-loaded via dynamic `import()`** only when the Guide mode
  mounts, so the main bundle stays ~789 KB and mermaid loads as a separate on-demand chunk.
- In `explorer/src/guide/GuideExplorer.tsx`, override the markdown `code` renderer: a
  `language-mermaid` fence is rendered to SVG with `mermaid.render()` (unique id per block) and
  injected; on parse error, fall back to showing the source verbatim (never blank).
- Theme to Observatory: `mermaid.initialize({ startOnLoad:false, theme:'base', themeVariables:… })`
  with void background, indigo/cyan lines, glass node fills, Fira Code font.
- Payoff: every diagram draws — existing §3/§7/§9/§15 plus the new §4 ones. Respect
  `prefers-reduced-motion` (mermaid is static, so no extra motion).

## Part 2 — The three deep dives

Each follows: **problem → model/mechanism → worked example → consequences**, grounded, terse.

### §4.1 Unified token model
- Problem: static/request-level batching → head-of-line blocking; iteration-level scheduling (Orca).
- Model: the two counters; per step, assign tokens so `num_computed_tokens` catches up to
  `num_tokens_with_spec`. Prefill = big catch-up, decode = catch-up of 1.
- Worked example: a **counter-trace table** (one request across ~4 steps) + a **mermaid** lifecycle
  diagram (repeated catch-up).
- Consequences: chunked prefill / prefix caching / spec decode are the *same loop* with different
  counter values (forward-ref §21).

### §4.2 Paged memory
- Problem (quantified): contiguous `max_model_len` slab → internal frag (4096-slab for 40 tokens ≈
  99% waste) + external frag (admission blocked unless a full slab is free).
- Model: the OS-VM mapping as a **table** (block=page, block-table=page-table, pool=physical RAM,
  shared prefix=shared page, `ref_cnt`); doubly-linked free queue; one-block-at-a-time growth.
- Worked example: a **mermaid logical→physical block map** with two requests sharing a prefix;
  a contiguous-vs-paged tokens-held comparison (ties to §10 numbers).
- Consequences: near-zero waste, prefix sharing, eviction only of `ref_cnt==0` (forward-ref §9, §11).

### §4.3 Prefill vs decode
- Model: arithmetic intensity vs the roofline; prefill = compute-bound (big GEMMs, O(n²) attention);
  decode = memory-bandwidth-bound (1 token, stream all weights + KV).
- Worked example: an **ASCII roofline** (reliable in a fenced block) + a **prefill-vs-decode table**
  + the **decode latency floor** — primary number on H100 (~3.35 TB/s): `14 GB ÷ 3.35 TB/s ≈
  4.2 ms/token` at batch 1; formula stated hardware-agnostic; batching amortizes the weight read.
- Code bridge: scheduler is unified, but the kernel distinguishes via `max_query_len==1` captured
  fast path (`flash_attn.py:281`, §19).
- Consequences: batching helps decode most; long prefills starve decode (→ chunked prefill);
  quantization/GQA/MLA shrink decode bytes (→ §10, §22).

## Part 3 — Mechanics & scope

- Edit only §4.1–4.3 in `HACKERS_GUIDE.md` (no renumbering; still 27 sections).
- Add `mermaid` dep; implement rendering + theme in `GuideExplorer.tsx`; regenerate
  `explorer/public/data/guide.md` and `components.json` via `build_component_manifest.py`.
- Out of scope: §4.4/§4.5 (left as-is), the other sections, hacks.

## Verification

1. Markdown consistency: 27 headers == 27 TOC entries; no stale `§N` (>27).
2. New anchors grep-verified at v0.22.0 (request.py, kv_cache_utils.py, flash_attn.py).
3. Regenerate manifest/page: 27 sections / 18 hacks, refs resolve; `guide.md` re-emitted with §4.
4. `npm install` → **`npm ci` consistent** (avoid the lockfile drift that broke CI before);
   `npx tsc -b --noEmit` + `npm run build` clean; confirm a separate mermaid chunk is emitted and
   `dist/data/guide.md` has the deep §4.
5. Commit + push to the `phi9t` fork; `explorer-pages.yml` green; site + `/data/guide.md` 200;
   **manually confirm a mermaid diagram draws** (not source) on the live guide page.
