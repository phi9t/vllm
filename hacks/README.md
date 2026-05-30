# hacks/

Runnable scripts that exercise individual V1 engine components in
isolation. Companion to [`../HACKERS_GUIDE.md`](../HACKERS_GUIDE.md)
(updated for vLLM **v0.22.0**).

**Run with the project venv:**

```bash
.venv/bin/python hacks/05_scheduler_step.py
```

| # | Script | Pairs with guide section |
| --- | --- | --- |
| 01 | `01_llm_smoke.py` | §4 — `LLM.generate` end-to-end (real model, `VLLM_HACK_RUN_MODEL=1`) |
| 02 | `02_request_lifecycle.py` | §14 — `Request` and `RequestStatus` |
| 03 | `03_kv_cache_manager.py` | §8 / §10 — `BlockPool` allocate / share / free |
| 04 | `04_prefix_cache_hashing.py` | §8 — `get_request_block_hasher` shows prefix sharing |
| 05 | `05_scheduler_step.py` | §7 — miniature `Scheduler.schedule()` |
| 06 | `06_chunked_prefill.py` | §7 — chunked prefill across multiple steps |
| 07 | `07_sampler.py` | §13 — temperature / top-k / top-p / bad-words side by side |
| 08 | `08_logits_processor.py` | §13 — custom `LogitsProcessor` that bans a token |
| 09 | `09_attention_backend_select.py` | §12 — backend inventory + selector inputs |
| 10 | `10_output_processor.py` | §14 — `OutputProcessor` API surface |
| 11 | `11_executor_uniproc.py` | §16 / §19 — `Executor` ABC + concrete implementations |
| 12 | `12_engine_step.py` | §6 / §17 — capstone: one full engine step, zero CUDA |
| 13 | `13_async_llm_stream.py` | §4 — `AsyncLLM` streaming (real model, `VLLM_HACK_RUN_MODEL=1`) |
| 14 | `14_input_processor.py` | §5 — prompt → token ids → `EngineCoreRequest` shape |
| 15 | `15_kv_cache_sizing.py` | §9 — KV-cache sizing arithmetic (GQA vs MLA) |
| 16 | `16_batch_packing.py` | §11 — flatten a mixed prefill+decode batch |
| 17 | `17_model_registry.py` | §15 / §21 — registry/`get_model` + param & KV footprint |
| 18 | `18_cudagraph_bucketing.py` | §18 — pad batch sizes to captured CUDA-graph buckets |

Shared helpers live in [`_stubs.py`](_stubs.py).

## Why the heavier components don't instantiate the real class

Many scripts (05, 06, 09, 10, 11, 12, 14, 15, 16, 18) demonstrate concepts
without constructing a real `Scheduler`, `OutputProcessor`, `Executor`, or
`GPUModelRunner`. That's deliberate: those classes pull in a
`VllmConfig` / `KVCacheConfig` / loadable model, which is too much
setup for a tutorial script. The hacks either:

- **Mirror** the real algorithm in <100 lines (e.g. `MiniScheduler` in
  `05_scheduler_step.py`, the sizing math in `15_kv_cache_sizing.py`, the
  batch packing in `16_batch_packing.py`) — annotated with line-number
  references to the real implementation, so you can read both side by side.
- **Inspect** the class via `inspect.signature` to surface the API
  contract you'd hit when integrating (e.g. `10_output_processor.py`,
  `17_model_registry.py`).

Scripts 02, 03, 04, 07, and 08 use the real V1 types directly. The
arithmetic hacks (14–18) need no vLLM import at all and run anywhere.

## Smoke test

`tests/hacks/test_hacks_smoke.py` runs every script except 01 and 13
with a 30 s timeout and asserts exit code 0. If you change an internal
signature and the hacks stop running, that's the first signal that
this guide needs an update.
