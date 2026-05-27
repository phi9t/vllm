# hacks/

Runnable scripts that exercise individual V1 engine components in
isolation. Companion to [`../HACKERS_GUIDE.md`](../HACKERS_GUIDE.md).

**Run with the project venv:**

```bash
.venv/bin/python hacks/05_scheduler_step.py
```

| # | Script | Pairs with guide section |
| --- | --- | --- |
| 01 | `01_llm_smoke.py` | §3 — `LLM.generate` end-to-end (real model, `VLLM_HACK_RUN_MODEL=1`) |
| 02 | `02_request_lifecycle.py` | §10 — `Request` and `RequestStatus` |
| 03 | `03_kv_cache_manager.py` | §6 — `BlockPool` allocate / share / free |
| 04 | `04_prefix_cache_hashing.py` | §6 — `get_request_block_hasher` shows prefix sharing |
| 05 | `05_scheduler_step.py` | §5 — miniature `Scheduler.schedule()` |
| 06 | `06_chunked_prefill.py` | §5 — chunked prefill across multiple steps |
| 07 | `07_sampler.py` | §9 — temperature / top-k / top-p / bad-words side by side |
| 08 | `08_logits_processor.py` | §9 — custom `LogitsProcessor` that bans a token |
| 09 | `09_attention_backend_select.py` | §8 — backend inventory + selector inputs |
| 10 | `10_output_processor.py` | §10 — `OutputProcessor` API surface |
| 11 | `11_executor_uniproc.py` | §11/§12 — `Executor` ABC + concrete implementations |
| 12 | `12_engine_step.py` | §4 — capstone: one full engine step, zero CUDA |
| 13 | `13_async_llm_stream.py` | §3 — `AsyncLLM` streaming (real model, `VLLM_HACK_RUN_MODEL=1`) |

Shared helpers live in [`_stubs.py`](_stubs.py).

## Why the heavier components don't instantiate the real class

A few scripts (05, 06, 09, 10, 11, 12) demonstrate concepts without
actually constructing a real `Scheduler`, `OutputProcessor`, or
`Executor`. That's deliberate: those classes pull in a
`VllmConfig` / `KVCacheConfig` / loadable model, which is too much
setup for a tutorial script. The hacks either:

- **Mirror** the real algorithm in <100 lines (e.g. `MiniScheduler` in
  `05_scheduler_step.py`) — annotated with line-number references to
  the real implementation, so you can read both side by side.
- **Inspect** the class via `inspect.signature` to surface the API
  contract you'd hit when integrating (e.g. `10_output_processor.py`).

Scripts 02, 03, 04, 07, and 08 use the real V1 types directly.

## Smoke test

`tests/hacks/test_hacks_smoke.py` runs every script except 01 and 13
with a 30 s timeout and asserts exit code 0. If you change an internal
signature and the hacks stop running, that's the first signal that
this guide needs an update.
