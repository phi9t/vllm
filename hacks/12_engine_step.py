# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capstone: drive one full engine step end-to-end with zero CUDA.

Pairs with HACKERS_GUIDE.md §4 (EngineCore: the step loop).

This is the "I understand V1" exercise. It composes the miniature
scheduler from hack 05, the BlockPool from hack 03, and the sampling
op call from hack 07, into one engine step:

   1. Scheduler.schedule()
   2. Executor.execute_model() — faked by a deterministic stub
   3. Scheduler.update_from_output()
   4. OutputProcessor — represented by a trivial token-concat per req

Cross-reference against `EngineCore.step()` at
vllm/v1/engine/core.py:406.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from _stubs import print_header

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
_spec = importlib.util.spec_from_file_location(
    "_hack05_engine", _THIS / "05_scheduler_step.py"
)
_h05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h05)  # type: ignore[union-attr]

MiniScheduler = _h05.MiniScheduler
MiniRequest = _h05.MiniRequest


def fake_model(num_scheduled_tokens: dict[str, int], vocab: int = 32) -> dict[str, int]:
    """A stub for `Executor.execute_model` — returns one sampled id per req.

    Real flow: GPUModelRunner.execute_model returns a ModelRunnerOutput
    with sampled_token_ids per req_index. We just argmax on noise so the
    test is deterministic.
    """
    torch.manual_seed(0)
    out: dict[str, int] = {}
    for rid in num_scheduled_tokens:
        logits = torch.randn(vocab)
        out[rid] = int(logits.argmax(dim=-1).item())
    return out


def collect_output(
    req: MiniRequest, sampled_id: int, ledger: dict[str, list[int]]
) -> None:
    """Stand-in for OutputProcessor.process_outputs."""
    ledger.setdefault(req.request_id, []).append(sampled_id)


def main() -> int:
    sch = MiniScheduler(max_running=2, token_budget=8)
    sch.add(MiniRequest("alpha", prompt_len=3, max_new_tokens=2))
    sch.add(MiniRequest("beta", prompt_len=5, max_new_tokens=3))
    ledger: dict[str, list[int]] = {}

    step = 0
    while sch.waiting or sch.running:
        step += 1
        print_header(f"Engine step {step}")

        # 1. Scheduler.schedule()
        scheduled = sch.schedule()
        print(f"  scheduler.schedule() -> {scheduled.num_scheduled_tokens}")

        # 2. Executor.execute_model() — faked
        sampled = fake_model(scheduled.num_scheduled_tokens)
        print(f"  executor.execute_model() -> {sampled}")

        # 4. OutputProcessor — collect into a per-request ledger
        for r in sch.running:
            if r.request_id in sampled and r.num_computed_tokens >= r.prompt_len:
                collect_output(r, sampled[r.request_id], ledger)

        # 3. Scheduler.update_from_output()
        sch.update_from_output(scheduled, sampled)
        print(f"  running after step: {[r.request_id for r in sch.running]}")

        if step >= 10:
            print("  (cap at 10 steps)")
            break

    print_header("Final output ledger")
    for rid, toks in ledger.items():
        print(f"  {rid}: sampled token ids = {toks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
