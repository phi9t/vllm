# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show a long prompt sliced across multiple steps (chunked prefill).

Pairs with HACKERS_GUIDE.md §5 (The Scheduler).

Reuses the miniature scheduler from hack 05 to demonstrate the
chunked-prefill knob effect (`enable_chunked_prefill`,
vllm/v1/core/sched/scheduler.py:643). With a small token budget and a
long prompt, the prefill is sliced; with `enable_chunked_prefill=False`
the long prompt would either fit in one step or block.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
_spec = importlib.util.spec_from_file_location(
    "_hack05", _THIS / "05_scheduler_step.py"
)
_h05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h05)  # type: ignore[union-attr]

MiniScheduler = _h05.MiniScheduler
MiniRequest = _h05.MiniRequest

from _stubs import print_header  # noqa: E402


def main() -> int:
    print_header("Setup: 1 long prompt, tiny token budget")
    sch = MiniScheduler(max_running=1, token_budget=64)
    sch.add(MiniRequest("long-doc", prompt_len=512, max_new_tokens=4))
    print("  prompt_len=512  token_budget=64  -> needs >= 8 prefill steps")

    step = 0
    while sch.waiting or sch.running:
        step += 1
        out = sch.schedule()
        # Snapshot the request's current computed-tokens count so we can
        # report progress before update_from_output.
        progress = {
            r.request_id: f"{r.num_computed_tokens}/{r.prompt_len}" for r in sch.running
        }
        print(
            f"  step {step:>2}: scheduled_tokens={out.num_scheduled_tokens} "
            f"prompt_progress={progress}"
        )
        # No model: only mark a request "running for decode" once its
        # prompt is fully consumed.
        sampled = {}
        for r in sch.running:
            if r.num_computed_tokens >= r.prompt_len:
                sampled[r.request_id] = 1
        sch.update_from_output(out, sampled)
        if step >= 16:
            print("  (capped at 16 steps for the demo)")
            break

    print_header("Takeaway")
    print("  Chunked prefill lets long-prompt requests share the engine")
    print("  with decode-only requests by slicing the prefill across")
    print("  many steps. Real scheduler: scheduler.py:643 reads the knob.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
