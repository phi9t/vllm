# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simulate one Scheduler step.

Pairs with HACKERS_GUIDE.md §5 (The Scheduler).

Building a real `vllm.v1.core.sched.scheduler.Scheduler` requires a
fully-configured `VllmConfig` + `KVCacheConfig` + structured-output manager
+ model config. That's far too much setup for a hack script. Instead, we
implement a *miniature* scheduler that mirrors the real algorithm closely
enough to teach the moving parts. Cross-reference the line numbers below
against `vllm/v1/core/sched/scheduler.py` to follow along.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _stubs import print_header


@dataclass
class MiniRequest:
    request_id: str
    prompt_len: int  # tokens in the prompt
    max_new_tokens: int  # decode budget
    num_computed_tokens: int = 0
    num_output_tokens: int = 0
    status: str = "WAITING"  # WAITING | RUNNING | FINISHED


@dataclass
class MiniSchedulerOutput:
    scheduled_new_reqs: list[str] = field(default_factory=list)
    scheduled_running_reqs: list[str] = field(default_factory=list)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0


class MiniScheduler:
    """The shape of `Scheduler.schedule()` at ~scheduler.py:310.

    Two passes:
      1) Decode pass: each RUNNING request gets at most 1 new token.
      2) Prefill pass: WAITING requests are promoted with as many of
         their prompt tokens as fit in the remaining token budget.
         (`enable_chunked_prefill=True` is implicit here.)
    """

    def __init__(self, *, max_running: int, token_budget: int):
        self.max_running = max_running
        self.token_budget = token_budget
        self.waiting: list[MiniRequest] = []
        self.running: list[MiniRequest] = []

    def add(self, req: MiniRequest) -> None:
        self.waiting.append(req)

    def schedule(self) -> MiniSchedulerOutput:
        out = MiniSchedulerOutput()
        budget = self.token_budget

        # Decode pass — like Scheduler.schedule() at scheduler.py:310.
        for req in list(self.running):
            if budget <= 0:
                break
            out.scheduled_running_reqs.append(req.request_id)
            out.num_scheduled_tokens[req.request_id] = 1
            budget -= 1

        # Prefill pass — promote waiting under remaining budget and
        # max_num_running_reqs (scheduler.py uses self.max_num_running_reqs
        # and the chunked-prefill knob at line 643).
        while self.waiting and len(self.running) < self.max_running and budget > 0:
            req = self.waiting.pop(0)
            req.status = "RUNNING"
            self.running.append(req)
            chunk = min(req.prompt_len - req.num_computed_tokens, budget)
            req.num_computed_tokens += chunk
            out.scheduled_new_reqs.append(req.request_id)
            out.num_scheduled_tokens[req.request_id] = chunk
            budget -= chunk

        out.total_tokens = self.token_budget - budget
        return out

    def update_from_output(
        self, scheduled: MiniSchedulerOutput, sampled: dict[str, int]
    ) -> None:
        """Mirror of Scheduler.update_from_output at scheduler.py:1248.

        Appends 1 sampled token per request, retires anything that hit
        its max_new_tokens budget.
        """
        for rid in scheduled.scheduled_running_reqs:
            if rid not in sampled:
                continue
            for req in self.running:
                if req.request_id != rid:
                    continue
                req.num_output_tokens += 1
                if req.num_output_tokens >= req.max_new_tokens:
                    req.status = "FINISHED"
        self.running = [r for r in self.running if r.status != "FINISHED"]


def main() -> int:
    sch = MiniScheduler(max_running=4, token_budget=16)
    sch.add(MiniRequest("short", prompt_len=4, max_new_tokens=2))
    sch.add(MiniRequest("medium", prompt_len=8, max_new_tokens=3))
    sch.add(MiniRequest("longish", prompt_len=12, max_new_tokens=1))

    for step in range(1, 5):
        print_header(f"Engine step {step}")
        out = sch.schedule()
        print(f"  new      : {out.scheduled_new_reqs}")
        print(f"  running  : {out.scheduled_running_reqs}")
        print(f"  per-req  : {out.num_scheduled_tokens}")
        print(f"  total tok: {out.total_tokens} / budget {sch.token_budget}")

        # Pretend the model sampled one token per RUNNING request.
        sampled = {rid: 999 for rid in out.scheduled_running_reqs}
        sch.update_from_output(out, sampled)
        print(f"  running after step: {[r.request_id for r in sch.running]}")
        if not sch.running and not sch.waiting:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
