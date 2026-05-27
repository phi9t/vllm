"""Walk a Request through the RequestStatus state machine.

Pairs with HACKERS_GUIDE.md §10 (Request lifecycle & output).

Demonstrates: vllm/v1/request.py — Request, RequestStatus.
No engine, no model, no GPU.
"""

from __future__ import annotations

from _stubs import print_header

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def main() -> int:
    print_header("Constructing a Request")
    sampling = SamplingParams(max_tokens=4)
    req = Request(
        request_id="hack-2",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=sampling,
        pooling_params=None,
    )
    print(f"  request_id      = {req.request_id}")
    print(f"  num_prompt_toks = {req.num_prompt_tokens}")
    print(f"  status (init)   = {req.status}")
    print(f"  is_finished()   = {req.is_finished()}")

    print_header("Walking the state machine")
    transitions = [
        ("admit to scheduler", RequestStatus.RUNNING),
        ("preempted by KV pressure", RequestStatus.PREEMPTED),
        ("re-admitted", RequestStatus.RUNNING),
        ("hit EOS", RequestStatus.FINISHED_STOPPED),
    ]
    for note, new_status in transitions:
        req.status = new_status
        finished = req.is_finished()
        reason = req.get_finished_reason()
        print(
            f"  {note:30s} -> status={new_status:32s}"
            f" finished={finished} reason={reason}"
        )

    print_header("Appending sampled tokens")
    req.append_output_token_ids([42, 43])
    print(f"  num_output_tokens = {req.num_output_tokens}")
    print(f"  output_token_ids  = {list(req.output_token_ids)}")
    print(f"  all_token_ids     = {list(req.all_token_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
