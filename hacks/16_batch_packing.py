# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack a mixed prefill+decode batch into one flat token tensor.

Pairs with HACKERS_GUIDE.md §11 (Continuous batching, in code).

V1 doesn't batch per-request — it flattens the whole step into a single
1-D token sequence and tells the attention kernel where each request
lives via `query_start_loc` and `seq_lens`. The real packing happens in
vllm/v1/worker/gpu_input_batch.py; here we build the same arrays from a
synthetic batch so the layout is obvious. No torch, no CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

from _stubs import print_header

BLOCK_SIZE = 16


@dataclass
class ReqInBatch:
    req_id: str
    num_query_tokens: int  # tokens computed THIS step (prefill chunk or 1 for decode)
    context_len: int       # tokens already in the KV cache before this step
    block_ids: list[int]   # physical KV blocks owned by this request


def pack(reqs: list[ReqInBatch]) -> dict:
    """Mirror of the SchedulerOutput -> flat-tensor packing."""
    token_ids: list[int] = []          # the single 1-D sequence for the step
    query_start_loc = [0]              # prefix-sum of per-req query lengths
    seq_lens: list[int] = []           # total sequence length per req (ctx + query)
    slot_mapping: list[int] = []       # flat KV slot for each query token

    next_tok = 1000
    for r in reqs:
        for _ in range(r.num_query_tokens):
            token_ids.append(next_tok)
            next_tok += 1
        query_start_loc.append(query_start_loc[-1] + r.num_query_tokens)
        seq_lens.append(r.context_len + r.num_query_tokens)
        # Each query token writes into the next free slot of the request's blocks.
        for pos in range(r.context_len, r.context_len + r.num_query_tokens):
            block = r.block_ids[pos // BLOCK_SIZE]
            slot_mapping.append(block * BLOCK_SIZE + (pos % BLOCK_SIZE))

    return {
        "num_reqs": len(reqs),
        "total_query_tokens": len(token_ids),
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "slot_mapping": slot_mapping,
    }


def main() -> int:
    # Two prefills (a long chunk + a short one) and one decode, in one step.
    reqs = [
        ReqInBatch("prefill-A", num_query_tokens=20, context_len=0,
                   block_ids=[10, 11]),                 # 20 prompt tokens, 2 blocks
        ReqInBatch("decode-B", num_query_tokens=1, context_len=33,
                   block_ids=[20, 21, 22]),             # 1 new token at pos 33
        ReqInBatch("prefill-C", num_query_tokens=6, context_len=0,
                   block_ids=[30]),                     # short prompt
    ]

    print_header("Per-request inputs (what the scheduler decided)")
    for r in reqs:
        kind = "decode " if r.num_query_tokens == 1 and r.context_len else "prefill"
        print(f"  {r.req_id:10s} {kind}  query={r.num_query_tokens:2d}  "
              f"ctx={r.context_len:2d}  blocks={r.block_ids}")

    out = pack(reqs)
    print_header("Flattened step tensors (one batch, no padding)")
    print(f"  num_reqs            = {out['num_reqs']}")
    print(f"  total_query_tokens  = {out['total_query_tokens']}  "
          "(prefill and decode share ONE forward pass)")
    print(f"  query_start_loc     = {out['query_start_loc']}  "
          "(req i = [qsl[i]:qsl[i+1]])")
    print(f"  seq_lens            = {out['seq_lens']}  (ctx + query, for attention)")
    print(f"  slot_mapping[:8]    = {out['slot_mapping'][:8]} ...  "
          f"(len {len(out['slot_mapping'])})")

    # Sanity: query_start_loc is a prefix sum; last == total.
    assert out["query_start_loc"][-1] == out["total_query_tokens"]
    assert len(out["slot_mapping"]) == out["total_query_tokens"]
    print_header("Takeaway")
    print("  query_start_loc partitions the flat tensor per request; seq_lens")
    print("  gives attention each request's full length. No prefill/decode")
    print("  split — the block table + slot_mapping let one kernel serve both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
