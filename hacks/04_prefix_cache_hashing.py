# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate that two prompts with a shared prefix produce equal block hashes.

Pairs with HACKERS_GUIDE.md §6 (Paged attention & the KV cache manager).

Demonstrates: vllm/v1/core/kv_cache_utils.py — get_request_block_hasher,
init_none_hash. No KVCacheManager, no Scheduler.
"""

from __future__ import annotations

import hashlib

from _stubs import print_header

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request


def sha256_bytes(obj: object) -> bytes:
    """Tiny stable hash function compatible with the hasher signature."""
    return hashlib.sha256(repr(obj).encode()).digest()


def make_request(req_id: str, tokens: list[int], hasher) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hasher,
    )


def main() -> int:
    BLOCK_SIZE = 4
    init_none_hash(sha256_bytes)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256_bytes)

    shared_prefix = [10, 20, 30, 40, 50, 60, 70, 80]  # 2 full blocks
    a = make_request("A", shared_prefix + [101, 102, 103, 104], hasher)
    b = make_request("B", shared_prefix + [201, 202, 203, 204], hasher)
    c = make_request("C", [99, 99, 99, 99] + [201, 202, 203, 204], hasher)

    print_header("Block hashes (per full BLOCK_SIZE=4 chunk)")
    for r in (a, b, c):
        ids = [h.hex()[:12] for h in r.block_hashes]
        print(f"  {r.request_id}: tokens={list(r.prompt_token_ids)}")
        print(f"     blocks -> {ids}")

    print_header("Conclusions")
    print(
        f"  A[0] == B[0]: {a.block_hashes[0] == b.block_hashes[0]} "
        "(same first prefix block, hashes equal)"
    )
    print(
        f"  A[1] == B[1]: {a.block_hashes[1] == b.block_hashes[1]} "
        "(same second prefix block, hashes equal)"
    )
    print(
        f"  A[2] == B[2]: {a.block_hashes[2] == b.block_hashes[2]} "
        "(diverging suffix, hashes differ)"
    )
    print(
        f"  A[0] == C[0]: {a.block_hashes[0] == c.block_hashes[0]} "
        "(different prefix, no sharing)"
    )
    print(
        f"  B[2] == C[1]: {b.block_hashes[2] == c.block_hashes[1]} "
        "(same suffix tokens but different prefix -> chained hash differs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
