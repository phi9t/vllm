# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocate, share, and free KV blocks using BlockPool directly.

Pairs with HACKERS_GUIDE.md §6 (Paged attention & the KV cache manager).

Demonstrates: vllm/v1/core/block_pool.py — BlockPool.get_new_blocks,
free_blocks, get_cached_block. No model, no GPU, no real KV cache.
"""

from __future__ import annotations

from _stubs import print_header

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)


def main() -> int:
    BLOCK_SIZE = 16
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )
    print_header("Pool state at boot")
    print(f"  num_gpu_blocks = {pool.num_gpu_blocks}")
    print(f"  free blocks    = {pool.get_num_free_blocks()}")
    print(f"  null_block id  = {pool.null_block.block_id}")

    print_header("Request A: allocate 3 blocks (e.g. 48 prompt tokens)")
    blocks_a = pool.get_new_blocks(3)
    print(f"  got block_ids  = {[b.block_id for b in blocks_a]}")
    print(f"  free remaining = {pool.get_num_free_blocks()}")

    print_header("Cache the first block under a synthetic hash")
    fake_hash = BlockHash(b"shared-prefix-block-0")
    keyed_hash = make_block_hash_with_group_id(fake_hash, 0)
    blocks_a[0].block_hash = keyed_hash  # uses the property setter
    pool.cached_block_hash_to_block.insert(keyed_hash, blocks_a[0])
    print(f"  block {blocks_a[0].block_id} cached under hash {fake_hash!r}")

    print_header("Request B: look up the same prefix block before allocating")
    hit = pool.get_cached_block(fake_hash, kv_cache_group_ids=[0])
    print(f"  cache hit -> blocks {[b.block_id for b in hit] if hit else None}")
    if hit:
        pool.touch(hit)
        print(
            f"  block {hit[0].block_id} ref_cnt now = {hit[0].ref_cnt}"
            f" (shared between A and B)"
        )

    print_header("Free request A's tail blocks")
    pool.free_blocks(reversed(blocks_a[1:]))
    print(f"  free blocks    = {pool.get_num_free_blocks()}")
    print(f"  shared block ref_cnt = {blocks_a[0].ref_cnt} (still held by B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
