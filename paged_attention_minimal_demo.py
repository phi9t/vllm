#!/usr/bin/env python3
"""
PagedAttention: Minimalist Core Demo & Test
============================================

This is the simplest possible demonstration of vLLM's PagedAttention.
Shows the core mechanics without any framework complexity.

Core Question: How do we store K/V caches for variable-length sequences
               without wasting memory on fragmentation and padding?

Answer: PagedAttention - store K/V in fixed-size blocks, like OS paging.
"""

import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class MinimalPagedAttention:
    """
    Minimalist PagedAttention implementation.

    Key Components:
    ---------------
    1. KV Cache: Fixed-size blocks instead of contiguous arrays
    2. Block Table: Maps logical positions -> physical block IDs
    3. Slot Mapping: Computes physical address from logical position

    The Innovation:
    ---------------
    Traditional: Contiguous K/V arrays per sequence
                  [seq_len, num_heads, head_dim]

    PagedAttention: Blocked K/V storage
                  [num_blocks, num_heads, head_dim, block_size]

    Benefit: Non-contiguous allocation, memory sharing, no padding waste
    """

    def __init__(
        self,
        num_blocks: int = 10,
        block_size: int = 4,
        num_heads: int = 2,
        head_dim: int = 8,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical storage: [num_blocks, 2 (K/V), num_heads, head_dim, block_size]
        # This is the KV cache memory pool
        self.kv_cache = torch.zeros(num_blocks, 2, num_heads, head_dim, block_size)

        # Block allocation state
        self.free_blocks = list(range(num_blocks))

        # Block table: sequence_id -> ordered list of block IDs
        self.block_tables = {}

        print(f"🚀 MinimalPagedAttention initialized")
        print(
            f"   Blocks: {num_blocks} × {block_size} tokens = {num_blocks * block_size} max tokens"
        )
        print(f"   Per token: {num_heads} heads × {head_dim} dims")
        print(
            f"   Memory: {num_blocks * block_size * num_heads * head_dim * 2 * 4 / 1024:.1f} KB (fp32)"
        )

    def allocate_sequence(self, seq_id: str, length: int) -> List[int]:
        """
        Allocate blocks for a sequence.

        Like OS allocating page frames for a process.
        Block table is the "page table" for the sequence.
        """
        num_blocks_needed = (length + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(
                f"Out of blocks: need {num_blocks_needed}, have {len(self.free_blocks)}"
            )

        # Allocate blocks (FIFO from free list)
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated

        print(
            f"\n📦 Allocated {num_blocks_needed} blocks for '{seq_id}' (length={length})"
        )
        print(f"   Block table: {allocated}")

        return allocated

    def logical_to_physical(self, seq_id: str, logical_pos: int) -> Tuple[int, int]:
        """
        Convert logical position to physical (block_id, offset).

        This is the core translation:
        - logical_pos: 0, 1, 2, 3, 4, 5, ... (contiguous, per-sequence)
        - block_id: which physical block stores this token
        - offset: position within that block

        Example with block_size=4:
            Position 0 -> (block_0, offset_0)
            Position 3 -> (block_0, offset_3)
            Position 4 -> (block_1, offset_0)  <- new block!
            Position 7 -> (block_1, offset_3)
        """
        block_table = self.block_tables[seq_id]

        block_idx = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        block_id = block_table[block_idx]

        return block_id, offset

    def write_kv(
        self, seq_id: str, position: int, key: torch.Tensor, value: torch.Tensor
    ) -> int:
        """
        Write K/V vectors to cache at given position.

        Returns: slot_id (for debugging)
        """
        block_table = self.block_tables[seq_id]
        block_idx = position // self.block_size

        # Allocate more blocks if needed
        while block_idx >= len(block_table):
            if not self.free_blocks:
                raise MemoryError(
                    f"Out of blocks for '{seq_id}' at position {position}"
                )
            new_block = self.free_blocks.pop(0)
            block_table.append(new_block)

        block_id, offset = self.logical_to_physical(seq_id, position)

        # Store in physical cache
        self.kv_cache[block_id, 0, :, :, offset] = key  # K
        self.kv_cache[block_id, 1, :, :, offset] = value  # V

        # Compute slot (linearized address for kernel)
        slot = block_id * self.block_size + offset

        if position < 3 or position % 5 == 0:
            print(
                f"   💾 Pos {position:2d} -> Block {block_id}, Offset {offset} (slot={slot})"
            )

        return slot

    def read_kv(self, seq_id: str, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K/V vectors from cache at given position."""
        block_id, offset = self.logical_to_physical(seq_id, position)

        key = self.kv_cache[block_id, 0, :, :, offset]
        value = self.kv_cache[block_id, 1, :, :, offset]

        return key, value

    def paged_attention(
        self, seq_id: str, query: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Compute attention using paged K/V cache.

        This is what the CUDA kernel does:
        1. Gather K/V from non-contiguous blocks
        2. Compute attention scores
        3. Return weighted sum
        """
        # Gather all K/V for this sequence from blocks
        keys = []
        values = []

        for pos in range(seq_len):
            k, v = self.read_kv(seq_id, pos)
            keys.append(k)
            values.append(v)

        # Stack: [seq_len, num_heads, head_dim]
        keys = torch.stack(keys)
        values = torch.stack(values)

        # Compute attention (scaled dot-product)
        # query: [num_heads, head_dim]
        # keys: [seq_len, num_heads, head_dim]
        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.einsum("hd,shd->hs", query, keys) * scale

        # Softmax and apply to values
        weights = F.softmax(scores, dim=-1)
        output = torch.einsum("hs,shd->hd", weights, values)

        return output


def demo_basic_paging():
    """
    Demo 1: Show basic paging mechanism.

    How PagedAttention stores 10 tokens in 2 blocks of size 4.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Paging (10 tokens in blocks of 4)")
    print("=" * 60)

    attn = MinimalPagedAttention(
        num_blocks=5,
        block_size=4,  # Each block holds 4 tokens
        num_heads=2,
        head_dim=8,
    )

    # Allocate for sequence of 10 tokens
    seq_id = "demo_seq"
    attn.allocate_sequence(seq_id, length=10)

    # Write 10 tokens worth of K/V
    print("\n📝 Writing K/V cache:")
    for pos in range(10):
        k = torch.randn(2, 8)
        v = torch.randn(2, 8)
        attn.write_kv(seq_id, pos, k, v)

    print(f"\n📊 Block breakdown:")
    print(f"   Block 0: positions 0-3 (4 tokens)")
    print(f"   Block 1: positions 4-7 (4 tokens)")
    print(f"   Block 2: positions 8-9 (2 tokens, 2 empty)")
    print(f"   Waste: 2/12 = 16.7% (much better than 50% with padding!)")

    # Compute attention for new token
    print("\n🎯 Computing attention for new token:")
    query = torch.randn(2, 8)
    output = attn.paged_attention(seq_id, query, seq_len=10)
    print(f"   Output shape: {output.shape}")
    print(f"   ✓ Successfully attended to all 10 tokens from paged cache!")


def demo_memory_comparison():
    """
    Demo 2: Compare PagedAttention vs traditional contiguous allocation.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Memory Efficiency Comparison")
    print("=" * 60)

    # Three sequences of different lengths
    seq_lens = [10, 25, 7]
    block_size = 8
    num_heads = 8
    head_dim = 64

    print(f"\n📋 Setup: 3 sequences with lengths {seq_lens}")
    print(f"   Block size: {block_size}")
    print(
        f"   Per token: {num_heads} heads × {head_dim} dims = {num_heads * head_dim * 2 / 1024:.1f} KB (K+V)"
    )

    # Traditional approach: allocate max length for all, truncate
    max_len = max(seq_lens)
    traditional_tokens = len(seq_lens) * max_len

    print(f"\n🏛️  TRADITIONAL (contiguous allocation):")
    print(f"   Allocate {max_len} for each sequence (max length padding)")
    print(
        f"   Total tokens allocated: {len(seq_lens)} × {max_len} = {traditional_tokens}"
    )
    print(f"   Actual tokens needed: {sum(seq_lens)}")
    print(
        f"   Waste: {traditional_tokens - sum(seq_lens)} tokens ({(traditional_tokens - sum(seq_lens)) / traditional_tokens * 100:.1f}%)"
    )

    # PagedAttention: allocate only needed blocks
    blocks_per_seq = [(l + block_size - 1) // block_size for l in seq_lens]
    total_blocks = sum(blocks_per_seq)
    paged_tokens = total_blocks * block_size

    print(f"\n⚡ PAGED ATTENTION:")
    print(f"   Blocks per sequence: {blocks_per_seq}")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Total token slots: {total_blocks} × {block_size} = {paged_tokens}")
    print(f"   Actual tokens: {sum(seq_lens)}")
    print(
        f"   Waste: {paged_tokens - sum(seq_lens)} tokens ({(paged_tokens - sum(seq_lens)) / paged_tokens * 100:.1f}%)"
    )

    # Comparison
    traditional_waste = (traditional_tokens - sum(seq_lens)) / traditional_tokens * 100
    paged_waste = (paged_tokens - sum(seq_lens)) / paged_tokens * 100

    print(f"\n📈 Comparison:")
    print(f"   Traditional waste: {traditional_waste:.1f}%")
    print(f"   PagedAttention waste: {paged_waste:.1f}%")
    print(f"   Improvement: {traditional_waste / paged_waste:.1f}x less waste!")


def demo_copy_on_write():
    """
    Demo 3: Show copy-on-write sharing for parallel sampling.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Copy-on-Write Sharing (Parallel Sampling)")
    print("=" * 60)

    attn = MinimalPagedAttention(num_blocks=20, block_size=4, num_heads=2, head_dim=8)

    # Parent sequence with 12 tokens
    parent_id = "parent"
    attn.allocate_sequence(parent_id, length=12)

    print(f"\n👨‍👧‍👦 Parent sequence: 12 tokens (3 blocks)")
    for pos in range(12):
        attn.write_kv(parent_id, pos, torch.randn(2, 8), torch.randn(2, 8))

    # Fork 3 children (in real vLLM, they share block table initially)
    children = ["child_0", "child_1", "child_2"]

    print(f"\n🔄 Forking {len(children)} children:")
    for child_id in children:
        # Copy block table (reference, not data)
        attn.block_tables[child_id] = attn.block_tables[parent_id].copy()
        print(f"   {child_id}: shares blocks {attn.block_tables[child_id]}")

    # Children diverge (write different tokens)
    print(f"\n✏️  Children generating different tokens (divergence):")
    for i, child_id in enumerate(children):
        # Write 2 unique tokens per child
        for j in range(2):
            pos = 12 + j
            # In real CoW, this would trigger block copy if shared
            attn.write_kv(child_id, pos, torch.randn(2, 8), torch.randn(2, 8))
        print(f"   {child_id}: added 2 unique tokens at positions 12-13")

    # Memory stats
    print(f"\n💾 Memory Analysis:")
    print(f"   Without sharing: 4 sequences × 14 tokens = 56 tokens worth of storage")
    print(f"   With CoW: ~36 tokens worth (shared 12 + 4×2 unique)")
    print(f"   Savings: ~{56 - 36} tokens (36% reduction!)")


def run_tests():
    """
    Unit tests for MinimalPagedAttention.
    """
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    attn = MinimalPagedAttention(num_blocks=10, block_size=4, num_heads=2, head_dim=8)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic allocation
    try:
        blocks = attn.allocate_sequence("test1", length=6)
        assert len(blocks) == 2  # 6 tokens needs 2 blocks of size 4
        assert blocks == [0, 1]
        print("✓ Test 1: Block allocation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        tests_failed += 1

    # Test 2: Logical to physical mapping
    try:
        block_id, offset = attn.logical_to_physical("test1", 5)
        assert block_id == 1  # Position 5 is in 2nd block (block index 1)
        assert offset == 1  # Position 5 % 4 = 1
        print("✓ Test 2: Logical to physical mapping")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        tests_failed += 1

    # Test 3: KV cache write/read
    try:
        k_in = torch.randn(2, 8)
        v_in = torch.randn(2, 8)
        attn.write_kv("test1", 3, k_in, v_in)
        k_out, v_out = attn.read_kv("test1", 3)
        assert torch.allclose(k_in, k_out)
        assert torch.allclose(v_in, v_out)
        print("✓ Test 3: KV cache write/read")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        tests_failed += 1

    # Test 4: Attention computation
    try:
        # Fill with some data
        for pos in range(5):
            k = torch.randn(2, 8)
            v = torch.randn(2, 8)
            attn.write_kv("test1", pos, k, v)

        query = torch.randn(2, 8)
        output = attn.paged_attention("test1", query, seq_len=5)
        assert output.shape == (2, 8)
        assert not torch.isnan(output).any()
        print("✓ Test 4: Paged attention computation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        tests_failed += 1

    # Test 5: Multiple sequences
    try:
        attn.allocate_sequence("test2", length=3)
        for pos in range(3):
            attn.write_kv("test2", pos, torch.randn(2, 8), torch.randn(2, 8))

        # Both sequences should coexist
        k1, _ = attn.read_kv("test1", 0)
        k2, _ = attn.read_kv("test2", 0)
        assert k1.shape == k2.shape == (2, 8)
        print("✓ Test 5: Multiple sequences")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        tests_failed += 1

    print(f"\n📊 Results: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


if __name__ == "__main__":
    # Run all demos
    demo_basic_paging()
    demo_memory_comparison()
    demo_copy_on_write()

    # Run tests
    success = run_tests()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: PagedAttention Core Concept")
    print("=" * 60)
    print("""
🎯 THE PROBLEM:
   - LLM inference needs KV cache for each token
   - Traditional: Contiguous arrays per sequence
   - Problem: Fragmentation, padding waste, no sharing

⚡ THE SOLUTION (PagedAttention):
   1. Divide KV cache into fixed-size blocks
   2. Block table maps logical -> physical locations  
   3. Non-contiguous allocation reduces waste
   4. Copy-on-write enables memory sharing

🔑 KEY INSIGHTS:
   - Like OS virtual memory paging
   - Slot = block_id × block_size + offset
   - Enables efficient batching of variable-length sequences
   - Prefix caching comes naturally from block sharing

💡 IN VLLM:
   - GPU kernels read from paged cache using block tables
   - Scheduler manages block allocation/lifetime
   - Automatic prefix caching and memory sharing
    """)

    sys.exit(0 if success else 1)
