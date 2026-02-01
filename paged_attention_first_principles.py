"""
PagedAttention: First Principles Implementation
===============================================

This module demonstrates the core innovation of vLLM: PagedAttention.

Key Concept:
------------
Traditional attention requires contiguous KV cache memory per sequence.
PagedAttention uses fixed-size blocks (like OS virtual memory paging), allowing:
1. Non-contiguous memory allocation
2. Memory sharing between sequences
3. Reduced fragmentation
4. Efficient prefix caching

The block table maps logical positions -> physical block indices.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class Block:
    """A single KV cache block storing key-value pairs for multiple tokens."""

    block_id: int
    num_heads: int
    head_size: int
    block_size: int  # Number of tokens per block

    def __post_init__(self):
        # KV cache storage: [num_heads, head_size, block_size]
        # Each position can store one token's K/V vectors
        self.k_cache = torch.zeros(self.num_heads, self.head_size, self.block_size)
        self.v_cache = torch.zeros(self.num_heads, self.head_size, self.block_size)
        self.num_tokens = 0  # How many tokens are actually stored
        self.ref_count = 0  # For copy-on-write sharing

    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def add_token(self, k: torch.Tensor, v: torch.Tensor, position: int) -> None:
        """Store K/V vectors for a token at a specific position within the block."""
        if position >= self.block_size:
            raise ValueError(
                f"Position {position} exceeds block size {self.block_size}"
            )
        self.k_cache[:, :, position] = k
        self.v_cache[:, :, position] = v
        self.num_tokens = max(self.num_tokens, position + 1)

    def get_kv(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V vectors for a token at a specific position."""
        return self.k_cache[:, :, position], self.v_cache[:, :, position]

    def fork(self, new_block_id: int) -> "Block":
        """Create a copy-on-write copy of this block."""
        new_block = Block(new_block_id, self.num_heads, self.head_size, self.block_size)
        new_block.k_cache = self.k_cache.clone()
        new_block.v_cache = self.v_cache.clone()
        new_block.num_tokens = self.num_tokens
        new_block.ref_count = 1
        self.ref_count += 1
        return new_block

    def __repr__(self):
        return f"Block(id={self.block_id}, tokens={self.num_tokens}/{self.block_size}, refs={self.ref_count})"


class BlockAllocator:
    """Manages the pool of available blocks (like OS page frame allocator)."""

    def __init__(
        self, num_blocks: int, num_heads: int, head_size: int, block_size: int
    ):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        # Free list - blocks available for allocation
        self.free_blocks: List[int] = list(range(num_blocks))

        # All blocks storage
        self.blocks: Dict[int, Block] = {}

        print(f"[BlockAllocator] Initialized with {num_blocks} blocks")
        print(f"  - Block size: {block_size} tokens")
        print(f"  - Per block: {num_heads} heads × {head_size} dims")
        print(f"  - Total capacity: {num_blocks * block_size} tokens")

    def allocate(self) -> Optional[Block]:
        """Allocate a new block from the free list."""
        if not self.free_blocks:
            return None

        block_id = self.free_blocks.pop(0)
        block = Block(block_id, self.num_heads, self.head_size, self.block_size)
        self.blocks[block_id] = block
        print(
            f"[BlockAllocator] Allocated block {block_id} (remaining: {len(self.free_blocks)})"
        )
        return block

    def free(self, block_id: int) -> None:
        """Return a block to the free list."""
        if block_id in self.blocks:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count <= 0:
                del self.blocks[block_id]
                self.free_blocks.append(block_id)
                print(
                    f"[BlockAllocator] Freed block {block_id} (available: {len(self.free_blocks)})"
                )

    def get_usage_stats(self) -> Dict:
        """Get current memory usage statistics."""
        used = len(self.blocks)
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used,
            "free_blocks": len(self.free_blocks),
            "utilization": used / self.num_blocks * 100,
        }


class BlockTable:
    """
    Maps logical token positions to physical block locations.

    This is the core of PagedAttention - like a page table in OS virtual memory.
    Logical positions are contiguous, but physical storage is in fixed-size blocks.
    """

    def __init__(self, block_allocator: BlockAllocator):
        self.allocator = block_allocator
        self.block_ids: List[int] = []  # Ordered list of physical block IDs
        self.logical_len = 0  # Number of tokens stored

    def append_token(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a new token's K/V to the sequence."""
        logical_pos = self.logical_len
        block_idx, offset = self._logical_to_physical(logical_pos)

        # Do we need a new block?
        if block_idx >= len(self.block_ids):
            block = self.allocator.allocate()
            if block is None:
                raise RuntimeError("Out of memory - no blocks available")
            self.block_ids.append(block.block_id)

        # Store the K/V vectors
        block_id = self.block_ids[block_idx]
        block = self.allocator.blocks[block_id]
        block.add_token(k, v, offset)
        self.logical_len += 1

    def _logical_to_physical(self, logical_pos: int) -> Tuple[int, int]:
        """
        Convert logical token position to (block_index, offset_within_block).

        Example: block_size=16
        - Logical pos 0 -> block 0, offset 0
        - Logical pos 15 -> block 0, offset 15
        - Logical pos 16 -> block 1, offset 0
        """
        block_idx = logical_pos // self.allocator.block_size
        offset = logical_pos % self.allocator.block_size
        return block_idx, offset

    def get_kv(self, logical_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V vectors for a logical position."""
        block_idx, offset = self._logical_to_physical(logical_pos)
        if block_idx >= len(self.block_ids):
            raise IndexError(f"Position {logical_pos} not allocated")

        block_id = self.block_ids[block_idx]
        block = self.allocator.blocks[block_id]
        return block.get_kv(offset)

    def get_block_table_array(self, max_len: int) -> List[int]:
        """Get the block table as an array (for kernel execution)."""
        num_blocks_needed = (
            max_len + self.allocator.block_size - 1
        ) // self.allocator.block_size
        table = self.block_ids[:num_blocks_needed]
        # Pad with -1 for remaining positions
        return table + [-1] * (num_blocks_needed - len(table))

    def fork(self) -> "BlockTable":
        """Create a copy-on-write copy of this block table (for parallel sampling)."""
        new_table = BlockTable(self.allocator)
        new_table.block_ids = self.block_ids.copy()
        new_table.logical_len = self.logical_len

        # Increment ref counts
        for block_id in self.block_ids:
            if block_id in self.allocator.blocks:
                self.allocator.blocks[block_id].ref_count += 1

        print(f"[BlockTable] Forked - shared {len(self.block_ids)} blocks")
        return new_table

    def __repr__(self):
        return f"BlockTable(tokens={self.logical_len}, blocks={len(self.block_ids)}, block_ids={self.block_ids})"


def ref_paged_attention(
    query: torch.Tensor,
    block_table: BlockTable,
    seq_len: int,
    scale: float,
) -> torch.Tensor:
    """
    Reference implementation of PagedAttention.

    Args:
        query: [num_heads, head_size] - query vector for current token
        block_table: BlockTable containing cached K/V
        seq_len: Length of sequence to attend to
        scale: Attention scale factor (1/sqrt(head_size))

    Returns:
        output: [num_heads, head_size] - attention output
    """
    num_heads, head_size = query.shape

    # Collect all keys and values from blocks
    keys = []
    values = []

    for pos in range(seq_len):
        k, v = block_table.get_kv(pos)
        keys.append(k)
        values.append(v)

    # Stack into [seq_len, num_heads, head_size]
    keys = torch.stack(keys, dim=0)
    values = torch.stack(values, dim=0)

    # Compute attention: Q @ K^T
    # query: [num_heads, head_size]
    # keys: [seq_len, num_heads, head_size]
    # We want: [num_heads, seq_len]
    attn_scores = torch.einsum("hd,shd->hs", query, keys) * scale

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply to values: [num_heads, seq_len] @ [seq_len, num_heads, head_size]
    output = torch.einsum("hs,shd->hd", attn_weights, values)

    return output


class SimplePagedAttentionDemo:
    """
    End-to-end demonstration of PagedAttention mechanism.
    """

    def __init__(
        self,
        num_blocks: int = 100,
        block_size: int = 16,
        num_heads: int = 4,
        head_size: int = 64,
    ):
        self.block_allocator = BlockAllocator(
            num_blocks=num_blocks,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
        )
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

    def run_single_sequence(self, seq_len: int = 50) -> None:
        """Demonstrate PagedAttention for a single sequence."""
        print("\n" + "=" * 60)
        print("SINGLE SEQUENCE PAGED ATTENTION DEMO")
        print("=" * 60)

        # Create block table for the sequence
        block_table = BlockTable(self.block_allocator)

        # Simulate generating tokens
        print(f"\nGenerating {seq_len} tokens...")

        for i in range(seq_len):
            # Simulate computing K/V for current token
            k = torch.randn(self.num_heads, self.head_size)
            v = torch.randn(self.num_heads, self.head_size)

            # Store in block table
            block_table.append_token(k, v)

            if i < 5 or i == seq_len - 1:
                print(
                    f"  Token {i}: stored in block_table, now has {block_table.logical_len} tokens"
                )

        print(f"\nFinal Block Table: {block_table}")

        # Demonstrate attention computation
        print("\nComputing attention for new token...")
        query = torch.randn(self.num_heads, self.head_size)
        scale = 1.0 / (self.head_size**0.5)

        output = ref_paged_attention(query, block_table, seq_len, scale)
        print(f"Attention output shape: {output.shape}")

        # Show memory stats
        stats = self.block_allocator.get_usage_stats()
        print(f"\nMemory Usage:")
        print(
            f"  Blocks used: {stats['used_blocks']}/{stats['total_blocks']} ({stats['utilization']:.1f}%)"
        )
        print(
            f"  Wasted space: {stats['used_blocks'] * self.block_size - seq_len} tokens"
        )

    def run_parallel_sampling(self, seq_len: int = 50, num_samples: int = 4) -> None:
        """Demonstrate PagedAttention with parallel sampling (sharing)."""
        print("\n" + "=" * 60)
        print("PARALLEL SAMPLING WITH SHARED PREFIX (COPY-ON-WRITE)")
        print("=" * 60)

        # Create shared prefix
        shared_block_table = BlockTable(self.block_allocator)

        print(f"\nGenerating shared prefix of {seq_len} tokens...")
        for i in range(seq_len):
            k = torch.randn(self.num_heads, self.head_size)
            v = torch.randn(self.num_heads, self.head_size)
            shared_block_table.append_token(k, v)

        print(f"Shared prefix: {shared_block_table}")

        # Fork for parallel sampling
        print(f"\nForking {num_samples} parallel samples...")
        sample_tables = [shared_block_table.fork() for _ in range(num_samples)]

        # Each sample generates a few different tokens (divergence)
        print("\nEach sample generating 3 unique tokens...")
        for i, table in enumerate(sample_tables):
            for j in range(3):
                k = torch.randn(self.num_heads, self.head_size)
                v = torch.randn(self.num_heads, self.head_size)
                table.append_token(k, v)
            print(f"  Sample {i}: {seq_len}+3 tokens, blocks={table.block_ids}")

        # Memory stats
        stats = self.block_allocator.get_usage_stats()
        print(f"\nMemory Usage:")
        print(f"  Blocks used: {stats['used_blocks']}/{stats['total_blocks']}")
        print(f"  Without sharing would need: {num_samples * (seq_len + 3)} tokens")
        print(f"  Actual storage: ~{stats['used_blocks'] * self.block_size} tokens")
        print(f"  Memory saved: ~{(num_samples - 1) * seq_len} tokens!")


if __name__ == "__main__":
    # Run the demonstration
    demo = SimplePagedAttentionDemo(
        num_blocks=50, block_size=16, num_heads=4, head_size=64
    )

    demo.run_single_sequence(seq_len=35)
    demo.run_parallel_sampling(seq_len=30, num_samples=4)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. PagedAttention uses fixed-size blocks (like OS pages)")
    print("2. Block tables map logical positions to physical blocks")
    print("3. Non-contiguous allocation reduces memory fragmentation")
    print("4. Copy-on-write enables efficient sharing for parallel sampling")
    print("5. Prefix caching is natural - shared blocks for common prefixes")
