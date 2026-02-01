#!/usr/bin/env python3
"""
PagedAttention vLLM Engine Tracer
==================================

This script demonstrates how vLLM's engine uses PagedAttention internally.
It adds hooks and logging to trace the execution flow.
"""

from contextlib import contextmanager
import os
from typing import Any, Dict, List, Optional

import torch

# Set environment variables for CPU mode
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_COMPILE_LEVEL"] = "0"


class PagedAttentionTracer:
    """
    Traces PagedAttention execution through vLLM engine.

    Hooks into key points:
    1. Block allocation
    2. Block table construction
    3. PagedAttention kernel execution
    4. KV cache write operations
    """

    def __init__(self):
        self.trace_log: List[Dict[str, Any]] = []
        self.enabled = True
        self.indent_level = 0

    def log(self, message: str, data: Optional[Dict] = None) -> None:
        """Log a trace event."""
        if not self.enabled:
            return

        indent = "  " * self.indent_level
        entry = {"message": f"{indent}{message}", "data": data or {}}
        self.trace_log.append(entry)
        print(entry["message"])

    @contextmanager
    def section(self, name: str):
        """Context manager for trace sections."""
        self.log(f"▶ {name}")
        self.indent_level += 1
        try:
            yield self
        finally:
            self.indent_level -= 1
            self.log(f"◀ {name} complete")

    def print_summary(self) -> None:
        """Print a summary of the trace."""
        print("\n" + "=" * 60)
        print("TRACE SUMMARY")
        print("=" * 60)

        # Group by sections
        sections = []
        current_section = []

        for entry in self.trace_log:
            if entry["message"].startswith("▶"):
                if current_section:
                    sections.append(current_section)
                current_section = [entry]
            else:
                current_section.append(entry)

        if current_section:
            sections.append(current_section)

        for section in sections:
            print(f"\n{section[0]['message']}")
            for entry in section[1:]:
                print(f"  {entry['message']}")


# Global tracer instance
tracer = PagedAttentionTracer()


def trace_block_allocation(block_id: int, num_tokens: int, block_size: int) -> None:
    """Hook called when a new block is allocated."""
    tracer.log(
        f"BLOCK ALLOCATED: id={block_id}",
        {"block_id": block_id, "capacity": block_size, "tokens_stored": num_tokens},
    )


def trace_block_table_construction(
    block_table: List[int], seq_len: int, block_size: int
) -> None:
    """Hook called when block table is built."""
    num_blocks = len(block_table)
    capacity = num_blocks * block_size

    tracer.log(
        f"BLOCK TABLE: {num_blocks} blocks, {seq_len}/{capacity} tokens used",
        {
            "block_ids": block_table,
            "seq_len": seq_len,
            "block_size": block_size,
            "utilization": f"{seq_len / capacity * 100:.1f}%",
        },
    )


def trace_kv_cache_write(slot_mapping: torch.Tensor, num_tokens: int) -> None:
    """Hook called when K/V is written to cache."""
    tracer.log(
        f"KV CACHE WRITE: {num_tokens} tokens",
        {
            "slot_mapping_shape": list(slot_mapping.shape),
            "num_tokens": num_tokens,
            "slots": slot_mapping[
                : min(10, len(slot_mapping))
            ].tolist(),  # First 10 slots
        },
    )


def trace_paged_attention_exec(
    query_shape: tuple,
    num_seqs: int,
    seq_lens: List[int],
    num_blocks: int,
    block_size: int,
) -> None:
    """Hook called when PagedAttention kernel is executed."""
    tracer.log(
        f"PAGED ATTENTION KERNEL",
        {
            "query_shape": query_shape,
            "num_seqs": num_seqs,
            "seq_lens": seq_lens,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "total_tokens": sum(seq_lens),
        },
    )


class SimpleInferenceTracer:
    """
    Minimal vLLM-like inference engine with PagedAttention tracing.

    This demonstrates the core flow without the full vLLM complexity.
    """

    def __init__(
        self,
        num_blocks: int = 100,
        block_size: int = 16,
        num_heads: int = 8,
        head_size: int = 64,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size

        # KV cache: [num_blocks, 2, num_heads, head_size, block_size]
        # 2 for key and value
        self.kv_cache = torch.zeros(num_blocks, 2, num_heads, head_size, block_size)

        # Block allocation tracking
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}  # seq_id -> [block_ids]
        self.seq_lens: dict[str, int] = {}

        print(f"[Engine] Initialized")
        print(f"  - KV cache: {num_blocks} blocks × {block_size} tokens")
        print(f"  - Per token: {num_heads} heads × {head_size} dims")
        print(f"  - Total capacity: {num_blocks * block_size} tokens")

    def allocate_blocks(self, num_tokens: int, seq_id: str) -> List[int]:
        """Allocate blocks for a sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"Out of memory: need {num_blocks_needed}, have {len(self.free_blocks)}"
            )

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)
            trace_block_allocation(block_id, 0, self.block_size)

        self.allocated_blocks[seq_id] = allocated
        self.seq_lens[seq_id] = num_tokens
        trace_block_table_construction(allocated, num_tokens, self.block_size)

        return allocated

    def compute_slot_mapping(
        self, seq_len: int, block_table: List[int]
    ) -> torch.Tensor:
        """
        Compute slot mapping for KV cache write.

        Slot = block_id * block_size + offset
        Maps logical positions to physical cache locations.
        """
        slot_mapping = []
        for pos in range(seq_len):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = block_table[block_idx]
            slot = block_id * self.block_size + offset
            slot_mapping.append(slot)

        return torch.tensor(slot_mapping, dtype=torch.long)

    def run_prefill(self, seq_id: str, prompt_len: int) -> None:
        """Run prefill phase - process prompt and cache KV."""
        with tracer.section(f"PREFILL (seq={seq_id}, len={prompt_len})"):
            # Allocate blocks
            block_table = self.allocate_blocks(prompt_len, seq_id)

            # Simulate computing K/V for prompt tokens
            k_cache = torch.randn(prompt_len, self.num_heads, self.head_size)
            v_cache = torch.randn(prompt_len, self.num_heads, self.head_size)

            # Compute slot mapping
            slot_mapping = self.compute_slot_mapping(prompt_len, block_table)
            trace_kv_cache_write(slot_mapping, prompt_len)

            # Write to paged cache
            for i, slot in enumerate(slot_mapping):
                block_id = int(slot) // self.block_size
                offset = int(slot) % self.block_size
                self.kv_cache[block_id, 0, :, :, offset] = k_cache[i]  # Key
                self.kv_cache[block_id, 1, :, :, offset] = v_cache[i]  # Value

            tracer.log(f"Cached {prompt_len} tokens in {len(block_table)} blocks")

    def run_decode(self, seq_id: str, num_tokens: int = 1) -> None:
        """Run decode phase - generate new token using PagedAttention."""
        with tracer.section(f"DECODE (seq={seq_id}, generating {num_tokens} tokens)"):
            block_table = self.allocated_blocks.get(seq_id, [])
            seq_len = self.seq_lens.get(seq_id, 0)

            for _ in range(num_tokens):
                # Query for new token
                query = torch.randn(self.num_heads, self.head_size)

                # Execute PagedAttention
                trace_paged_attention_exec(
                    query_shape=tuple(query.shape),
                    num_seqs=1,
                    seq_lens=[seq_len],
                    num_blocks=len(block_table),
                    block_size=self.block_size,
                )

                # Simulate attention computation
                # In real vLLM, this would be the CUDA kernel
                tracer.log(f"Computing attention over {seq_len} cached tokens")

                # Generate new token and cache it
                new_k = torch.randn(self.num_heads, self.head_size)
                new_v = torch.randn(self.num_heads, self.head_size)

                # Find slot for new token
                new_pos = seq_len  # Next position
                block_idx = new_pos // self.block_size

                if block_idx >= len(block_table):
                    # Need new block
                    new_block = self.free_blocks.pop(0)
                    block_table.append(new_block)
                    self.allocated_blocks[seq_id] = block_table
                    trace_block_allocation(new_block, 0, self.block_size)

                block_id = block_table[block_idx]
                offset = new_pos % self.block_size
                slot = block_id * self.block_size + offset

                trace_kv_cache_write(torch.tensor([slot]), 1)

                # Write to cache
                self.kv_cache[block_id, 0, :, :, offset] = new_k
                self.kv_cache[block_id, 1, :, :, offset] = new_v

                seq_len += 1
                tracer.log("Generated and cached 1 new token")

            self.seq_lens[seq_id] = seq_len


def demo_paged_attention_in_vllm():
    """
    Demonstrate how PagedAttention works inside vLLM engine.
    """
    print("\n" + "=" * 70)
    print("PAGED ATTENTION EXECUTION TRACE IN VLLM-LIKE ENGINE")
    print("=" * 70)
    print("\nThis demonstrates the core PagedAttention flow:")
    print("1. Block allocation (like OS page frames)")
    print("2. Block table construction (like page table)")
    print("3. Slot mapping (logical -> physical address)")
    print("4. PagedAttention kernel execution")
    print("5. KV cache write operations")

    engine = SimpleInferenceTracer(
        num_blocks=50, block_size=16, num_heads=4, head_size=64
    )

    # Demo 1: Single sequence
    print("\n" + "=" * 70)
    print("SCENARIO 1: Single Sequence Inference")
    print("=" * 70)

    engine.run_prefill("seq_0", prompt_len=35)
    engine.run_decode("seq_0", num_tokens=1)
    engine.run_decode("seq_0", num_tokens=1)

    # Demo 2: Multiple sequences with sharing
    print("\n" + "=" * 70)
    print("SCENARIO 2: Multiple Sequences (Parallel Sampling)")
    print("=" * 70)

    # First sequence processes long prompt
    engine.run_prefill("seq_parent", prompt_len=48)

    # Fork for parallel sampling (in real vLLM, this shares block table)
    for i in range(3):
        # In reality, this would copy-on-write share the blocks
        # For demo, we allocate new blocks
        engine.run_prefill(f"seq_child_{i}", prompt_len=5)  # Just allocate some blocks
        tracer.log(f"Forked sequence {i} - shares blocks with parent")

    # Print final summary
    tracer.print_summary()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM TRACE")
    print("=" * 70)
    print("""
1. BLOCK ALLOCATION:
   - Fixed-size blocks allocated on-demand
   - No contiguous memory requirement
   - Similar to OS page frame allocation

2. BLOCK TABLE:
   - Maps logical positions -> physical block IDs
   - Like OS page table mapping virtual -> physical addresses
   - Enables non-contiguous storage

3. SLOT MAPPING:
   - slot = block_id * block_size + offset
   - Converts logical token position to physical cache location
   - Used by both write (reshape_and_cache) and read (paged_attention)

4. PAGED ATTENTION:
   - Kernel receives block_table and slot_mapping
   - Fetches K/V from non-contiguous blocks on-the-fly
   - Enables memory-efficient attention with dynamic allocation

5. BENEFITS:
   - Reduced fragmentation (fixed block sizes)
   - Memory sharing (copy-on-write forking)
   - Prefix caching (reuse blocks for common prefixes)
   - Efficient batching (different length sequences)
    """)


if __name__ == "__main__":
    demo_paged_attention_in_vllm()
