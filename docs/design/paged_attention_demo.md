# PagedAttention Demo

A self-contained demonstration of vLLM's core innovation: **PagedAttention**.

For the real kernel implementation details, see [paged_attention.md](./paged_attention.md).

## Overview

PagedAttention is vLLM's key technical contribution -- a memory-efficient approach to storing and accessing Key-Value (K/V) caches during LLM inference. Instead of allocating contiguous memory for each sequence, it uses fixed-size blocks similar to OS virtual memory paging.

## Running the Demo

```bash
# All demos + tests
python examples/offline_inference/paged_attention_demo.py

# Tests only
python examples/offline_inference/paged_attention_demo.py --test-only
```

The demo includes three validated scenarios and a unit test suite:

1. **Basic Paging** -- Block allocation, slot mapping, attention computation
2. **Memory Efficiency Comparison** -- PagedAttention vs traditional contiguous allocation
3. **Copy-on-Write Sharing** -- Parallel sampling with shared prefix blocks

## Key Concepts

### The Problem
Traditional attention requires contiguous K/V cache storage:
```
[seq_len, num_heads, head_dim] - contiguous memory
```
- Wastes memory on fragmentation
- Requires padding to max length
- Cannot share memory between sequences

### The Solution (PagedAttention)
Blocked K/V storage:
```
[num_blocks, num_heads, head_dim, block_size] - blocked memory
```

Benefits:
- Non-contiguous allocation (reduces fragmentation)
- Memory sharing (copy-on-write for parallel sampling)
- No padding waste
- Prefix caching (reuse blocks)

### Core Mechanism

1. **Block Allocation**: Allocate fixed-size blocks from a pool
2. **Block Table**: Map logical positions to physical block IDs
   ```
   Position 0-15  -> Block 0
   Position 16-31 -> Block 1
   Position 32-47 -> Block 2
   ```
3. **Slot Mapping**: Compute physical address
   ```
   slot = block_id * block_size + offset
   ```
4. **PagedAttention Kernel**: Read K/V from non-contiguous blocks using block table

## Memory Efficiency

For 3 sequences with lengths [10, 25, 7] and block_size=8:

| Approach | Allocated | Actual | Waste |
|----------|-----------|--------|-------|
| Traditional (padding) | 75 tokens | 42 tokens | 44% |
| **PagedAttention** | 56 tokens | 42 tokens | **25%** |

**Improvement: 1.8x less waste!**

## In Real vLLM

Key files:
- `vllm/v1/attention/ops/paged_attn.py` -- Core PagedAttention ops
- `vllm/v1/worker/gpu/block_table.py` -- Block table + slot mapping
- `vllm/v1/worker/gpu/model_runner.py` -- Block table updates per batch
