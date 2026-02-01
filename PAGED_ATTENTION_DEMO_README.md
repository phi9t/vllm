# PagedAttention Demo Suite

This directory contains a comprehensive demonstration of vLLM's core innovation: **PagedAttention**.

## Overview

PagedAttention is vLLM's key technical contribution - a memory-efficient approach to storing and accessing Key-Value (K/V) caches during LLM inference. Instead of allocating contiguous memory for each sequence, it uses fixed-size blocks similar to OS virtual memory paging.

## Files

### 1. `paged_attention_first_principles.py`
**First Principles Implementation**

A pedagogical implementation showing the core mechanics of PagedAttention:
- `Block` class: Represents a K/V cache block
- `BlockAllocator`: Manages block pool (like OS page frame allocator)
- `BlockTable`: Maps logical positions to physical blocks (like page table)
- `ref_paged_attention()`: Reference attention implementation using paged cache

**Run:**
```bash
source .venv/bin/activate
python paged_attention_first_principles.py
```

**Output:** Demonstrates single sequence generation and parallel sampling with block sharing.

### 2. `paged_attention_vllm_tracer.py`
**vLLM Engine Execution Tracer**

Shows how PagedAttention is used inside vLLM's inference engine:
- `PagedAttentionTracer`: Logs and traces all PagedAttention operations
- `SimpleInferenceTracer`: Minimal vLLM-like engine with tracing hooks
- Traces block allocation, slot mapping, and kernel execution

**Run:**
```bash
source .venv/bin/activate
python paged_attention_vllm_tracer.py
```

**Output:** Detailed trace showing prefill and decode phases with PagedAttention.

### 3. `paged_attention_minimal_demo.py`
**Minimalist Core Demo & Tests**

The simplest possible demonstration with unit tests:
- `MinimalPagedAttention`: Minimal working implementation
- Demo 1: Basic paging mechanics
- Demo 2: Memory efficiency comparison (traditional vs PagedAttention)
- Demo 3: Copy-on-write sharing for parallel sampling
- Unit tests: Verifies correctness

**Run:**
```bash
source .venv/bin/activate
python paged_attention_minimal_demo.py
```

**Output:** 3 demos + 5 unit tests (all should pass ✓).

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
- ✓ Non-contiguous allocation (reduces fragmentation)
- ✓ Memory sharing (copy-on-write for parallel sampling)
- ✓ No padding waste
- ✓ Prefix caching (reuse blocks)

### Core Mechanism

1. **Block Allocation**: Allocate fixed-size blocks from a pool
2. **Block Table**: Map logical positions → physical block IDs
   ```
   Position 0-15  → Block 0
   Position 16-31 → Block 1
   Position 32-47 → Block 2
   ```
3. **Slot Mapping**: Compute physical address
   ```
   slot = block_id × block_size + offset
   ```
4. **PagedAttention Kernel**: Read K/V from non-contiguous blocks using block table

### Example Flow

```python
# 1. Allocate blocks for sequence
block_table = [0, 1, 2]  # 3 blocks

# 2. Write K/V at position 20
block_idx = 20 // 16 = 1  # Block 1
offset    = 20 % 16  = 4  # Offset 4
slot      = 1 × 16 + 4 = 20

# 3. Compute attention
output = paged_attention(query, block_table, seq_len=35)
```

## Memory Efficiency

For 3 sequences with lengths [10, 25, 7] and block_size=8:

| Approach | Allocated | Actual | Waste |
|----------|-----------|--------|-------|
| Traditional (padding) | 75 tokens | 42 tokens | 44% |
| **PagedAttention** | 56 tokens | 42 tokens | **25%** |

**Improvement: 1.8x less waste!**

## In Real vLLM

The actual vLLM implementation:
1. Uses CUDA kernels for PagedAttention (see `vllm/v1/attention/ops/`)
2. Manages block tables + slot mapping in the v1 worker (`vllm/v1/worker/`)
3. Supports copy-on-write forking
4. Implements automatic prefix caching

Key files in vLLM:
- `vllm/v1/attention/ops/paged_attn.py` - Core PagedAttention ops
- `vllm/v1/worker/block_table.py` - Block table + slot mapping
- `vllm/v1/worker/gpu_model_runner.py` - Block table updates per batch

## Quick Start

```bash
# 1. Setup environment
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# 2. Run demos
python paged_attention_minimal_demo.py      # Fast, with tests
python paged_attention_first_principles.py  # Educational
python paged_attention_vllm_tracer.py       # Shows vLLM integration
```

## Summary

PagedAttention enables vLLM to:
- Serve more concurrent requests (better memory efficiency)
- Support longer sequences (no contiguous allocation requirement)
- Enable parallel sampling (memory sharing via copy-on-write)
- Cache prefixes automatically (block-level sharing)

**Result: Up to 2-4x throughput improvement over traditional approaches!**
