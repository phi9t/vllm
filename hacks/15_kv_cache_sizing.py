# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How big is my KV cache? The sizing arithmetic, no GPU.

Pairs with HACKERS_GUIDE.md §9 (KV cache sizing & memory profiling).

At startup vLLM profiles peak memory
(vllm/v1/worker/gpu_worker.py:354 determine_available_memory), then
get_kv_cache_configs (vllm/v1/core/kv_cache_utils.py:1922) divides the
free bytes by the per-block cost (KVCacheSpec.page_size_bytes,
vllm/v1/kv_cache_interface.py:82) to get num_gpu_blocks. This script
reproduces that division for a few real models so you can see where
gpu_memory_utilization actually goes — and why GQA/MLA shrink the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from _stubs import print_header

GiB = 1 << 30
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4}


@dataclass
class ModelKV:
    name: str
    num_layers: int
    num_kv_heads: int  # GQA: < num_attention_heads; MLA: see latent below
    head_dim: int
    dtype: str = "bf16"
    # MLA stores one compressed latent per token instead of K/V per head:
    mla_latent_dim: int | None = None  # kv_lora_rank + qk_rope_head_dim


def bytes_per_token(m: ModelKV) -> int:
    db = DTYPE_BYTES[m.dtype]
    if m.mla_latent_dim is not None:
        # MLA: a single latent vector per token per layer (no ×2, no ×heads).
        return m.num_layers * m.mla_latent_dim * db
    # Standard / GQA attention: K and V, per kv-head.
    return 2 * m.num_layers * m.num_kv_heads * m.head_dim * db


def report(m: ModelKV, total_vram_gib: float, util: float, block_size: int) -> None:
    free = int(total_vram_gib * GiB * util)  # rough: ignores weights/activations
    bpt = bytes_per_token(m)
    block_bytes = bpt * block_size
    num_blocks = free // block_bytes
    kv_tokens = num_blocks * block_size
    kind = "MLA latent" if m.mla_latent_dim is not None else "GQA/MHA"
    print(f"  {m.name:18s} [{kind}]")
    print(f"    bytes/token        = {bpt:,}  ({m.dtype})")
    print(f"    block_bytes (bs={block_size:<3d}) = {block_bytes:,}")
    print(f"    num_gpu_blocks     = {num_blocks:,}")
    print(f"    KV capacity        = {kv_tokens:,} tokens "
          f"(~{kv_tokens // 1000}K)")


def main() -> int:
    total_vram_gib = 80.0   # e.g. one H100-80GB
    util = 0.90             # gpu_memory_utilization, minus a nominal weights slice
    block_size = 16         # vllm/config/cache.py:47 default

    print_header(f"KV cache sizing @ {total_vram_gib:.0f}GiB · util={util} · "
                 f"block_size={block_size}")
    print("  (free-for-KV approximated as total*util; real vLLM subtracts "
          "weights+activations+cudagraphs from a profiling run)\n")

    models = [
        # name, layers, kv_heads, head_dim, dtype, mla_latent
        ModelKV("Qwen3-0.6B", 28, 8, 128),
        ModelKV("Qwen3-8B (GQA)", 36, 8, 128),
        ModelKV("Llama-3-70B (GQA)", 80, 8, 128),
        # DeepSeek-V3/V4: MLA latent = kv_lora_rank(512) + qk_rope_head_dim(64)
        ModelKV("DeepSeek (MLA)", 61, 0, 0, mla_latent_dim=512 + 64),
    ]
    for m in models:
        report(m, total_vram_gib, util, block_size)
        print()

    print_header("Takeaway")
    print("  KV bytes/token scales with layers × kv_heads × head_dim. GQA cuts")
    print("  kv_heads; MLA replaces per-head K/V with ONE small latent — which")
    print("  is why DeepSeek holds far more tokens of KV in the same VRAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
