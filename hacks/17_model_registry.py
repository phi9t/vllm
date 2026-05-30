# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""From a config to a model: the registry + a parameter/KV footprint.

Pairs with HACKERS_GUIDE.md §15 (Model loading) and §21 (DeepSeek V4).

Two parts:
  1. Best-effort peek at the real ModelRegistry / get_model API
     (vllm/model_executor/models/registry.py:1319, model_loader/__init__.py:128)
     via inspect — graceful if vLLM isn't importable here.
  2. Pure-Python parameter + KV math from a config dict, mirroring the
     explorer's blockTypes.ts (total vs active MoE params; GQA vs MLA KV).
"""

from __future__ import annotations

from _stubs import print_header


# --- Part 2 math: mirrors explorer/src/architecture/blockTypes.ts ------------
def dense_params(cfg: dict) -> int:
    d, L, H, kv, hd = (cfg["hidden_size"], cfg["num_hidden_layers"],
                       cfg["num_attention_heads"], cfg["num_key_value_heads"],
                       cfg["head_dim"])
    V, inter = cfg["vocab_size"], cfg["intermediate_size"]
    q, kvd = H * hd, kv * hd
    attn = d * (q + 2 * kvd) + q * d + 2 * hd + 2 * d        # qkv + o + qk-norm + 2 norms
    ffn = d * 2 * inter + inter * d                          # gate_up + down
    embed = V * d
    head = 0 if cfg.get("tie_word_embeddings") else V * d
    return embed + L * (attn + ffn) + d + head


def moe_params(cfg: dict) -> tuple[int, int]:
    """Returns (total, active-per-token) for an MoE model."""
    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    H, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    V = cfg["vocab_size"]
    E, k = cfg["num_experts"], cfg["num_experts_per_tok"]
    I = cfg["moe_intermediate_size"]
    q, kvd = H * hd, kv * hd
    attn = d * (q + 2 * kvd) + q * d + 2 * hd + 2 * d
    router = d * E
    per_expert = 3 * d * I
    embed = V * d
    head = 0 if cfg.get("tie_word_embeddings") else V * d
    total = embed + L * (attn + router + E * per_expert) + d + head
    active = embed + L * (attn + router + k * per_expert) + d + head
    return total, active


def kv_bytes_per_token(cfg: dict, dtype_bytes: int = 2) -> int:
    L = cfg["num_hidden_layers"]
    if "kv_lora_rank" in cfg:  # MLA: one latent per token per layer
        latent = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        return L * latent * dtype_bytes
    return 2 * L * cfg["num_key_value_heads"] * cfg["head_dim"] * dtype_bytes


def fmt(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def real_registry_peek() -> None:
    try:
        import inspect

        from vllm.model_executor.model_loader import get_model  # type: ignore
        from vllm.model_executor.models.registry import ModelRegistry  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        print(f"  (vLLM not importable here: {type(exc).__name__}; skipping live peek)")
        print("  registry: vllm/model_executor/models/registry.py:1319 (ModelRegistry)")
        print("  loader  : vllm/model_executor/model_loader/__init__.py:128 (get_model)")
        return
    archs = ModelRegistry.get_supported_archs()
    print(f"  ModelRegistry knows {len(archs)} architectures, e.g.:")
    for a in sorted(archs):
        if "Qwen3" in a or "DeepseekV4" in a or "Llama" in a:
            print(f"    - {a}")
    print(f"  get_model signature: {inspect.signature(get_model)}")


def main() -> int:
    print_header("1. Architecture string -> class (the registry)")
    real_registry_peek()

    print_header("2. Parameter + KV footprint from a config (no weights)")
    qwen3_8b = dict(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, intermediate_size=12288,
                    vocab_size=151936, tie_word_embeddings=False)
    qwen3_moe = dict(hidden_size=2048, num_hidden_layers=48, num_attention_heads=32,
                     num_key_value_heads=4, head_dim=128, intermediate_size=6144,
                     vocab_size=151936, tie_word_embeddings=False,
                     num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
    deepseek = dict(hidden_size=7168, num_hidden_layers=61, num_attention_heads=128,
                    num_key_value_heads=128, head_dim=192, intermediate_size=18432,
                    vocab_size=129280, tie_word_embeddings=False,
                    num_experts=256, num_experts_per_tok=8, moe_intermediate_size=2048,
                    kv_lora_rank=512, qk_rope_head_dim=64)

    print(f"  Qwen3-8B (dense GQA): total={fmt(dense_params(qwen3_8b))} params, "
          f"KV={kv_bytes_per_token(qwen3_8b)} bytes/token")
    t, a = moe_params(qwen3_moe)
    print(f"  Qwen3-30B-A3B (MoE) : total={fmt(t)} params, active/token={fmt(a)}, "
          f"KV={kv_bytes_per_token(qwen3_moe)} bytes/token")
    t, a = moe_params(deepseek)
    print(f"  DeepSeek (MLA+MoE)  : total={fmt(t)} params, active/token={fmt(a)}, "
          f"KV={kv_bytes_per_token(deepseek)} bytes/token (latent!)")

    print_header("Takeaway")
    print("  The registry maps config.architectures[0] -> (module, class), then")
    print("  get_model builds + load_weights fills it. MoE total >> active (only")
    print("  top-k experts run); MLA's per-token KV is a single small latent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
