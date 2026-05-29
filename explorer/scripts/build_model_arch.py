#!/usr/bin/env python3
"""Generate per-model architecture manifests for the Explorer Architecture SPA.

For each model in MODEL_LIST this script:
  1. Attempts to load the real HuggingFace config via ``transformers.AutoConfig``.
  2. Falls back to hard-coded values if transformers or the network is unavailable.
  3. Instantiates the ``dense-qknorm`` block template, resolving each block's
     ``ref`` to a real ``file:line`` via ``resolve_line`` (symbol-grep, drift-proof).
  4. Writes ``explorer/public/data/models/<slug>.json`` per model.
  5. Writes ``explorer/public/data/models/index.json`` (slug / label / family /
     totalParams) for the SPA model switcher.

Run via the repo venv (AGENTS.md rule — never bare python3):

    .venv/bin/python explorer/scripts/build_model_arch.py \\
        [--repo-root .] [--out-dir explorer/public/data/models]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure resolve_line is importable whether we are called from the repo
# root or from inside explorer/scripts/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _refs import resolve_line  # noqa: E402  (after sys.path fix)

# ---------------------------------------------------------------------------
# Hard-coded fallback configs (offline / no-network mode)
# ---------------------------------------------------------------------------
FALLBACKS: dict[str, dict] = {
    "qwen3-0_6b": {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
    },
    "qwen3-8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
    },
    "qwen3-30b-a3b": {
        # Verified against Qwen/Qwen3-30B-A3B HF config
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,   # dense MLP layers (mlp_only_layers)
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
        # MoE-specific
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "num_shared_experts": 0,
    },
    "deepseek-v3": {
        # Verified against deepseek-ai/DeepSeek-V3 architecture spec
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim": 192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        "intermediate_size": 18432,   # dense MLP (first_k_dense_replace layers)
        "vocab_size": 129280,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 10000.0,
        # MLA-specific
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        # MoE-specific
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "num_shared_experts": 1,
        "first_k_dense_replace": 3,
    },
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
CONFIG_KEYS = [
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
    "tie_word_embeddings", "torch_dtype", "rope_theta",
    # MoE fields (optional — only present for MoE models)
    "num_experts", "num_experts_per_tok", "moe_intermediate_size", "num_shared_experts",
    # MLA fields (optional — only present for MLA models)
    "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim", "first_k_dense_replace",
]

MODEL_LIST: list[tuple[str, str, str, str]] = [
    # (slug, hf_model_id, family, human_label)
    ("qwen3-0_6b",    "Qwen/Qwen3-0.6B",    "dense-qknorm", "Qwen3-0.6B"),
    ("qwen3-8b",      "Qwen/Qwen3-8B",      "dense-qknorm", "Qwen3-8B"),
    ("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", "moe-qknorm",   "Qwen3-30B-A3B"),
    ("deepseek-v3",   "deepseek-ai/DeepSeek-V3", "mla-moe", "DeepSeek-V3"),
]

# ---------------------------------------------------------------------------
# dense-qknorm block template
#
# Each entry: (id, block_type, kind, label, symbol, source_file, desc, note?)
# source_file is relative to repo root — resolve_line will grep it.
# ---------------------------------------------------------------------------

# Helper type alias for clarity
# Tuple: (id, btype, kind, label, display_symbol, grep_symbol, source_file, desc, note)
# display_symbol  = human-readable name shown in the SPA (matches qwen3Blocks.ts)
# grep_symbol     = string used by resolve_line to anchor to the correct source line
_BlockDef = tuple


def _b(
    id: str, btype: str, kind: str, label: str,
    display_symbol: str, grep_symbol: str, source_file: str,
    desc: str, note: str | None = None,
) -> _BlockDef:
    return (id, btype, kind, label, display_symbol, grep_symbol, source_file, desc, note)


DENSE_QKNORM_PRELUDE: list[_BlockDef] = [
    _b(
        "embed", "embed", "embed",
        "Token embeddings",
        "VocabParallelEmbedding (embed_tokens)",  # display
        "self.embed_tokens = VocabParallelEmbedding",  # grep anchor
        "vllm/model_executor/models/qwen2.py",
        "input_ids → hidden_states. Looks up a row of the embedding matrix per token.",
        "tie_word_embeddings reuses this matrix as the lm_head.",
    ),
]

DENSE_QKNORM_ATT_STEPS: list[_BlockDef] = [
    _b(
        "qkv_proj", "qkv_linear", "proj",
        "QKV projection",
        "QKVParallelLinear (qkv_proj)",  # display
        "self.qkv_proj = QKVParallelLinear",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "One fused linear producing Q, K, V. Split into q_dim + 2·kv_dim (GQA).",
    ),
    _b(
        "q_norm", "qk_norm", "norm",
        "Q head-norm",
        "RMSNorm (q_norm, head_dim)",  # display
        "self.q_norm = RMSNorm",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "Per-head RMSNorm over head_dim applied to queries before RoPE.",
        "Qwen3-specific: QK-Norm stabilizes attention logits.",
    ),
    _b(
        "k_norm", "qk_norm", "norm",
        "K head-norm",
        "RMSNorm (k_norm, head_dim)",  # display
        "self.k_norm = RMSNorm",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "Per-head RMSNorm over head_dim applied to keys before RoPE.",
        "Qwen3-specific: paired with q_norm.",
    ),
    _b(
        "rope", "rope", "rope",
        "RoPE (Q, K)",
        "RotaryEmbedding (get_rope)",  # display
        "def get_rope",  # grep anchor
        "vllm/model_executor/layers/rotary_embedding/__init__.py",
        "Rotary position embedding rotates Q and K by position. Uses a cached cos/sin table.",
        "rope_theta = 1e6 for long context.",
    ),
    _b(
        "attn", "attention_gqa", "attn",
        "Attention (paged KV)",
        "Attention.forward",  # display
        "def forward",  # grep anchor (in attention.py, forward method)
        "vllm/model_executor/layers/attention/attention.py",
        "Scaled dot-product attention over the paged KV cache; backend chosen at init. GQA: kv_heads < heads.",
        "KV cache lives inside this layer (block tables, slot mapping).",
    ),
    _b(
        "o_proj", "o_proj", "proj",
        "Output projection",
        "RowParallelLinear (o_proj)",  # display
        "self.o_proj = RowParallelLinear",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "Projects the attention output (q_dim) back to hidden_size.",
    ),
]

DENSE_QKNORM_MLP_STEPS: list[_BlockDef] = [
    _b(
        "gate_up", "gate_up", "mlp",
        "Gate+Up projection",
        "MergedColumnParallelLinear (gate_up_proj)",  # display
        "self.gate_up_proj = MergedColumnParallelLinear",  # grep anchor
        "vllm/model_executor/models/qwen2.py",
        "Fused gate and up projection: hidden_size → 2·intermediate_size.",
    ),
    _b(
        "silu", "activation", "act",
        "SiLU + gating",
        "SiluAndMul",  # display
        "class SiluAndMul",  # grep anchor
        "vllm/model_executor/layers/activation.py",
        "SwiGLU activation: silu(gate) · up → intermediate_size.",
    ),
    _b(
        "down", "down", "mlp",
        "Down projection",
        "RowParallelLinear (down_proj)",  # display
        "self.down_proj = RowParallelLinear",  # grep anchor
        "vllm/model_executor/models/qwen2.py",
        "Projects intermediate_size back to hidden_size.",
    ),
]

DENSE_QKNORM_HEAD: list[_BlockDef] = [
    _b(
        "final_norm", "rmsnorm", "norm",
        "Final RMSNorm",
        "RMSNorm (norm)",  # display
        "self.norm = RMSNorm",  # grep anchor
        "vllm/model_executor/models/qwen2.py",
        "Normalizes the final hidden state before the LM head.",
    ),
    _b(
        "lm_head", "lm_head", "head",
        "LM head",
        "ParallelLMHead",  # display
        "self.lm_head = ParallelLMHead",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "Projects hidden_size → vocab_size to produce logits.",
        "Weight tied to embed_tokens when tie_word_embeddings is set.",
    ),
    _b(
        "logits", "logits", "head",
        "Logits processor",
        "LogitsProcessor",  # display
        "self.logits_processor = LogitsProcessor",  # grep anchor
        "vllm/model_executor/models/qwen3.py",
        "Gathers logits for the sampled positions; hands off to the Sampler.",
    ),
]

DENSE_QKNORM_ATT_PRENORM: _BlockDef = _b(
    "input_norm", "rmsnorm", "norm",
    "Input RMSNorm",
    "RMSNorm (input_layernorm)",  # display
    "self.input_layernorm = RMSNorm",  # grep anchor
    "vllm/model_executor/models/qwen3.py",
    "Pre-attention normalization, fused with the residual add on all but the first layer.",
)

DENSE_QKNORM_MLP_PRENORM: _BlockDef = _b(
    "post_norm", "rmsnorm", "norm",
    "Post-attn RMSNorm",
    "RMSNorm (post_attention_layernorm)",  # display
    "self.post_attention_layernorm = RMSNorm",  # grep anchor
    "vllm/model_executor/models/qwen3.py",
    "Pre-MLP normalization, fused with the residual add.",
)

# ---------------------------------------------------------------------------
# moe-qknorm block template
#
# Attention branch: identical to dense-qknorm (same symbols, same files).
# MoE branch: router (gate Linear) + experts (FusedMoE) from qwen3_moe.py.
# Prelude and head are also grounded in qwen3_moe.py where symbols exist.
# ---------------------------------------------------------------------------

MOE_QKNORM_PRELUDE: list[_BlockDef] = [
    _b(
        "embed", "embed", "embed",
        "Token embeddings",
        "VocabParallelEmbedding (embed_tokens)",         # display
        "self.embed_tokens = VocabParallelEmbedding",    # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "input_ids → hidden_states. Looks up a row of the embedding matrix per token.",
        "tie_word_embeddings reuses this matrix as the lm_head.",
    ),
]

# Attention prenorm — grounded in qwen3_moe.py (Qwen3MoeDecoderLayer)
MOE_QKNORM_ATT_PRENORM: _BlockDef = _b(
    "input_norm", "rmsnorm", "norm",
    "Input RMSNorm",
    "RMSNorm (input_layernorm)",                       # display
    "self.input_layernorm = RMSNorm",                  # grep anchor
    "vllm/model_executor/models/qwen3_moe.py",
    "Pre-attention normalization, fused with the residual add on all but the first layer.",
)

# Attention steps: reuse same grep anchors — all present in qwen3_moe.py
MOE_QKNORM_ATT_STEPS: list[_BlockDef] = [
    _b(
        "qkv_proj", "qkv_linear", "proj",
        "QKV projection",
        "QKVParallelLinear (qkv_proj)",                # display
        "self.qkv_proj = QKVParallelLinear",           # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "One fused linear producing Q, K, V. Split into q_dim + 2·kv_dim (GQA).",
    ),
    _b(
        "q_norm", "qk_norm", "norm",
        "Q head-norm",
        "RMSNorm (q_norm, head_dim)",                  # display
        "self.q_norm = RMSNorm",                       # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Per-head RMSNorm over head_dim applied to queries before RoPE.",
        "Qwen3-specific: QK-Norm stabilizes attention logits.",
    ),
    _b(
        "k_norm", "qk_norm", "norm",
        "K head-norm",
        "RMSNorm (k_norm, head_dim)",                  # display
        "self.k_norm = RMSNorm",                       # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Per-head RMSNorm over head_dim applied to keys before RoPE.",
        "Qwen3-specific: paired with q_norm.",
    ),
    _b(
        "rope", "rope", "rope",
        "RoPE (Q, K)",
        "RotaryEmbedding (get_rope)",                  # display
        "def get_rope",                                # grep anchor
        "vllm/model_executor/layers/rotary_embedding/__init__.py",
        "Rotary position embedding rotates Q and K by position. Uses a cached cos/sin table.",
        "rope_theta = 1e6 for long context.",
    ),
    _b(
        "attn", "attention_gqa", "attn",
        "Attention (paged KV)",
        "Attention.forward",                           # display
        "def forward",                                 # grep anchor
        "vllm/model_executor/layers/attention/attention.py",
        "Scaled dot-product attention over the paged KV cache; backend chosen at init. GQA: kv_heads < heads.",
        "KV cache lives inside this layer (block tables, slot mapping).",
    ),
    _b(
        "o_proj", "o_proj", "proj",
        "Output projection",
        "RowParallelLinear (o_proj)",                  # display
        "self.o_proj = RowParallelLinear",             # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Projects the attention output (q_dim) back to hidden_size.",
    ),
]

# MoE prenorm — post-attention layernorm in qwen3_moe.py
MOE_QKNORM_MOE_PRENORM: _BlockDef = _b(
    "post_norm", "rmsnorm", "norm",
    "Post-attn RMSNorm",
    "RMSNorm (post_attention_layernorm)",              # display
    "self.post_attention_layernorm = RMSNorm",         # grep anchor
    "vllm/model_executor/models/qwen3_moe.py",
    "Pre-MoE normalization, fused with the residual add.",
)

# MoE steps: router gate + FusedMoE experts block
MOE_QKNORM_MOE_STEPS: list[_BlockDef] = [
    _b(
        "router", "moe_router", "router",
        "MoE router (gate)",
        "ReplicatedLinear (gate)",                     # display
        "self.gate = ReplicatedLinear",                # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Top-k gating: projects hidden_size → num_experts and selects the top-k experts per token.",
        "norm_topk_prob=True: softmax over top-k scores is renormalized so they sum to 1.",
    ),
    _b(
        "experts", "moe_experts", "moe",
        "Sparse expert FFNs",
        "FusedMoE (experts)",                         # display
        "self.experts = FusedMoE",                    # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "128 independent SwiGLU FFNs (moe_intermediate_size=768); only top-8 run per token.",
        "Total params: E×3×d×I across all experts; active params: k×3×d×I per token.",
    ),
]

MOE_QKNORM_HEAD: list[_BlockDef] = [
    _b(
        "final_norm", "rmsnorm", "norm",
        "Final RMSNorm",
        "RMSNorm (norm)",                             # display
        "self.norm = RMSNorm",                        # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Normalizes the final hidden state before the LM head.",
    ),
    _b(
        "lm_head", "lm_head", "head",
        "LM head",
        "ParallelLMHead",                             # display
        "self.lm_head = ParallelLMHead",              # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Projects hidden_size → vocab_size to produce logits.",
        "Weight tied to embed_tokens when tie_word_embeddings is set.",
    ),
    _b(
        "logits", "logits", "head",
        "Logits processor",
        "LogitsProcessor",                            # display
        "self.logits_processor = LogitsProcessor",    # grep anchor
        "vllm/model_executor/models/qwen3_moe.py",
        "Gathers logits for the sampled positions; hands off to the Sampler.",
    ),
]

# ---------------------------------------------------------------------------
# mla-moe block template (DeepSeek-V3)
#
# All symbols grounded in vllm/model_executor/models/deepseek_v2.py
# (the file implements both DeepseekV2 and DeepseekV3).
# ---------------------------------------------------------------------------

# Source file shorthand
_DSV2 = "vllm/model_executor/models/deepseek_v2.py"

MLA_MOE_PRELUDE: list[_BlockDef] = [
    _b(
        "embed", "embed", "embed",
        "Token embeddings",
        "VocabParallelEmbedding (embed_tokens)",
        "self.embed_tokens = VocabParallelEmbedding",
        _DSV2,
        "input_ids → hidden_states. Looks up a row of the embedding matrix per token.",
        "tie_word_embeddings=False for DeepSeek-V3.",
    ),
]

# MLA attention pre-norm (DeepseekV2DecoderLayer.input_layernorm)
MLA_MOE_ATT_PRENORM: _BlockDef = _b(
    "input_norm", "rmsnorm", "norm",
    "Input RMSNorm",
    "RMSNorm (input_layernorm)",
    "self.input_layernorm = RMSNorm",
    _DSV2,
    "Pre-attention normalization over hidden_size.",
)

# MLA attention steps — grounded in DeepseekV2Attention (lines ~448–488)
MLA_MOE_ATT_STEPS: list[_BlockDef] = [
    _b(
        "q_a", "q_a_linear", "latent",
        "Q down-proj (A)",
        "ReplicatedLinear (q_a_proj)",
        "self.q_a_proj = ReplicatedLinear",
        _DSV2,
        "Q latent projection (A): projects hidden_size → q_lora_rank, compressing queries into a low-rank latent space (MLA down-projection).",
    ),
    _b(
        "q_a_norm", "mla_q_norm", "norm",
        "Q latent RMSNorm",
        "RMSNorm (q_a_layernorm, q_lora_rank)",
        "self.q_a_layernorm = RMSNorm",
        _DSV2,
        "RMSNorm over q_lora_rank before decompression.",
    ),
    _b(
        "q_b", "q_b_linear", "latent",
        "Q up-proj (B)",
        "ColumnParallelLinear (q_b_proj)",
        "self.q_b_proj = ColumnParallelLinear",
        _DSV2,
        "Q decompression projection (B): decompresses q_lora_rank → H·(qk_nope_head_dim + qk_rope_head_dim), expanding queries back to full multi-head dimension (MLA up-projection).",
    ),
    _b(
        "kv_a", "kv_a_linear", "latent",
        "KV down-proj (A)",
        "ReplicatedLinear (kv_a_proj_with_mqa)",
        "self.kv_a_proj_with_mqa = ReplicatedLinear",
        _DSV2,
        "KV latent projection (A): projects hidden_size → kv_lora_rank + qk_rope_head_dim, compressing keys/values into a shared low-rank latent plus a decoupled RoPE key (MLA down-projection).",
    ),
    _b(
        "kv_a_norm", "mla_kv_norm", "norm",
        "KV latent RMSNorm",
        "RMSNorm (kv_a_layernorm, kv_lora_rank)",
        "self.kv_a_layernorm = RMSNorm",
        _DSV2,
        "RMSNorm over kv_lora_rank before decompression.",
    ),
    _b(
        "kv_b", "kv_b_linear", "latent",
        "KV up-proj (B)",
        "ColumnParallelLinear (kv_b_proj)",
        "self.kv_b_proj = ColumnParallelLinear",
        _DSV2,
        "KV decompression projection (B): decompresses kv_lora_rank → H·(qk_nope_head_dim + v_head_dim), reconstructing full key/value heads for attention (MLA up-projection).",
    ),
    _b(
        "rope", "rope", "rope",
        "RoPE (Q, K)",
        "RotaryEmbedding (get_rope)",
        "def get_rope",
        "vllm/model_executor/layers/rotary_embedding/__init__.py",
        "Rotary position embedding on decoupled RoPE portions of Q and K.",
        "rope_theta=10000 for DeepSeek-V3.",
    ),
    _b(
        "attn", "attention_mla", "attn",
        "MLA attention",
        "Attention.forward",
        "def forward",
        "vllm/model_executor/layers/attention/attention.py",
        "MLA Attention (paged KV): scaled dot-product attention with compressed KV cache. KV cache stores the latent vector (kv_lora_rank + qk_rope_head_dim dims) per token — far smaller than GQA.",
        "KV cache is (kvl+qr)·bytes per token — far smaller than GQA.",
    ),
    _b(
        "o_proj", "mla_o_proj", "proj",
        "Output projection",
        "RowParallelLinear (o_proj)",
        "self.o_proj = RowParallelLinear",
        _DSV2,
        "Projects H·v_head_dim → hidden_size.",
    ),
]

# Dense MLP pre-norm (used in the first first_k_dense_replace layers)
MLA_MOE_DENSE_MLP_PRENORM: _BlockDef = _b(
    "post_norm", "rmsnorm", "norm",
    "Post-attn RMSNorm",
    "RMSNorm (post_attention_layernorm)",
    "self.post_attention_layernorm = RMSNorm",
    _DSV2,
    "Pre-MLP normalization.",
)

# Dense MLP steps — grounded in DeepseekV2MLP (lines ~215–230)
MLA_MOE_DENSE_MLP_STEPS: list[_BlockDef] = [
    _b(
        "gate_up", "gate_up", "mlp",
        "Gate+Up projection",
        "MergedColumnParallelLinear (gate_up_proj)",
        "self.gate_up_proj = MergedColumnParallelLinear",
        _DSV2,
        "Fused gate and up projection: hidden_size → 2·intermediate_size.",
    ),
    _b(
        "silu", "activation", "act",
        "SiLU + gating",
        "SiluAndMul",
        "class SiluAndMul",
        "vllm/model_executor/layers/activation.py",
        "SwiGLU activation: silu(gate) · up → intermediate_size.",
    ),
    _b(
        "down", "down", "mlp",
        "Down projection",
        "RowParallelLinear (down_proj)",
        "self.down_proj = RowParallelLinear",
        _DSV2,
        "Projects intermediate_size back to hidden_size.",
    ),
]

# MoE pre-norm (used in MoE layers)
MLA_MOE_MOE_PRENORM: _BlockDef = _b(
    "post_norm", "rmsnorm", "norm",
    "Post-attn RMSNorm",
    "RMSNorm (post_attention_layernorm)",
    "self.post_attention_layernorm = RMSNorm",
    _DSV2,
    "Pre-MoE normalization.",
)

# MoE steps — grounded in DeepseekV2MoE (lines ~245–350)
MLA_MOE_MOE_STEPS: list[_BlockDef] = [
    _b(
        "router", "moe_router", "router",
        "MoE router (gate)",
        "GateLinear (gate)",
        "self.gate = GateLinear",
        _DSV2,
        "Top-k gating: projects hidden_size → num_experts; selects top-8 experts per token.",
        "norm_topk_prob=True: renormalizes top-k softmax scores.",
    ),
    _b(
        "experts", "moe_experts", "moe",
        "Sparse expert FFNs",
        "FusedMoE (experts)",
        "self.experts = FusedMoE",
        _DSV2,
        "256 independent SwiGLU FFNs (moe_intermediate_size=2048); only top-8 run per token.",
        "Total params: E×3×d×I across all experts; active params: k×3×d×I per token.",
    ),
    _b(
        "shared_expert", "shared_expert", "moe",
        "Shared expert FFN",
        "DeepseekV2MLP (shared_experts)",
        "self.shared_experts = DeepseekV2MLP",
        _DSV2,
        "Always-active shared expert (1×moe_intermediate_size). Runs for every token.",
        "Unlike routed experts, shared_experts always contributes to every token.",
    ),
]

MLA_MOE_HEAD: list[_BlockDef] = [
    _b(
        "final_norm", "rmsnorm", "norm",
        "Final RMSNorm",
        "RMSNorm (norm)",
        "self.norm = RMSNorm",
        _DSV2,
        "Normalizes the final hidden state before the LM head.",
    ),
    _b(
        "lm_head", "lm_head", "head",
        "LM head",
        "ParallelLMHead",
        "self.lm_head = ParallelLMHead",
        _DSV2,
        "Projects hidden_size → vocab_size to produce logits.",
        "tie_word_embeddings=False for DeepSeek-V3.",
    ),
    _b(
        "logits", "logits", "head",
        "Logits processor",
        "LogitsProcessor",
        "self.logits_processor = LogitsProcessor",
        _DSV2,
        "Gathers logits for the sampled positions; hands off to the Sampler.",
    ),
]

TEMPLATES: dict[str, dict] = {
    "dense-qknorm": {
        "prelude": DENSE_QKNORM_PRELUDE,
        "att_prenorm": DENSE_QKNORM_ATT_PRENORM,
        "att_steps": DENSE_QKNORM_ATT_STEPS,
        "mlp_prenorm": DENSE_QKNORM_MLP_PRENORM,
        "mlp_steps": DENSE_QKNORM_MLP_STEPS,
        "head": DENSE_QKNORM_HEAD,
    },
    "moe-qknorm": {
        "prelude": MOE_QKNORM_PRELUDE,
        "att_prenorm": MOE_QKNORM_ATT_PRENORM,
        "att_steps": MOE_QKNORM_ATT_STEPS,
        "moe_prenorm": MOE_QKNORM_MOE_PRENORM,
        "moe_steps": MOE_QKNORM_MOE_STEPS,
        "head": MOE_QKNORM_HEAD,
    },
    "mla-moe": {
        "prelude": MLA_MOE_PRELUDE,
        "att_prenorm": MLA_MOE_ATT_PRENORM,
        "att_steps": MLA_MOE_ATT_STEPS,
        "dense_mlp_prenorm": MLA_MOE_DENSE_MLP_PRENORM,
        "dense_mlp_steps": MLA_MOE_DENSE_MLP_STEPS,
        "moe_prenorm": MLA_MOE_MOE_PRENORM,
        "moe_steps": MLA_MOE_MOE_STEPS,
        "head": MLA_MOE_HEAD,
    },
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_hf_config(hf_model_id: str) -> dict | None:
    """Try to fetch config via transformers.AutoConfig; return None on any failure."""
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as exc:
        print(f"[warn] transformers unavailable ({exc}); will use fallback config")
        return None
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    except Exception as exc:
        print(f"[warn] could not fetch {hf_model_id} ({exc}); will use fallback config")
        return None

    out: dict = {}
    for k in CONFIG_KEYS:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            if v is not None:  # skip None MoE fields for dense models
                out[k] = v
    # Derive head_dim if missing.
    # For MLA models (qk_nope_head_dim present), head_dim = nope + rope.
    if "head_dim" not in out:
        qk_nope = getattr(cfg, "qk_nope_head_dim", None)
        qk_rope = getattr(cfg, "qk_rope_head_dim", None)
        if qk_nope is not None and qk_rope is not None:
            out["head_dim"] = qk_nope + qk_rope
        elif out.get("hidden_size") and out.get("num_attention_heads"):
            out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]
    if "rope_theta" not in out:
        rope = getattr(cfg, "rope_theta", None)
        if rope is not None:
            out["rope_theta"] = float(rope)
    if "torch_dtype" in out:
        out["torch_dtype"] = str(out["torch_dtype"]).replace("torch.", "")
    elif hasattr(cfg, "torch_dtype"):
        out["torch_dtype"] = str(getattr(cfg, "torch_dtype")).replace("torch.", "")
    # DeepSeek-V3 uses HF field names n_routed_experts / n_shared_experts
    # Map them into the canonical manifest field names used by blockTypes.ts
    if "num_experts" not in out and hasattr(cfg, "n_routed_experts"):
        v = getattr(cfg, "n_routed_experts", None)
        if v is not None:
            out["num_experts"] = v
    if "num_shared_experts" not in out and hasattr(cfg, "n_shared_experts"):
        v = getattr(cfg, "n_shared_experts", None)
        if v is not None:
            out["num_shared_experts"] = v
    if "num_experts_per_tok" not in out and hasattr(cfg, "num_experts_per_tok"):
        # Some configs use num_experts_per_tok already; others use top_k
        pass
    if "num_experts_per_tok" not in out and hasattr(cfg, "top_k"):
        v = getattr(cfg, "top_k", None)
        if v is not None:
            out["num_experts_per_tok"] = v
    return out


# ---------------------------------------------------------------------------
# Block instantiation
# ---------------------------------------------------------------------------

def _make_block(bdef: _BlockDef, repo_root: Path, hint_map: dict[str, int]) -> dict:
    """Convert a template block definition into a manifest Block dict."""
    bid, btype, kind, label, display_symbol, grep_symbol, source_file, desc, note = bdef
    hint = hint_map.get(bid, 1)
    line = resolve_line(repo_root, source_file, grep_symbol, hint)
    block: dict = {
        "id": bid,
        "type": btype,
        "label": label,
        "symbol": display_symbol,
        "ref": f"{source_file}:{line}",
        "kind": kind,
        "desc": desc,
    }
    if note:
        block["note"] = note
    return block


# Hint lines — fallback if symbol grep fails (file:approx-line)
_HINTS: dict[str, int] = {
    # dense-qknorm hints (qwen3.py / qwen2.py symbols)
    "embed":       358,
    "input_norm":  211,
    "qkv_proj":     98,
    "q_norm":      142,
    "k_norm":      143,
    "rope":         33,
    "attn":        409,
    "o_proj":      107,
    "post_norm":   212,
    "gate_up":      93,
    "silu":        118,
    "down":        100,
    "final_norm":  382,
    "lm_head":     298,
    "logits":      307,
    # moe-qknorm hints (qwen3_moe.py symbols)
    "router":      179,   # self.gate = ReplicatedLinear
    "experts":     211,   # self.experts = FusedMoE
    # mla-moe hints (deepseek_v2.py symbols)
    "q_a":         448,   # self.q_a_proj = ReplicatedLinear
    "q_a_norm":    455,   # self.q_a_layernorm = RMSNorm
    "q_b":         456,   # self.q_b_proj = ColumnParallelLinear
    "kv_a":        472,   # self.kv_a_proj_with_mqa = ReplicatedLinear
    "kv_a_norm":   479,   # self.kv_a_layernorm = RMSNorm
    "kv_b":        480,   # self.kv_b_proj = ColumnParallelLinear
    "shared_expert": 317, # self.shared_experts = DeepseekV2MLP
}


def build_manifest(
    slug: str,
    hf_model_id: str,
    family: str,
    label: str,
    repo_root: Path,
) -> dict:
    """Build the full manifest dict for one model."""
    # --- config ---------------------------------------------------------------
    raw = load_hf_config(hf_model_id)
    fallback_base = FALLBACKS.get(slug, {})
    if raw:
        source = "transformers"
        cfg: dict = {}
        for k in CONFIG_KEYS:
            v = raw.get(k, fallback_base.get(k))
            if v is not None:
                cfg[k] = v
        print(f"[ok] {hf_model_id}: loaded from HuggingFace (transformers)")
    else:
        source = "fallback"
        cfg = {k: v for k, v in fallback_base.items() if v is not None}
        print(f"[warn] {hf_model_id}: using hard-coded fallback config")

    # --- template -------------------------------------------------------------
    tmpl = TEMPLATES[family]

    def mb(bdef: _BlockDef) -> dict:
        return _make_block(bdef, repo_root, _HINTS)

    prelude = [mb(b) for b in tmpl["prelude"]]
    att_prenorm = mb(tmpl["att_prenorm"])
    att_steps   = [mb(b) for b in tmpl["att_steps"]]
    head        = [mb(b) for b in tmpl["head"]]

    if family == "moe-qknorm":
        moe_prenorm = mb(tmpl["moe_prenorm"])
        moe_steps   = [mb(b) for b in tmpl["moe_steps"]]
        layers_entry = {
            "repeat": cfg["num_hidden_layers"],
            "label": "MoE decoder layer",
            "branches": [
                {
                    "name": "attn",
                    "accent": "#10b981",
                    "preNorm": att_prenorm,
                    "steps": att_steps,
                },
                {
                    "name": "moe",
                    "accent": "#f472b6",
                    "preNorm": moe_prenorm,
                    "steps": moe_steps,
                },
            ],
        }
        layers = [layers_entry]
    elif family == "mla-moe":
        # Two layer groups: k dense layers + (L - k) MoE layers
        k = cfg.get("first_k_dense_replace", 0)
        rest = cfg["num_hidden_layers"] - k

        dense_mlp_prenorm = mb(tmpl["dense_mlp_prenorm"])
        dense_mlp_steps   = [mb(b) for b in tmpl["dense_mlp_steps"]]
        moe_prenorm        = mb(tmpl["moe_prenorm"])
        moe_steps          = [mb(b) for b in tmpl["moe_steps"]]

        mla_attn_branch = {
            "name": "attn",
            "accent": "#10b981",
            "preNorm": att_prenorm,
            "steps": att_steps,
        }

        dense_group = {
            "repeat": k,
            "label": f"dense layer × first {k}",
            "branches": [
                mla_attn_branch,
                {
                    "name": "mlp",
                    "accent": "#6366f1",
                    "preNorm": dense_mlp_prenorm,
                    "steps": dense_mlp_steps,
                },
            ],
        }

        moe_group = {
            "repeat": rest,
            "label": f"MoE layer × {rest}",
            "branches": [
                mla_attn_branch,
                {
                    "name": "moe",
                    "accent": "#f472b6",
                    "preNorm": moe_prenorm,
                    "steps": moe_steps,
                },
            ],
        }
        layers = [dense_group, moe_group]
    else:
        # dense-qknorm (and future dense families)
        mlp_prenorm = mb(tmpl["mlp_prenorm"])
        mlp_steps   = [mb(b) for b in tmpl["mlp_steps"]]
        layers_entry = {
            "repeat": cfg["num_hidden_layers"],
            "label": "decoder layer",
            "branches": [
                {
                    "name": "attn",
                    "accent": "#10b981",
                    "preNorm": att_prenorm,
                    "steps": att_steps,
                },
                {
                    "name": "mlp",
                    "accent": "#6366f1",
                    "preNorm": mlp_prenorm,
                    "steps": mlp_steps,
                },
            ],
        }
        layers = [layers_entry]

    return {
        "model": hf_model_id,
        "slug": slug,
        "family": family,
        "source": source,
        "config": cfg,
        "prelude": prelude,
        "layers": layers,
        "head": head,
    }


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def compute_total_params(cfg: dict) -> int:
    """Transformer parameter count (no bias assumed; tied embeddings counted once).

    For dense models:
      embeddings:  vocab * hidden
      per layer:
        qkv_proj:  hidden * (q_dim + 2*kv_dim)
        q_norm + k_norm: head_dim each
        o_proj:    q_dim * hidden
        input_norm + post_norm: hidden each
        gate_up:   hidden * 2 * inter
        down:      inter * hidden
      final_norm:  hidden
      lm_head:     vocab * hidden  (0 if tied)

    For MoE models (num_experts present in cfg):
      MLP params replaced per layer by:
        router:    hidden * num_experts
        experts:   num_experts * 3 * hidden * moe_intermediate_size  (all expert weights)
      Attention params identical to dense.

    For MLA-MoE models (q_lora_rank present in cfg):
      k dense layers: MLA attn + dense MLP
      (L-k) MoE layers: MLA attn + router + routed experts + shared experts
      MLA attn params/layer:
        q_a_proj:    hidden * q_lora_rank
        q_a_norm:    q_lora_rank
        q_b_proj:    q_lora_rank * H * (qk_nope + qk_rope)
        kv_a_proj:   hidden * (kv_lora_rank + qk_rope)
        kv_a_norm:   kv_lora_rank
        kv_b_proj:   kv_lora_rank * H * (qk_nope + v_head)
        o_proj:      H * v_head * hidden
        input_norm + post_norm: hidden each
    """
    d     = cfg["hidden_size"]
    L     = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_h  = cfg["num_key_value_heads"]
    hd    = cfg["head_dim"]
    vocab = cfg["vocab_size"]
    tied  = cfg.get("tie_word_embeddings", False)

    q_dim  = heads * hd
    kv_dim = kv_h  * hd

    # Embeddings (counted once)
    embed = vocab * d

    # Head
    final_norm = d
    lm_head    = 0 if tied else vocab * d

    ql = cfg.get("q_lora_rank")
    if ql is not None:
        # MLA-MoE model (e.g. DeepSeek-V3)
        kvl = cfg["kv_lora_rank"]
        qn  = cfg["qk_nope_head_dim"]
        qr  = cfg["qk_rope_head_dim"]
        vd  = cfg["v_head_dim"]
        H   = heads

        # MLA attention params per layer
        q_a       = d * ql
        q_a_norm  = ql
        q_b       = ql * H * (qn + qr)
        kv_a      = d * (kvl + qr)
        kv_a_norm = kvl
        kv_b      = kvl * H * (qn + vd)
        o_proj    = H * vd * d
        norms     = 2 * d   # input_norm + post_norm
        mla_attn_params = q_a + q_a_norm + q_b + kv_a + kv_a_norm + kv_b + o_proj + norms

        k    = cfg.get("first_k_dense_replace", 0)
        rest = L - k

        # Dense MLP (first k layers)
        inter   = cfg["intermediate_size"]
        dense_mlp = d * 2 * inter + inter * d   # gate_up + down

        # MoE FFN (remaining layers)
        E       = cfg["num_experts"]
        I_moe   = cfg["moe_intermediate_size"]
        ns      = cfg.get("num_shared_experts", 0) or 0
        router  = d * E
        experts = E * 3 * d * I_moe
        shared  = ns * 3 * d * I_moe
        moe_ffn = router + experts + shared

        total = (
            embed
            + k    * (mla_attn_params + dense_mlp)
            + rest * (mla_attn_params + moe_ffn)
            + final_norm
            + lm_head
        )
        return total

    # Attention params — same for dense and MoE (GQA)
    qkv    = d * (q_dim + 2 * kv_dim)
    q_norm = hd
    k_norm = hd
    o_proj = q_dim * d
    norms  = 2 * d          # input_norm + post_norm
    attn_params = qkv + q_norm + k_norm + o_proj + norms

    E = cfg.get("num_experts")
    if E:
        # MoE model: replace dense MLP with router + expert FFNs
        I = cfg["moe_intermediate_size"]
        router  = d * E                 # gate Linear: hidden → num_experts
        experts = E * 3 * d * I        # all E expert SwiGLU FFNs (gate+up+down)
        ffn_params = router + experts
    else:
        # Dense model: standard SwiGLU FFN
        inter = cfg["intermediate_size"]
        gate_up  = d * 2 * inter
        down     = inter * d
        ffn_params = gate_up + down

    per_layer = attn_params + ffn_params

    return embed + L * per_layer + final_norm + lm_head


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-model architecture manifests for the Explorer SPA.",
    )
    ap.add_argument("--repo-root", default=".", help="Path to vLLM repo root")
    ap.add_argument(
        "--out-dir",
        default="explorer/public/data/models",
        help="Directory to write <slug>.json and index.json",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir   = (
        Path(args.out_dir)
        if Path(args.out_dir).is_absolute()
        else repo_root / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []

    for slug, hf_model_id, family, label in MODEL_LIST:
        manifest = build_manifest(slug, hf_model_id, family, label, repo_root)
        total_params = compute_total_params(manifest["config"])

        out_path = out_dir / f"{slug}.json"
        out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        cfg = manifest["config"]
        print(
            f"wrote {out_path} : {hf_model_id} ({manifest['source']}) "
            f"L={cfg['num_hidden_layers']} d={cfg['hidden_size']} "
            f"heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']} "
            f"hd={cfg['head_dim']}  totalParams={total_params:,}"
        )

        index.append({
            "slug": slug,
            "label": label,
            "family": family,
            "totalParams": total_params,
        })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index_path} : {len(index)} models")


if __name__ == "__main__":
    main()
