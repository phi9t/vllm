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
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
CONFIG_KEYS = [
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
    "tie_word_embeddings", "torch_dtype", "rope_theta",
]

MODEL_LIST: list[tuple[str, str, str, str]] = [
    # (slug, hf_model_id, family, human_label)
    ("qwen3-0_6b", "Qwen/Qwen3-0.6B", "dense-qknorm", "Qwen3-0.6B"),
    ("qwen3-8b",   "Qwen/Qwen3-8B",   "dense-qknorm", "Qwen3-8B"),
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

TEMPLATES: dict[str, dict] = {
    "dense-qknorm": {
        "prelude": DENSE_QKNORM_PRELUDE,
        "att_prenorm": DENSE_QKNORM_ATT_PRENORM,
        "att_steps": DENSE_QKNORM_ATT_STEPS,
        "mlp_prenorm": DENSE_QKNORM_MLP_PRENORM,
        "mlp_steps": DENSE_QKNORM_MLP_STEPS,
        "head": DENSE_QKNORM_HEAD,
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
        cfg = AutoConfig.from_pretrained(hf_model_id)
    except Exception as exc:
        print(f"[warn] could not fetch {hf_model_id} ({exc}); will use fallback config")
        return None

    out: dict = {}
    for k in CONFIG_KEYS:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    # Derive head_dim if missing
    if "head_dim" not in out and out.get("hidden_size") and out.get("num_attention_heads"):
        out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]
    if "rope_theta" not in out:
        rope = getattr(cfg, "rope_theta", None)
        if rope is not None:
            out["rope_theta"] = float(rope)
    if "torch_dtype" in out:
        out["torch_dtype"] = str(out["torch_dtype"]).replace("torch.", "")
    elif hasattr(cfg, "torch_dtype"):
        out["torch_dtype"] = str(getattr(cfg, "torch_dtype")).replace("torch.", "")
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


# Hint lines from qwen3Blocks.ts (fall-through if symbol not found)
_HINTS: dict[str, int] = {
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
    if raw:
        source = "transformers"
        fallback_base = FALLBACKS[slug]
        cfg: dict = {}
        for k in CONFIG_KEYS:
            cfg[k] = raw.get(k, fallback_base.get(k))
        print(f"[ok] {hf_model_id}: loaded from HuggingFace (transformers)")
    else:
        source = "fallback"
        cfg = dict(FALLBACKS[slug])
        print(f"[warn] {hf_model_id}: using hard-coded fallback config")

    # --- template -------------------------------------------------------------
    tmpl = TEMPLATES[family]

    def mb(bdef: _BlockDef) -> dict:
        return _make_block(bdef, repo_root, _HINTS)

    prelude = [mb(b) for b in tmpl["prelude"]]
    att_prenorm = mb(tmpl["att_prenorm"])
    att_steps   = [mb(b) for b in tmpl["att_steps"]]
    mlp_prenorm = mb(tmpl["mlp_prenorm"])
    mlp_steps   = [mb(b) for b in tmpl["mlp_steps"]]
    head        = [mb(b) for b in tmpl["head"]]

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

    return {
        "model": hf_model_id,
        "slug": slug,
        "family": family,
        "source": source,
        "config": cfg,
        "prelude": prelude,
        "layers": [layers_entry],
        "head": head,
    }


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def compute_total_params(cfg: dict) -> int:
    """Dense Transformer parameter count (no training/backward, no bias assumed).

    Formula:
      embeddings:  vocab * hidden          (counted once; tied => lm_head reuses)
      per layer:
        qkv_proj:  hidden * (q_dim + 2*kv_dim)
        q_norm:    head_dim
        k_norm:    head_dim
        o_proj:    q_dim * hidden
        input_norm: hidden
        post_norm:  hidden
        gate_up:   hidden * 2 * inter
        down:      inter * hidden
      final_norm:  hidden
      lm_head:     vocab * hidden          (tied => already counted)
    """
    d     = cfg["hidden_size"]
    L     = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_h  = cfg["num_key_value_heads"]
    hd    = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    tied  = cfg.get("tie_word_embeddings", False)

    q_dim  = heads * hd
    kv_dim = kv_h  * hd

    # Embeddings (counted once)
    embed = vocab * d

    # Per-layer params
    qkv    = d * (q_dim + 2 * kv_dim)
    q_norm = hd
    k_norm = hd
    o_proj = q_dim * d
    norms  = 2 * d          # input_norm + post_norm
    gate_up = d * 2 * inter
    down   = inter * d
    per_layer = qkv + q_norm + k_norm + o_proj + norms + gate_up + down

    # Head
    final_norm = d
    lm_head    = 0 if tied else vocab * d

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
