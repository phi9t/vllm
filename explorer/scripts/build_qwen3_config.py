#!/usr/bin/env python3
"""Snapshot a Qwen3 dense model config for the Architecture Deep Dive's in-browser math.

Tries to load the real config via ``transformers.AutoConfig``; if transformers or
the network is unavailable, falls back to the known Qwen3-0.6B values so the UI
still renders. Run via the repo venv (AGENTS.md):

    .venv/bin/python explorer/scripts/build_qwen3_config.py \
        --model Qwen/Qwen3-0.6B --out explorer/public/data/qwen3_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Known Qwen3-0.6B config (offline fallback / default).
FALLBACK = {
    "model": "Qwen/Qwen3-0.6B",
    "source": "fallback",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}

KEYS = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "tie_word_embeddings",
]


def load_real(model: str) -> dict | None:
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] transformers unavailable ({exc}); using fallback config")
        return None
    try:
        cfg = AutoConfig.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not fetch {model} ({exc}); using fallback config")
        return None

    out: dict = {"model": model, "source": "transformers"}
    for k in KEYS:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    # head_dim may be derived if not explicit.
    if "head_dim" not in out and out.get("hidden_size") and out.get("num_attention_heads"):
        out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]
    rope = getattr(cfg, "rope_theta", None)
    if rope is not None:
        out["rope_theta"] = float(rope)
    out["torch_dtype"] = str(getattr(cfg, "torch_dtype", "bfloat16")).replace("torch.", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="explorer/public/data/qwen3_config.json")
    args = ap.parse_args()

    cfg = load_real(args.model) or dict(FALLBACK)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} : {cfg['model']} ({cfg['source']}) "
          f"L={cfg['num_hidden_layers']} d={cfg['hidden_size']} "
          f"heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']} hd={cfg['head_dim']}")


if __name__ == "__main__":
    main()
