"""Inventory the V1 attention backends and the selector's inputs.

Pairs with HACKERS_GUIDE.md §8 (Attention backends).

`vllm/v1/attention/selector.py:get_attn_backend` requires a live
`VllmConfig` context (`get_current_vllm_config()`), which we don't have
in a hack script. Instead we enumerate the backend files and the inputs
the selector uses (the `AttentionSelectorConfig` fields) so you know
which knob to wiggle when debugging backend selection.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from _stubs import print_header

BACKENDS_DIR = Path(
    importlib.import_module("vllm.v1.attention.backends").__file__
).parent


def main() -> int:
    print_header("Available attention backend files")
    for path in sorted(BACKENDS_DIR.iterdir()):
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
            print(f"  vllm/v1/attention/backends/{path.name}")
        elif path.is_dir() and not path.name.startswith("_"):
            print(f"  vllm/v1/attention/backends/{path.name}/  (subpackage)")

    print_header("Selector inputs (AttentionSelectorConfig fields)")
    from vllm.v1.attention.selector import AttentionSelectorConfig

    for name in inspect.signature(AttentionSelectorConfig).parameters:
        print(f"  - {name}")

    print_header("Selector signature")
    from vllm.v1.attention.selector import get_attn_backend

    sig = inspect.signature(get_attn_backend)
    print(f"  get_attn_backend{sig}")
    print(
        "\n  The selector reads dtype, head size, kv_cache_dtype, MLA-ness,"
        "\n  and VLLM_ATTENTION_BACKEND, then dispatches to one of the"
        "\n  backend files listed above. See selector.py:52."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
