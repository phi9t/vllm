# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which captured CUDA graph does a batch replay? The bucketing logic.

Pairs with HACKERS_GUIDE.md §18 (CUDA graphs & torch.compile).

vLLM captures the forward at a fixed list of batch sizes
(`cudagraph_capture_sizes`, vllm/config/compilation.py:622) during
capture_model() (vllm/v1/worker/gpu_model_runner.py:6150). At run time a
batch is padded UP to the nearest captured size and that graph is
replayed; batches larger than the max fall back to eager. This script
reproduces that selection — pure logic, no CUDA.
"""

from __future__ import annotations

from bisect import bisect_left

from _stubs import print_header


def default_capture_sizes(max_bs: int = 512) -> list[int]:
    """Mirror of vLLM's default-ish ladder: 1, 2, 4, then steps of 8/16."""
    sizes = [1, 2, 4]
    s = 8
    while s <= max_bs:
        sizes.append(s)
        s += 8 if s < 32 else 16
    return sizes


def pick_graph(batch_size: int, capture_sizes: list[int]) -> tuple[str, int, int]:
    """Return (mode, padded_size, padding) for a batch."""
    if batch_size > capture_sizes[-1]:
        return ("eager", batch_size, 0)
    i = bisect_left(capture_sizes, batch_size)
    padded = capture_sizes[i]
    return ("cudagraph", padded, padded - batch_size)


def main() -> int:
    sizes = default_capture_sizes(max_bs=256)
    print_header("Captured CUDA-graph batch sizes")
    print(f"  {sizes}")
    print(f"  ({len(sizes)} graphs captured at startup; each costs memory)")

    print_header("Runtime: pad each batch up to the nearest captured size")
    print(f"  {'batch':>6} | {'mode':<10} | {'replays graph':>13} | wasted")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*13}-+-------")
    for bs in [1, 3, 7, 8, 9, 31, 33, 100, 256, 300]:
        mode, padded, pad = pick_graph(bs, sizes)
        graph = "-" if mode == "eager" else f"bs={padded}"
        note = "(> max → eager)" if mode == "eager" else f"+{pad} pad tokens"
        print(f"  {bs:>6} | {mode:<10} | {graph:>13} | {note}")

    print_header("Takeaway")
    print("  More capture sizes = less padding waste but more capture memory &")
    print("  startup time. FULL_AND_PIECEWISE (the V1 default) captures full")
    print("  graphs for pure-decode batches and piecewise for mixed prefill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
