# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step through the sampling pipeline op by op.

Pairs with HACKERS_GUIDE.md §9 (Sampling).

Demonstrates: vllm/v1/sample/ops/topk_topp_sampler.py (`apply_top_k_top_p`),
ops/bad_words.py (`apply_bad_words`). We call the operations directly
instead of going through `Sampler.forward`, which would need a fully
populated `SamplingMetadata`.
"""

from __future__ import annotations

import torch
from _stubs import print_header

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def show(name: str, logits: torch.Tensor) -> None:
    probs = logits.softmax(dim=-1)
    top = torch.topk(probs[0], k=5)
    items = ", ".join(
        f"id={i.item()} p={v.item():.3f}" for v, i in zip(top.values, top.indices)
    )
    print(f"  {name:24s} top-5: [{items}]")


def main() -> int:
    torch.manual_seed(0)
    vocab = 16
    base = torch.randn(1, vocab) * 2

    print_header("Raw logits")
    show("raw", base)

    print_header("Temperature scaling (lower temperature -> sharper)")
    for t in (0.7, 1.0, 1.5):
        show(f"temperature={t}", base / t)

    print_header("Top-k")
    for k in (1, 3, 8):
        masked = apply_top_k_top_p_pytorch(
            base.clone(), k=torch.tensor([k]), p=None, allow_cpu_sync=True
        )
        show(f"top_k={k}", masked)

    print_header("Top-p (nucleus)")
    for p in (0.5, 0.9):
        masked = apply_top_k_top_p_pytorch(
            base.clone(), k=None, p=torch.tensor([p]), allow_cpu_sync=True
        )
        show(f"top_p={p}", masked)

    print_header("Bad-words mask (ban tokens 0 and 1)")
    banned = base.clone()
    banned[0, [0, 1]] = -float("inf")
    show("after ban", banned)

    print_header("Sampling")
    probs = (base / 0.7).softmax(dim=-1)
    drawn = torch.multinomial(probs, num_samples=5, replacement=True)
    samples = drawn[0].tolist()
    greedy = base.argmax(dim=-1).item()
    print(f"  5 random samples (temperature=0.7): {samples}")
    print(f"  argmax (greedy)                   : {greedy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
