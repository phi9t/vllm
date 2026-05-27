"""Write a custom LogitsProcessor that bans a token.

Pairs with HACKERS_GUIDE.md §9 (Sampling).

Demonstrates: vllm/v1/sample/logits_processor/interface.py (LogitsProcessor
ABC). We don't load the processor through a full Sampler — we just
implement the interface and call `apply()` to show before/after logits.
"""

from __future__ import annotations

import torch
from _stubs import print_header

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)


class BanToken(LogitsProcessor):
    """Ban one specific token id by masking its logit to -inf."""

    def __init__(self, vllm_config=None, device=None, is_pin_memory=False):
        self.banned_id: int | None = None

    def set_banned(self, token_id: int) -> None:
        self.banned_id = token_id

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.banned_id is not None:
            logits[..., self.banned_id] = -float("inf")
        return logits

    def is_argmax_invariant(self) -> bool:
        return False  # banning a token can change argmax

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        return None


def show_top(name: str, logits: torch.Tensor) -> None:
    probs = logits.softmax(dim=-1)
    top = torch.topk(probs[0], k=5)
    items = ", ".join(
        f"id={i.item()} p={v.item():.3f}" for v, i in zip(top.values, top.indices)
    )
    print(f"  {name:14s} top-5: [{items}]")


def main() -> int:
    torch.manual_seed(0)
    logits = torch.randn(1, 16) * 2

    proc = BanToken()
    proc.set_banned(int(logits.argmax(dim=-1).item()))

    print_header(f"Banning argmax token {proc.banned_id}")
    show_top("before", logits.clone())
    show_top("after", proc.apply(logits.clone()))

    print(
        "\nWiring tip: real custom processors are registered via "
        "the `logits_processors` field in SamplingParams; see "
        "docs/design/logits_processors.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
