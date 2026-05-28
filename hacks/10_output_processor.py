# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect OutputProcessor — what each method does in the request flow.

Pairs with HACKERS_GUIDE.md §10 (Request lifecycle & output).

Driving OutputProcessor with real engine outputs needs a live tokenizer,
detokenizer state, and `EngineCoreOutputs` instances. We instantiate the
class with `tokenizer=None` and inspect its public API surface so you
know what to look up when chasing a streaming bug.
"""

from __future__ import annotations

import inspect

from _stubs import print_header

from vllm.v1.engine.output_processor import (
    OutputProcessor,
    RequestOutputCollector,
)


def main() -> int:
    op = OutputProcessor(tokenizer=None, log_stats=False)
    print_header("OutputProcessor instance")
    print(f"  has_unfinished_requests() = {op.has_unfinished_requests()}")
    print(f"  request_states            = {op.request_states}")
    print(f"  parent_requests           = {op.parent_requests}")

    print_header("Public methods")
    for name, member in inspect.getmembers(
        OutputProcessor, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        sig = inspect.signature(member)
        print(f"  {name}{sig}")

    print_header("RequestOutputCollector")
    print("  Lives next door — accumulates `RequestOutput` chunks per request.")
    for name, member in inspect.getmembers(
        RequestOutputCollector, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        sig = inspect.signature(member)
        print(f"  {name}{sig}")

    print(
        "\n  Read these in vllm/v1/engine/output_processor.py around line 438"
        "\n  to see how `EngineCoreOutputs` are turned into the streaming"
        "\n  `RequestOutput`s the OpenAI API serves."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
