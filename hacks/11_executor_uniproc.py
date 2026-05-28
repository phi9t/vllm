# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tour the Executor abstraction and its concrete implementations.

Pairs with HACKERS_GUIDE.md §11 (Workers & executors) and §12.

Instantiating a real `UniProcExecutor` requires a complete `VllmConfig`
plus a loadable model — far too much for a hack script. We instead
walk the `Executor` ABC and list the four concrete implementations,
which is what you actually look at when picking the right executor for
a deployment.
"""

from __future__ import annotations

import inspect

from _stubs import print_header

from vllm.v1.executor.abstract import Executor


def main() -> int:
    print_header("vllm.v1.executor.abstract.Executor (ABC)")
    for name in sorted(dir(Executor)):
        if name.startswith("_"):
            continue
        attr = getattr(Executor, name)
        if not callable(attr):
            continue
        try:
            sig = inspect.signature(attr)
        except (TypeError, ValueError):
            sig = "(...)"
        print(f"  {name}{sig}")

    print_header("Concrete executors")
    impls = [
        (
            "vllm.v1.executor.uniproc_executor",
            "UniProcExecutor",
            "one process, one GPU — easiest to step through in a debugger",
        ),
        (
            "vllm.v1.executor.multiproc_executor",
            "MultiprocExecutor",
            "one OS process per GPU — default multi-GPU on a single node",
        ),
        (
            "vllm.v1.executor.ray_executor",
            "RayDistributedExecutor",
            "Ray actors — multi-node serving",
        ),
    ]
    for mod, cls, why in impls:
        print(f"  {mod}.{cls}")
        print(f"      {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
