# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream tokens from AsyncLLM on a tiny model.

Pairs with HACKERS_GUIDE.md §3 (LLM.generate entry point).

Like hack 01 this loads a real model and is gated behind
`VLLM_HACK_RUN_MODEL=1`. Demonstrates `AsyncLLM.generate` returning an
async iterator of `RequestOutput` chunks (see
vllm/v1/engine/async_llm.py:524).
"""

from __future__ import annotations

import asyncio

from _stubs import print_header, require_model_run


async def stream() -> int:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    print_header("Booting AsyncLLM(facebook/opt-125m)")
    engine_args = AsyncEngineArgs(model="facebook/opt-125m")
    llm = AsyncLLM.from_engine_args(engine_args)

    print_header("Streaming")
    request_id = "hack-13"
    params = SamplingParams(temperature=0.0, max_tokens=12)
    async for out in llm.generate(
        prompt="The capital of France is",
        sampling_params=params,
        request_id=request_id,
    ):
        text = out.outputs[0].text
        finished = out.finished
        print(f"  chunk: finished={finished} text={text!r}")
    return 0


def main() -> int:
    require_model_run()
    return asyncio.run(stream())


if __name__ == "__main__":
    raise SystemExit(main())
