"""Full pipeline smoke test on a tiny model.

Pairs with HACKERS_GUIDE.md §3 (LLM.generate entry point).

The only hack that loads real weights. Gated behind
`VLLM_HACK_RUN_MODEL=1` so CI skips it by default. Requires a working
torch + CUDA install (or CPU build), and ~250 MB of disk for opt-125m.
"""

from __future__ import annotations

from _stubs import print_header, require_model_run


def main() -> int:
    require_model_run()

    from vllm import LLM, SamplingParams

    print_header("Booting facebook/opt-125m")
    llm = LLM(model="facebook/opt-125m")

    print_header("Generating")
    params = SamplingParams(temperature=0.0, max_tokens=12)
    outs = llm.generate(["The capital of France is"], params)
    for out in outs:
        print(f"  prompt   : {out.prompt!r}")
        print(f"  output   : {out.outputs[0].text!r}")
        print(f"  token_ids: {list(out.outputs[0].token_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
