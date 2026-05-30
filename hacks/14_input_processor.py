# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace the input path: prompt text -> token ids -> EngineCoreRequest.

Pairs with HACKERS_GUIDE.md §4 (Input processing & tokenization).

The front door of the engine turns a user prompt into an
`EngineCoreRequest` (vllm/v1/engine/__init__.py:80) before anything is
scheduled. The real path is:

    prompt --tokenizer--> token_ids
           --InputPreprocessor (vllm/inputs/preprocess.py:48)-->
           ProcessorInputs
           --InputProcessor (vllm/v1/engine/input_processor.py:36)-->
           EngineCoreRequest

Building the real `InputProcessor` needs a full `VllmConfig` + tokenizer,
so here we (1) mirror the prompt->ids->request *shape* with a tiny
deterministic byte tokenizer (no weights, no network), and (2) best-effort
introspect the real `EngineCoreRequest` fields if vLLM is importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _stubs import print_header


# --- A tiny deterministic tokenizer (stand-in for the real HF tokenizer) -----
# Real tokenization happens in vllm/transformers_utils/tokenizer.py; here we
# only need *a* reproducible prompt -> ids mapping to show the request shape.
class ByteTokenizer:
    """Maps each byte to an id in [256, 511]; ids 0..255 are reserved/special."""

    eos_id = 0
    bos_id = 1

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        ids = [256 + b for b in text.encode("utf-8")]
        return [self.bos_id, *ids] if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        body = bytes(i - 256 for i in ids if i >= 256)
        return body.decode("utf-8", errors="replace")


@dataclass
class MiniEngineCoreRequest:
    """Mirror of the fields the scheduler actually reads off the real
    EngineCoreRequest (vllm/v1/engine/__init__.py:80)."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: dict
    arrival_time: float
    lora_request: None = None
    mm_features: list = field(default_factory=list)


def real_engine_core_request_fields() -> list[str] | None:
    """Best-effort: list the real EngineCoreRequest fields if vLLM imports."""
    try:
        from vllm.v1.engine import EngineCoreRequest  # type: ignore
    except Exception:
        return None
    # msgspec.Struct exposes its field names here.
    return list(getattr(EngineCoreRequest, "__struct_fields__", []))


def main() -> int:
    tok = ByteTokenizer()
    prompt = "The capital of France is"

    print_header("1. Tokenize the prompt")
    ids = tok.encode(prompt)
    print(f"  prompt      : {prompt!r}")
    print(f"  token_ids   : {ids}")
    print(f"  num_tokens  : {len(ids)}  (1 BOS + {len(ids) - 1} byte tokens)")
    print(f"  round-trip  : {tok.decode(ids[1:])!r}")

    print_header("2. Wrap into an EngineCoreRequest (the scheduler's input)")
    req = MiniEngineCoreRequest(
        request_id="req-0",
        prompt_token_ids=ids,
        sampling_params={"max_tokens": 8, "temperature": 0.0},
        arrival_time=0.0,
    )
    print(f"  request_id        : {req.request_id}")
    print(f"  prompt_token_ids  : {len(req.prompt_token_ids)} ids")
    print(f"  sampling_params   : {req.sampling_params}")
    print("  -> handed to EngineCore.add_request() -> Scheduler.waiting")

    print_header("3. The real EngineCoreRequest contract")
    fields = real_engine_core_request_fields()
    if fields is None:
        print("  (vLLM not importable here — showing the documented field set)")
        fields = [
            "request_id", "prompt_token_ids", "mm_features", "sampling_params",
            "pooling_params", "arrival_time", "lora_request", "cache_salt",
            "data_parallel_rank",
        ]
    for f in fields:
        print(f"    - {f}")
    print("\n  Source: vllm/v1/engine/__init__.py:80 (class EngineCoreRequest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
