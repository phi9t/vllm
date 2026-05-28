# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared stubs/helpers for the hacks/ directory.

These exist so each hack can exercise one V1 component without booting
the full engine. Nothing here is part of vLLM's public API.

Pairs with HACKERS_GUIDE.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def require_model_run() -> None:
    """Exit cleanly if the user hasn't opted into model-loading hacks."""
    if os.environ.get("VLLM_HACK_RUN_MODEL") != "1":
        print(
            "skipped: this hack loads a real model. Set "
            "VLLM_HACK_RUN_MODEL=1 to run it."
        )
        raise SystemExit(0)


@dataclass
class FakeSamplingParams:
    """Just enough for hand-rolled scheduling/sampler demos."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class FakeModelRunnerOutput:
    """Minimal shape compatible with our scheduler/output_processor demos."""

    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: Any | None = None
    prompt_logprobs_dict: dict[str, Any] = field(default_factory=dict)
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
