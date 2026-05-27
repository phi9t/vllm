# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for the hacks/ directory.

Runs every non-model-loading script with a 30 s timeout and asserts
exit code 0. Catches drift when an internal signature shifts under us.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HACKS_DIR = ROOT / "hacks"
TIMEOUT_S = 30

# Scripts that load a real model — gated behind VLLM_HACK_RUN_MODEL.
MODEL_GATED = {"01_llm_smoke.py", "13_async_llm_stream.py"}


def _hack_scripts() -> list[Path]:
    return sorted(p for p in HACKS_DIR.glob("[0-9][0-9]_*.py") if p.is_file())


@pytest.mark.parametrize("script", _hack_scripts(), ids=lambda p: p.name)
def test_hack_runs(script: Path) -> None:
    if script.name in MODEL_GATED and os.environ.get("VLLM_HACK_RUN_MODEL") != "1":
        pytest.skip(f"{script.name} requires VLLM_HACK_RUN_MODEL=1")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=HACKS_DIR,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, (
        f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
