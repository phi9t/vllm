# vLLM Agent Guidelines

> Instructions for AI agents working in the vLLM repository.

## Build Commands

```bash
# Install from source (editable)
pip install -e .

# Install dependencies
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

# Build extensions only
python setup.py build_ext --inplace
```

## Lint/Format Commands

```bash
# Install pre-commit hooks (one-time setup)
pip install -r requirements-lint.txt
pre-commit install --hook-type pre-commit --hook-type commit-msg

# Run all linters manually
pre-commit run --all-files

# Run specific linter
pre-commit run yapf --all-files
pre-commit run ruff --all-files
pre-commit run isort --all-files

# Run mypy type checker
./tools/mypy.sh
```

## Test Commands

```bash
# Run all tests
pytest tests/

# Run single test
pytest tests/path/to/test_file.py::test_function_name -v

# Run specific test directory
pytest tests/core/ -v

# Run with specific markers
pytest tests/ -m "not distributed" -v

# Run with async support
pytest tests/async_engine/ --asyncio-mode=auto -v
```

## Code Style Guidelines

### Formatting
- **Line length**: 80 characters max (ruff enforces)
- **Formatter**: yapf (Google Python style)
- **Import sorter**: isort with parentheses
- **Linter**: ruff (replaces flake8, pycodestyle, pyupgrade)

### Import Order
```python
# 1. Standard library
import os
from typing import List

# 2. Third-party
import torch
import numpy as np
from transformers import AutoModel

# 3. vllm (absolute imports only)
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `LlamaForCausalLM`)
- Functions/variables: `snake_case` (e.g., `get_model_config`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_JOBS`)
- Private: `_leading_underscore`
- Type variables: `TypeVar` with `_T` suffix

### Types & Type Hints
- Use `typing` imports: `List`, `Dict`, `Optional`, `Union`
- Function signatures should be typed
- Use `dataclass` for configuration classes
- mypy checks enabled for: `vllm/*.py`, `tests/`, `vllm/attention/`, `vllm/engine/`, etc.

### Logging Pattern
```python
from vllm.logger import init_logger
logger = init_logger(__name__)

# Use custom methods
logger.info_once("Message")  # Logs only once
logger.warning_once("Warning")  # Warns only once
```

### Error Handling
- Use specific exceptions over bare `except:`
- Raise with context: `raise RuntimeError("msg") from e`
- Use `assert` for internal invariants
- Check `envs.py` for environment-based configuration

### Documentation
- Docstrings: Google style (triple quotes)
- Type hints preferred over docstring types
- Comments explain "why", code explains "what"

### Pre-commit Issues
- If a pre-commit hook fails, fix the underlying issue rather than bypassing
- Run `pre-commit run --all-files` to check all files before committing

## Key Files

- `pyproject.toml`: Build config, tool settings (ruff, mypy, pytest)
- `setup.py`: Custom CMake build for C++/CUDA extensions
- `.pre-commit-config.yaml`: All linting/formatting tools
- `requirements-lint.txt`: Pre-commit dependency
- `requirements-test.txt`: Test dependencies
- `tools/mypy.sh`: Type checking script
- `vllm/envs.py`: Environment variables
- `vllm/logger.py`: Logging utilities

## Project Structure

```
vllm/               # Main package
├── v1/            # New v1 engine (alpha)
├── attention/     # Attention backends
├── core/          # Scheduler
├── engine/        # LLM engine
├── model_executor/ # Model execution
├── worker/        # Worker processes
tests/             # Test suite
├── conftest.py    # pytest fixtures
├── models/        # Model tests
├── kernels/       # Kernel tests
```

## Testing Markers

- `@pytest.mark.core_model`: Enable in PR tests
- `@pytest.mark.distributed`: Distributed GPU tests only
- `@pytest.mark.skip_v1`: Skip with v1 engine
- `@pytest.mark.optional`: Optional tests (use `--optional` to run)

## Architecture Documentation

For a comprehensive understanding of vLLM's building blocks, design patterns, and data flow, see:
- [vllm_architecture.md](./docs/design/vllm_architecture.md) - Deep dive into the v1 engine architecture

This document covers:
- Top 10 most important building blocks (ranked by criticality)
- Request lifecycle data flow
- Core primitives and data structures
- File-to-component mapping
- Common tasks for AI agents
- Design patterns used throughout the codebase

Refer to this document when:
- Working on a new component and need to understand the architecture
- Debugging issues related to scheduling, attention, or KV cache
- Adding new features that touch multiple layers
- Understanding how requests flow through the system
