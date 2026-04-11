# vLLM DevX Infra Exploration and Experimentation Design

## Purpose

Define the next `devx` iteration for GPU Linux users that optimizes for:

1. Fast first success (`time-to-first-success`).
2. Fast configuration/model iteration for exploration.
3. A clean path toward future contributor-focused reliability work.

This design intentionally prioritizes shell-first workflows and minimal control-plane complexity.

## Scope

In scope:

- NVIDIA GPU Linux only.
- One-command golden path for stack validation and first query success.
- Preset-driven model/config switching for repeat experiments.
- Declarative experiment runner with persisted artifacts and summary reporting.
- Reuse of existing `devx` scripts as execution backbone.

Out of scope:

- CPU fallback mode.
- Non-Linux developer platforms.
- GUI/control-plane service.
- Broad compose topology redesign.

## Users and Success Criteria

Primary users for this cycle:

- New users who want first successful query with minimal setup friction.
- Experimenters who want to rapidly swap models and compare outcomes.

Future users (deferred):

- vLLM contributors requiring deeper debugging and repeatable stability validation.

Primary success metrics:

- P0: first successful routed query in under 10 minutes on a fresh supported machine.
- P1: preset switch and next successful query in under 60 seconds (warm-cache target).
- P2 (later phase): repeated E2E runs with flake tracking and stability hardening.

## Design Principles

- Fail fast with precise remediation.
- Keep one obvious path for common workflows.
- Preserve script composability for power users.
- Prefer additive wrappers over disruptive rewrites.
- Persist experiment evidence by default.

## Command Surface (Phase A then B)

Single entrypoint command:

- `devx doctor`
- `devx up --preset <name>`
- `devx query --prompt "<text>" [--preset <name>]`
- `devx switch --preset <name>`
- `devx experiment run -f <manifest.yaml>`
- `devx experiment report <run-id>`

Behavior contract:

- `doctor` validates host/runtime prerequisites and exits non-zero on blocking issues.
- `up` launches stack with a validated preset and blocks until route+query smoke passes.
- `query` sends a canonical request through Dynamo frontend and prints normalized output.
- `switch` updates active preset with minimal restarts and warmup probe.
- `experiment run` executes variant matrix and writes deterministic artifacts.
- `experiment report` renders concise comparisons from stored artifacts.

## Architecture

### Components

1. `devx/bin/devx`
- Thin dispatcher parsing subcommands and options.
- Calls reusable library functions; does not duplicate business logic.

2. `devx/lib/presets.sh`
- Registry for supported presets.
- Stores model ID and required environment knobs.
- Includes metadata such as expected warmup behavior and resource notes.

3. `devx/lib/preflight.sh`
- Shared prerequisite checks used by `doctor` and `up`.
- Verifies GPU visibility, Docker availability, HF token presence, model resolvability, and required ports.

4. Existing execution scripts (reused)
- `devx/launch_container.sh` for compose lifecycle.
- `devx/verify_e2e.sh` and `devx/dynamo/start_backend_probe.sh` for readiness and route correctness.

5. `devx/lib/experiments.sh` (Phase B)
- Manifest parser and run coordinator.
- Artifact writer and summary helpers.

### Data and State

- Active session state file (e.g., under `~/.devx/.../state/`):
  - active preset
  - compose project namespace
  - last successful verification timestamp
- Experiment artifacts under `devx/runs/<run-id>/`:
  - `manifest.resolved.json`
  - `results.jsonl`
  - `summary.md`
  - optional raw response snapshots

## Data Flow

### Golden path (`devx up --preset qwen3-0.6b`)

1. Resolve preset from `presets.sh`.
2. Run preflight checks.
3. Export validated env to launch context.
4. Launch/rebuild stack via existing script wrappers.
5. Run readiness chain:
- runtime health
- worker health
- frontend route health
- `/v1/models` contains expected model
- chat completion smoke success
6. Persist session state and print ready status.

### Query path

1. Resolve active preset from session state unless explicitly provided.
2. Issue `/v1/chat/completions` to Dynamo frontend.
3. Print stable output structure:
- model ID
- elapsed time
- response excerpt
- request correlation id (if available)

### Switch path

1. Validate target preset.
2. Detect whether full rebuild is required.
3. Apply minimal restart/reconfigure path.
4. Perform warmup query and model-id assertion.
5. Update session state on success.

### Experiment path (Phase B)

1. Parse experiment manifest into ordered runs.
2. For each run:
- switch or up with target preset
- optional warmup calls
- execute query set with timing capture
3. Persist line-delimited results and summary.
4. Return run id and artifact location.

## Error Handling

All failures return:

- category: `preflight`, `startup`, `routing`, `inference`, or `experiment`
- failed check name
- observed value and expected value
- one immediate next action

Fail-fast defaults:

- `up` halts on first hard failure.
- `switch` rolls no further once consistency checks fail.
- `experiment run` can support `--continue-on-failure` later, but default remains fail-fast.

## Testing Strategy

### Script-level coverage

Add shell tests in `devx/tests` for:

- preset lookup and validation behavior.
- preflight pass/fail classification.
- command dispatch and argument parsing.
- switch decision logic (minimal restart vs full rebuild).
- experiment manifest parse and artifact contract.

### Required E2E gates

1. Golden path gate:
- `doctor -> up(preset) -> query` succeeds.

2. Switching gate:
- `up(preset A) -> switch(preset B) -> query` succeeds.
- `/v1/models` asserts expected active model each step.

3. Experiment gate (Phase B):
- matrix run produces valid `results.jsonl` and deterministic `summary.md`.

## Rollout Plan

Phase A1:

- Implement `devx` dispatcher with `doctor`, `up`, `query`.
- Introduce preset registry and shared preflight.

Phase A2:

- Implement `switch` with warmup and session state update.
- Tighten error classification and operator hints.

Phase B1:

- Add experiment manifest runner and artifact persistence.

Phase B2:

- Add reporting command and quick comparison summaries.

Post-B stability track (future priority 3):

- repeat-run flake detection and reliability baselines.
- contributor-oriented diagnostics and deeper test loops.

## Risks and Mitigations

Risk: preset drift between runtime, worker, and probe scripts.
Mitigation: single preset source of truth and strict model-id assertions in readiness checks.

Risk: startup latency exceeds iteration target.
Mitigation: explicit warmup behavior, minimal restart path, and measured timing output.

Risk: hidden infrastructure failures produce opaque errors.
Mitigation: categorized failure output with expected/observed values and immediate next action.

## Acceptance Criteria

1. New user can run `doctor -> up --preset <default> -> query` and get a routed response in under 10 minutes on supported hardware.
2. Experimenter can run `switch --preset <other>` and get next successful query within 60 seconds in warm-cache conditions.
3. Experiment manifest run stores artifacts and produces a readable summary without manual log scraping.
4. Failures are categorized and actionable without requiring source-code inspection.
