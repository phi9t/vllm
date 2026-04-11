# vLLM Devx Build + Runtime + Dynamo End-to-End Setup Design

## Purpose

Define a fail-fast, operator-friendly flow that:

1. Builds (optionally rebuilds) and launches the current `devx` stack.
2. Brings up source-backed `vllm-runtime`.
3. Verifies Dynamo frontend + Dynamo worker routing.
4. Executes a simple routed query for `Qwen/Qwen3.5-7B-Instruct`.
5. Produces hop-by-hop checkpoint output so failures are localized immediately.

This is a setup-and-verification design, not a topology expansion.

## Scope

In scope:

- Local Compose-based `devx` flow only.
- Existing pinned local Dynamo v1 contract (`file` discovery, one worker).
- Fail-fast verification for service readiness and routed inference success.
- Explicit `default` vs `rebuild` startup behavior.

Out of scope:

- Multi-worker or multi-model Dynamo topologies.
- Non-`file` discovery backends.
- Kubernetes/production hardening.
- Broader `devx` architecture changes unrelated to e2e setup verification.

## Chosen Approach

Primary approach:

- Keep script-driven orchestration as source of truth.
- Add a dedicated e2e verification path and optional ergonomics via `just`.

Rationale:

- Lowest behavior churn for current `devx` workflow.
- Reuses existing contracts and wrappers.
- Keeps diagnostics close to existing launch surface.

Rejected alternatives:

- New standalone orchestration layer (higher drift risk).
- Heavy refactor to make `just` the control plane (not required for milestone).

## Runtime Contract

### Model target

- Default and test model: `Qwen/Qwen3.5-7B-Instruct`.

### Token source and policy

- Hugging Face token source:  
  `~/.devx/special-circ-phi9t-vllm/secrets/huggingface_token`
- Policy: fail closed in preflight if missing or empty.
- `CREDS.yaml` is not part of token sourcing in this design.

### Startup behavior

- `default` mode:
  - starts stack without forced rebuild.
- `rebuild` mode:
  - explicitly rebuilds runtime-relevant images/services before launch, then starts stack.

## End-to-End Verification Sequence (Fail-Fast)

The verifier runs checkpoints in strict order and exits non-zero on first hard failure:

1. Preflight token check passes.
2. Docker + Compose availability check passes.
3. `devshell` is healthy (`sshd` running).
4. `vllm-runtime` is healthy (`/health` reachable).
5. `dynamo-vllm-worker` health endpoint reports ready.
6. Dynamo frontend `/health` contains
   `dyn://${DYNAMO_NAMESPACE}.backend.generate`.
7. Dynamo frontend `/v1/models` includes `Qwen/Qwen3.5-7B-Instruct`.
8. Dynamo frontend `/v1/chat/completions` succeeds with that model.

Success is defined as all checkpoints passing in one run.

## Diagnostics and Operator UX

For each checkpoint:

- emit one short PASS/FAIL line with checkpoint id and target.
- on FAIL, include immediate next command(s) to inspect relevant logs/status.

Expected operator commands:

- launch default
- launch rebuild
- verify e2e
- inspect per-service logs/status

`just` targets may be added as thin wrappers around script truth, not as a second orchestration source.

## Configuration Alignment

Defaults should be consistent across runtime/worker/probe wrappers:

- model id: `Qwen/Qwen3.5-7B-Instruct`
- namespace and endpoint assumptions: unchanged from pinned local Dynamo v1 contract
- no topology widening beyond current frontend + single worker path

## Testing Strategy

Add/update shell-level tests to cover:

- fail-closed token preflight behavior.
- checkpoint sequence and first-failure exit behavior.
- mode parsing (`default` and `rebuild`).
- model/default propagation consistency where touched.

Existing `devx` tests remain required to pass.

## Acceptance Criteria

1. A documented default launch + verify flow passes end-to-end with token present.
2. A documented rebuild launch + verify flow passes end-to-end.
3. Missing/empty token fails immediately before costly startup work.
4. On induced failure, verifier stops at first broken hop and prints targeted diagnostics.
5. Routed query for `Qwen/Qwen3.5-7B-Instruct` succeeds through Dynamo.
