# devx

## Onboarding

Start with the user-facing guide for first-time setup and a quick happy-path inference flow:

- [devx/onboarding.md](devx/onboarding.md)

## Regression checks for SSH and container validation

For full command-by-command validation evidence and outputs, use:

- [devx/validation-runbook.md](devx/validation-runbook.md)

For SSH/auth behavior and compose output hardening:

```bash
bash devx/tests/test_compose_runner.sh
```

For main dev container verification checkpoints (`verify-e2e`):

```bash
bash devx/tests/test_verify_e2e.sh
```

There is also a shortcut bundle:

```bash
just test-devx
```

## Reference Wrappers and Core Commands

```bash
just doctor
just devx-up qwen3-0.6b
just devx-switch qwen3-4b
just devx-query "Reply with exactly: ok"
just devx-exp-run devx/experiments/sample.yaml
just devx-exp-report <run-id>
```

## Backward-Compatible Command Examples

```bash
./devx/bin/devx doctor --preset qwen3-0.6b
./devx/bin/devx up --preset qwen3-0.6b
./devx/bin/devx query --prompt "Reply with exactly: ok"
```

```bash
./devx/bin/devx switch --preset qwen3-4b
./devx/bin/devx query --prompt "Give a one-line summary of vLLM."
```

```bash
./devx/bin/devx experiment run -f devx/experiments/sample.yaml
./devx/bin/devx experiment report <run-id>
```
