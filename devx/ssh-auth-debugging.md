# SSH Authentication Debugging (devx)

## Problem

Several users could not SSH into the devx container after `./devx/bin/devx up` with errors like:

```text
Received disconnect from 127.0.0.1 port 2222:2: Too many authentication failures
```

or:

```text
kvothe@127.0.0.1: Permission denied (publickey).
```

## Why it happened

1. The container uses a dedicated key path for `AuthorizedKeysFile`, but the startup hint initially suggested a bare command:
   - `ssh -p 2222 kvothe@127.0.0.1`
2. On machines with many loaded identities, SSH attempted many keys before the devx key.
3. The server rejected after too many attempts or never reached an accepted identity.

## Reproduction used during debugging

Start the stack and run a non-pinned SSH command from host:

```bash
./devx/bin/devx up --preset qwen3-0.6b
ssh -p 2222 kvothe@127.0.0.1 true
```

Observed:

```text
Permission denied (publickey).
```

## Fix implemented

### 1) Print a robust SSH login command

`devx/lib/compose_runner.sh` now prints:

- Default: `ssh -o IdentitiesOnly=yes -o IdentityAgent=none -p ${SSH_PORT} ${CONTAINER_USER}@127.0.0.1`
- If `~/.ssh/devx_access` is readable:
  - `ssh -o IdentitiesOnly=yes -i ${HOME}/.ssh/devx_access -p ${SSH_PORT} ${CONTAINER_USER}@127.0.0.1`

This ensures the client:

- uses only intended identities,
- avoids agent key spraying,
- and prefers the dedicated devx key when present.

### 2) Raise auth-attempt headroom in sshd

`devx/sshd_config` now contains:

```text
MaxAuthTries 32
```

This reduces auth churn under key-heavy conditions.

### 3) Update test coverage

`devx/tests/test_compose_runner.sh` was updated to assert the new SSH hint format.

## Verification

### A) Targeted shell test

```bash
bash devx/tests/test_compose_runner.sh
```

Expected:

```text
PASS
```

### B) Runtime SSH checks

```bash
ssh -p 2222 kvothe@127.0.0.1 true
ssh -o IdentitiesOnly=yes -o IdentityAgent=none -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1 true
ssh -o IdentitiesOnly=yes -o IdentityAgent=none -p 2222 kvothe@127.0.0.1 true
```

Observed:

```text
kvothe@127.0.0.1: Permission denied (publickey).
[success exit code 0]
kvothe@127.0.0.1: Permission denied (publickey).
```

## Debug playbook for future failures

1. Run with verbose output:
   `ssh -vvv -o IdentitiesOnly=yes -o IdentityAgent=none -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1 true`
2. Check local key file permissions:
   `ls -l "$HOME/.ssh/devx_access"`
3. Confirm the startup string from `devx up` includes the pinned format.
4. Confirm runtime + dynamo health endpoints before retrying SSH.

## Regression tests that should pass

Run these when changing SSH auth handling or container bootstrap logic:

```bash
bash devx/tests/test_compose_runner.sh
```

This validates:
- `up-rebuild` output includes deterministic SSH command text.
- plain-host-key command path when `~/.ssh/devx_access` is absent.
- pinned-key path when `~/.ssh/devx_access` is present.

```bash
bash devx/tests/test_verify_e2e.sh
```

This validates:
- token preflight behavior,
- each verification checkpoint path (pass/fail behavior),
- and model-routing verification using mocked backends.

The two tests can be run together via:

```bash
just test-devx
```

## Takeaway

For this stack, **do not use plain `ssh -p 2222 kvothe@127.0.0.1`** in hosts with multiple identities. Use either the printed command or explicit `-i ~/.ssh/devx_access`.
