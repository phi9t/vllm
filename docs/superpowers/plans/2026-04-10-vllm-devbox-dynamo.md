# vLLM Devbox + Dynamo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vLLM-first Docker Compose developer environment with a robust SSH/tmux devbox, a source-backed `vllm-runtime`, and a minimal Dynamo integration path.

**Architecture:** Extend the existing `devx/` stack instead of creating a parallel infrastructure system. Keep `main` and `vllm-runtime` close to the upstream vLLM runtime image family, make `main` the interactive SSH entrypoint, and run `vllm-runtime` from the mounted repo in source-overlay mode by default. Add Dynamo only after first pinning the exact v1 topology needed to route to one local vLLM backend.

**Tech Stack:** Docker Compose, NVIDIA container runtime, OpenSSH, tmux, upstream vLLM runtime image family, repo-local wrapper scripts, `just`

---

## File map

### Existing files to modify

- `devx/compose.yaml`
  Current compose stack. Extend this instead of creating a new parallel stack.
- `devx/dev_container.dockerfile`
  Base for the `main` devbox image. Rework from current devshell assumptions to the new `main` contract.
- `devx/run_hermetic_sshd.sh`
  Reuse and simplify for robust SSH startup, home bootstrap, and host UID/GID enforcement.
- `devx/sshd_config`
  Tighten for the interactive long-lived SSH contract.
- `devx/lib/host_mounts.sh`
  Update for the new `~/.devx/special-circ-phi9t-vllm` state root and shared cache paths.
- `devx/lib/naming.sh`
  Adjust stack/service naming if needed for the new `main` / `vllm-runtime` / Dynamo terminology.
- `devx/launch_container.sh`
  Either simplify into the new stack launcher or reduce to a compatibility shim that delegates to `docker compose`.

### New files to create

- `devx/vllm_runtime.dockerfile`
  Runtime image extension staying close to upstream vLLM image behavior.
- `devx/start_vllm_runtime.sh`
  Explicit source-overlay runtime wrapper with health/readiness handling.
- `devx/start_main.sh`
  Thin entrypoint wrapper for `main` if `run_hermetic_sshd.sh` should stay narrower.
- `devx/prepare_host_state.sh`
  Creates `~/.devx/special-circ-phi9t-vllm` and cache directories with the right modes.
- `devx/hf_token.env.sh`
  Reads `~/.devx/special-circ-phi9t-vllm/secrets/huggingface_token` and exports `HF_TOKEN` safely for compose.
- `devx/dynamo/README.md`
  Short maintainer note on the exact chosen Dynamo v1 topology and wrapper expectations.
- `devx/dynamo/start_frontend.sh`
  Repo-owned Dynamo startup wrapper.
- `devx/dynamo/start_backend_probe.sh`
  Lightweight integration probe or backend registration wrapper, depending on actual Dynamo v1 needs.
- `justfile`
  Standard host-side control surface.

### New docs/tests to create

- `docs/deployment/integrations/dynamo.md`
  Update with the non-Kubernetes local dev integration note once topology is pinned.
- `devx/tests/test_prepare_host_state.sh`
  Smoke-style shell test for directory creation and permissions.
- `devx/tests/test_start_vllm_runtime.sh`
  Smoke-style shell test for runtime mode selection and env validation.
- `devx/tests/test_run_hermetic_sshd.sh`
  Shell test for SSH bootstrap invariants that matter to the new stack.

## Planning assumptions

- The implementation extends `devx/` rather than creating a new `infra/` tree.
- The blessed v1 `vllm-runtime` baseline is:
  - same mounted repo as `main`
  - repo mounted read-only
  - minimal-mount mode by default
  - source-overlay code path by default
- The image-native runtime path may remain as an explicit fallback flag, but it is not the default path in v1.
- The exact Dynamo topology is still implementation-discovery work, but the outcome must be encoded concretely before broader compose wiring lands.

## Task 1: Pin the Dynamo v1 topology and implementation boundary

**Files:**
- Modify: `docs/deployment/integrations/dynamo.md`
- Create: `devx/dynamo/README.md`
- Create: `devx/dynamo/start_frontend.sh`
- Create: `devx/dynamo/start_backend_probe.sh`

- [ ] **Step 1: Inspect upstream Dynamo packaging and local integration requirements**

Run:

```bash
git grep -n "Dynamo" docs/deployment/integrations devx || true
git grep -n "ai-dynamo" . || true
```

Also inspect the upstream Dynamo repo manually to answer:

- which process is the frontend
- what support services are mandatory for one local backend
- what backend protocol/registration path is expected
- what endpoint or probe can prove routed success

Expected: a short written answer in working notes that names the exact v1 services.

- [ ] **Step 2: Record the chosen v1 topology**

Write `devx/dynamo/README.md` with:

- exact service list
- process role for each service
- required environment variables
- health/readiness contract
- exact success condition for “Dynamo is routing to local vLLM”

Expected content includes concrete service names, not “mandatory support services.”

- [ ] **Step 3: Update the existing Dynamo integration doc**

Add a short section to `docs/deployment/integrations/dynamo.md` covering:

- local Compose-based developer integration
- the chosen v1 topology
- what is intentionally out of scope

- [ ] **Step 4: Create wrapper scripts for the pinned topology**

Implement `devx/dynamo/start_frontend.sh` and `devx/dynamo/start_backend_probe.sh` so they:

- validate required env vars
- print the selected topology/mode
- fail loudly on missing binaries or incompatible upstream packaging

- [ ] **Step 5: Commit**

```bash
git add docs/deployment/integrations/dynamo.md devx/dynamo/README.md devx/dynamo/start_frontend.sh devx/dynamo/start_backend_probe.sh
git commit -m "devx: pin local dynamo v1 topology"
```

## Task 2: Prepare host-state layout and secret handling

**Files:**
- Create: `devx/prepare_host_state.sh`
- Create: `devx/hf_token.env.sh`
- Modify: `devx/lib/host_mounts.sh`
- Test: `devx/tests/test_prepare_host_state.sh`

- [ ] **Step 1: Write the failing shell test for host-state creation**

Create `devx/tests/test_prepare_host_state.sh` with checks for:

- `~/.devx/special-circ-phi9t-vllm/home`
- `~/.devx/special-circ-phi9t-vllm/ssh-hostkeys`
- `~/.devx/special-circ-phi9t-vllm/cache/{uv,pip,bazel}`
- per-service `tmp`, `var-tmp`, and `logs`
- `~/.cache/huggingface`
- correct `1777` mode on tmp dirs

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
bash devx/tests/test_prepare_host_state.sh
```

Expected: FAIL because the script does not exist yet.

- [ ] **Step 3: Implement host-state preparation**

Create `devx/prepare_host_state.sh` to:

- create the full `~/.devx/special-circ-phi9t-vllm` tree
- create `~/.cache/huggingface`
- set per-service temp dirs to `1777`
- keep ownership aligned to the invoking host user

- [ ] **Step 4: Add safe HF token loading**

Create `devx/hf_token.env.sh` to:

- read `~/.devx/special-circ-phi9t-vllm/secrets/huggingface_token`
- export `HF_TOKEN` only if the file exists and is readable
- avoid echoing the token

- [ ] **Step 5: Wire mount-path helpers**

Modify `devx/lib/host_mounts.sh` so path resolution prefers:

- `~/.devx/special-circ-phi9t-vllm/...`
- `~/.cache/huggingface`

and no longer assumes the older ad hoc state layout.

- [ ] **Step 6: Run the test to verify it passes**

Run:

```bash
bash devx/prepare_host_state.sh
bash devx/tests/test_prepare_host_state.sh
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add devx/prepare_host_state.sh devx/hf_token.env.sh devx/lib/host_mounts.sh devx/tests/test_prepare_host_state.sh
git commit -m "devx: add host state bootstrap"
```

## Task 3: Harden `main` as the SSH-first devbox

**Files:**
- Modify: `devx/dev_container.dockerfile`
- Modify: `devx/run_hermetic_sshd.sh`
- Modify: `devx/sshd_config`
- Create: `devx/start_main.sh`
- Test: `devx/tests/test_run_hermetic_sshd.sh`

- [ ] **Step 1: Write the failing SSH bootstrap test**

Create `devx/tests/test_run_hermetic_sshd.sh` to assert:

- login user must be `kvothe`
- host UID/GID must be provided and must match `kvothe`
- host keys must live under the stack root
- seed logic is non-destructive
- SSH startup config rejects password auth and root login

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
bash devx/tests/test_run_hermetic_sshd.sh
```

Expected: FAIL until the new entrypoint contract is in place.

- [ ] **Step 3: Update the devbox Dockerfile**

Modify `devx/dev_container.dockerfile` to:

- keep `USERNAME=kvothe`
- keep host-mapped UID/GID build args
- install the approved interactive tool set only
- prepare image skeleton content for `/home/kvothe`
- keep the image close to the upstream vLLM runtime family

- [ ] **Step 4: Tighten SSH bootstrap**

Modify `devx/run_hermetic_sshd.sh` and/or add `devx/start_main.sh` so startup:

- enforces host UID/GID matching
- seeds only missing home files
- preserves persistent host keys
- launches `sshd` as the foreground process

- [ ] **Step 5: Harden `sshd_config` for interactive use**

Update `devx/sshd_config` with:

- key-only authentication
- password auth off
- root login off
- keepalives suitable for long tmux sessions
- support for non-interactive SSH commands

- [ ] **Step 6: Run the SSH bootstrap test**

Run:

```bash
bash devx/tests/test_run_hermetic_sshd.sh
```

Expected: PASS.

- [ ] **Step 7: Build the `main` image and verify SSH starts**

Run:

```bash
docker compose -f devx/compose.yaml build devshell
docker compose -f devx/compose.yaml up -d devshell
docker compose -f devx/compose.yaml logs --tail=100 devshell
```

Expected: `sshd` is healthy and listening.

- [ ] **Step 8: Commit**

```bash
git add devx/dev_container.dockerfile devx/run_hermetic_sshd.sh devx/sshd_config devx/start_main.sh devx/tests/test_run_hermetic_sshd.sh
git commit -m "devx: harden main ssh devbox"
```

## Task 4: Implement tmux-first login behavior with shell fallback

**Files:**
- Modify: `devx/dev_container.dockerfile`
- Modify: `devx/run_hermetic_sshd.sh`
- Modify: any seeded shell files under `devx/legacy` or new skeleton content

- [ ] **Step 1: Add the tmux attach helper to the home skeleton**

Ensure the skeleton contains a small helper that:

- checks for interactive login
- skips auto-attach when `TMUX_AUTO_ATTACH=0`
- attaches or creates session `main`

- [ ] **Step 2: Wire shell startup to use the helper**

Update skeleton shell init so interactive SSH logins run the helper while
non-interactive commands do not.

- [ ] **Step 3: Verify interactive and non-interactive behavior**

Run:

```bash
ssh -p 2222 kvothe@127.0.0.1
ssh -p 2222 kvothe@127.0.0.1 "echo ok"
ssh -p 2222 kvothe@127.0.0.1 "TMUX_AUTO_ATTACH=0 exec zsh -l"
```

Expected:

- first command lands in tmux
- second prints `ok` without tmux interference
- third lands in a plain shell

- [ ] **Step 4: Commit**

```bash
git add devx/dev_container.dockerfile devx/run_hermetic_sshd.sh devx/legacy
git commit -m "devx: add tmux-first ssh login flow"
```

## Task 5: Build the `vllm-runtime` image and bless one baseline

**Files:**
- Create: `devx/vllm_runtime.dockerfile`
- Create: `devx/start_vllm_runtime.sh`
- Test: `devx/tests/test_start_vllm_runtime.sh`
- Modify: `devx/compose.yaml`

- [ ] **Step 1: Write the failing runtime wrapper test**

Create `devx/tests/test_start_vllm_runtime.sh` to check:

- mode selection is explicit
- default mode is source-overlay
- repo mount must be present and readable
- write access to the repo path is rejected
- missing `HF_TOKEN` produces a useful warning or failure depending on model access

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
bash devx/tests/test_start_vllm_runtime.sh
```

Expected: FAIL until the wrapper exists.

- [ ] **Step 3: Create the runtime Dockerfile**

Create `devx/vllm_runtime.dockerfile` that:

- starts from the upstream vLLM runtime image family
- adds only the minimum tooling needed for wrapper execution and health checks
- runs as `kvothe` with host-mapped UID/GID expectations

- [ ] **Step 4: Implement the runtime wrapper**

Create `devx/start_vllm_runtime.sh` to:

- print the selected code path
- default to source-overlay mode
- install/build from the mounted read-only source into the container’s own
  runtime environment
- launch the direct vLLM server on the internal Compose network
- expose a health/readiness endpoint contract

- [ ] **Step 5: Add the runtime service to compose**

Modify `devx/compose.yaml` so the service:

- mounts the repo read-only
- mounts `~/.cache/huggingface`
- mounts per-service tmp/log dirs
- gets strong GPU/IPC/SHM settings
- stays internal-only

- [ ] **Step 6: Run the wrapper test**

Run:

```bash
bash devx/tests/test_start_vllm_runtime.sh
```

Expected: PASS.

- [ ] **Step 7: Build and start `vllm-runtime`**

Run:

```bash
docker compose -f devx/compose.yaml build vllm-runtime
docker compose -f devx/compose.yaml up -d vllm-runtime
docker compose -f devx/compose.yaml logs --tail=200 vllm-runtime
```

Expected: model load begins or completes, and the health endpoint becomes ready.

- [ ] **Step 8: Verify direct vLLM reachability from `main`**

From inside `main`, run:

```bash
curl -sf http://vllm-runtime:<PORT>/health
```

Expected: success.

- [ ] **Step 9: Commit**

```bash
git add devx/vllm_runtime.dockerfile devx/start_vllm_runtime.sh devx/tests/test_start_vllm_runtime.sh devx/compose.yaml
git commit -m "devx: add source-backed vllm runtime"
```

## Task 6: Add explicit GPU, IPC, and SHM configuration

**Files:**
- Modify: `devx/compose.yaml`
- Modify: `devx/lib/host_mounts.sh` if env helpers are needed

- [ ] **Step 1: Add env-controlled GPU visibility**

Introduce env variables for:

- `MAIN_VISIBLE_DEVICES`
- `VLLM_RUNTIME_VISIBLE_DEVICES`

Default both to the same visible set for v1.

- [ ] **Step 2: Add strong IPC/shm defaults**

Ensure `main` and `vllm-runtime` use:

- strong IPC configuration
- explicitly large SHM
- any ulimits that inference workloads need

- [ ] **Step 3: Validate the compose config**

Run:

```bash
docker compose -f devx/compose.yaml config
```

Expected: valid output with the intended resources present.

- [ ] **Step 4: Commit**

```bash
git add devx/compose.yaml devx/lib/host_mounts.sh
git commit -m "devx: pin gpu ipc and shm policy"
```

## Task 7: Wire the minimal Dynamo integration path

**Files:**
- Modify: `devx/compose.yaml`
- Create or modify: `devx/dynamo/start_frontend.sh`
- Create or modify: `devx/dynamo/start_backend_probe.sh`

- [ ] **Step 1: Add the Dynamo services chosen in Task 1**

Modify `devx/compose.yaml` to include only the pinned v1 Dynamo topology.

- [ ] **Step 2: Connect Dynamo to `vllm-runtime`**

Wire service discovery, env vars, and startup order so Dynamo points to the
direct local vLLM backend.

- [ ] **Step 3: Add routed-health verification**

Ensure at least one health or probe command validates that Dynamo can actually
route a request to the backend rather than only checking process liveness.

- [ ] **Step 4: Bring up the full stack**

Run:

```bash
docker compose -f devx/compose.yaml up -d
docker compose -f devx/compose.yaml ps
docker compose -f devx/compose.yaml logs --tail=200
```

Expected: all intended services are up or clearly failing with actionable logs.

- [ ] **Step 5: Verify direct and routed paths**

From inside `main`, run one direct request and one routed request.

Expected:

- direct request succeeds
- routed request succeeds
- routed response can be attributed to the local `vllm-runtime`

- [ ] **Step 6: Commit**

```bash
git add devx/compose.yaml devx/dynamo
git commit -m "devx: add minimal dynamo routing path"
```

## Task 8: Add host-side `just` commands and operator flow

**Files:**
- Create: `justfile`
- Modify: `devx/launch_container.sh` if retained

- [ ] **Step 1: Create the `justfile`**

Add recipes for:

- `up`
- `down`
- `restart`
- `ps`
- `logs service=...`
- `ssh`
- `rebuild-main`
- `rebuild-runtime`

- [ ] **Step 2: Ensure `just ssh` matches the SSH contract**

It should target the loopback-published `main` port and document the tmux-first
behavior.

- [ ] **Step 3: Validate operator flow**

Run:

```bash
just up
just ps
just logs service=devshell
just ssh
```

Expected: the stack can be operated entirely from the host with the `justfile`.

- [ ] **Step 4: Commit**

```bash
git add justfile devx/launch_container.sh
git commit -m "devx: add host operator commands"
```

## Task 9: Final docs and verification

**Files:**
- Modify: `docs/deployment/integrations/dynamo.md`
- Modify: `docs/superpowers/specs/2026-04-10-vllm-devbox-dynamo-design.md` only if implementation forced a justified correction

- [ ] **Step 1: Verify spec coverage against implementation**

Check that the implementation covers:

- SSH/tmux contract
- source-backed runtime contract
- GPU/IPC/SHM policy
- host-state layout
- direct vLLM path
- Dynamo-routed path
- `just` control surface

- [ ] **Step 2: Run end-to-end verification**

Run:

```bash
bash devx/prepare_host_state.sh
docker compose -f devx/compose.yaml up -d --build
docker compose -f devx/compose.yaml ps
```

Then verify:

- SSH into `main`
- tmux auto-attach
- direct request to `vllm-runtime`
- routed request through Dynamo

- [ ] **Step 3: Document any design drift**

If implementation forced a change to the approved spec, update the spec
immediately and keep the drift explicit.

- [ ] **Step 4: Commit**

```bash
git add docs/deployment/integrations/dynamo.md docs/superpowers/specs/2026-04-10-vllm-devbox-dynamo-design.md devx justfile
git commit -m "docs: finalize devbox and dynamo workflow"
```

## Self-review

### Spec coverage

- `main` SSH/tmux ergonomics: Tasks 3, 4, 8, 9
- host state and secrets: Task 2
- source-backed `vllm-runtime`: Tasks 5, 6
- GPU/IPC/SHM: Task 6
- minimal Dynamo integration: Tasks 1, 7
- extend-not-fork `devx/`: file map plus Tasks 1 through 8

### Placeholder scan

- No `TBD`/`TODO`
- Commands are concrete
- Each task names exact files

### Type/contract consistency

- Blessed runtime baseline in this plan is source-overlay + minimal mount mode
- `devx/` is treated as the implementation base, not a parallel stack
- `main` remains the only host-exposed service

