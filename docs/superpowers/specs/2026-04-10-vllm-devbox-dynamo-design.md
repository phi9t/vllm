# vLLM Devbox + Runtime + Minimal Dynamo Design

## Purpose

Define a vLLM-focused Docker Compose development environment that provides:

- a stable SSH-first CUDA devbox for interactive work
- a deployment-like vLLM runtime container driven from the same source tree
- a minimal Dynamo integration path that routes to the local vLLM backend

The environment is optimized for one primary developer using Codex and Claude
Code inside the devbox, with persistence and ergonomics closer to a remote VM
than a disposable container.

## Goals

- Make `main` the stable daily-driving development container.
- SSH into `main` and work there with `zsh`, `tmux`, Codex, and Claude Code.
- Develop against the mounted `vllm` source tree from inside `main`.
- Run `vllm-runtime` from the same source tree, mounted read-only.
- Keep `main` and `vllm-runtime` as close as practical on CUDA, Python,
  PyTorch, and core runtime dependencies.
- Keep `vllm-runtime` close to the official upstream vLLM runtime image.
- Run the smallest useful Dynamo setup needed to route to one local vLLM
  backend.
- Keep state outside the repo so Git operations and worktrees do not endanger
  persistence.
- Share large reusable caches, especially Hugging Face cache, across host and
  containers.
- Make the first version strong on GPU, IPC, and SHM so both interactive
  development and inference workloads can run without artificially constrained
  container defaults.

## Non-goals

- Multi-tenant isolation.
- Production hardening.
- Kubernetes deployment.
- Supporting multiple model runtimes in the first version.
- Generalizing the stack beyond `vllm` before the first version works.

## Service Topology

### `main`

Primary SSH-accessible CUDA devbox. Responsibilities:

- host entrypoint for the stack
- persistent workstation-like environment
- editing, testing, debugging, and agent execution
- tmux-centric interactive workflow
- optional internal port forwarding base for runtime and Dynamo inspection

`main` is the only service with a host-exposed port in the first version.

### `vllm-runtime`

Deployment-like runtime-under-test. Responsibilities:

- run vLLM from the same mounted repo source as `main`
- expose an internal-only direct vLLM endpoint on the Compose network
- pull model artifacts if missing
- stay close to the official upstream runtime image and runtime assumptions

### Minimal Dynamo stack

Only the components needed to route requests to the local vLLM backend.

Responsibilities:

- provide an internal-only Dynamo entrypoint
- validate that the local vLLM backend can serve behind Dynamo
- avoid expanding into a full general Dynamo development environment in version
  one

Any Dynamo UI remains internal-only and is accessed via SSH port forwarding
through `main` if needed.

### Dynamo v1 contract

The first version must stop calling Dynamo “minimal” in the abstract and define
the actual contract:

- source of truth is `ai-dynamo/dynamo`
- the stack contains only the Dynamo frontend plus the exact mandatory support
  services it needs to route to one local vLLM backend
- all Dynamo startup logic lives behind repo-owned wrapper scripts
- no Kubernetes assumption exists in version one even if upstream docs are
  Kubernetes-first

Success criteria for v1:

1. direct request to `vllm-runtime` succeeds on the Compose network
2. request through the Dynamo frontend also succeeds
3. the response proves Dynamo reached the local vLLM backend rather than some
   unrelated model service

If upstream Dynamo packaging makes this impossible without a larger topology,
the wrappers must fail loudly and document the missing dependency rather than
silently broadening scope.

## Network Model

- All services are on the same private Compose network.
- Only SSH for `main` is published to the host.
- `main` SSH binds to host loopback only.
- `vllm-runtime` and Dynamo endpoints are internal-only.
- All containers retain outbound internet access for package installs, Git
  operations, model downloads, and API access.

## GPU, IPC, and SHM Model

This environment is explicitly performance-oriented rather than default-safe.

### GPU policy

- `vllm-runtime` gets dedicated GPU access and is the primary serving workload
- `main` is CUDA-capable and may access the same GPU family for development,
  debugging, and local verification
- first version should default to making the same visible GPU set available to
  `main` and `vllm-runtime`, then narrow that later only if contention proves
  painful
- exact device selection must be controlled through env values, not hard-coded

### IPC policy

- inference-oriented containers should use strong IPC semantics equivalent to
  host IPC or another explicitly large-capacity configuration
- the design must not rely on Docker defaults for CUDA inference workloads

### SHM policy

- `main` and `vllm-runtime` require explicitly large shared memory settings
- version one should optimize for not falling over under heavy development or
  inference demand rather than for minimal host footprint

This section exists because weak container IPC/SHM defaults are a known source
of vLLM friction.

## SSH and Interactive Model

### SSH access

`main` runs `sshd` as its primary service process.

Required SSH behavior:

- key-only authentication
- password auth disabled
- root login disabled
- long-running interactive sessions supported
- keepalive settings tuned for tmux-centric workflows
- non-interactive SSH commands must still work correctly

### Jump-host model

The host machine remains the real SSH endpoint from the outside. The container
has its own stable host keys. The baseline access mode is direct SSH to the
loopback-published container port from the host itself, with `ProxyJump` as the
remote-access pattern when the host is itself being SSHed into.

Recommended pattern:

- host forwards `127.0.0.1:2222` to `main:22`
- local users may connect directly to that loopback port
- remote users may connect with `ProxyJump` through the host
- host and container use separate host keys

This avoids reusing the host's SSH host identity inside the container while
still keeping the host as the jump point.

### tmux behavior

Interactive logins to `main` should auto-attach to a well-known tmux session,
for example `main`.

Requirements:

- attach if the session exists
- create it if it does not
- do not auto-attach for non-interactive SSH commands
- provide an explicit shell fallback such as `TMUX_AUTO_ATTACH=0`

tmux is part of the core product of the devbox, not an optional convenience.

## User and Identity Model

Use one consistent non-root user everywhere:

- username: `kvothe`
- same host-mapped UID:GID in all cooperating containers
- same ownership expectations on shared bind mounts

Rationale:

- the distinct username makes it obvious that the user is in the container
- consistent UID:GID avoids shared-mount permission drift
- same-user semantics match the single-user same-trust development model

All cooperating containers that touch shared repo or cache state run as
`kvothe`.

The design inherits the current `devx` requirement that container UID:GID match
the host UID:GID. This is not optional in version one because the repo, home,
and caches are bind-mounted from the host.

## Source Layout and Mounting

### Repo source

First version mounts only the `vllm` repo as the primary working tree.

- `main` mounts the repo read/write
- `vllm-runtime` mounts the same repo read-only

This preserves a clean boundary:

- all edits happen in `main`
- runtime execution sees the same source snapshot
- runtime cannot mutate the working tree

Extra worktrees can be managed from inside `main`, but Compose is not
responsible for dynamically mounting multiple worktrees in version one.

## Persistence Model

### Stack root

All stack-specific mutable state lives outside the repo under:

```text
~/.devx/special-circ-phi9t-vllm
```

This keeps state safe from branch switches, repo replacement, worktree churn,
and accidental Git operations.

### Global shared cache

Hugging Face cache is shared globally at:

```text
~/.cache/huggingface
```

This allows reuse across host workflows and other containerized environments.

### Persist all of `/home/kvothe`

Persist the entire devbox home directory:

```text
~/.devx/special-circ-phi9t-vllm/home
```

Reasoning:

- this is a single-user personal workstation-style environment
- full-home persistence maximizes continuity for Codex, Claude Code, shell
  state, editor state, and auth
- the image should only seed missing files, never overwrite an existing home

### Home drift policy

Persisting the full home is a deliberate convenience tradeoff. To keep it from
turning into an unsupported pet machine, version one adopts this policy:

- the image owns only the skeleton in the image
- the mounted home is user-owned state after first boot
- bootstrap may add missing files and perform narrowly-scoped migrations, but
  must not silently rewrite user-modified files
- any intentional migration logic must be versioned and explicit
- the stack must document a supported “reset home to skeleton” path for users
  who want to start clean

This keeps onboarding low-friction while acknowledging that full-home
persistence weakens strict image parity over time.

### Other persisted state

Under `~/.devx/special-circ-phi9t-vllm`, persist:

- `ssh-hostkeys/`
- `cache/uv/`
- `cache/pip/`
- `cache/bazel/`
- `main/tmp/`
- `main/var-tmp/`
- `main/logs/`
- `vllm-runtime/tmp/`
- `vllm-runtime/var-tmp/`
- `vllm-runtime/logs/`
- `dynamo/tmp/`
- `dynamo/var-tmp/`
- `dynamo/logs/`
- `secrets/`

### Scratch isolation

Even though this is one trusted user environment, `/tmp`, `/var/tmp`, and
service logs should remain per-service. They are more collision-prone than home
or caches.

## Secrets Model

Do not bake secrets into images or tracked files.

For the first version, place the Hugging Face token under:

```text
~/.devx/special-circ-phi9t-vllm/secrets/huggingface_token
```

Inject it into relevant containers at startup as `HF_TOKEN`.

The token is available to:

- `main`
- `vllm-runtime`
- any minimal Dynamo component that directly needs model access

Rationale:

- `main` needs parity with runtime and ad hoc Hugging Face CLI use
- the environment is single-user and same-trust
- secret material stays outside the repo and outside the image

Managed-by-stack secrets in v1:

- `HF_TOKEN`

User-managed but expected-to-work state in the persistent home:

- SSH client keys and config
- Git config and credentials
- Codex/Claude auth state
- other personal CLI/tool auth material

The stack should avoid trying to centralize all auth in version one.

## Image Strategy

### `vllm-runtime`

Start from the official upstream vLLM OpenAI server image pattern. Layer only
the minimum wiring needed for:

- mounted source consumption
- internal network integration
- model configuration
- cache and secret mounting

### `vllm-runtime` code path contract

The runtime must not be ambiguous about what code it is executing.

Version one contract:

- base image remains close to the official vLLM runtime image
- mounted repo is read-only and available at a fixed path
- startup logic explicitly chooses one of the supported execution paths rather
  than mixing them implicitly

Supported execution paths for experimentation:

1. image-native path: run the vLLM already installed in the runtime image
2. source-overlay path: build/install vLLM from the mounted source into the
   runtime container's own Python environment at startup, then run that result

The startup wrapper must print which path is active. Version one should keep
both paths available so experimentation can determine which one is smoother,
but each individual run must choose exactly one path.

### `main`

Start from the same upstream vLLM runtime base and add:

- `openssh-server`
- `zsh`
- `tmux`
- core dev tools
- editor/dev utilities
- Codex and Claude Code support
- a home bootstrap entrypoint

This preserves maximal similarity between `main` and `vllm-runtime` while still
allowing `main` to be a better interactive workstation.

### Environment parity

`main` and `vllm-runtime` should align as much as practical on:

- CUDA family/version
- Python family/version
- PyTorch family/version
- core runtime libraries

They should begin with independent Python/package environments even if the base
image family is shared.

## Runtime Execution Modes

The design keeps two runtime-mounting modes available for `vllm-runtime`:

### Minimal mount mode

- read-only repo mount
- shared Hugging Face cache
- only the additional build/cache dirs actually needed

### Shared-home compatibility mode

- same as minimal mode
- plus access to the persistent `kvothe` home if runtime tooling requires it

Preferred baseline is minimal mode. Shared-home mode remains available as an
experiment if compatibility friction appears.

## Model Target

First version targets one small Qwen 3.5 8B-class model.

Requirements:

- same model for direct vLLM and Dynamo-routed testing
- if missing, pull automatically on first run
- reuse the global host Hugging Face cache if present

This keeps the first version focused while still exercising realistic model
download and serving behavior.

## Compose Operational Model

Default stack should bring up:

- `main`
- `vllm-runtime`
- minimal Dynamo services needed to route to the local vLLM backend

The stack is intentionally small enough that one `docker compose up` should be
the normal developer entrypoint.

### Readiness contract

Version one must define readiness explicitly:

- `main` is ready when SSH is accepting connections
- `vllm-runtime` is ready only when its health endpoint passes and the selected
  model is actually loaded enough to serve requests
- Dynamo services are ready only when they can successfully route a request to
  `vllm-runtime`

Cold start on first model download may be long. Dependent services should
retry/wait based on health checks instead of assuming immediate availability.

## Host-side Control Surface

Do not mount the host Docker socket into `main` in version one.

Rationale:

- keep `main` cleaner and less privileged
- keep container lifecycle outside the container initially
- reserve rootless in-container image build or validation for later if it
  proves useful

Provide a host-side `justfile` as the standard control surface.

Initial commands should include:

- `just up`
- `just down`
- `just restart`
- `just ps`
- `just logs service=...`
- `just ssh`
- `just rebuild-main`
- `just rebuild-runtime`

## Relationship to existing `devx/`

The current `devx/` stack is reference material, not ignored prior art.

Version one should treat `devx/` this way:

- reuse proven ideas such as host UID:GID mapping, SSH bootstrap, and persistent
  state handling
- do not blindly fork the entire `devx/` shape if it does not fit the new
  vLLM-first design
- make an explicit implementation-time decision whether the new stack extends
  `devx/` or supersedes it

The design intent is to avoid ending the repo with two permanently overlapping
container systems that solve the same problem differently.

## Main entrypoint behavior

The `main` entrypoint should:

1. verify or create user/group for `kvothe`
2. ensure the mounted home exists
3. seed only missing files from an image skeleton
4. ensure ownership and permissions
5. load or generate persistent SSH host keys under the stack root
6. write the intended `sshd_config`
7. launch `sshd` in foreground

No other resident helper daemons are required in version one.

## Risks and Guardrails

### Home persistence drift

Persisting the full home means image updates do not automatically propagate to
 existing dotfiles. This is acceptable. Bootstrap must remain non-destructive.

### Shared mutable state

Because all cooperating containers run as the same user and may share selected
bind mounts, this stack is explicitly a same-user same-trust system, not a
security boundary.

### Runtime/build friction

Independent Python environments may increase startup or iteration cost. That is
acceptable for version one; experimentation will determine whether further
sharing is worth the complexity.

### GPU contention

Giving both `main` and `vllm-runtime` strong GPU/IPC/SHM access may cause
contention. This is acceptable in version one because the goal is to optimize
for a powerful personal environment first, then narrow constraints based on real
usage.

### Dynamo packaging drift

Dynamo should be wrapped behind repo-owned startup logic rather than relying on
fragile Compose inline commands. That wrapper layer absorbs version drift.

## Initial file layout

The infrastructure repo or subdirectory should contain at least:

```text
infra/
  docker-compose.yml
  .env.example
  main/
    Dockerfile
    entrypoint.sh
    home-skel/
  vllm-runtime/
    Dockerfile
    start.sh
  dynamo/
    Dockerfile
    start-frontend.sh
    start-backend.sh
  scripts/
    prepare-host-state.sh
justfile
```

This layout is intentionally small. The first version should not overbuild the
platform.

## Recommended next step

Translate this spec into an implementation plan that covers:

- exact bind mounts
- Compose services and networks
- `main` SSH/tmux bootstrap details
- `vllm-runtime` startup contract
- minimal Dynamo dependency graph
- host state preparation
- `justfile` command definitions
