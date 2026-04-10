# Dynamo local v1 topology

This directory pins the local developer Dynamo contract for `devx/`.

The goal is narrow: one Dynamo frontend routing to one local `vllm-runtime`
backend. If upstream Dynamo packaging stops supporting this exact shape, the
wrappers in this directory must fail loudly instead of silently growing the
topology.

Pinned against the Dynamo v1.0.1 packaging surface, using upstream source
checkout `fd8410da6afcc2ad2b40819867f6780cfdcf0628` for implementation
confirmation.

## Working notes

The upstream `ai-dynamo/dynamo` sources and docs inspected for this contract
were:

- `README.md`
- `docs/getting-started/quickstart.md`
- `docs/getting-started/local-installation.md`
- `docs/design-docs/discovery-plane.md`
- `docs/components/frontend/README.md`
- `docs/observability/health-checks.md`
- `docs/reference/release-artifacts.md`
- `components/src/dynamo/vllm/args.py`
- `components/src/dynamo/vllm/main.py`
- `lib/llm/src/http/service/health.rs`

Upstream answers for the local v1 contract:

- Frontend process: `python -m dynamo.frontend`
- Mandatory support services for one local vLLM backend: none external when all
  participants use `--discovery-backend file` and the vLLM worker runs with
  `--kv-events-config '{"enable_kv_cache_events": false}'`
- Expected backend registration path: `dyn://<namespace>.backend.generate`
  from the Dynamo vLLM worker (`python -m dynamo.vllm`); the same worker may
  also register `clear_kv_blocks` and `load_metrics`
- Routed-success probe: the frontend `/health` endpoint must list
  `dyn://<namespace>.backend.generate`, `/v1/models` must expose the expected
  model id, and `/v1/chat/completions` against that model must succeed

## Exact service list

### `main`

Role: Compose service that hosts the Dynamo frontend process for local
developer routing.

Exact process:

```bash
python -m dynamo.frontend \
  --discovery-backend file \
  --router-mode round-robin \
  --namespace "${DYN_NAMESPACE}" \
  --http-host "${DYN_HTTP_HOST}" \
  --http-port "${DYN_HTTP_PORT}"
```

### `vllm-runtime`

Role: the single local backend that registers itself with Dynamo.

Exact process:

```bash
python -m dynamo.vllm \
  --model "${VLLM_MODEL}" \
  --discovery-backend file \
  --namespace "${DYN_NAMESPACE}" \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### Explicitly absent in local v1

These upstream services are not part of the pinned local topology:

- `etcd-server`
- `nats-server`
- standalone `dynamo.router`
- planner / operator / Kubernetes-only services

The local topology depends on a shared filesystem path for file discovery, not
on extra network services.

## Process roles

| Service | Process | Role in v1 |
| --- | --- | --- |
| `main` | `python -m dynamo.frontend` | OpenAI-compatible HTTP entrypoint and worker discovery consumer |
| `vllm-runtime` | `python -m dynamo.vllm` | Registers `backend.generate` and serves the actual model through Dynamo |

## Required environment

### Repo-owned wrapper variables

The local `devx/` wrappers intentionally use a small repo-owned env surface and
translate it into upstream `DYN_*` variables:

- `start_frontend.sh`
  - requires `DYNAMO_FILE_KV`
  - accepts `DYNAMO_PYTHON_BIN`
  - accepts `DYNAMO_NAMESPACE`, `DYNAMO_FRONTEND_HOST`, `DYNAMO_FRONTEND_PORT`
  - rejects non-`file` discovery and non-`round-robin` routing
- `start_backend_probe.sh`
  - requires `DYNAMO_EXPECTED_MODEL`
  - accepts `DYNAMO_FRONTEND_URL`
  - accepts `DYNAMO_NAMESPACE`
  - rejects non-`file` discovery

These wrappers do not try to infer broader compose wiring.

### Frontend

Required for the pinned topology:

- `DYN_DISCOVERY_BACKEND=file`
- `DYN_FILE_KV=<shared directory visible to both main and vllm-runtime>`
- `DYN_NAMESPACE=<stack-scoped namespace>`
- `DYN_ROUTER_MODE=round-robin`

Expected but allowed to use defaults:

- `DYN_HTTP_HOST` defaults to `0.0.0.0`
- `DYN_HTTP_PORT` defaults to `8000`

### `vllm-runtime`

Required for the pinned topology:

- `DYN_DISCOVERY_BACKEND=file`
- `DYN_FILE_KV=<same shared directory as frontend>`
- `DYN_NAMESPACE=<same namespace as frontend>`
- `DYN_SYSTEM_PORT=<worker health port>`
- `DYN_SYSTEM_STARTING_HEALTH_STATUS=notready`
- `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]`

Required worker CLI behavior:

- use `python -m dynamo.vllm`
- keep `--kv-events-config '{"enable_kv_cache_events": false}'`
- do not switch to KV-aware routing or durable KV events in this topology

## Health and readiness contract

### Frontend

- `GET /health` on `http://${DYN_HTTP_HOST}:${DYN_HTTP_PORT}/health`
- Ready enough for routing when the response contains
  `dyn://${DYN_NAMESPACE}.backend.generate` in `endpoints`

### `vllm-runtime`

- `GET /health` on `http://<vllm-runtime-host>:${DYN_SYSTEM_PORT}/health`
- Ready when the worker reports endpoint state `generate=ready`
- The readiness contract assumes
  `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]`

## Exact success condition

"Dynamo is routing to local vLLM" means all of the following are true:

1. The frontend `/health` response lists
   `dyn://${DYN_NAMESPACE}.backend.generate`.
2. `GET /v1/models` on the frontend exposes the expected local model id.
3. `POST /v1/chat/completions` to the frontend succeeds for that model id.

In other words: the frontend has discovered the local worker and can route an
OpenAI-compatible request through Dynamo to that worker.

## Implementation boundary

The wrappers in this directory deliberately do not:

- start `vllm-runtime`
- start `etcd-server` or `nats-server`
- build a larger Dynamo graph
- approximate alternate upstream packaging layouts

If upstream Dynamo stops exposing `python -m dynamo.frontend`,
`python -m dynamo.vllm`, `file` discovery, or the current HTTP health / model
surface, treat that as an incompatible packaging change for this local v1
contract.
