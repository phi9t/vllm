# NVIDIA Dynamo

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is an open-source framework for distributed LLM inference that can run vLLM on Kubernetes with flexible serving architectures (e.g. aggregated/disaggregated, optional router/planner).

## Local developer integration

The local `devx/` integration is intentionally pinned to a single concrete
version-one topology instead of a vague "minimal Dynamo" claim:

- `main` runs the Dynamo frontend process via
  [`devx/dynamo/start_frontend.sh`](../../../devx/dynamo/start_frontend.sh).
- `dynamo-vllm-worker` is the Dynamo backend and runs the Dynamo vLLM worker
  (`python -m dynamo.vllm`).
- the direct source-backed `vllm-runtime` remains outside Dynamo so local repo
  work can still compare direct vLLM behavior against the routed Dynamo path.
- Discovery is `file`-backed through a shared `DYN_FILE_KV` path, and KV events
  stay disabled for this topology so local development does not require
  `etcd-server` or `nats-server`.

This split is intentional: upstream `ai-dynamo==1.0.1` currently targets an
older vLLM API surface than the repo's `vllm==0.19.x`, so the local routed path
uses a Dynamo-pinned worker image while the direct path keeps the repo-under-
test runtime.

The topology contract, required environment, and routed-success probe are
recorded in
[`devx/dynamo/README.md`](../../../devx/dynamo/README.md).

Out of scope for this pinned v1 integration:

- disaggregated prefill/decode Dynamo graphs
- KV-aware routing and any NATS-backed event plane
- standalone `dynamo.router`, planner, or Kubernetes operator wiring
- general multi-backend or multi-model local developer topologies

For Kubernetes deployment instructions and examples (including vLLM), see the [Deploying Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/README.md) guide.

Background reading: InfoQ news coverage — [NVIDIA Dynamo simplifies Kubernetes deployment for LLM inference](https://www.infoq.com/news/2025/12/nvidia-dynamo-kubernetes/).
