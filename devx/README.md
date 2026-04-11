# devx

## E2E Verification Transcript (2026-04-11 UTC)

This is a command-and-output transcript from a real run of the local devx + Dynamo E2E verification flow.

Notes:
- `HF_TOKEN` is redacted.
- The default model `Qwen/Qwen3.5-7B-Instruct` did not resolve in this environment, so the successful verification run uses:
  - `VLLM_MODEL=Qwen/Qwen3-0.6B`
  - `DYNAMO_MODEL=Qwen/Qwen3-0.6B`

### 1. Environment checks

```bash
$ pwd
/data00/home/philip.yang/special_circumstances/vllm

$ git rev-parse --abbrev-ref HEAD
phi9t-mainline

$ git status --short --branch
## phi9t-mainline
?? devx/authorized_keys
?? devx/tests/test_run_hermetic_sshd_actual_ssh.sh
?? docs/superpowers/plans/2026-04-10-cleanup-agentic-hack-frontier.md
?? docs/superpowers/plans/2026-04-11-vllm-devx-dynamo-e2e-setup-plan.md

$ docker version
Client: Docker Engine - Community
 Version:           26.1.4
 API version:       1.45
 Go version:        go1.21.11
 Git commit:        5650f9b
 Built:             Wed Jun  5 11:29:15 2024
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          26.1.4
  API version:      1.45 (minimum version 1.24)
  Go version:       go1.21.11
  Git commit:       de5c9cf
  Built:            Wed Jun  5 11:29:15 2024
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.6.33
  GitCommit:        d2d58213f83a351ca8f528a95fbd145f5654e957
 runc:
  Version:          1.1.12
  GitCommit:        v1.1.12-0-g51d5e94
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0

$ docker compose version
Docker Compose version v2.27.1
```

### 2. Start stack (initial run)

```bash
$ ./devx/launch_container.sh up-rebuild
...
Developer stack is starting.
SSH login: ssh -p 2222 kvothe@127.0.0.1
Compose project: phi9t-vllm-devx
```

### 3. Verify run #1 (fails at checkpoint 1)

```bash
$ ./devx/launch_container.sh verify-e2e
CHECKPOINT 1 FAIL: missing or empty huggingface token file: /data00/home/philip.yang/.devx/special-circ-phi9t-vllm/secrets/huggingface_token
```

### 4. Verify run #2 (after token file, fails at checkpoint 4)

```bash
$ ./devx/launch_container.sh verify-e2e
CHECKPOINT 1 PASS: huggingface token file present
CHECKPOINT 2 PASS: docker and docker compose available
CHECKPOINT 3 PASS: compose services visible
CHECKPOINT 4 FAIL: vllm-runtime /health probe failed
```

Runtime and worker diagnostics showed:
- runtime previously launched without `HF_TOKEN` in env (`warning: HF_TOKEN is not set`)
- Dynamo worker repeatedly failed to fetch default model `Qwen/Qwen3.5-7B-Instruct` with Hugging Face 404

Example worker log excerpt:

```text
Failed to fetch model 'Qwen/Qwen3.5-7B-Instruct' ... 404 Not Found
https://huggingface.co/api/models/Qwen/Qwen3.5-7B-Instruct/revision/main
```

### 5. Validate available Qwen IDs

```bash
$ for m in Qwen/Qwen3-8B Qwen/Qwen3-4B Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct; do code=$(curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/api/models/$m"); echo "$m $code"; done
Qwen/Qwen3-8B 200
Qwen/Qwen3-4B 200
Qwen/Qwen3-1.7B 200
Qwen/Qwen3-0.6B 200
Qwen/Qwen2.5-7B-Instruct 200
Qwen/Qwen2.5-3B-Instruct 200
```

### 6. Restart stack with model override

```bash
$ HF_TOKEN=<redacted> VLLM_MODEL=Qwen/Qwen3-0.6B DYNAMO_MODEL=Qwen/Qwen3-0.6B ./devx/launch_container.sh up-rebuild
...
Developer stack is starting.
SSH login: ssh -p 2222 kvothe@127.0.0.1
Compose project: phi9t-vllm-devx
```

### 7. Retry verification until services are warm, then pass

```bash
$ HF_TOKEN=<redacted> VLLM_MODEL=Qwen/Qwen3-0.6B DYNAMO_MODEL=Qwen/Qwen3-0.6B ./devx/launch_container.sh verify-e2e
CHECKPOINT 1 PASS: huggingface token file present
CHECKPOINT 2 PASS: docker and docker compose available
CHECKPOINT 3 PASS: compose services visible
CHECKPOINT 4 PASS: vllm-runtime /health reachable
CHECKPOINT 5 PASS: dynamo-vllm-worker /health reachable
CHECKPOINT 6 PASS: dynamo frontend route/models/chat path verified for Qwen/Qwen3-0.6B
```

### 8. Manual proof probes

```bash
$ ./devx/launch_container.sh ps --services
devshell
dynamo-frontend
dynamo-vllm-worker
ollama
sglang
vllm-runtime
```

```bash
$ ./devx/launch_container.sh exec -T vllm-runtime python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"
200

$ ./devx/launch_container.sh exec -T dynamo-vllm-worker python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8081/health').status)"
200
```

```bash
$ ./devx/launch_container.sh exec -T dynamo-frontend env DYNAMO_EXPECTED_MODEL="Qwen/Qwen3-0.6B" bash /workspace/vllm/devx/dynamo/start_backend_probe.sh
Pinned Dynamo local v1 topology probe
  frontend url     : http://127.0.0.1:8000
  namespace        : phi9t-vllm-devx
  expected endpoint: dyn://phi9t-vllm-devx.backend.generate
  expected model   : Qwen/Qwen3-0.6B
  discovery        : file
  curl timeout     : connect=3s total=15s
Routed success confirmed: Dynamo frontend discovered dyn://phi9t-vllm-devx.backend.generate and served Qwen/Qwen3-0.6B.
```

```bash
$ ./devx/launch_container.sh exec -T dynamo-frontend python3 - <<'PY'
import json, urllib.request
base='http://127.0.0.1:8000'
health=json.load(urllib.request.urlopen(base+'/health'))
print('HEALTH_ENDPOINTS=', health.get('endpoints'))
models=json.load(urllib.request.urlopen(base+'/v1/models'))
ids=[m.get('id') for m in models.get('data',[])]
print('MODEL_IDS=', ids)
payload={
  'model':'Qwen/Qwen3-0.6B',
  'messages':[{'role':'user','content':'Reply with exactly: ok'}],
  'max_tokens':8,
  'temperature':0,
}
req=urllib.request.Request(base+'/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
resp=json.load(urllib.request.urlopen(req))
msg=resp.get('choices',[{}])[0].get('message',{}).get('content','')
print('CHAT_MODEL=', resp.get('model'))
print('CHAT_REPLY=', msg)
PY
HEALTH_ENDPOINTS= ['dyn://phi9t-vllm-devx.backend.clear_kv_blocks', 'dyn://phi9t-vllm-devx.backend.generate']
MODEL_IDS= ['Qwen/Qwen3-0.6B']
CHAT_MODEL= Qwen/Qwen3-0.6B
CHAT_REPLY= <think>
Okay, the user wants me
```

## Result

The E2E flow is verified as working in this environment when using a valid model override (`Qwen/Qwen3-0.6B`) plus `HF_TOKEN`.
