# DevX Validation Runbook

Use this runbook to reproduce all verification work and evidence used for the latest onboarding completion.

## 0) Scope

- Runtime/container bootstrap: `devx` compose and image/container start path.
- SSH authentication behavior under the dedicated devx key.
- E2E inference path through Dynamo + vLLM.
- Complete local test-suite execution.

## 1) Full test suite (all `devx/tests/*.sh`)

The complete suite was executed in order with per-test logs written under `devx/runs/`.

```bash
set -euo pipefail
TS=20260413-complete-tests
mkdir -p devx/runs
for f in \
  devx/tests/test_prepare_host_state.sh \
  devx/tests/test_devx_cli.sh \
  devx/tests/test_devx_switch.sh \
  devx/tests/test_devx_up_query.sh \
  devx/tests/test_experiments.sh \
  devx/tests/test_preflight.sh \
  devx/tests/test_run_hermetic_sshd.sh \
  devx/tests/test_run_hermetic_sshd_actual_ssh.sh \
  devx/tests/test_start_frontend.sh \
  devx/tests/test_start_vllm_runtime.sh \
  devx/tests/test_compose_runner.sh \
  devx/tests/test_verify_e2e.sh
do
  bn=\"$(basename \"$f\" .sh)\"
  log=\"devx/runs/${TS}-${bn}.log\"
  bash \"$f\" 2>&1 | tee \"$log\"
done
```

Observed summary:

```text
test_prepare_host_state.sh  PASS
test_devx_cli.sh           PASS
test_devx_switch.sh        PASS
test_devx_up_query.sh      PASS
test_experiments.sh        PASS
test_preflight.sh          PASS
test_run_hermetic_sshd.sh  PASS
test_run_hermetic_sshd_actual_ssh.sh  PASS
test_start_frontend.sh      PASS
test_start_vllm_runtime.sh  PASS
test_compose_runner.sh      PASS
test_verify_e2e.sh          PASS
```

Artifacts:

- Canonical suite log: `devx/runs/20260413-complete-tests-rerun-071335.log`

The canonical suite file was produced by the loop shown in this section, so all test
pass/fail output is already captured there.

## 2) Dedicated onboarding execution with outputs

```bash
export CREDS_FILE="${CREDS_FILE:-$HOME/workspace/CREDS.yaml}"
export HF_TOKEN="$(awk '/huggingface:/{f=1;next} f && /token:/{print $2; exit}' "$CREDS_FILE")"
mkdir -p "$HOME/.devx/special-circ-phi9t-vllm/secrets"
printf '%s\n' "$HF_TOKEN" > "$HOME/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"
chmod 600 "$HOME/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

./devx/bin/devx up --preset qwen3-0.6b
./devx/launch_container.sh exec -T vllm-runtime curl -is http://127.0.0.1:8000/health
./devx/launch_container.sh exec -T dynamo-vllm-worker curl -fsS http://127.0.0.1:8081/health
./devx/launch_container.sh exec -T dynamo-frontend curl -fsS http://127.0.0.1:8000/v1/models | jq .
ssh -o IdentitiesOnly=yes -o IdentityAgent=none -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1 whoami
./devx/bin/devx query --prompt "Reply with exactly: ok"

# multi-turn + tool probe uses the same snippet in onboarding.md section 6
```

Observed output highlights:

```text
CHECKPOINT 4 PASS: vllm-runtime /health reachable
CHECKPOINT 5 PASS: dynamo-vllm-worker /health reachable
CHECKPOINT 6 PASS: dynamo frontend route/models/chat path verified for Qwen/Qwen3-0.6B
DEVX UP PASS: preset=qwen3-0.6b model=Qwen/Qwen3-0.6B

HTTP/1.1 200 OK
content-length: 0

{"status":"ready","uptime":{"secs":68,"nanos":994191258},"endpoints":{"clear_kv_blocks":"ready","generate":"ready"}}

{
  "object": "list",
  "data": [
    {"id":"Qwen/Qwen3-0.6B","object":"model","owned_by":"nvidia"}
  ]
}

ssh result: kvothe

single-turn result includes model id, latency, and reply content

tool probe result:
  tool message content contains a JSON code block for get_weather(city=Paris)
  tool_calls field: null
```

Full transcript:

- `devx/runs/onboarding-rerun-20260413-072036.log`

## 3) SSH/auth troubleshooting runbook summary

`devx/tests/test_run_hermetic_sshd_actual_ssh.sh` and `devx/tests/test_compose_runner.sh` validate that SSH auth is reachable with the intended key and that startup prints explicit pinned identity instructions. This behavior is documented in `devx/ssh-auth-debugging.md`.
