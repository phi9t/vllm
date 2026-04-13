# DevX Onboarding: Launch, SSH, and Qwen3-0.6B Inference

This guide walks through a deterministic happy path for running the dev stack and validating the vLLM + Dynamo flow.

## 1) Prepare Hugging Face token

```bash
export CREDS_FILE="${CREDS_FILE:-$HOME/workspace/CREDS.yaml}"
export HF_TOKEN="$(awk '/huggingface:/{f=1;next} f && /token:/{print $2; exit}' "$CREDS_FILE")"
mkdir -p "$HOME/.devx/special-circ-phi9t-vllm/secrets"
printf '%s\n' "$HF_TOKEN" > "$HOME/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"
chmod 600 "$HOME/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"
ls -l "$HOME/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"
```

## 2) Start the stack (preset qwen3-0.6b)

```bash
./devx/bin/devx up --preset qwen3-0.6b
```

Expected checkpoints to pass in this environment:

```text
CHECKPOINT 1 PASS: huggingface token file present
CHECKPOINT 2 PASS: docker and docker compose available
CHECKPOINT 3 PASS: compose services visible
CHECKPOINT 4 PASS: vllm-runtime /health reachable
CHECKPOINT 5 PASS: dynamo-vllm-worker /health reachable
CHECKPOINT 6 PASS: dynamo frontend route/models/chat path verified for Qwen/Qwen3-0.6B
DEVX UP PASS: preset=qwen3-0.6b model=Qwen/Qwen3-0.6B
```

The stack prints an SSH hint; always use it. It pins the dedicated key and limits identity
probes so SSH does not fail under multiple loaded identities.

```text
SSH login: ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1
```

## 3) Validate services and endpoints

```bash
./devx/launch_container.sh exec -T vllm-runtime curl -is http://127.0.0.1:8000/health
./devx/launch_container.sh exec -T dynamo-vllm-worker curl -fsS http://127.0.0.1:8081/health
./devx/launch_container.sh exec -T dynamo-frontend curl -fsS http://127.0.0.1:8000/v1/models | jq .
```

Observed output (latest successful run):

```text
HTTP/1.1 200 OK
content-length: 0

{"status":"ready","uptime":{"secs":68,"nanos":994191258},"endpoints":{"clear_kv_blocks":"ready","generate":"ready"}}
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": 1776063452,
      "owned_by": "nvidia"
    }
  ]
}
```

## 4) SSH into the container

```bash
ssh -o IdentitiesOnly=yes -o IdentityAgent=none -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1 whoami
```

Expected:

```text
kvothe
```

## 5) Single-turn query to confirm inference path

```bash
./devx/bin/devx query --prompt "Reply with exactly: ok"
```

Observed:

```text
{"model":"Qwen/Qwen3-0.6B","latency_ms":550,"reply":"<think>\nOkay, the user said \"Reply with exactly: ok\". Let me think. They probably want me to just output \"ok\" without any extra text. But maybe they meant somet"}
```

## 6) Multi-turn + tool-call probe

```bash
./devx/launch_container.sh exec -T dynamo-frontend python3 - <<'PY'
import json
import urllib.request

base = 'http://127.0.0.1:8000'

def call(payload):
    req = urllib.request.Request(
        base + '/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())

turn1 = call({
    'model': 'Qwen/Qwen3-0.6B',
    'messages': [{'role': 'user', 'content': 'I am building a small CLI demo. Keep responses short.'}],
    'temperature': 0.0,
    'max_tokens': 80,
})

turn2 = call({
    'model': 'Qwen/Qwen3-0.6B',
    'messages': [
        {'role': 'user', 'content': 'I am building a small CLI demo. Keep responses short.'},
        {'role': 'assistant', 'content': 'Got it. Keep responses brief and action oriented.'},
        {'role': 'user', 'content': 'Now give me one concrete next step for this project.'},
    ],
    'temperature': 0.0,
    'max_tokens': 100,
})

tools = call({
    'model': 'Qwen/Qwen3-0.6B',
    'messages': [{'role': 'user', 'content': 'Tell me the current weather summary for Paris in JSON.'}],
    'tools': [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get a short weather summary for a city.',
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            }
        },
    }],
    'tool_choice': 'auto',
    'max_tokens': 120,
})

print('turn1=', json.dumps(turn1['choices'][0]['message'], indent=2))
print('turn2=', json.dumps(turn2['choices'][0]['message'], indent=2))
print('tool_message=', json.dumps(tools['choices'], indent=2))
PY
```

Observed:

```text
turn1= { ... "content": "<think>..." }
turn2= { ... "content": "<think>..." }
tool_message= [
  {
    "index": 0,
    "message": {
      "content": "<think>... <json>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</json>",
      "role": "assistant",
      "reasoning_content": null
    },
    "finish_reason": "stop"
  }
]
```

`tool_calls` is not emitted in this model’s JSON envelope; the tool request appears embedded in assistant content for this preset. If you need strict JSON `tool_calls` output, switch to a model/serve config that supports it.

## 7) Teardown

```bash
./devx/launch_container.sh down
```

## Evidence

The canonical evidence log for this onboarding run is:

- `devx/runs/onboarding-rerun-20260413-072036.log`

For the broader regression suite, see `devx/validation-runbook.md`.
