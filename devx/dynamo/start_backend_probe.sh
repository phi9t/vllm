#!/bin/bash

set -euo pipefail

die() {
  echo "start_backend_probe.sh: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_command curl
require_command jq

DYNAMO_DISCOVERY_BACKEND="${DYNAMO_DISCOVERY_BACKEND:-file}"
DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-dynamo}"
DYNAMO_FRONTEND_URL="${DYNAMO_FRONTEND_URL:-http://127.0.0.1:8000}"
DYNAMO_EXPECTED_MODEL="${DYNAMO_EXPECTED_MODEL:-}"
DYNAMO_PROBE_MAX_TOKENS="${DYNAMO_PROBE_MAX_TOKENS:-16}"
DYNAMO_PROBE_PROMPT="${DYNAMO_PROBE_PROMPT:-Reply with the single word local.}"

[ "${DYNAMO_DISCOVERY_BACKEND}" = "file" ] || die "pinned local v1 topology only supports DYNAMO_DISCOVERY_BACKEND=file"
[ -n "${DYNAMO_EXPECTED_MODEL}" ] || die "DYNAMO_EXPECTED_MODEL must be set to the model id registered by vllm-runtime"
[[ "${DYNAMO_PROBE_MAX_TOKENS}" =~ ^[0-9]+$ ]] || die "DYNAMO_PROBE_MAX_TOKENS must be an integer"

EXPECTED_ENDPOINT="dyn://${DYNAMO_NAMESPACE}.backend.generate"

cat <<EOF
Pinned Dynamo local v1 topology probe
  frontend url     : ${DYNAMO_FRONTEND_URL}
  namespace        : ${DYNAMO_NAMESPACE}
  expected endpoint: ${EXPECTED_ENDPOINT}
  expected model   : ${DYNAMO_EXPECTED_MODEL}
  discovery        : file
EOF

health_json="$(curl -fsS "${DYNAMO_FRONTEND_URL%/}/health")" \
  || die "frontend health request failed: ${DYNAMO_FRONTEND_URL%/}/health"
echo "${health_json}" | jq -e --arg endpoint "${EXPECTED_ENDPOINT}" \
  '(.endpoints // []) | index($endpoint) != null' >/dev/null \
  || die "frontend /health does not list ${EXPECTED_ENDPOINT}"

models_json="$(curl -fsS "${DYNAMO_FRONTEND_URL%/}/v1/models")" \
  || die "frontend models request failed: ${DYNAMO_FRONTEND_URL%/}/v1/models"
echo "${models_json}" | jq -e --arg model "${DYNAMO_EXPECTED_MODEL}" \
  '[(.data // [])[]?.id] | index($model) != null' >/dev/null \
  || die "frontend /v1/models does not expose ${DYNAMO_EXPECTED_MODEL}"

request_payload="$(
  jq -cn \
    --arg model "${DYNAMO_EXPECTED_MODEL}" \
    --arg prompt "${DYNAMO_PROBE_PROMPT}" \
    --argjson max_tokens "${DYNAMO_PROBE_MAX_TOKENS}" \
    '{
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: $max_tokens
    }'
)"

response_json="$(
  curl -fsS "${DYNAMO_FRONTEND_URL%/}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${request_payload}"
)" || die "frontend routed chat completion failed"

echo "${response_json}" | jq -e --arg model "${DYNAMO_EXPECTED_MODEL}" \
  '.model == $model and ((.choices // []) | length) > 0' >/dev/null \
  || die "chat completion response did not match the expected model contract"

echo "Routed success confirmed: Dynamo frontend discovered ${EXPECTED_ENDPOINT} and served ${DYNAMO_EXPECTED_MODEL}."
