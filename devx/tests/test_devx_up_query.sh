#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"

cleanup() {
  chmod -R u+w "${TEST_ROOT}" >/dev/null 2>&1 || true
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT

assert_contains() {
  local haystack_file="$1"
  local needle="$2"

  grep -F -- "${needle}" "${haystack_file}" >/dev/null 2>&1 || {
    echo "missing expected output: ${needle}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  }
}

assert_json_field() {
  local json_file="$1"
  local field_path="${2#.}"
  local expected="$3"

  local actual
  actual="$(python3 - "${json_file}" "${field_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

value = data
for part in sys.argv[2].split("."):
    value = value[part]

print(value)
PY
)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "unexpected JSON field value for ${field_path}: ${actual} (expected ${expected})" >&2
    echo "--- ${json_file} ---" >&2
    cat "${json_file}" >&2
    exit 1
  fi
}

assert_json_number_field() {
  local json_file="$1"
  local field_path="${2#.}"

  python3 - "${json_file}" "${field_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

value = data
for part in sys.argv[2].split("."):
    value = value[part]

if not isinstance(value, int):
    raise SystemExit(f"field {sys.argv[2]} is not a JSON integer: {value!r}")
PY
}

mkdir -p \
  "${TEST_ROOT}/bin" \
  "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

cat <<'EOF_DOCKER' > "${TEST_ROOT}/bin/docker"
#!/bin/bash
set -euo pipefail

case "${1:-}" in
  version)
    exit 0
    ;;
  compose)
    if [[ "${2:-}" == "version" ]]; then
      exit 0
    fi
    exit 0
    ;;
  *)
    exit 1
    ;;
esac
EOF_DOCKER
chmod +x "${TEST_ROOT}/bin/docker"

cat <<'EOF_CURL' > "${TEST_ROOT}/bin/curl"
#!/bin/bash
set -euo pipefail

exit 0
EOF_CURL
chmod +x "${TEST_ROOT}/bin/curl"

cat <<'EOF_LAUNCH' > "${TEST_ROOT}/fake_launch_container.sh"
#!/bin/bash
set -euo pipefail

echo "ARGS: $*" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "HF_TOKEN=${HF_TOKEN:-}" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "VLLM_MODEL=${VLLM_MODEL:-}" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "DYNAMO_MODEL=${DYNAMO_MODEL:-}" >> "${DEVX_TEST_LAUNCH_LOG}"

case "${1:-}" in
  up-rebuild|verify-e2e)
    exit 0
    ;;
  exec)
    model="${DYNAMO_MODEL:-${VLLM_MODEL:-missing/model}}"
    prompt="$(
      DEVX_FAKE_QUERY_PAYLOAD="${DEVX_QUERY_PAYLOAD:-}" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["DEVX_FAKE_QUERY_PAYLOAD"])
print(payload["messages"][0]["content"])
PY
    )"
    printf '{"model":"%s","choices":[{"message":{"content":"reply for %s: %s"}}]}\n' \
      "${model}" "${model}" "${prompt}"
    exit 0
    ;;
  *)
    echo "unexpected launcher command: ${1:-}" >&2
    exit 1
    ;;
esac
EOF_LAUNCH
chmod +x "${TEST_ROOT}/fake_launch_container.sh"

printf 'hf_test_token\n' > "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

STATE_FILE="${TEST_ROOT}/state.env"
LAUNCH_LOG="${TEST_ROOT}/launch.log"

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  bash "${REPO_ROOT}/devx/bin/devx" up --preset qwen3-4b \
  >"${TEST_ROOT}/up.stdout" \
  2>"${TEST_ROOT}/up.stderr"; then
  echo "devx up should succeed for a known preset" >&2
  cat "${TEST_ROOT}/up.stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/up.stdout" "DEVX UP PASS: preset=qwen3-4b model=Qwen/Qwen3-4B"
assert_contains "${STATE_FILE}" "active_preset=qwen3-4b"
assert_contains "${STATE_FILE}" "updated_at_unix="
assert_contains "${LAUNCH_LOG}" "ARGS: up-rebuild"
assert_contains "${LAUNCH_LOG}" "HF_TOKEN=hf_test_token"
assert_contains "${LAUNCH_LOG}" "VLLM_MODEL=Qwen/Qwen3-4B"
assert_contains "${LAUNCH_LOG}" "DYNAMO_MODEL=Qwen/Qwen3-4B"

if bash "${REPO_ROOT}/devx/bin/devx" query \
  >"${TEST_ROOT}/query-missing.stdout" \
  2>"${TEST_ROOT}/query-missing.stderr"; then
  echo "devx query should fail without --prompt" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/query-missing.stderr" "query: missing required --prompt"

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  bash "${REPO_ROOT}/devx/bin/devx" query --prompt "default preset prompt" \
  >"${TEST_ROOT}/query-default.stdout" \
  2>"${TEST_ROOT}/query-default.stderr"; then
  echo "devx query should succeed with the session preset" >&2
  cat "${TEST_ROOT}/query-default.stderr" >&2
  exit 1
fi

assert_json_number_field "${TEST_ROOT}/query-default.stdout" '.latency_ms'
assert_json_field "${TEST_ROOT}/query-default.stdout" '.model' 'Qwen/Qwen3-4B'
assert_json_field "${TEST_ROOT}/query-default.stdout" '.reply' 'reply for Qwen/Qwen3-4B: default preset prompt'
assert_contains "${LAUNCH_LOG}" "ARGS: exec -T dynamo-frontend"
assert_contains "${LAUNCH_LOG}" "VLLM_MODEL=Qwen/Qwen3-4B"
assert_contains "${LAUNCH_LOG}" "DYNAMO_MODEL=Qwen/Qwen3-4B"

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  bash "${REPO_ROOT}/devx/bin/devx" query --prompt "explicit preset prompt" --preset qwen3-0.6b \
  >"${TEST_ROOT}/query-explicit.stdout" \
  2>"${TEST_ROOT}/query-explicit.stderr"; then
  echo "devx query should succeed with an explicit preset override" >&2
  cat "${TEST_ROOT}/query-explicit.stderr" >&2
  exit 1
fi

assert_json_field "${TEST_ROOT}/query-explicit.stdout" '.model' 'Qwen/Qwen3-0.6B'
assert_json_field "${TEST_ROOT}/query-explicit.stdout" '.reply' 'reply for Qwen/Qwen3-0.6B: explicit preset prompt'
assert_contains "${LAUNCH_LOG}" "VLLM_MODEL=Qwen/Qwen3-0.6B"
assert_contains "${LAUNCH_LOG}" "DYNAMO_MODEL=Qwen/Qwen3-0.6B"

echo "PASS"
