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

assert_json_field_prefix() {
  local json_file="$1"
  local field_path="${2#.}"
  local expected_prefix="$3"

  python3 - "${json_file}" "${field_path}" "${expected_prefix}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

value = data
for part in sys.argv[2].split("."):
    value = value[part]

if not isinstance(value, str):
    raise SystemExit(f"field {sys.argv[2]} is not a string")

if not value.startswith(sys.argv[3]):
    raise SystemExit(f"field {sys.argv[2]} did not start with expected prefix: {sys.argv[3]!r}, actual={value!r}")
PY
}

assert_json_field_max_length() {
  local json_file="$1"
  local field_path="${2#.}"
  local max_length="$3"

  python3 - "${json_file}" "${field_path}" "${max_length}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

value = data
for part in sys.argv[2].split("."):
    value = value[part]

if not isinstance(value, str):
    raise SystemExit(f"field {sys.argv[2]} is not a string")

max_length = int(sys.argv[3])
if len(value) > max_length:
    raise SystemExit(f"field {sys.argv[2]} exceeded max length {max_length}: {len(value)}")
PY
}

assert_line_equals() {
  local haystack_file="$1"
  local line_number="$2"
  local expected="$3"
  local actual

  actual="$(sed -n "${line_number}p" "${haystack_file}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "unexpected line ${line_number} in ${haystack_file}" >&2
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  fi
}

assert_file_not_exists() {
  local file_path="$1"

  if [[ -e "${file_path}" ]]; then
    echo "file should not exist: ${file_path}" >&2
    if [[ -f "${file_path}" ]]; then
      cat "${file_path}" >&2
    fi
    exit 1
  fi
}

mkdir -p \
  "${TEST_ROOT}/bin" \
  "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

if ! DEVX_DUMP_LAUNCH_CONTAINER_SH=1 \
  bash "${REPO_ROOT}/devx/bin/devx" \
  >"${TEST_ROOT}/launch-path.stdout" \
  2>"${TEST_ROOT}/launch-path.stderr"; then
  echo "devx should print the default launch_container.sh path in dump mode" >&2
  cat "${TEST_ROOT}/launch-path.stderr" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/launch-path.stdout" "${REPO_ROOT}/devx/launch_container.sh"

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

expected_url="${DEVX_EXPECTED_HF_MODEL_URL:-}"
actual_url="${*: -1}"

if [[ -n "${DEVX_TEST_CURL_LOG:-}" ]]; then
  echo "${actual_url}" >> "${DEVX_TEST_CURL_LOG}"
fi

if [[ -z "${expected_url}" ]]; then
  echo "DEVX_EXPECTED_HF_MODEL_URL must be set for this test" >&2
  exit 1
fi

if [[ "${actual_url}" != "${expected_url}" ]]; then
  echo "unexpected curl URL: ${actual_url}" >&2
  echo "expected: ${expected_url}" >&2
  exit 1
fi

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
  up-rebuild)
    exit 0
    ;;
  verify-e2e)
    if [[ "${DEVX_FAIL_VERIFY_E2E:-0}" == "1" ]]; then
      exit 23
    fi
    exit 0
    ;;
  exec)
    request_json="$(cat)"
    request_model="$(
      python3 - "${request_json}" <<'PY'
import json
import sys

request = json.loads(sys.argv[1])
print(request["model"])
PY
    )"
    prompt="$(
      python3 - "${request_json}" <<'PY'
import json
import sys

request = json.loads(sys.argv[1])
print(request["messages"][0]["content"])
PY
    )"
    response_model="${request_model}-backend"
    long_tail="0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz"
    echo "REQUEST_MODEL=${request_model}" >> "${DEVX_TEST_LAUNCH_LOG}"
    echo "RESPONSE_MODEL=${response_model}" >> "${DEVX_TEST_LAUNCH_LOG}"
    printf '{"model":"%s","choices":[{"message":{"content":"reply for request=%s response=%s: %s"}}]}\n' \
      "${response_model}" "${request_model}" "${response_model}" "${prompt} ${long_tail}"
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
CURL_LOG="${TEST_ROOT}/curl.log"
EXPECTED_HF_MODEL_URL="https://huggingface.co/api/models/Qwen/Qwen3-4B"

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  DEVX_TEST_CURL_LOG="${CURL_LOG}" \
  DEVX_EXPECTED_HF_MODEL_URL="${EXPECTED_HF_MODEL_URL}" \
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
assert_line_equals "${LAUNCH_LOG}" 1 "ARGS: up-rebuild"
assert_line_equals "${LAUNCH_LOG}" 5 "ARGS: verify-e2e"
assert_contains "${CURL_LOG}" "${EXPECTED_HF_MODEL_URL}"

FAIL_STATE_FILE="${TEST_ROOT}/state-fail.env"
FAIL_LAUNCH_LOG="${TEST_ROOT}/launch-fail.log"
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DEVX_STATE_FILE="${FAIL_STATE_FILE}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_TEST_LAUNCH_LOG="${FAIL_LAUNCH_LOG}" \
  DEVX_TEST_CURL_LOG="${CURL_LOG}" \
  DEVX_EXPECTED_HF_MODEL_URL="${EXPECTED_HF_MODEL_URL}" \
  DEVX_FAIL_VERIFY_E2E=1 \
  bash "${REPO_ROOT}/devx/bin/devx" up --preset qwen3-4b \
  >"${TEST_ROOT}/up-fail.stdout" \
  2>"${TEST_ROOT}/up-fail.stderr"; then
  echo "devx up should fail when verify-e2e fails" >&2
  exit 1
fi
assert_line_equals "${FAIL_LAUNCH_LOG}" 1 "ARGS: up-rebuild"
assert_line_equals "${FAIL_LAUNCH_LOG}" 5 "ARGS: verify-e2e"
assert_file_not_exists "${FAIL_STATE_FILE}"

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
assert_json_field "${TEST_ROOT}/query-default.stdout" '.model' 'Qwen/Qwen3-4B-backend'
assert_json_field_prefix "${TEST_ROOT}/query-default.stdout" '.reply' 'reply for request=Qwen/Qwen3-4B response=Qwen/Qwen3-4B-backend: default preset prompt'
assert_json_field_max_length "${TEST_ROOT}/query-default.stdout" '.reply' 160
assert_contains "${LAUNCH_LOG}" "ARGS: exec -T dynamo-frontend"
assert_contains "${LAUNCH_LOG}" "REQUEST_MODEL=Qwen/Qwen3-4B"
assert_contains "${LAUNCH_LOG}" "RESPONSE_MODEL=Qwen/Qwen3-4B-backend"

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

assert_json_field "${TEST_ROOT}/query-explicit.stdout" '.model' 'Qwen/Qwen3-0.6B-backend'
assert_json_field_prefix "${TEST_ROOT}/query-explicit.stdout" '.reply' 'reply for request=Qwen/Qwen3-0.6B response=Qwen/Qwen3-0.6B-backend: explicit preset prompt'
assert_json_field_max_length "${TEST_ROOT}/query-explicit.stdout" '.reply' 160
assert_contains "${LAUNCH_LOG}" "REQUEST_MODEL=Qwen/Qwen3-0.6B"
assert_contains "${LAUNCH_LOG}" "RESPONSE_MODEL=Qwen/Qwen3-0.6B-backend"

echo "PASS"
