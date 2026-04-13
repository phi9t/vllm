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

assert_file_empty() {
  local path="$1"

  if [[ -s "${path}" ]]; then
    echo "expected empty file: ${path}" >&2
    cat "${path}" >&2
    exit 1
  fi
}

assert_file_absent() {
  local path="$1"

  if [[ -e "${path}" ]]; then
    echo "expected file to be removed: ${path}" >&2
    cat "${path}" >&2
    exit 1
  fi
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

mkdir -p \
  "${TEST_ROOT}/bin" \
  "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

cat <<'EOF_LAUNCH' > "${TEST_ROOT}/fake_launch_container.sh"
#!/bin/bash
set -euo pipefail

echo "ARGS: $*" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "HF_TOKEN=${HF_TOKEN:-}" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "VLLM_MODEL=${VLLM_MODEL:-}" >> "${DEVX_TEST_LAUNCH_LOG}"
echo "DYNAMO_MODEL=${DYNAMO_MODEL:-}" >> "${DEVX_TEST_LAUNCH_LOG}"

case "${1:-}" in
  up|verify-e2e)
    if [[ "${1:-}" == "verify-e2e" && "${DEVX_FAIL_VERIFY_E2E:-0}" == "1" ]]; then
      exit 23
    fi
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
touch "${LAUNCH_LOG}"

printf 'active_preset=qwen3-0.6b\nupdated_at_unix=111\n' > "${STATE_FILE}"

if DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" switch --preset does-not-exist \
  >"${TEST_ROOT}/invalid.stdout" 2>"${TEST_ROOT}/invalid.stderr"; then
  echo "switch should fail for an invalid preset" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/invalid.stderr" "invalid preset: does-not-exist"
assert_file_empty "${LAUNCH_LOG}"
assert_contains "${STATE_FILE}" "active_preset=qwen3-0.6b"
assert_contains "${STATE_FILE}" "updated_at_unix=111"

if bash "${REPO_ROOT}/devx/bin/devx" switch \
  >"${TEST_ROOT}/missing.stdout" 2>"${TEST_ROOT}/missing.stderr"; then
  echo "switch should fail when --preset is omitted" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/missing.stderr" "switch: missing required --preset"

printf 'active_preset=qwen3-0.6b\nupdated_at_unix=111\n' > "${STATE_FILE}"
: > "${LAUNCH_LOG}"

if ! DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" switch --preset qwen3-4b \
  >"${TEST_ROOT}/switch.stdout" 2>"${TEST_ROOT}/switch.stderr"; then
  echo "switch should succeed for a known preset" >&2
  cat "${TEST_ROOT}/switch.stderr" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/switch.stdout" "DEVX SWITCH PASS: qwen3-0.6b -> qwen3-4b"
assert_contains "${STATE_FILE}" "active_preset=qwen3-4b"
assert_line_equals "${LAUNCH_LOG}" 1 "ARGS: up"
assert_line_equals "${LAUNCH_LOG}" 2 "HF_TOKEN=hf_test_token"
assert_line_equals "${LAUNCH_LOG}" 3 "VLLM_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 4 "DYNAMO_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 5 "ARGS: verify-e2e"
assert_line_equals "${LAUNCH_LOG}" 6 "HF_TOKEN=hf_test_token"
assert_line_equals "${LAUNCH_LOG}" 7 "VLLM_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 8 "DYNAMO_MODEL=Qwen/Qwen3-4B"

printf 'active_preset=qwen3-4b\nupdated_at_unix=222\n' > "${STATE_FILE}"
: > "${LAUNCH_LOG}"

if ! DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" switch --preset qwen3-4b \
  >"${TEST_ROOT}/noop.stdout" 2>"${TEST_ROOT}/noop.stderr"; then
  echo "switch should succeed when the target preset is already active" >&2
  cat "${TEST_ROOT}/noop.stderr" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/noop.stdout" "DEVX SWITCH PASS: qwen3-4b -> qwen3-4b"
assert_file_empty "${LAUNCH_LOG}"
assert_contains "${STATE_FILE}" "active_preset=qwen3-4b"
assert_contains "${STATE_FILE}" "updated_at_unix=222"

printf 'active_preset=qwen3-0.6b\nupdated_at_unix=333\n' > "${STATE_FILE}"
: > "${LAUNCH_LOG}"

if DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  DEVX_FAIL_VERIFY_E2E=1 \
  DEVX_STATE_FILE="${STATE_FILE}" \
  DEVX_TEST_LAUNCH_LOG="${LAUNCH_LOG}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" switch --preset qwen3-4b \
  >"${TEST_ROOT}/fail.stdout" 2>"${TEST_ROOT}/fail.stderr"; then
  echo "switch should fail when verify-e2e fails" >&2
  exit 1
fi
assert_contains "${STATE_FILE}" "active_preset=qwen3-0.6b"
assert_contains "${STATE_FILE}" "updated_at_unix=333"
assert_line_equals "${LAUNCH_LOG}" 1 "ARGS: up"
assert_line_equals "${LAUNCH_LOG}" 2 "HF_TOKEN=hf_test_token"
assert_line_equals "${LAUNCH_LOG}" 3 "VLLM_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 4 "DYNAMO_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 5 "ARGS: verify-e2e"
assert_line_equals "${LAUNCH_LOG}" 6 "HF_TOKEN=hf_test_token"
assert_line_equals "${LAUNCH_LOG}" 7 "VLLM_MODEL=Qwen/Qwen3-4B"
assert_line_equals "${LAUNCH_LOG}" 8 "DYNAMO_MODEL=Qwen/Qwen3-4B"

echo "PASS"
