#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
RUN_DIR_PATHS=()

cleanup() {
  local run_dir_path

  for run_dir_path in "${RUN_DIR_PATHS[@]}"; do
    if [[ -n "${run_dir_path}" && -e "${run_dir_path}" ]]; then
      chmod -R u+w "${run_dir_path}" >/dev/null 2>&1 || true
      rm -rf "${run_dir_path}"
    fi
  done
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

assert_file_exists() {
  local path="$1"

  if [[ ! -f "${path}" ]]; then
    echo "missing expected file: ${path}" >&2
    exit 1
  fi
}

assert_file_not_exists() {
  local path="$1"

  if [[ -e "${path}" ]]; then
    echo "unexpected file: ${path}" >&2
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

assert_line_prefix() {
  local haystack_file="$1"
  local line_number="$2"
  local expected_prefix="$3"
  local actual

  actual="$(sed -n "${line_number}p" "${haystack_file}")"
  if [[ "${actual}" != "${expected_prefix}"* ]]; then
    echo "unexpected line ${line_number} in ${haystack_file}" >&2
    echo "expected prefix: ${expected_prefix}" >&2
    echo "actual:          ${actual}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  fi
}

mkdir -p "${TEST_ROOT}/bin"
mkdir -p "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

printf 'hf_test_token\n' > "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

cat <<'EOF_JQ' > "${TEST_ROOT}/bin/jq"
#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "-cn" && "${2:-}" == "--arg" && "${3:-}" == "model" && "${5:-}" == "--rawfile" && "${6:-}" == "prompt" ]]; then
  model_value="${4:-}"
  prompt_file="${7:-}"
  prompt_value="$(cat "${prompt_file}")"
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}]}\n' "${model_value}" "${prompt_value}"
  exit 0
fi

if [[ "${1:-}" == "-cn" && "${2:-}" == "--arg" && "${3:-}" == "model" && "${5:-}" == "--arg" && "${6:-}" == "prompt" ]]; then
  model_value="${4:-}"
  prompt_value="${7:-}"
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}]}\n' "${model_value}" "${prompt_value}"
  exit 0
fi

if [[ "${1:-}" == "-r" && "${2:-}" == "--arg" && "${3:-}" == "requested_model" ]]; then
  requested_model="${4:-}"
  response_file="${6:-}"
  model_value="$(sed -n 's/.*"model":"\([^"]*\)".*/\1/p' "${response_file}" | head -n 1)"
  if [[ -n "${model_value}" ]]; then
    printf '%s\n' "${model_value}"
  else
    printf '%s\n' "${requested_model}"
  fi
  exit 0
fi

if [[ "${1:-}" == "-r" ]]; then
  response_file="${3:-}"
  reply_value="$(sed -n 's/.*"content":"\([^"]*\)".*/\1/p' "${response_file}" | head -n 1)"
  if [[ -z "${reply_value}" ]]; then
    reply_value="$(sed -n 's/.*"reply":"\([^"]*\)".*/\1/p' "${response_file}" | head -n 1)"
  fi
  printf '%s\n' "${reply_value:0:160}"
  exit 0
fi

if [[ "${1:-}" == "-cn" && "${2:-}" == "--arg" && "${3:-}" == "model" && "${5:-}" == "--argjson" && "${6:-}" == "latency_ms" ]]; then
  model_value="${4:-}"
  latency_value="${7:-}"
  reply_value="${10:-}"
  printf '{"model":"%s","latency_ms":%s,"reply":"%s"}\n' "${model_value}" "${latency_value}" "${reply_value}"
  exit 0
fi

echo "unsupported jq invocation: $*" >&2
exit 1
EOF_JQ
chmod +x "${TEST_ROOT}/bin/jq"

cat <<'EOF_MANIFEST_REORDERED' > "${TEST_ROOT}/experiment-reordered.yaml"
runs:
  - preset: qwen3-0.6b
    name: small
    prompts:
      - "Reply with: ok"
EOF_MANIFEST_REORDERED

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
  exec)
    if [[ "${DEVX_FAIL_QUERY:-0}" == "1" ]]; then
      echo "simulated query failure" >&2
      exit 23
    fi
    request_json="$(cat)"
    request_model="$(printf '%s' "${request_json}" | sed -n 's/.*"model":"\([^"]*\)".*/\1/p')"
    prompt="$(printf '%s' "${request_json}" | sed -n 's/.*"content":"\([^"]*\)".*/\1/p')"
    response_model="${request_model}-backend"
    printf '{"model":"%s","choices":[{"message":{"content":"reply for request=%s response=%s: %s"}}]}\n' \
      "${response_model}" "${request_model}" "${response_model}" "${prompt}"
    ;;
  *)
    echo "unexpected launcher command: ${1:-}" >&2
    exit 1
    ;;
  esac
EOF_LAUNCH
chmod +x "${TEST_ROOT}/fake_launch_container.sh"

DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch.log" \
PATH="${TEST_ROOT}/bin:${PATH}" \
DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
HOME="${TEST_ROOT}/home" \
bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-reordered.yaml" \
  >"${TEST_ROOT}/run.out" 2>"${TEST_ROOT}/run.err" || {
  echo "experiment run should succeed" >&2
  cat "${TEST_ROOT}/run.err" >&2
  exit 1
}

assert_contains "${TEST_ROOT}/run.out" "RUN_ID="
assert_contains "${TEST_ROOT}/run.out" "RUN_DIR=devx/runs/"
assert_line_equals "${TEST_ROOT}/launch.log" 1 "ARGS: up"
assert_line_equals "${TEST_ROOT}/launch.log" 5 "ARGS: verify-e2e"
assert_line_prefix "${TEST_ROOT}/launch.log" 9 "ARGS: exec -T dynamo-frontend"

run_id="$(sed -n 's/^RUN_ID=//p' "${TEST_ROOT}/run.out" | head -n 1)"
run_dir_rel="$(sed -n 's/^RUN_DIR=//p' "${TEST_ROOT}/run.out" | head -n 1)"

if [[ -z "${run_id}" ]]; then
  echo "RUN_ID must not be empty" >&2
  cat "${TEST_ROOT}/run.out" >&2
  exit 1
fi

if [[ "${run_dir_rel}" != "devx/runs/${run_id}" ]]; then
  echo "unexpected RUN_DIR: ${run_dir_rel}" >&2
  cat "${TEST_ROOT}/run.out" >&2
  exit 1
fi

RUN_DIR_PATH="${REPO_ROOT}/${run_dir_rel}"
RUN_DIR_PATHS+=("${RUN_DIR_PATH}")
assert_file_exists "${RUN_DIR_PATH}/manifest.input.yaml"
assert_file_exists "${RUN_DIR_PATH}/results.jsonl"
assert_file_exists "${RUN_DIR_PATH}/summary.md"
assert_contains "${RUN_DIR_PATH}/manifest.input.yaml" "preset: qwen3-0.6b"
assert_contains "${RUN_DIR_PATH}/results.jsonl" '"preset":"qwen3-0.6b"'
assert_contains "${RUN_DIR_PATH}/results.jsonl" '"prompt":"Reply with: ok"'
assert_contains "${RUN_DIR_PATH}/summary.md" "preset: qwen3-0.6b"
assert_contains "${RUN_DIR_PATH}/summary.md" "prompts: 1"

bash "${REPO_ROOT}/devx/bin/devx" experiment report "${run_id}" \
  >"${TEST_ROOT}/report.out" 2>"${TEST_ROOT}/report.err" || {
  echo "experiment report should succeed" >&2
  cat "${TEST_ROOT}/report.err" >&2
  exit 1
}

assert_contains "${TEST_ROOT}/report.out" "preset: qwen3-0.6b"
assert_contains "${TEST_ROOT}/report.out" "prompts: 1"

rm -f "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

cat <<'EOF_MISSING_TOKEN' > "${TEST_ROOT}/experiment-missing-token.yaml"
runs:
  - name: small
    preset: qwen3-4b
    prompts:
      - "Reply with: ok"
EOF_MISSING_TOKEN

if DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-missing-token.log" \
  PATH="${TEST_ROOT}/bin:${PATH}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-missing-token.yaml" \
  >"${TEST_ROOT}/missing-token.out" 2>"${TEST_ROOT}/missing-token.err"; then
  echo "experiment run should fail when the token file is missing" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/missing-token.err" "preflight: missing or empty token file:"
assert_file_not_exists "${TEST_ROOT}/launch-missing-token.log"

printf 'hf_test_token\n' > "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

cat <<'EOF_BAD_RUN' > "${TEST_ROOT}/experiment-bad-run-id.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "Reply with: ok"
EOF_BAD_RUN

if DEVX_EXPERIMENT_RUN_ID="../cleanup" \
  DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-bad-run.log" \
  PATH="${TEST_ROOT}/bin:${PATH}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-bad-run-id.yaml" \
  >"${TEST_ROOT}/bad-run.out" 2>"${TEST_ROOT}/bad-run.err"; then
  echo "experiment run should reject traversal-like run ids" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/bad-run.err" "experiment run: invalid run-id: ../cleanup"

if bash "${REPO_ROOT}/devx/bin/devx" experiment report "../cleanup" \
  >"${TEST_ROOT}/bad-report.out" 2>"${TEST_ROOT}/bad-report.err"; then
  echo "experiment report should reject traversal-like run ids" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/bad-report.err" "experiment report: invalid run-id: ../cleanup"

cat <<'EOF_UNQUOTED' > "${TEST_ROOT}/experiment-unquoted.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - Reply with: ok
EOF_UNQUOTED

DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-unquoted.log" \
PATH="${TEST_ROOT}/bin:${PATH}" \
DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
HOME="${TEST_ROOT}/home" \
bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-unquoted.yaml" \
  >"${TEST_ROOT}/run-unquoted.out" 2>"${TEST_ROOT}/run-unquoted.err" || {
  echo "experiment run should accept unquoted prompt scalars" >&2
  cat "${TEST_ROOT}/run-unquoted.err" >&2
  exit 1
}

run_id_unquoted="$(sed -n 's/^RUN_ID=//p' "${TEST_ROOT}/run-unquoted.out" | head -n 1)"
run_dir_unquoted="${REPO_ROOT}/devx/runs/${run_id_unquoted}"
RUN_DIR_PATHS+=("${run_dir_unquoted}")
assert_file_exists "${run_dir_unquoted}/results.jsonl"
assert_contains "${run_dir_unquoted}/results.jsonl" '"prompt":"Reply with: ok"'

cat <<'EOF_NESTED_PROMPT' > "${TEST_ROOT}/experiment-nested-prompt.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      prompt:
        text: ok
EOF_NESTED_PROMPT

if bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-nested-prompt.yaml" \
  >"${TEST_ROOT}/nested-prompt.out" 2>"${TEST_ROOT}/nested-prompt.err"; then
  echo "experiment run should reject nested prompt structures" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/nested-prompt.err" "experiment run: unsupported manifest shape: nested prompt structures are not supported"

cat <<'EOF_QUOTED_ESCAPE' > "${TEST_ROOT}/experiment-quoted-escape.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "say \"ok\"\nline\t2"
EOF_QUOTED_ESCAPE

DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-quoted-escape.log" \
PATH="${TEST_ROOT}/bin:${PATH}" \
DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
HOME="${TEST_ROOT}/home" \
bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-quoted-escape.yaml" \
  >"${TEST_ROOT}/quoted-escape.out" 2>"${TEST_ROOT}/quoted-escape.err" || {
  echo "experiment run should accept quoted prompt scalars with escaped content" >&2
  cat "${TEST_ROOT}/quoted-escape.err" >&2
  exit 1
}

run_id_quoted_escape="$(sed -n 's/^RUN_ID=//p' "${TEST_ROOT}/quoted-escape.out" | head -n 1)"
run_dir_quoted_escape="${REPO_ROOT}/devx/runs/${run_id_quoted_escape}"
RUN_DIR_PATHS+=("${run_dir_quoted_escape}")
assert_file_exists "${run_dir_quoted_escape}/results.jsonl"
assert_contains "${run_dir_quoted_escape}/results.jsonl" '"prompt":"say \"ok\"\nline\t2"'

cat <<'EOF_BLOCK_SCALAR' > "${TEST_ROOT}/experiment-block-scalar.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - |
        Reply with: ok
EOF_BLOCK_SCALAR

if bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-block-scalar.yaml" \
  >"${TEST_ROOT}/block-scalar.out" 2>"${TEST_ROOT}/block-scalar.err"; then
  echo "experiment run should reject block scalar prompts" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/block-scalar.err" "experiment run: unsupported manifest shape: block scalar prompts are not supported"

cat <<'EOF_MULTI_RUN' > "${TEST_ROOT}/experiment-multi-run.yaml"
runs:
  - name: first
    preset: qwen3-0.6b
    prompts:
      - "Reply with: ok"
  - name: second
    preset: qwen3-4b
    prompts:
      - "Reply with: ok"
EOF_MULTI_RUN

if bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-multi-run.yaml" \
  >"${TEST_ROOT}/multi-run.out" 2>"${TEST_ROOT}/multi-run.err"; then
  echo "experiment run should fail for multiple runs" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/multi-run.err" "experiment run: unsupported manifest shape: multiple runs are not supported"

cat <<'EOF_MULTI_PROMPT' > "${TEST_ROOT}/experiment-multi-prompt.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "Reply with: ok"
      - "Reply with: ok again"
EOF_MULTI_PROMPT

if bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-multi-prompt.yaml" \
  >"${TEST_ROOT}/multi-prompt.out" 2>"${TEST_ROOT}/multi-prompt.err"; then
  echo "experiment run should fail for multiple prompts" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/multi-prompt.err" "experiment run: unsupported manifest shape: multiple prompts are not supported"

cat <<'EOF_FAIL' > "${TEST_ROOT}/experiment-fail.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "Reply with: ok"
EOF_FAIL

DEVX_EXPERIMENT_RUN_ID=cleanup-check \
DEVX_FAIL_QUERY=1 \
DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-fail.log" \
PATH="${TEST_ROOT}/bin:${PATH}" \
DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
HOME="${TEST_ROOT}/home" \
bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-fail.yaml" \
  >"${TEST_ROOT}/fail.out" 2>"${TEST_ROOT}/fail.err" && {
  echo "experiment run should fail when the query step fails" >&2
  exit 1
}

assert_contains "${TEST_ROOT}/fail.err" "experiment run: query failed"
assert_file_not_exists "${REPO_ROOT}/devx/runs/cleanup-check"

cat <<'EOF_DETERMINISTIC' > "${TEST_ROOT}/experiment-deterministic.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "Reply with: first"
EOF_DETERMINISTIC

DEVX_EXPERIMENT_RUN_ID=deterministic-run \
DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-deterministic.log" \
PATH="${TEST_ROOT}/bin:${PATH}" \
DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
HOME="${TEST_ROOT}/home" \
bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-deterministic.yaml" \
  >"${TEST_ROOT}/deterministic.out" 2>"${TEST_ROOT}/deterministic.err" || {
  echo "first deterministic experiment run should succeed" >&2
  cat "${TEST_ROOT}/deterministic.err" >&2
  exit 1
}
RUN_DIR_PATHS+=("${REPO_ROOT}/devx/runs/deterministic-run")

cat <<'EOF_DETERMINISTIC_2' > "${TEST_ROOT}/experiment-deterministic-2.yaml"
runs:
  - name: small
    preset: qwen3-0.6b
    prompts:
      - "Reply with: second"
EOF_DETERMINISTIC_2

if DEVX_EXPERIMENT_RUN_ID=deterministic-run \
  DEVX_TEST_LAUNCH_LOG="${TEST_ROOT}/launch-deterministic-2.log" \
  PATH="${TEST_ROOT}/bin:${PATH}" \
  DEVX_LAUNCH_CONTAINER_SH="${TEST_ROOT}/fake_launch_container.sh" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" experiment run -f "${TEST_ROOT}/experiment-deterministic-2.yaml" \
  >"${TEST_ROOT}/deterministic-2.out" 2>"${TEST_ROOT}/deterministic-2.err"; then
  echo "duplicate deterministic run id should be rejected" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/deterministic-2.err" "experiment run: run directory already exists with artifacts:"
assert_file_not_exists "${TEST_ROOT}/launch-deterministic-2.log"
assert_contains "${REPO_ROOT}/devx/runs/deterministic-run/results.jsonl" '"prompt":"Reply with: first"'

echo "PASS"
