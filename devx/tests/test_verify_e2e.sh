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

assert_not_contains() {
  local haystack_file="$1"
  local needle="$2"

  if grep -F -- "${needle}" "${haystack_file}" >/dev/null 2>&1; then
    echo "unexpected output: ${needle}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  fi
}

reset_logs() {
  : > "${TEST_ROOT}/git.log"
  : > "${TEST_ROOT}/docker.log"
}

mkdir -p "${TEST_ROOT}/bin" "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

# shellcheck disable=SC1091
source "${REPO_ROOT}/devx/lib/naming.sh"

EXPECTED_PROJECT_ID="$(basename "${REPO_ROOT}")"
EXPECTED_STACK_NAME="$(compute_stack_name "${EXPECTED_PROJECT_ID}")"
EXPECTED_COMPOSE_PROJECT_NAME="$(compute_compose_project_name "${EXPECTED_STACK_NAME}")"

cat <<'EOF_GIT' > "${TEST_ROOT}/bin/git"
#!/bin/bash
set -euo pipefail

echo "$*" >> "__GIT_LOG__"

args=("$@")
if [[ "${args[0]:-}" == "-C" ]]; then
  shift 2
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--path-format=absolute" && "${3:-}" == "--git-common-dir" ]]; then
  echo "old git does not support --path-format=absolute" >&2
  exit 129
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--git-common-dir" ]]; then
  printf '.git\n'
  exit 0
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--git-dir" ]]; then
  printf '.git\n'
  exit 0
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--show-toplevel" ]]; then
  printf '__REPO_ROOT__\n'
  exit 0
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--is-inside-work-tree" ]]; then
  exit 0
fi

if [[ "${1:-}" == "rev-parse" && "${2:-}" == "--show-prefix" ]]; then
  printf ''
  exit 0
fi

exit 0
EOF_GIT

sed -i "s|__GIT_LOG__|${TEST_ROOT}/git.log|g" "${TEST_ROOT}/bin/git"
sed -i "s|__REPO_ROOT__|${REPO_ROOT}|g" "${TEST_ROOT}/bin/git"
chmod +x "${TEST_ROOT}/bin/git"

cat <<'EOF_DOCKER' > "${TEST_ROOT}/bin/docker"
#!/bin/bash
set -euo pipefail

echo "$*" >> "__DOCKER_LOG__"

if [[ "$*" == "version" ]]; then
  exit 0
fi

if [[ "$*" == "compose version" ]]; then
  exit 0
fi

if [[ "$*" == *" ps --services" ]]; then
  cat <<'EOF_SERVICES'
vllm-runtime
dynamo-frontend
dynamo-vllm-worker
EOF_SERVICES
  exit 0
fi

if [[ "$*" == *"compose --project-name __COMPOSE_PROJECT_NAME__ -f __COMPOSE_FILE__ exec -T vllm-runtime"* ]]; then
  exit 0
fi

if [[ "$*" == *"compose --project-name __COMPOSE_PROJECT_NAME__ -f __COMPOSE_FILE__ exec -T dynamo-vllm-worker"* ]]; then
  if [[ "${DYNAMO_TEST_SCENARIO:-}" == "worker-fail" ]]; then
    exit 1
  fi
  exit 0
fi

if [[ "$*" == *"compose --project-name __COMPOSE_PROJECT_NAME__ -f __COMPOSE_FILE__ exec -T dynamo-frontend"* ]]; then
  if [[ "${DYNAMO_TEST_SCENARIO:-}" == "frontend-fail" ]]; then
    exit 1
  fi
  exit 0
fi

exit 0
EOF_DOCKER

sed -i "s|__DOCKER_LOG__|${TEST_ROOT}/docker.log|g" "${TEST_ROOT}/bin/docker"
sed -i "s|__COMPOSE_FILE__|${REPO_ROOT}/devx/compose.yaml|g" "${TEST_ROOT}/bin/docker"
sed -i "s|__COMPOSE_PROJECT_NAME__|${EXPECTED_COMPOSE_PROJECT_NAME}|g" "${TEST_ROOT}/bin/docker"
chmod +x "${TEST_ROOT}/bin/docker"

if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/verify_e2e.sh" \
  >"${TEST_ROOT}/missing-token.stdout" \
  2>"${TEST_ROOT}/missing-token.stderr"; then
  echo "verify_e2e should fail when the huggingface token file is missing" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/missing-token.stderr" "CHECKPOINT 1 FAIL"
assert_contains "${TEST_ROOT}/missing-token.stderr" "huggingface_token"

printf 'hf_test_token\n' > "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

reset_logs
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DYNAMO_TEST_SCENARIO=success \
  bash "${REPO_ROOT}/devx/launch_container.sh" verify-e2e \
  >"${TEST_ROOT}/launch-success.stdout" \
  2>"${TEST_ROOT}/launch-success.stderr"; then
  true
else
  echo "launch_container verify-e2e should succeed with old git fallback" >&2
  cat "${TEST_ROOT}/launch-success.stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 1 PASS"
assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 2 PASS"
assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 3 PASS"
assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 4 PASS"
assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 5 PASS"
assert_contains "${TEST_ROOT}/launch-success.stderr" "CHECKPOINT 6 PASS"
assert_contains "${TEST_ROOT}/docker.log" "/workspace/${EXPECTED_PROJECT_ID}/devx/dynamo/start_backend_probe.sh"
assert_contains "${TEST_ROOT}/docker.log" "DYNAMO_EXPECTED_MODEL=Qwen/Qwen3.5-7B-Instruct"
assert_contains "${TEST_ROOT}/git.log" "rev-parse --path-format=absolute --git-common-dir"
assert_contains "${TEST_ROOT}/git.log" "rev-parse --git-common-dir"

reset_logs
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DYNAMO_TEST_SCENARIO=worker-fail \
  bash "${REPO_ROOT}/devx/verify_e2e.sh" \
  >"${TEST_ROOT}/worker-fail.stdout" \
  2>"${TEST_ROOT}/worker-fail.stderr"; then
  echo "verify_e2e should fail when the worker health checkpoint fails" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 1 PASS"
assert_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 2 PASS"
assert_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 3 PASS"
assert_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 4 PASS"
assert_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 5 FAIL"
assert_not_contains "${TEST_ROOT}/worker-fail.stderr" "CHECKPOINT 6"
assert_not_contains "${TEST_ROOT}/docker.log" "start_backend_probe.sh"

reset_logs
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  DYNAMO_TEST_SCENARIO=frontend-fail \
  bash "${REPO_ROOT}/devx/verify_e2e.sh" \
  >"${TEST_ROOT}/frontend-fail.stdout" \
  2>"${TEST_ROOT}/frontend-fail.stderr"; then
  echo "verify_e2e should fail when the frontend routing probe fails" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 1 PASS"
assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 2 PASS"
assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 3 PASS"
assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 4 PASS"
assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 5 PASS"
assert_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 6 FAIL"
assert_not_contains "${TEST_ROOT}/frontend-fail.stderr" "CHECKPOINT 6 PASS"
assert_contains "${TEST_ROOT}/docker.log" "start_backend_probe.sh"

mkdir -p "${TEST_ROOT}/invalid-hdfs"
reset_logs
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  LOCAL_HDFS_CONF_DIR="${TEST_ROOT}/invalid-hdfs" \
  DYNAMO_TEST_SCENARIO=success \
  bash "${REPO_ROOT}/devx/verify_e2e.sh" \
  >"${TEST_ROOT}/invalid-hdfs.stdout" \
  2>"${TEST_ROOT}/invalid-hdfs.stderr"; then
  true
else
  echo "verify_e2e should ignore invalid LOCAL_HDFS_CONF_DIR during verification" >&2
  cat "${TEST_ROOT}/invalid-hdfs.stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/invalid-hdfs.stderr" "CHECKPOINT 6 PASS"
assert_not_contains "${TEST_ROOT}/invalid-hdfs.stderr" "host_mounts: LOCAL_HDFS_CONF_DIR"

reset_logs
if PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  VLLM_MODEL="ignored/model" \
  DYNAMO_MODEL="custom/model" \
  DYNAMO_TEST_SCENARIO=success \
  bash "${REPO_ROOT}/devx/verify_e2e.sh" \
  >"${TEST_ROOT}/dynamo-model.stdout" \
  2>"${TEST_ROOT}/dynamo-model.stderr"; then
  true
else
  echo "verify_e2e should pass with a DYNAMO_MODEL override" >&2
  cat "${TEST_ROOT}/dynamo-model.stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/dynamo-model.stderr" "CHECKPOINT 6 PASS: dynamo frontend route/models/chat path verified for custom/model"
assert_contains "${TEST_ROOT}/docker.log" "DYNAMO_EXPECTED_MODEL=custom/model"
assert_not_contains "${TEST_ROOT}/docker.log" "DYNAMO_EXPECTED_MODEL=ignored/model"

echo "PASS"
