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

assert_line_contains() {
  local haystack_file="$1"
  local line_number="$2"
  local needle="$3"

  local actual
  actual="$(sed -n "${line_number}p" "${haystack_file}")"
  if ! grep -F -- "${needle}" >/dev/null 2>&1 <<<"${actual}"; then
    echo "unexpected line ${line_number}" >&2
    echo "missing: ${needle}" >&2
    echo "actual:   ${actual}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  fi
}

mkdir -p "${TEST_ROOT}/bin" "${TEST_ROOT}/home"

# shellcheck disable=SC1091
source "${REPO_ROOT}/devx/lib/naming.sh"

EXPECTED_PROJECT_ID="$(basename "${REPO_ROOT}")"
EXPECTED_STACK_NAME="$(compute_stack_name "${EXPECTED_PROJECT_ID}")"
EXPECTED_COMPOSE_PROJECT_NAME="$(compute_compose_project_name "${EXPECTED_STACK_NAME}")"
REAL_REPO_ROOT="${REPO_ROOT}"

FAKE_REPO_ROOT="${TEST_ROOT}/fake-repo"
mkdir -p "${FAKE_REPO_ROOT}/devx"
cat <<'EOF_VERIFY' >"${FAKE_REPO_ROOT}/devx/verify_e2e.sh"
#!/bin/bash
exit 37
EOF_VERIFY
chmod +x "${FAKE_REPO_ROOT}/devx/verify_e2e.sh"

cat <<'EOF_DOCKER' >"${TEST_ROOT}/bin/docker"
#!/bin/bash
set -euo pipefail

echo "$*" >> "__DOCKER_LOG__"

if [[ "${1:-}" == "compose" && "${2:-}" == "version" ]]; then
  exit 0
fi

if [[ "${1:-}" == "compose" ]]; then
  exit 0
fi

exit 0
EOF_DOCKER

sed -i "s|__DOCKER_LOG__|${TEST_ROOT}/docker.log|g" "${TEST_ROOT}/bin/docker"
chmod +x "${TEST_ROOT}/bin/docker"

if (
  REPO_ROOT="${FAKE_REPO_ROOT}"
  # shellcheck disable=SC1090
  source "${REAL_REPO_ROOT}/devx/lib/compose_runner.sh"
  execute_compose_runner "${REPO_ROOT}/devx" verify-e2e
); then
  echo "verify-e2e should propagate the verifier's non-zero exit status" >&2
  exit 1
else
  status=$?
  if [[ "${status}" -ne 37 ]]; then
    echo "unexpected verify-e2e exit status: ${status}" >&2
    exit 1
  fi
fi

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/launch_container.sh" up-rebuild \
  >"${TEST_ROOT}/stdout" \
  2>"${TEST_ROOT}/stderr"; then
  echo "up-rebuild should have completed successfully" >&2
  cat "${TEST_ROOT}/stderr" >&2
  exit 1
fi

assert_line_contains "${TEST_ROOT}/docker.log" 1 "compose version"
assert_line_contains "${TEST_ROOT}/docker.log" 2 "compose --project-name ${EXPECTED_COMPOSE_PROJECT_NAME} -f ${REPO_ROOT}/devx/compose.yaml"
assert_line_contains "${TEST_ROOT}/docker.log" 2 "build devshell"
assert_line_contains "${TEST_ROOT}/docker.log" 3 "compose --project-name ${EXPECTED_COMPOSE_PROJECT_NAME} -f ${REPO_ROOT}/devx/compose.yaml"
assert_line_contains "${TEST_ROOT}/docker.log" 3 "build vllm-runtime"
assert_line_contains "${TEST_ROOT}/docker.log" 4 "compose --project-name ${EXPECTED_COMPOSE_PROJECT_NAME} -f ${REPO_ROOT}/devx/compose.yaml"
assert_line_contains "${TEST_ROOT}/docker.log" 4 "up -d"
assert_contains "${TEST_ROOT}/stdout" "Developer stack is starting."
assert_contains "${TEST_ROOT}/stdout" "SSH login: ssh -p 2222 kvothe@127.0.0.1"
assert_contains "${TEST_ROOT}/stdout" "Compose project: ${EXPECTED_COMPOSE_PROJECT_NAME}"

echo "PASS"
