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

  grep -F "${needle}" "${haystack_file}" >/dev/null 2>&1 || {
    echo "missing expected output: ${needle}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  }
}

run_wrapper() {
  local stdout_file="$1"
  local stderr_file="$2"
  shift 2

  "$@" bash "${WRAPPER}" --help >"${stdout_file}" 2>"${stderr_file}"
}

REPO_DIR="${TEST_ROOT}/repo"
mkdir -p "${REPO_DIR}/devx/dynamo" "${REPO_DIR}/.venv/bin" "${TEST_ROOT}/bin" "${TEST_ROOT}/file-kv"
cp "${REPO_ROOT}/devx/dynamo/start_frontend.sh" "${REPO_DIR}/devx/dynamo/start_frontend.sh"
WRAPPER="${REPO_DIR}/devx/dynamo/start_frontend.sh"

cat <<'EOF_REPO_PYTHON' > "${REPO_DIR}/.venv/bin/python"
#!/bin/bash
if [[ "$1" == "-c" ]] && [[ "$2" == *"find_spec"* ]]; then
  exit 0
fi

if [[ "$1" == "-c" ]] && [[ "$2" == *"importlib.import_module"* ]]; then
  if [[ "$3" == "dynamo.frontend" ]]; then
    exit 0
  fi

  if [[ "$3" == "dynamo.vllm" ]]; then
    exit 1
  fi
fi

if [[ "$*" == *"-m dynamo.frontend"* ]]; then
  echo "BROKEN_SELECTED"
  exit 0
fi

exit 0
EOF_REPO_PYTHON
chmod +x "${REPO_DIR}/.venv/bin/python"

cat <<'EOF_BAD_OVERRIDE' > "${TEST_ROOT}/bin/bad-override-python"
#!/bin/bash
if [[ "$1" == "-c" ]] && [[ "$2" == *"importlib.import_module"* ]]; then
  if [[ "$3" == "dynamo.frontend" ]]; then
    exit 0
  fi

  if [[ "$3" == "dynamo.vllm" ]]; then
    exit 1
  fi
fi

exit 0
EOF_BAD_OVERRIDE
chmod +x "${TEST_ROOT}/bin/bad-override-python"

cat <<'EOF_PATH_PYTHON' > "${TEST_ROOT}/bin/python3"
#!/bin/bash
if [[ "$1" == "-c" ]] && [[ "$2" == *"importlib.import_module"* ]]; then
  exit 0
fi

if [[ "$1" == "-m" ]] && [[ "$2" == "dynamo.frontend" ]]; then
  echo "GOOD_SELECTED"
  exit 0
fi

exit 0
EOF_PATH_PYTHON
chmod +x "${TEST_ROOT}/bin/python3"

if ! DYNAMO_FILE_KV="${TEST_ROOT}/file-kv" \
  DYNAMO_FRONTEND_PORT=8000 \
  PATH="${TEST_ROOT}/bin:${PATH}" \
  REPO_ROOT="${REPO_DIR}" \
  run_wrapper "${TEST_ROOT}/stdout" "${TEST_ROOT}/stderr" env; then
  echo "frontend script did not fall back to PATH python3" >&2
  cat "${TEST_ROOT}/stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/stdout" "GOOD_SELECTED"
if grep -F "BROKEN_SELECTED" "${TEST_ROOT}/stdout" >/dev/null 2>&1; then
  echo "frontend script selected the discoverable-but-unimportable repo interpreter" >&2
  cat "${TEST_ROOT}/stdout" >&2
  exit 1
fi

if DYNAMO_FILE_KV="${TEST_ROOT}/file-kv" \
  DYNAMO_FRONTEND_PORT=8000 \
  DYNAMO_PYTHON_BIN="${TEST_ROOT}/bin/bad-override-python" \
  PATH="${TEST_ROOT}/bin:${PATH}" \
  REPO_ROOT="${REPO_DIR}" \
  run_wrapper "${TEST_ROOT}/override.stdout" "${TEST_ROOT}/override.stderr" env; then
  echo "frontend script unexpectedly fell back from a broken DYNAMO_PYTHON_BIN override" >&2
  cat "${TEST_ROOT}/override.stdout" >&2
  cat "${TEST_ROOT}/override.stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/override.stderr" "DYNAMO_PYTHON_BIN cannot import dynamo.vllm"

echo "PASS"
