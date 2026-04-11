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

WRAPPER="${REPO_ROOT}/devx/start_vllm_runtime.sh"
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

assert_not_contains() {
  local haystack_file="$1"
  local needle="$2"

  if grep -F "${needle}" "${haystack_file}" >/dev/null 2>&1; then
    echo "unexpected output: ${needle}" >&2
    echo "--- ${haystack_file} ---" >&2
    cat "${haystack_file}" >&2
    exit 1
  fi
}

run_wrapper() {
  local stdout_file="$1"
  local stderr_file="$2"
  shift 2

  "$@" bash "${WRAPPER}" >"${stdout_file}" 2>"${stderr_file}"
}

REPO_DIR="${TEST_ROOT}/repo"
mkdir -p "${REPO_DIR}/vllm"
printf 'print("hello")\n' > "${REPO_DIR}/vllm/__init__.py"

INSTALLED_PKG_DIR="${TEST_ROOT}/installed/vllm"
mkdir -p "${INSTALLED_PKG_DIR}/vllm_flash_attn"
cat <<'EOF_INIT' > "${INSTALLED_PKG_DIR}/__init__.py"
print("installed")
EOF_INIT
cat <<'EOF_INSTALLED' > "${INSTALLED_PKG_DIR}/_version.py"
__version__ = "installed"
EOF_INSTALLED
printf 'binary\n' > "${INSTALLED_PKG_DIR}/_C.abi3.so"
printf 'flash-attn\n' > "${INSTALLED_PKG_DIR}/vllm_flash_attn/_vllm_fa2_C.abi3.so"

OVERLAY_ROOT="${TEST_ROOT}/overlay"

WRITABLE_STDOUT="${TEST_ROOT}/writable.stdout"
WRITABLE_STDERR="${TEST_ROOT}/writable.stderr"
if VLLM_RUNTIME_DRY_RUN=1 \
  VLLM_RUNTIME_MODE=source-overlay \
  VLLM_RUNTIME_SOURCE_DIR="${REPO_DIR}" \
  VLLM_MODEL="Qwen/Qwen3-8B" \
  run_wrapper "${WRITABLE_STDOUT}" "${WRITABLE_STDERR}" env; then
  echo "source-overlay unexpectedly accepted a writable source tree" >&2
  exit 1
fi
assert_contains "${WRITABLE_STDERR}" "source-overlay requires a read-only source tree"

chmod -R a-w "${REPO_DIR}"

SOURCE_STDOUT="${TEST_ROOT}/source.stdout"
SOURCE_STDERR="${TEST_ROOT}/source.stderr"
VLLM_RUNTIME_DRY_RUN=1 \
VLLM_RUNTIME_MODE=source-overlay \
VLLM_RUNTIME_SOURCE_DIR="${REPO_DIR}" \
VLLM_RUNTIME_INSTALLED_PKG_DIR="${INSTALLED_PKG_DIR}" \
VLLM_RUNTIME_OVERLAY_ROOT="${OVERLAY_ROOT}" \
VLLM_MODEL="Qwen/Qwen3-8B" \
run_wrapper "${SOURCE_STDOUT}" "${SOURCE_STDERR}" env
assert_contains "${SOURCE_STDOUT}" "mode=source-overlay"
assert_contains "${SOURCE_STDOUT}" "installed=${INSTALLED_PKG_DIR}"
assert_contains "${SOURCE_STDOUT}" "overlay=${OVERLAY_ROOT}"
assert_contains "${SOURCE_STDOUT}" "PYTHONPATH=${OVERLAY_ROOT}"
assert_contains "${SOURCE_STDOUT}" "launch=python3 -m vllm.entrypoints.openai.api_server"
assert_contains "${SOURCE_STDERR}" "warning: HF_TOKEN is not set"
cmp -s "${REPO_DIR}/vllm/__init__.py" "${OVERLAY_ROOT}/vllm/__init__.py"
cmp -s "${INSTALLED_PKG_DIR}/_version.py" "${OVERLAY_ROOT}/vllm/_version.py"
cmp -s "${INSTALLED_PKG_DIR}/_C.abi3.so" "${OVERLAY_ROOT}/vllm/_C.abi3.so"
cmp -s "${INSTALLED_PKG_DIR}/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
  "${OVERLAY_ROOT}/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"

REQUIRED_STDOUT="${TEST_ROOT}/required.stdout"
REQUIRED_STDERR="${TEST_ROOT}/required.stderr"
if VLLM_RUNTIME_DRY_RUN=1 \
  VLLM_RUNTIME_MODE=source-overlay \
  VLLM_RUNTIME_REQUIRE_HF_TOKEN=1 \
  VLLM_RUNTIME_SOURCE_DIR="${REPO_DIR}" \
  VLLM_RUNTIME_INSTALLED_PKG_DIR="${INSTALLED_PKG_DIR}" \
  VLLM_RUNTIME_OVERLAY_ROOT="${OVERLAY_ROOT}" \
  run_wrapper "${REQUIRED_STDOUT}" "${REQUIRED_STDERR}" env; then
  echo "HF token requirement unexpectedly passed without a token" >&2
  exit 1
fi
assert_contains "${REQUIRED_STDERR}" "HF_TOKEN is required"

IMAGE_STDOUT="${TEST_ROOT}/image.stdout"
IMAGE_STDERR="${TEST_ROOT}/image.stderr"
VLLM_RUNTIME_DRY_RUN=1 \
VLLM_RUNTIME_MODE=image-native \
VLLM_RUNTIME_SOURCE_DIR="${REPO_DIR}" \
VLLM_MODEL="Qwen/Qwen3-8B" \
HF_TOKEN="hf_test_token" \
run_wrapper "${IMAGE_STDOUT}" "${IMAGE_STDERR}" env
assert_contains "${IMAGE_STDOUT}" "mode=image-native"
assert_not_contains "${IMAGE_STDOUT}" "PYTHONPATH="
assert_contains "${IMAGE_STDOUT}" "hf-token=present"
assert_not_contains "${IMAGE_STDERR}" "warning: HF_TOKEN is not set"

DEFAULT_IMAGE_STDOUT="${TEST_ROOT}/default-image.stdout"
DEFAULT_IMAGE_STDERR="${TEST_ROOT}/default-image.stderr"
VLLM_RUNTIME_DRY_RUN=1 \
VLLM_RUNTIME_MODE=image-native \
VLLM_RUNTIME_SOURCE_DIR="${REPO_DIR}" \
HF_TOKEN="hf_test_token" \
run_wrapper "${DEFAULT_IMAGE_STDOUT}" "${DEFAULT_IMAGE_STDERR}" env
assert_contains "${DEFAULT_IMAGE_STDOUT}" "model=Qwen/Qwen3.5-7B-Instruct"

echo "PASS"
