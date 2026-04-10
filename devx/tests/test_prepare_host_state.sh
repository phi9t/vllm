#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "${TEST_ROOT}"' EXIT

TEST_HOME="${TEST_ROOT}/home"
STATE_ROOT="${TEST_HOME}/.devx/special-circ-phi9t-vllm"
HF_CACHE_ROOT="${TEST_HOME}/.cache/huggingface"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

mkdir -p "${TEST_HOME}"

assert_exists() {
  local path="$1"
  [ -e "${path}" ] || {
    echo "missing path: ${path}" >&2
    exit 1
  }
}

assert_mode() {
  local expected="$1"
  local path="$2"
  local actual

  actual="$(stat -c '%a' "${path}")"
  [ "${actual}" = "${expected}" ] || {
    echo "unexpected mode for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  }
}

assert_owner() {
  local path="$1"
  local actual

  actual="$(stat -c '%u:%g' "${path}")"
  [ "${actual}" = "${HOST_UID}:${HOST_GID}" ] || {
    echo "unexpected owner for ${path}: expected ${HOST_UID}:${HOST_GID}, got ${actual}" >&2
    exit 1
  }
}

HOME="${TEST_HOME}" bash "${REPO_ROOT}/devx/prepare_host_state.sh"

assert_exists "${STATE_ROOT}/home"
assert_exists "${STATE_ROOT}/ssh-hostkeys"
assert_exists "${STATE_ROOT}/cache/uv"
assert_exists "${STATE_ROOT}/cache/pip"
assert_exists "${STATE_ROOT}/cache/bazel"
assert_exists "${HF_CACHE_ROOT}"

for service in main vllm-runtime dynamo; do
  assert_exists "${STATE_ROOT}/${service}/tmp"
  assert_exists "${STATE_ROOT}/${service}/var-tmp"
  assert_exists "${STATE_ROOT}/${service}/logs"
  assert_mode 1777 "${STATE_ROOT}/${service}/tmp"
  assert_mode 1777 "${STATE_ROOT}/${service}/var-tmp"
  assert_owner "${STATE_ROOT}/${service}/tmp"
  assert_owner "${STATE_ROOT}/${service}/var-tmp"
  assert_owner "${STATE_ROOT}/${service}/logs"
done

assert_mode 1777 "${STATE_ROOT}/main/tmp"
assert_mode 1777 "${STATE_ROOT}/vllm-runtime/tmp"
assert_mode 1777 "${STATE_ROOT}/dynamo/tmp"
assert_mode 1777 "${STATE_ROOT}/main/var-tmp"
assert_mode 1777 "${STATE_ROOT}/vllm-runtime/var-tmp"
assert_mode 1777 "${STATE_ROOT}/dynamo/var-tmp"

assert_owner "${STATE_ROOT}/home"
assert_owner "${STATE_ROOT}/ssh-hostkeys"
assert_owner "${STATE_ROOT}/cache/uv"
assert_owner "${STATE_ROOT}/cache/pip"
assert_owner "${STATE_ROOT}/cache/bazel"
assert_owner "${HF_CACHE_ROOT}"

printf 'hf_test_token\n' > "${STATE_ROOT}/secrets/huggingface_token"

HOME="${TEST_HOME}" bash -c "
  set -euo pipefail
  source '${REPO_ROOT}/devx/lib/naming.sh'
  export_identity_context '${REPO_ROOT}'
  export_naming_context '${REPO_ROOT}/devx'
  source '${REPO_ROOT}/devx/lib/host_mounts.sh'
  export_host_mount_context
  unset HF_TOKEN
  source '${REPO_ROOT}/devx/hf_token.env.sh'
  [ \"\${HF_TOKEN:-}\" = 'hf_test_token' ]
  source '${REPO_ROOT}/devx/hf_token.env.sh'
  [ \"\${HF_TOKEN:-}\" = 'hf_test_token' ]
"

echo "PASS"
