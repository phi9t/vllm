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

PRESET_SCRIPT="${REPO_ROOT}/devx/lib/presets.sh"

if bash "${REPO_ROOT}/devx/bin/devx" help >"${TEST_ROOT}/help.out" 2>"${TEST_ROOT}/help.err"; then
  assert_contains "${TEST_ROOT}/help.out" "Usage: devx"
else
  echo "devx help should succeed once dispatcher exists" >&2
  exit 1
fi

if bash "${REPO_ROOT}/devx/bin/devx" unknown >"${TEST_ROOT}/unknown.out" 2>"${TEST_ROOT}/unknown.err"; then
  echo "unknown command must fail" >&2
  exit 1
else
  unknown_exit_code=$?
fi
if [[ "${unknown_exit_code}" -ne 1 ]]; then
  echo "unknown command must exit 1, got ${unknown_exit_code}" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/unknown.err" "unknown subcommand"

for cmd in doctor up query switch experiment; do
  stdout_file="${TEST_ROOT}/${cmd}.out"
  stderr_file="${TEST_ROOT}/${cmd}.err"
  if bash "${REPO_ROOT}/devx/bin/devx" "${cmd}" >"${stdout_file}" 2>"${stderr_file}"; then
    echo "${cmd} must fail until implemented" >&2
    exit 1
  else
    cmd_exit_code=$?
  fi
  if [[ "${cmd_exit_code}" -ne 2 ]]; then
    echo "${cmd} must exit 2, got ${cmd_exit_code}" >&2
    exit 1
  fi
  assert_contains "${stderr_file}" "not yet implemented: ${cmd}"
  assert_not_contains "${stdout_file}" "not yet implemented: ${cmd}"
done

# shellcheck disable=SC1090
source "${PRESET_SCRIPT}"

printf '%s\n' "${DEVX_PRESETS[@]}" >"${TEST_ROOT}/presets.txt"
assert_contains "${TEST_ROOT}/presets.txt" "qwen3-0.6b"
assert_contains "${TEST_ROOT}/presets.txt" "qwen3-4b"

for preset in "${DEVX_PRESETS[@]}"; do
  model_id="$(preset_model_id "${preset}")"
  case "${preset}" in
    qwen3-0.6b)
      [[ "${model_id}" == "Qwen/Qwen3-0.6B" ]] || {
        echo "preset_model_id ${preset} returned the wrong model id" >&2
        exit 1
      }
      ;;
    qwen3-4b)
      [[ "${model_id}" == "Qwen/Qwen3-4B" ]] || {
        echo "preset_model_id ${preset} returned the wrong model id" >&2
        exit 1
      }
      ;;
    *)
      echo "unexpected preset in DEVX_PRESETS: ${preset}" >&2
      exit 1
      ;;
  esac
done

if preset_model_id does-not-exist >/dev/null 2>&1; then
  echo "preset_model_id must fail for unknown presets" >&2
  exit 1
fi

if ! require_preset qwen3-0.6b >/dev/null 2>"${TEST_ROOT}/require.ok.err"; then
  echo "require_preset should accept known presets" >&2
  exit 1
fi

if require_preset does-not-exist >"${TEST_ROOT}/require.fail.out" 2>"${TEST_ROOT}/require.fail.err"; then
  echo "require_preset must fail for unknown presets" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/require.fail.err" "invalid preset: does-not-exist"

echo "PASS"
