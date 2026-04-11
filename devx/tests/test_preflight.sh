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

mkdir -p "${TEST_ROOT}/bin" \
  "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

touch "${TEST_ROOT}/curl.log"

cat <<'EOF_DOCKER' > "${TEST_ROOT}/bin/docker"
#!/bin/bash
set -euo pipefail

case "${1:-}" in
  version)
    echo "Docker version 27.0.0"
    ;;
  compose)
    if [[ "${2:-}" == "version" ]]; then
      echo "Docker Compose version v2.27.0"
      exit 0
    fi
    echo "unexpected docker compose invocation: $*" >&2
    exit 1
    ;;
  *)
    echo "unexpected docker invocation: $*" >&2
    exit 1
    ;;
esac
EOF_DOCKER
chmod +x "${TEST_ROOT}/bin/docker"

cat <<'EOF_CURL' > "${TEST_ROOT}/bin/curl"
#!/bin/bash
set -euo pipefail

echo "$*" >> "__CURL_LOG__"

auth_header=""
args=("$@")
for ((i = 0; i < $#; i++)); do
  if [[ "${args[i]}" == "-H" ]]; then
    next_index=$((i + 1))
    auth_header="${args[next_index]:-}"
  fi
done

if [[ "${auth_header}" != "Authorization: Bearer hf_test_token" ]]; then
  echo "unexpected authorization header: ${auth_header}" >&2
  exit 1
fi

url="${args[$(( $# - 1 ))]}"
case "${url}" in
  *"https://huggingface.co/api/models/Qwen/Qwen3-0.6B"*)
    exit 0
    ;;
  *"https://huggingface.co/api/models/Qwen/Qwen3-4B"*)
    exit 0
    ;;
  *)
    echo "unexpected curl invocation: $*" >&2
    exit 1
    ;;
esac
EOF_CURL
sed -i "s|__CURL_LOG__|${TEST_ROOT}/curl.log|g" "${TEST_ROOT}/bin/curl"
chmod +x "${TEST_ROOT}/bin/curl"

cat <<'EOF_JQ' > "${TEST_ROOT}/bin/jq"
#!/bin/bash
set -euo pipefail

exit 0
EOF_JQ
chmod +x "${TEST_ROOT}/bin/jq"

run_doctor() {
  local home_dir="$1"
  shift

  PATH="${TEST_ROOT}/bin:${PATH}" \
    HOME="${home_dir}" \
    bash "${REPO_ROOT}/devx/bin/devx" doctor "$@"
}

assert_file_empty() {
  local path="$1"

  if [[ -s "${path}" ]]; then
    echo "expected empty file: ${path}" >&2
    cat "${path}" >&2
    exit 1
  fi
}

reset_curl_log() {
  : > "${TEST_ROOT}/curl.log"
}

token_dir="${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets"

reset_curl_log
if run_doctor "${TEST_ROOT}/home" \
  >"${TEST_ROOT}/missing.stdout" \
  2>"${TEST_ROOT}/missing.stderr"; then
  echo "doctor should fail when the token file is missing" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/missing.stderr" "missing or empty token file"
assert_file_empty "${TEST_ROOT}/curl.log"

printf '   \n\t\n' > "${token_dir}/huggingface_token"

reset_curl_log
if run_doctor "${TEST_ROOT}/home" \
  >"${TEST_ROOT}/whitespace.stdout" \
  2>"${TEST_ROOT}/whitespace.stderr"; then
  echo "doctor should fail when the token file is whitespace-only" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/whitespace.stderr" "missing or empty token file"
assert_file_empty "${TEST_ROOT}/curl.log"

printf 'hf_test_token\n' > "${token_dir}/huggingface_token"

reset_curl_log
if ! run_doctor "${TEST_ROOT}/home" \
  >"${TEST_ROOT}/default.stdout" \
  2>"${TEST_ROOT}/default.stderr"; then
  echo "doctor should pass with the default preset" >&2
  cat "${TEST_ROOT}/default.stderr" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/default.stdout" "PREFLIGHT PASS: model=Qwen/Qwen3-0.6B"
assert_contains "${TEST_ROOT}/curl.log" "Authorization: Bearer hf_test_token"
assert_contains "${TEST_ROOT}/curl.log" "https://huggingface.co/api/models/Qwen/Qwen3-0.6B"

printf 'hf_test_token\n' > "${token_dir}/huggingface_token"

reset_curl_log
if ! run_doctor "${TEST_ROOT}/home" --preset qwen3-0.6b \
  >"${TEST_ROOT}/explicit.stdout" \
  2>"${TEST_ROOT}/explicit.stderr"; then
  echo "doctor should pass with an explicit preset" >&2
  cat "${TEST_ROOT}/explicit.stderr" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/explicit.stdout" "PREFLIGHT PASS: model=Qwen/Qwen3-0.6B"
assert_contains "${TEST_ROOT}/curl.log" "Authorization: Bearer hf_test_token"

reset_curl_log
if run_doctor "${TEST_ROOT}/home" --preset does-not-exist \
  >"${TEST_ROOT}/invalid.stdout" \
  2>"${TEST_ROOT}/invalid.stderr"; then
  echo "doctor should fail for an invalid preset" >&2
  exit 1
fi
assert_contains "${TEST_ROOT}/invalid.stderr" "invalid preset: does-not-exist"
assert_file_empty "${TEST_ROOT}/curl.log"

echo "PASS"
