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

url="${@: -1}"
case "${url}" in
  *"https://huggingface.co/api/models/Qwen/Qwen3-0.6B"*)
    exit 0
    ;;
  *)
    echo "unexpected curl invocation: $*" >&2
    exit 1
    ;;
esac
EOF_CURL
chmod +x "${TEST_ROOT}/bin/curl"

printf 'hf_test_token\n' > "${TEST_ROOT}/home/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"

if ! PATH="${TEST_ROOT}/bin:${PATH}" \
  HOME="${TEST_ROOT}/home" \
  bash "${REPO_ROOT}/devx/bin/devx" doctor --preset qwen3-0.6b \
  >"${TEST_ROOT}/stdout" \
  2>"${TEST_ROOT}/stderr"; then
  echo "doctor should pass once preflight is implemented" >&2
  cat "${TEST_ROOT}/stderr" >&2
  exit 1
fi

assert_contains "${TEST_ROOT}/stdout" "PREFLIGHT PASS: model=Qwen/Qwen3-0.6B"

echo "PASS"
