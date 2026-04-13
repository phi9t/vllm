#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/devx/compose.yaml"
EXPECTED_MODEL="${DYNAMO_MODEL:-${VLLM_MODEL:-Qwen/Qwen3.5-7B-Instruct}}"
VLLM_RUNTIME_PORT="${VLLM_RUNTIME_PORT:-8000}"
DYNAMO_SYSTEM_PORT="${DYNAMO_SYSTEM_PORT:-8081}"
VERIFY_E2E_ATTEMPTS="${VERIFY_E2E_ATTEMPTS:-60}"
VERIFY_E2E_DELAY_SECONDS="${VERIFY_E2E_DELAY_SECONDS:-3}"
FAIL_REASON=""

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/naming.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/git_context.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/host_mounts.sh"

initialize_context() {
  if ! export_identity_context "${REPO_ROOT}"; then
    FAIL_REASON="unable to initialize identity context"
    echo "verify_e2e: ${FAIL_REASON}" >&2
    exit 1
  fi

  if ! export_naming_context "${SCRIPT_DIR}"; then
    FAIL_REASON="unable to initialize naming context"
    echo "verify_e2e: ${FAIL_REASON}" >&2
    exit 1
  fi

  if ! export_git_context "${REPO_ROOT}" "${PWD}"; then
    FAIL_REASON="unable to initialize git context"
    echo "verify_e2e: ${FAIL_REASON}" >&2
    exit 1
  fi

  if ! export_host_mount_context; then
    FAIL_REASON="unable to initialize host mount context"
    echo "verify_e2e: ${FAIL_REASON}" >&2
    exit 1
  fi
}

initialize_context

TOKEN_FILE="${DEVX_HOST_SECRETS_DIR}/huggingface_token"
DOCKER_COMPOSE=(docker compose --project-name "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}")

pass_checkpoint() {
  local checkpoint="$1"
  local message="$2"

  echo "CHECKPOINT ${checkpoint} PASS: ${message}" >&2
}

fail_checkpoint() {
  local checkpoint="$1"

  echo "CHECKPOINT ${checkpoint} FAIL: ${FAIL_REASON}" >&2
  exit 1
}

run_checkpoint() {
  local checkpoint="$1"
  local message="$2"
  shift 2

  if "$@"; then
    pass_checkpoint "${checkpoint}" "${message}"
    return 0
  fi

  fail_checkpoint "${checkpoint}"
}

check_token_file() {
  local token

  if [ ! -s "${TOKEN_FILE}" ]; then
    FAIL_REASON="missing or empty huggingface token file: ${TOKEN_FILE}"
    return 1
  fi

  token="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
  if [ -z "${token}" ]; then
    FAIL_REASON="missing or empty huggingface token file: ${TOKEN_FILE}"
    return 1
  fi

  export HF_TOKEN="${token}"
  return 0
}

check_docker_and_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    FAIL_REASON="docker is not installed or not on PATH"
    return 1
  fi

  if ! docker version >/dev/null 2>&1; then
    FAIL_REASON="docker is installed but unavailable"
    return 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    FAIL_REASON="docker compose is unavailable"
    return 1
  fi

  return 0
}

check_compose_services_visibility() {
  local services
  local required_service

  if ! services="$("${DOCKER_COMPOSE[@]}" ps --services 2>/dev/null)"; then
    FAIL_REASON="docker compose services are not visible"
    return 1
  fi

  for required_service in vllm-runtime dynamo-frontend dynamo-vllm-worker; do
    if ! grep -Fx "${required_service}" >/dev/null 2>&1 <<< "${services}"; then
      FAIL_REASON="docker compose service is not visible: ${required_service}"
      return 1
    fi
  done

  return 0
}

run_with_retries() {
  local name="$1"
  local max_tries="$2"
  local delay_seconds="$3"
  shift 3

  local attempt
  for attempt in $(seq 1 "${max_tries}"); do
    if "$@"; then
      return 0
    fi

    if (( attempt < max_tries )); then
      echo "${name}: retrying (${attempt}/${max_tries}) in ${delay_seconds}s" >&2
      sleep "${delay_seconds}"
    fi
  done

  return 1
}

check_vllm_runtime_health() {
  if ! run_with_retries "vllm-runtime /health" "${VERIFY_E2E_ATTEMPTS}" "${VERIFY_E2E_DELAY_SECONDS}" "${DOCKER_COMPOSE[@]}" exec -T vllm-runtime \
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${VLLM_RUNTIME_PORT}/health')" \
    >/dev/null 2>&1; then
    FAIL_REASON="vllm-runtime /health probe failed"
    return 1
  fi

  return 0
}

check_dynamo_worker_health() {
  if ! run_with_retries "dynamo-vllm-worker /health" "${VERIFY_E2E_ATTEMPTS}" "${VERIFY_E2E_DELAY_SECONDS}" "${DOCKER_COMPOSE[@]}" exec -T dynamo-vllm-worker \
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${DYNAMO_SYSTEM_PORT}/health')" \
    >/dev/null 2>&1; then
    FAIL_REASON="dynamo-vllm-worker /health probe failed"
    return 1
  fi

  return 0
}

check_frontend_routing() {
  if ! run_with_retries "dynamo frontend route/models/chat" "${VERIFY_E2E_ATTEMPTS}" "${VERIFY_E2E_DELAY_SECONDS}" "${DOCKER_COMPOSE[@]}" exec -T dynamo-frontend \
    env DYNAMO_EXPECTED_MODEL="${EXPECTED_MODEL}" \
    bash "/workspace/${PROJECT_ID}/devx/dynamo/start_backend_probe.sh" \
    >/dev/null 2>&1; then
    FAIL_REASON="dynamo frontend route/models/chat probe failed for ${EXPECTED_MODEL}"
    return 1
  fi

  return 0
}

run_checkpoint 1 "huggingface token file present" check_token_file
run_checkpoint 2 "docker and docker compose available" check_docker_and_compose
run_checkpoint 3 "compose services visible" check_compose_services_visibility
run_checkpoint 4 "vllm-runtime /health reachable" check_vllm_runtime_health
run_checkpoint 5 "dynamo-vllm-worker /health reachable" check_dynamo_worker_health
run_checkpoint 6 "dynamo frontend route/models/chat path verified for ${EXPECTED_MODEL}" check_frontend_routing
