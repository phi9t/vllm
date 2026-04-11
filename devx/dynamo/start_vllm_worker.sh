#!/bin/bash

set -euo pipefail

cd /tmp

die() {
  echo "start_vllm_worker.sh: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_file_kv_dir() {
  local dir_path="$1"
  local probe_file

  [ -n "${dir_path}" ] || die "DYNAMO_FILE_KV must be set to the shared file-discovery directory"
  [ -d "${dir_path}" ] || die "DYNAMO_FILE_KV does not exist or is not a directory: ${dir_path}"
  [ -r "${dir_path}" ] || die "DYNAMO_FILE_KV is not readable: ${dir_path}"
  [ -x "${dir_path}" ] || die "DYNAMO_FILE_KV is not searchable/executable as a directory: ${dir_path}"
  [ -w "${dir_path}" ] || die "DYNAMO_FILE_KV is not writable: ${dir_path}"

  probe_file="$(mktemp "${dir_path%/}/.dynamo-worker-file-kv-probe.XXXXXX")" \
    || die "DYNAMO_FILE_KV is not usable for shared file discovery writes: ${dir_path}"
  rm -f "${probe_file}" \
    || die "DYNAMO_FILE_KV probe cleanup failed; directory may be misconfigured: ${dir_path}"
}

require_command python3

DYNAMO_DISCOVERY_BACKEND="${DYNAMO_DISCOVERY_BACKEND:-file}"
DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-dynamo}"
DYNAMO_FILE_KV="${DYNAMO_FILE_KV:-}"
DYNAMO_MODEL="${DYNAMO_MODEL:-Qwen/Qwen3-8B}"
DYNAMO_SYSTEM_PORT="${DYNAMO_SYSTEM_PORT:-8081}"
DYNAMO_KV_EVENTS_CONFIG="${DYNAMO_KV_EVENTS_CONFIG:-{\"enable_kv_cache_events\": false}}"

[ "${DYNAMO_DISCOVERY_BACKEND}" = "file" ] || die "pinned local v1 topology only supports DYNAMO_DISCOVERY_BACKEND=file"
validate_file_kv_dir "${DYNAMO_FILE_KV}"
[[ "${DYNAMO_SYSTEM_PORT}" =~ ^[0-9]+$ ]] || die "DYNAMO_SYSTEM_PORT must be an integer"

python3 - <<'PY'
import importlib.util
import sys

missing = [
    module for module in ("dynamo.vllm", "vllm")
    if importlib.util.find_spec(module) is None
]
if missing:
    raise SystemExit(f"missing modules: {', '.join(missing)}")
PY

cat <<EOF
Pinned Dynamo local v1 worker
  worker process   : python -m dynamo.vllm
  discovery        : file
  namespace        : ${DYNAMO_NAMESPACE}
  file store       : ${DYNAMO_FILE_KV}
  model            : ${DYNAMO_MODEL}
  system port      : ${DYNAMO_SYSTEM_PORT}
  kv events config : ${DYNAMO_KV_EVENTS_CONFIG}
EOF

export DYN_DISCOVERY_BACKEND="file"
export DYN_FILE_KV="${DYNAMO_FILE_KV}"
export DYN_NAMESPACE="${DYNAMO_NAMESPACE}"
export DYN_SYSTEM_PORT="${DYNAMO_SYSTEM_PORT}"
export DYN_SYSTEM_STARTING_HEALTH_STATUS="notready"
export DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]'
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${HOME:-/tmp}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME:-/tmp}/.config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}/vllm" \
  "${XDG_CONFIG_HOME}"

exec python3 -m dynamo.vllm \
  --model "${DYNAMO_MODEL}" \
  --discovery-backend file \
  --namespace "${DYNAMO_NAMESPACE}" \
  --kv-events-config "${DYNAMO_KV_EVENTS_CONFIG}" \
  "$@"
