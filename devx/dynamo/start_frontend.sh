#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd /tmp

die() {
  echo "start_frontend.sh: $*" >&2
  exit 1
}

resolve_python_bin() {
  if [ -n "${DYNAMO_PYTHON_BIN:-}" ]; then
    [ -x "${DYNAMO_PYTHON_BIN}" ] || die "DYNAMO_PYTHON_BIN is not executable: ${DYNAMO_PYTHON_BIN}"
    printf '%s\n' "${DYNAMO_PYTHON_BIN}"
    return 0
  fi

  local repo_python="${REPO_ROOT}/.venv/bin/python"
  if [ -x "${repo_python}" ]; then
    printf '%s\n' "${repo_python}"
    return 0
  fi

  local path_python
  for path_python in python3 python; do
    if command -v "${path_python}" >/dev/null 2>&1; then
      command -v "${path_python}"
      return 0
    fi
  done

  die "set DYNAMO_PYTHON_BIN or install python3/python with ai-dynamo on PATH; no repo-local interpreter found at ${repo_python}"
}

validate_file_kv_dir() {
  local dir_path="$1"
  local probe_file

  [ -n "${dir_path}" ] || die "DYNAMO_FILE_KV must be set to the shared file-discovery directory"
  [ -d "${dir_path}" ] || die "DYNAMO_FILE_KV does not exist or is not a directory: ${dir_path}"
  [ -r "${dir_path}" ] || die "DYNAMO_FILE_KV is not readable: ${dir_path}"
  [ -x "${dir_path}" ] || die "DYNAMO_FILE_KV is not searchable/executable as a directory: ${dir_path}"
  [ -w "${dir_path}" ] || die "DYNAMO_FILE_KV is not writable: ${dir_path}"

  probe_file="$(mktemp "${dir_path%/}/.dynamo-file-kv-probe.XXXXXX")" \
    || die "DYNAMO_FILE_KV is not usable for shared file discovery writes: ${dir_path}"
  rm -f "${probe_file}" \
    || die "DYNAMO_FILE_KV probe cleanup failed; directory may be misconfigured: ${dir_path}"
}

require_python_module() {
  local python_bin="$1"
  local module_name="$2"

  "${python_bin}" - "${module_name}" <<'PY'
import importlib.util
import sys

module_name = sys.argv[1]
if importlib.util.find_spec(module_name) is None:
    raise SystemExit(f"missing module: {module_name}")
PY
}

PYTHON_BIN="$(resolve_python_bin)"
DYNAMO_DISCOVERY_BACKEND="${DYNAMO_DISCOVERY_BACKEND:-file}"
DYNAMO_ROUTER_MODE="${DYNAMO_ROUTER_MODE:-round-robin}"
DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-dynamo}"
DYNAMO_FRONTEND_HOST="${DYNAMO_FRONTEND_HOST:-0.0.0.0}"
DYNAMO_FRONTEND_PORT="${DYNAMO_FRONTEND_PORT:-8000}"
DYNAMO_FILE_KV="${DYNAMO_FILE_KV:-}"

[ "${DYNAMO_DISCOVERY_BACKEND}" = "file" ] || die "pinned local v1 topology only supports DYNAMO_DISCOVERY_BACKEND=file"
[ "${DYNAMO_ROUTER_MODE}" = "round-robin" ] || die "pinned local v1 topology only supports DYNAMO_ROUTER_MODE=round-robin"
validate_file_kv_dir "${DYNAMO_FILE_KV}"

if ! require_python_module "${PYTHON_BIN}" "dynamo.frontend"; then
  die "upstream packaging is incompatible: ${PYTHON_BIN} cannot import dynamo.frontend"
fi

if ! require_python_module "${PYTHON_BIN}" "dynamo.vllm"; then
  die "upstream packaging is incompatible: ${PYTHON_BIN} cannot import dynamo.vllm"
fi

cat <<EOF
Pinned Dynamo local v1 topology
  frontend process : python -m dynamo.frontend
  backend contract : one dynamo.vllm worker in vllm-runtime
  discovery        : file
  router mode      : round-robin
  namespace        : ${DYNAMO_NAMESPACE}
  file store       : ${DYNAMO_FILE_KV}
  listen           : ${DYNAMO_FRONTEND_HOST}:${DYNAMO_FRONTEND_PORT}
EOF

export DYN_DISCOVERY_BACKEND="file"
export DYN_FILE_KV="${DYNAMO_FILE_KV}"
export DYN_NAMESPACE="${DYNAMO_NAMESPACE}"
export DYN_ROUTER_MODE="round-robin"
export DYN_HTTP_HOST="${DYNAMO_FRONTEND_HOST}"
export DYN_HTTP_PORT="${DYNAMO_FRONTEND_PORT}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME:-/tmp}/.config}"
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

exec "${PYTHON_BIN}" -m dynamo.frontend \
  --discovery-backend file \
  --router-mode round-robin \
  --namespace "${DYNAMO_NAMESPACE}" \
  --http-host "${DYNAMO_FRONTEND_HOST}" \
  --http-port "${DYNAMO_FRONTEND_PORT}" \
  "$@"
