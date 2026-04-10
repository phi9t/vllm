#!/bin/bash

set -euo pipefail

MODE="${VLLM_RUNTIME_MODE:-source-overlay}"
SOURCE_DIR="${VLLM_RUNTIME_SOURCE_DIR:-${VLLM_SOURCE_ROOT:-}}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
HOST="${VLLM_RUNTIME_HOST:-0.0.0.0}"
PORT="${VLLM_RUNTIME_PORT:-8000}"
DRY_RUN="${VLLM_RUNTIME_DRY_RUN:-0}"
REQUIRE_HF_TOKEN="${VLLM_RUNTIME_REQUIRE_HF_TOKEN:-0}"
OVERLAY_ROOT="${VLLM_RUNTIME_OVERLAY_ROOT:-/var/tmp/vllm-runtime-overlay}"
INSTALLED_PKG_DIR="${VLLM_RUNTIME_INSTALLED_PKG_DIR:-}"

warn() {
  echo "warning: $*" >&2
}

die() {
  echo "$*" >&2
  exit 1
}

is_read_only_tree() {
  local path="$1"

  [ -d "${path}" ] || return 1
  [ ! -w "${path}" ]
}

resolve_installed_pkg_dir() {
  if [ -n "${INSTALLED_PKG_DIR}" ]; then
    printf '%s\n' "${INSTALLED_PKG_DIR}"
    return 0
  fi

  python3 - <<'PY'
import site
from pathlib import Path

site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except AttributeError:
    pass

user_site = site.getusersitepackages()
if user_site:
    site_dirs.append(user_site)

for base in site_dirs:
    candidate = Path(base) / "vllm"
    if (candidate / "__init__.py").is_file():
        print(candidate)
        break
else:
    raise SystemExit(1)
PY
}

build_source_overlay() {
  local source_pkg_dir="$1"
  local installed_pkg_dir="$2"
  local overlay_root="$3"

  OVERLAY_SOURCE_PKG_DIR="${source_pkg_dir}" \
  OVERLAY_INSTALLED_PKG_DIR="${installed_pkg_dir}" \
  OVERLAY_ROOT_DIR="${overlay_root}" \
    python3 - <<'PY'
import os
import shutil
from pathlib import Path

source_pkg = Path(os.environ["OVERLAY_SOURCE_PKG_DIR"]).resolve()
installed_pkg = Path(os.environ["OVERLAY_INSTALLED_PKG_DIR"]).resolve()
overlay_root = Path(os.environ["OVERLAY_ROOT_DIR"]).resolve()
overlay_pkg = overlay_root / "vllm"

if overlay_root.exists():
    shutil.rmtree(overlay_root)
overlay_pkg.mkdir(parents=True, exist_ok=True)

for path in sorted(installed_pkg.rglob("*")):
    rel = path.relative_to(installed_pkg)
    dest = overlay_pkg / rel
    if path.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(path)

for path in sorted(source_pkg.rglob("*")):
    rel = path.relative_to(source_pkg)
    dest = overlay_pkg / rel
    if path.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(path)
PY
}

describe_launch() {
  printf '%s\n' \
    "launch=python3 -m vllm.entrypoints.openai.api_server --host ${HOST} --port ${PORT} --model ${MODEL}"
}

case "${MODE}" in
  source-overlay|image-native) ;;
  *)
    die "unsupported runtime mode: ${MODE}"
    ;;
esac

if [ "${REQUIRE_HF_TOKEN}" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then
  die "HF_TOKEN is required"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  warn "HF_TOKEN is not set"
fi

if [ "${MODE}" = "source-overlay" ]; then
  local_source_pkg_dir="${SOURCE_DIR}/vllm"
  [ -n "${SOURCE_DIR}" ] || die "VLLM_RUNTIME_SOURCE_DIR must be set for source-overlay mode"
  [ -d "${SOURCE_DIR}" ] || die "source directory does not exist: ${SOURCE_DIR}"
  [ -r "${SOURCE_DIR}" ] || die "source directory is not readable: ${SOURCE_DIR}"
  [ -r "${local_source_pkg_dir}/__init__.py" ] || die "source directory is not a vLLM checkout: ${SOURCE_DIR}"
  is_read_only_tree "${SOURCE_DIR}" || die "source-overlay requires a read-only source tree"
  INSTALLED_PKG_DIR="$(resolve_installed_pkg_dir)" || die "unable to locate installed vllm package"
  [ -d "${INSTALLED_PKG_DIR}" ] || die "installed vllm package directory does not exist: ${INSTALLED_PKG_DIR}"
  [ -r "${INSTALLED_PKG_DIR}/__init__.py" ] || die "installed vllm package is invalid: ${INSTALLED_PKG_DIR}"
  build_source_overlay "${local_source_pkg_dir}" "${INSTALLED_PKG_DIR}" "${OVERLAY_ROOT}"
  export PYTHONPATH="${OVERLAY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [ "${DRY_RUN}" = "1" ]; then
  printf '%s\n' "mode=${MODE}"
  printf '%s\n' "model=${MODEL}"
  printf '%s\n' "hf-token=$( [ -n "${HF_TOKEN:-}" ] && echo present || echo absent )"
  if [ "${MODE}" = "source-overlay" ]; then
    printf '%s\n' "source=${SOURCE_DIR}"
    printf '%s\n' "installed=${INSTALLED_PKG_DIR}"
    printf '%s\n' "overlay=${OVERLAY_ROOT}"
    printf '%s\n' "PYTHONPATH=${PYTHONPATH}"
  fi
  describe_launch
  exit 0
fi

unset VLLM_MODEL
unset VLLM_RUNTIME_HOST
unset VLLM_RUNTIME_MODE
unset VLLM_RUNTIME_OVERLAY_ROOT
unset VLLM_RUNTIME_PORT
unset VLLM_RUNTIME_REQUIRE_HF_TOKEN
unset VLLM_RUNTIME_SOURCE_DIR
unset VLLM_RUNTIME_INSTALLED_PKG_DIR

cd /tmp

exec python3 -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}"
