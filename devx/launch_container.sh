#!/bin/bash
# -*- shell-script -*-

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/naming.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/git_context.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/host_mounts.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/compose_runner.sh"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

cleanup() {
  # Intentionally do NOT delete the worktree `.git` shim file.
  # It is a bind-mount source for the running container, and deleting it on
  # exit can break restarts/recreates that re-resolve bind sources.
  cleanup_compose_runner
}

trap cleanup EXIT

require_command git
require_command docker
docker compose version >/dev/null 2>&1 || { echo "docker compose is required" >&2; exit 1; }

REPO_ROOT="$(resolve_repo_root "${SCRIPT_DIR}/..")"
export REPO_ROOT
readonly REPO_ROOT

export_identity_context "${REPO_ROOT}"
export_naming_context "${SCRIPT_DIR}"
export_git_context "${REPO_ROOT}" "${PWD}"
export_host_mount_context
execute_compose_runner "${SCRIPT_DIR}" "$@"
