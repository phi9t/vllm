#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CONTAINER_USER:-}" ]]; then
  CONTAINER_USER="kvothe"
elif [[ "${CONTAINER_USER}" != "kvothe" ]]; then
  echo "prepare_host_state.sh: CONTAINER_USER must be kvothe (got: ${CONTAINER_USER})" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/host_mounts.sh"

export_host_mount_context
ensure_host_state_dirs

echo "Prepared host state under ${DEVX_HOST_STATE_ROOT}"
