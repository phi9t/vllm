#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/host_mounts.sh"

export_host_mount_context

HF_TOKEN_FILE="${DEVX_HOST_SECRETS_DIR}/huggingface_token"

if [[ -r "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
  if [[ -n "${HF_TOKEN}" ]]; then
    export HF_TOKEN
  fi
fi
