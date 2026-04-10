#!/bin/bash

load_hf_token_env() {
  local script_dir hf_token_file hf_token

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck disable=SC1091
  source "${script_dir}/lib/host_mounts.sh"

  export_host_mount_context

  hf_token_file="${DEVX_HOST_SECRETS_DIR}/huggingface_token"

  if [[ -r "${hf_token_file}" ]]; then
    hf_token="$(tr -d '\r\n' < "${hf_token_file}")"
    if [[ -n "${hf_token}" ]]; then
      HF_TOKEN="${hf_token}"
      export HF_TOKEN
    fi
  fi
}

load_hf_token_env
