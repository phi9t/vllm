#!/bin/bash

normalize_prefixed_name() {
  local value="$1"

  if [[ "${value}" == phi9t-* ]]; then
    printf '%s\n' "${value}"
  else
    printf 'phi9t-%s\n' "${value}"
  fi
}

compute_project_id() {
  local repo_root="$1"
  local project_id_override="${2:-}"

  if [[ -n "${project_id_override}" ]]; then
    printf '%s\n' "${project_id_override}"
    return 0
  fi

  local git_common_dir
  if git_common_dir="$(git -C "${repo_root}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    printf '%s\n' "$(basename "$(dirname "${git_common_dir}")")"
    return 0
  fi

  if git_common_dir="$(git -C "${repo_root}" rev-parse --git-common-dir 2>/dev/null)"; then
    local git_common_dir_abs

    if [[ "${git_common_dir}" == /* ]]; then
      git_common_dir_abs="${git_common_dir}"
    else
      git_common_dir_abs="$(cd "${repo_root}" && cd "$(dirname "${git_common_dir}")" && pwd -P)/$(basename "${git_common_dir}")"
    fi

    printf '%s\n' "$(basename "$(dirname "${git_common_dir_abs}")")"
    return 0
  fi

  echo "naming: unable to resolve git common-dir for ${repo_root}" >&2
  return 1
}

compute_stack_name() {
  local project_id="$1"
  local stack_name_override="${2:-}"

  normalize_prefixed_name "${stack_name_override:-${project_id//_/-}-devx}"
}

compute_container_user() {
  local container_user_override="${1:-}"

  printf '%s\n' "${container_user_override:-kvothe}"
}

compute_compose_project_name() {
  local stack_name="$1"
  local compose_project_name_override="${2:-}"

  normalize_prefixed_name "${compose_project_name_override:-${stack_name}}"
}

compute_user_uid() {
  id -u
}

compute_user_gid() {
  id -g
}

export_identity_context() {
  local repo_root="$1"
  local project_id
  local stack_name
  local container_user
  local compose_project_name
  local user_uid
  local user_gid

  project_id="$(compute_project_id "${repo_root}" "${PROJECT_ID:-}")"
  stack_name="$(compute_stack_name "${project_id}" "${STACK_NAME:-}")"
  container_user="$(compute_container_user "${CONTAINER_USER:-}")"
  compose_project_name="$(compute_compose_project_name "${stack_name}" "${COMPOSE_PROJECT_NAME:-}")"
  user_uid="$(compute_user_uid)"
  user_gid="$(compute_user_gid)"

  PROJECT_ID="${project_id}"
  readonly PROJECT_ID

  STACK_NAME="${stack_name}"
  readonly STACK_NAME

  CONTAINER_USER="${container_user}"
  readonly CONTAINER_USER

  COMPOSE_PROJECT_NAME="${compose_project_name}"
  readonly COMPOSE_PROJECT_NAME

  USER_UID="${user_uid}"
  USER_GID="${user_gid}"
  readonly USER_UID USER_GID

  export PROJECT_ID
  export STACK_NAME
  export CONTAINER_USER
  export COMPOSE_PROJECT_NAME
  export USER_UID
  export USER_GID
}

# Additional variables that were in launch_container.sh
export_naming_context() {
  local script_dir="$1"

  DEVX_BASE_IMAGE="${DEVX_BASE_IMAGE:-vllm/vllm-openai:latest}"
  SSH_PORT="${SSH_PORT:-2222}"
  SGLANG_PORT="${SGLANG_PORT:-30001}"
  OLLAMA_PORT="${OLLAMA_PORT:-11434}"

  local local_authorized_keys="${script_dir}/authorized_keys"
  if [[ -f "${local_authorized_keys}" ]]; then
    AUTHORIZED_KEYS_SOURCE="${AUTHORIZED_KEYS_SOURCE:-${local_authorized_keys}}"
  else
    AUTHORIZED_KEYS_SOURCE="${AUTHORIZED_KEYS_SOURCE:-${script_dir}/authorized_keys.placeholder}"
  fi
  SSHD_CONFIG_SOURCE="${SSHD_CONFIG_SOURCE:-${script_dir}/sshd_config}"

  if [[ "${CONTAINER_USER:-}" != "kvothe" ]]; then
    echo "CONTAINER_USER is fixed to 'kvothe' for the lean devx runtime (got: ${CONTAINER_USER:-})" >&2
    exit 1
  fi

  # Explicit host identity contract for runtime writable roots.
  # (Matches the user identity used to create the host-backed mount dirs.)
  HOST_UID="${USER_UID}"
  HOST_GID="${USER_GID}"
  HOST_USER_NAME="$(id -un)"

  export DEVX_BASE_IMAGE
  export SSH_PORT
  export SGLANG_PORT
  export OLLAMA_PORT
  export AUTHORIZED_KEYS_PATH="${AUTHORIZED_KEYS_SOURCE}"
  export SSHD_CONFIG_PATH="${SSHD_CONFIG_SOURCE}"
  export HOST_UID
  export HOST_GID
  export HOST_USER_NAME
}
