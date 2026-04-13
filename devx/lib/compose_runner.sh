#!/bin/bash

# This file is sourced by other scripts.
# It intentionally avoids setting shell options (e.g. set -euo pipefail).

_compose_runner_init_override_context() {
  # These globals are part of the public API for this helper.
  # Declare them defensively to make the implicit contract explicit.
  # NOTE: avoid `declare -g` so callers can keep these arrays local
  # (bash uses dynamic scoping, so locals remain visible here).
  if ! declare -p OVERRIDE_VOLUMES >/dev/null 2>&1; then
    OVERRIDE_VOLUMES=()
  fi
  if ! declare -p OVERRIDE_ENV_KEYS >/dev/null 2>&1; then
    OVERRIDE_ENV_KEYS=()
  fi
  if ! declare -p OVERRIDE_ENV_VALUES >/dev/null 2>&1; then
    OVERRIDE_ENV_VALUES=()
  fi
}

_compose_runner_validate_override_context() {
  _compose_runner_init_override_context

  if [[ ${#OVERRIDE_ENV_KEYS[@]} -ne ${#OVERRIDE_ENV_VALUES[@]} ]]; then
    echo "compose_runner: OVERRIDE_ENV_KEYS and OVERRIDE_ENV_VALUES must have the same length" >&2
    return 1
  fi
}

_compose_runner_reset_state() {
  _compose_runner_init_override_context

  OVERRIDE_VOLUMES=()
  OVERRIDE_ENV_KEYS=()
  OVERRIDE_ENV_VALUES=()

  local override_file="${OPTIONAL_COMPOSE_FILE:-}"
  if [[ -n "${override_file}" ]]; then
    rm -f "${override_file}" 2>/dev/null || true
    unset OPTIONAL_COMPOSE_FILE || true
  fi
}

append_override_volume() {
  _compose_runner_init_override_context
  OVERRIDE_VOLUMES+=("$1")
}

append_override_env_var() {
  _compose_runner_init_override_context
  OVERRIDE_ENV_KEYS+=("$1")
  OVERRIDE_ENV_VALUES+=("$2")
}

quote_yaml_string() {
  local value="$1"

  value=${value//\'/\'\'}
  printf "'%s'" "${value}"
}

write_yaml_string_list_item() {
  local indent="$1"
  local value="$2"

  printf "%s- %s\n" "${indent}" "$(quote_yaml_string "${value}")"
}

build_override_file() {
  _compose_runner_validate_override_context

  if [[ ${#OVERRIDE_VOLUMES[@]} -eq 0 && ${#OVERRIDE_ENV_KEYS[@]} -eq 0 ]]; then
    cleanup_compose_runner
    return 0
  fi

  if [[ -z "${OPTIONAL_COMPOSE_FILE:-}" || ! -f "${OPTIONAL_COMPOSE_FILE}" ]]; then
    OPTIONAL_COMPOSE_FILE="$(mktemp "${TMPDIR:-/tmp}/devx-compose-override.XXXXXX.yaml")"
  fi

  {
    echo "services:"
    echo "  devshell:"
    if [[ ${#OVERRIDE_VOLUMES[@]} -gt 0 ]]; then
      echo "    volumes:"
      local volume
      for volume in "${OVERRIDE_VOLUMES[@]}"; do
        write_yaml_string_list_item "      " "${volume}"
      done
    fi
    if [[ ${#OVERRIDE_ENV_KEYS[@]} -gt 0 ]]; then
      echo "    environment:"
      local env_index
      for env_index in "${!OVERRIDE_ENV_KEYS[@]}"; do
        write_yaml_string_list_item "      " "${OVERRIDE_ENV_KEYS[env_index]}=${OVERRIDE_ENV_VALUES[env_index]}"
      done
    fi
  } >"${OPTIONAL_COMPOSE_FILE}"
}

cleanup_compose_runner() {
  _compose_runner_reset_state
}

resolve_compose_files() {
  local compose_file="$1"
  local -a compose_files=( -f "${compose_file}" )

  if [[ -n "${OPTIONAL_COMPOSE_FILE:-}" ]]; then
    compose_files+=( -f "${OPTIONAL_COMPOSE_FILE}" )
  fi

  printf '%s\n' "${compose_files[@]}"
}

resolve_compose_args() {
  local action="${1:-up}"
  shift || true

  case "${action}" in
    up)
      if [[ $# -eq 0 ]]; then
        printf '%s\n' up --build -d
      else
        printf '%s\n' up "$@"
      fi
      ;;
    shell)
      printf '%s\n' exec devshell /bin/bash
      ;;
    *)
      printf '%s\n' "${action}" "$@"
      ;;
  esac
}

run_compose_action() {
  local compose_project_name="$1"
  local compose_file="$2"
  local action="${3:-up}"
  shift 3 || true

  local -a compose_files
  mapfile -t compose_files < <(resolve_compose_files "${compose_file}")

  local -a compose_args
  mapfile -t compose_args < <(resolve_compose_args "${action}" "$@")

  docker compose --project-name "${compose_project_name}" "${compose_files[@]}" "${compose_args[@]}"
}

execute_compose_runner() {
  local script_dir="$1"
  shift

  local action="${1:-up}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  if [[ "${action}" == "smoke-harness" ]]; then
    bash "${REPO_ROOT}/tests/devx/operator_checklist.sh" "$@"
    exit 0
  fi

  if [[ "${action}" == "verify-e2e" ]]; then
    bash "${REPO_ROOT}/devx/verify_e2e.sh" "$@"
    return $?
  fi

  local config_render_only="false"
  if [[ "${action}" == "config" ]]; then
    config_render_only="true"
  fi

  COMPOSE_FILE="${COMPOSE_FILE:-${script_dir}/compose.yaml}"
  
  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "Compose file not found: ${COMPOSE_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${AUTHORIZED_KEYS_PATH}" ]]; then
    echo "authorized_keys file not found: ${AUTHORIZED_KEYS_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${SSHD_CONFIG_PATH}" ]]; then
    echo "sshd_config file not found: ${SSHD_CONFIG_PATH}" >&2
    exit 1
  fi

  if [[ "${config_render_only}" != "true" ]]; then
    ensure_host_mount_dirs
  fi
  configure_host_mount_sources
  export_host_mount_exports

  local env_index
  for env_index in "${!HOST_MOUNT_OVERRIDE_ENV_KEYS[@]}"; do
    append_override_env_var "${HOST_MOUNT_OVERRIDE_ENV_KEYS[env_index]}" "${HOST_MOUNT_OVERRIDE_ENV_VALUES[env_index]}"
  done

  if [[ -n "${SSH_AUTH_SOCK:-}" && -S "${SSH_AUTH_SOCK}" ]]; then
    append_override_volume "${SSH_AUTH_SOCK}:${SSH_AUTH_SOCK}"
    append_override_env_var "SSH_AUTH_SOCK" "${SSH_AUTH_SOCK}"
  fi

  if [[ -n "${HOST_METRIC_SOCKET:-}" && -S "${HOST_METRIC_SOCKET}" ]]; then
    append_override_volume "${HOST_METRIC_SOCKET}:/tmp/metric.sock"
  fi

  if [[ "${IS_WORKTREE}" == "TRUE" ]]; then
    local worktree_git_dir="${DEVX_HOST_WORKSPACE_DIR}/.worktrees/${PROJECT_ID}"
    local worktree_git_file="${worktree_git_dir}/${WORKTREE_NAME}.git"
    if [[ "${config_render_only}" != "true" ]]; then
      install -d -m 755 "${worktree_git_dir}"
      printf 'gitdir: /repo_dot_git/worktrees/%s\n' "${WORKTREE_NAME}" >"${worktree_git_file}"
    fi
    append_override_volume "${MAIN_GIT_DIR}:/repo_dot_git"
    append_override_volume "${worktree_git_file}:/workspace/${PROJECT_ID}/.git:ro"
  fi

  build_override_file

  if [[ "${action}" == "up-rebuild" ]]; then
    run_compose_action "${COMPOSE_PROJECT_NAME}" "${COMPOSE_FILE}" build devshell
    run_compose_action "${COMPOSE_PROJECT_NAME}" "${COMPOSE_FILE}" build vllm-runtime
    run_compose_action "${COMPOSE_PROJECT_NAME}" "${COMPOSE_FILE}" up -d
  else
    run_compose_action "${COMPOSE_PROJECT_NAME}" "${COMPOSE_FILE}" "${action}" "$@"
  fi

  if [[ "${action}" == "up" || "${action}" == "up-rebuild" ]]; then
    local devx_ssh_cmd="ssh -o IdentitiesOnly=yes -o IdentityAgent=none -p ${SSH_PORT} ${CONTAINER_USER}@127.0.0.1"
    if [[ -r "${HOME}/.ssh/devx_access" ]]; then
      devx_ssh_cmd="ssh -o IdentitiesOnly=yes -i ${HOME}/.ssh/devx_access -p ${SSH_PORT} ${CONTAINER_USER}@127.0.0.1"
    fi

    cat <<PRINT_EOF

Developer stack is starting.
SSH login: ${devx_ssh_cmd}
Compose project: ${COMPOSE_PROJECT_NAME}
PRINT_EOF
  fi
}
