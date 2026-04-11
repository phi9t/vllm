#!/bin/bash
# shellcheck shell=bash

resolve_session_state_file() {
  local state_root="${DEVX_STATE_ROOT:-${DEVX_HOST_STATE_ROOT:-${HOME}/.devx/special-circ-phi9t-vllm}}"
  printf '%s\n' "${DEVX_STATE_FILE:-${state_root}/session_state.env}"
}

read_session_state_value() {
  local key="${1:-}"
  local state_file

  state_file="$(resolve_session_state_file)"
  if [[ ! -f "${state_file}" ]]; then
    return 1
  fi

  sed -n "s/^${key}=//p" "${state_file}" | head -n 1
}

read_session_active_preset() {
  read_session_state_value "active_preset"
}

read_session_updated_at_unix() {
  read_session_state_value "updated_at_unix"
}

write_session_state() {
  local active_preset="${1:-}"
  local updated_at_unix="${2:-}"
  local state_file
  local state_dir
  local tmp_file

  state_file="$(resolve_session_state_file)"
  state_dir="$(dirname "${state_file}")"

  mkdir -p "${state_dir}"
  tmp_file="$(mktemp "${state_dir}/.session_state.XXXXXX")"

  {
    printf 'active_preset=%s\n' "${active_preset}"
    printf 'updated_at_unix=%s\n' "${updated_at_unix}"
  } >"${tmp_file}"

  mv "${tmp_file}" "${state_file}"
}
