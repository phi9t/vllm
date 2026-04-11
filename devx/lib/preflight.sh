#!/bin/bash
# shellcheck shell=bash

require_command() {
  local command_name="${1:-}"

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "preflight: missing required command: ${command_name}" >&2
    return 1
  fi
}

resolve_huggingface_token_file() {
  printf '%s\n' "${HOME}/.devx/special-circ-phi9t-vllm/secrets/huggingface_token"
}

require_huggingface_token_file() {
  local token_file

  token_file="$(resolve_huggingface_token_file)"
  if [[ ! -s "${token_file}" ]]; then
    echo "preflight: missing or empty token file: ${token_file}" >&2
    return 1
  fi
}

read_huggingface_token() {
  local token_file
  local token

  token_file="$(resolve_huggingface_token_file)"
  token="$(tr -d '[:space:]' < "${token_file}")"
  if [[ -z "${token}" ]]; then
    echo "preflight: missing or empty token file: ${token_file}" >&2
    return 1
  fi

  printf '%s\n' "${token}"
}

require_model_id_resolves() {
  local model_id="${1:-}"
  local token

  token="$(read_huggingface_token)" || return 1

  if ! curl \
    --silent \
    --show-error \
    --fail \
    -H "Authorization: Bearer ${token}" \
    "https://huggingface.co/api/models/${model_id}" \
    >/dev/null; then
    echo "preflight: unable to resolve model id via HuggingFace API: ${model_id}" >&2
    return 1
  fi
}

run_preflight() {
  local model_id="${1:-}"

  require_command docker
  require_command curl
  require_command jq

  docker version >/dev/null
  docker compose version >/dev/null
  require_huggingface_token_file
  require_model_id_resolves "${model_id}"

  echo "PREFLIGHT PASS: model=${model_id}"
}
