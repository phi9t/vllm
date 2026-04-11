#!/bin/bash
# shellcheck shell=bash

set -euo pipefail

DEVX_PRESETS=("qwen3-0.6b" "qwen3-4b")

preset_model_id() {
  case "${1}" in
    qwen3-0.6b)
      echo "Qwen/Qwen3-0.6B"
      ;;
    qwen3-4b)
      echo "Qwen/Qwen3-4B"
      ;;
    *)
      return 1
      ;;
  esac
}

require_preset() {
  local preset="${1}"
  if ! preset_model_id "${preset}" >/dev/null; then
    echo "invalid preset: ${preset}" >&2
    return 1
  fi
}
