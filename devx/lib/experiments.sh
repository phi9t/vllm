#!/bin/bash
# shellcheck shell=bash

DEVX_EXPERIMENTS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVX_EXPERIMENTS_REPO_ROOT="$(cd "${DEVX_EXPERIMENTS_LIB_DIR}/.." && pwd -P)"

resolve_experiment_run_root() {
  printf '%s\n' "${DEVX_EXPERIMENTS_REPO_ROOT}/runs"
}

experiment_validate_run_id() {
  local run_id="${1:-}"
  local context="${2:-experiment}"

  if [[ -z "${run_id}" ]]; then
    echo "${context}: run-id required" >&2
    return 1
  fi

  case "${run_id}" in
    *[!A-Za-z0-9_-]*|*..*|*/*)
      echo "${context}: invalid run-id: ${run_id}" >&2
      return 1
      ;;
  esac
}

resolve_experiment_run_dir_relative() {
  local run_id="${1:-}"

  printf '%s\n' "devx/runs/${run_id}"
}

resolve_experiment_run_dir_absolute() {
  local run_id="${1:-}"

  printf '%s\n' "$(resolve_experiment_run_root)/${run_id}"
}

resolve_devx_bin() {
  if [[ -n "${DEVX_BIN_PATH:-}" ]]; then
    printf '%s\n' "${DEVX_BIN_PATH}"
    return 0
  fi

  printf '%s\n' "${DEVX_EXPERIMENTS_REPO_ROOT}/bin/devx"
}

json_escape_string() {
  local value="${1:-}"

  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "${value}"
}

experiment_decode_quoted_scalar() {
  local value="${1:-}"
  local decoded=""
  local index=0
  local length="${#value}"
  local char
  local next_char

  while [[ "${index}" -lt "${length}" ]]; do
    char="${value:index:1}"
    if [[ "${char}" == "\\" && $((index + 1)) -lt "${length}" ]]; then
      next_char="${value:index+1:1}"
      case "${next_char}" in
        "\\")
          decoded+="\\"
          ;;
        '"')
          decoded+='"'
          ;;
        n)
          decoded+=$'\n'
          ;;
        r)
          decoded+=$'\r'
          ;;
        t)
          decoded+=$'\t'
          ;;
        b)
          decoded+=$'\b'
          ;;
        f)
          decoded+=$'\f'
          ;;
        *)
          decoded+="\\${next_char}"
          ;;
      esac
      index=$((index + 2))
      continue
    fi

    decoded+="${char}"
    index=$((index + 1))
  done

  printf '%s\n' "${decoded}"
}

experiment_trim_line() {
  local value="${1:-}"

  printf '%s' "${value}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

experiment_indent_width() {
  local value="${1:-}"
  local indent=0

  while [[ "${value:0:1}" == " " ]]; do
    indent=$((indent + 1))
    value="${value# }"
  done

  printf '%s\n' "${indent}"
}

experiment_scalar_value() {
  local value

  value="$(experiment_trim_line "${1:-}")"

  if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
    printf '%s\n' "${value:1:${#value}-2}"
    return 0
  fi

  if [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
    printf '%s\n' "${value:1:${#value}-2}"
    return 0
  fi

  printf '%s\n' "${value}"
}

experiment_new_run_id() {
  if [[ -n "${DEVX_EXPERIMENT_RUN_ID:-}" ]]; then
    if ! experiment_validate_run_id "${DEVX_EXPERIMENT_RUN_ID}" "experiment run"; then
      return 1
    fi
    printf '%s\n' "${DEVX_EXPERIMENT_RUN_ID}"
    return 0
  fi

  date -u +"%Y%m%dT%H%M%SZ"-$$
}

experiment_parse_prompt_scalar() {
  local value="${1:-}"
  local trimmed
  local unescaped

  trimmed="$(experiment_trim_line "${value}")"

  if [[ -z "${trimmed}" ]]; then
    echo "experiment run: unsupported manifest shape: empty prompt scalars are not supported" >&2
    return 1
  fi

  case "${trimmed}" in
    \#*)
      echo "experiment run: unsupported manifest shape: commented prompt scalars are not supported" >&2
      return 1
      ;;
    \|*)
      echo "experiment run: unsupported manifest shape: block scalar prompts are not supported" >&2
      return 1
      ;;
    \>*)
      echo "experiment run: unsupported manifest shape: block scalar prompts are not supported" >&2
      return 1
      ;;
  esac

  if [[ "${trimmed}" == \"*\" && "${trimmed}" == *\" ]]; then
    unescaped="${trimmed:1:${#trimmed}-2}"
    unescaped="$(experiment_decode_quoted_scalar "${unescaped}")"
    printf '%s\n' "${unescaped}"
    return 0
  fi

  if [[ "${trimmed}" == \'*\' && "${trimmed}" == *\' ]]; then
    unescaped="${trimmed:1:${#trimmed}-2}"
    unescaped="${unescaped//\'\'/\'}"
    printf '%s\n' "${unescaped}"
    return 0
  fi

  if [[ "${trimmed}" =~ ^-[[:space:]] ]]; then
    echo "experiment run: unsupported manifest shape: nested prompt structures are not supported" >&2
    return 1
  fi

  if [[ "${trimmed}" =~ [[:space:]]# ]]; then
    echo "experiment run: unsupported manifest shape: commented prompt scalars are not supported" >&2
    return 1
  fi

  printf '%s\n' "${trimmed}"
}

experiment_manifest_parse() {
  local manifest="${1:-}"
  local line
  local trimmed
  local indent
  local run_count=0
  local prompt_count=0
  local seen_runs_key=0
  local in_run=0
  local in_prompts=0
  local prompts_indent=-1

  EXPERIMENT_MANIFEST_PRESET=""
  EXPERIMENT_MANIFEST_PROMPT=""

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    trimmed="$(experiment_trim_line "${line}")"

    if [[ -z "${trimmed}" || "${trimmed:0:1}" == "#" ]]; then
      continue
    fi

    if [[ "${trimmed}" == "runs:" ]]; then
      if [[ "${seen_runs_key}" -eq 1 ]]; then
        echo "experiment run: unsupported manifest shape: multiple runs blocks are not supported" >&2
        return 1
      fi

      seen_runs_key=1
      in_run=0
      in_prompts=0
      prompts_indent=-1
      continue
    fi

    indent="$(experiment_indent_width "${line}")"

    if [[ "${indent}" -eq 2 && "${trimmed}" =~ ^-[[:space:]]*(.*)$ ]]; then
      if [[ "${run_count}" -ge 1 ]]; then
        echo "experiment run: unsupported manifest shape: multiple runs are not supported" >&2
        return 1
      fi

      run_count=1
      in_run=1
      in_prompts=0
      prompts_indent=-1
      trimmed="$(experiment_trim_line "${BASH_REMATCH[1]}")"

      if [[ -n "${trimmed}" ]]; then
        if [[ "${trimmed}" =~ ^preset:[[:space:]]*(.*)$ ]]; then
          EXPERIMENT_MANIFEST_PRESET="$(experiment_scalar_value "${BASH_REMATCH[1]}")"
          continue
        fi

        if [[ "${trimmed}" == "prompts:" ]]; then
          in_prompts=1
          prompts_indent="${indent}"
          continue
        fi
      fi

      continue
    fi

    if [[ "${in_run}" -eq 1 && "${indent}" -eq 4 && "${trimmed}" =~ ^preset:[[:space:]]*(.*)$ ]]; then
      EXPERIMENT_MANIFEST_PRESET="$(experiment_scalar_value "${BASH_REMATCH[1]}")"
      continue
    fi

    if [[ "${in_run}" -eq 1 && "${indent}" -eq 4 && "${trimmed}" == "prompts:" ]]; then
      in_prompts=1
      prompts_indent="${indent}"
      continue
    fi

    if [[ "${in_prompts}" -eq 1 && "${indent}" -le "${prompts_indent}" ]]; then
      in_prompts=0
    fi

    if [[ "${in_prompts}" -eq 1 ]]; then
      if [[ "${prompt_count}" -ge 1 ]]; then
        if [[ "${indent}" -eq $((prompts_indent + 2)) && "${trimmed}" =~ ^-[[:space:]]*(.*)$ ]]; then
          echo "experiment run: unsupported manifest shape: multiple prompts are not supported" >&2
          return 1
        fi

        echo "experiment run: unsupported manifest shape: nested prompt structures are not supported" >&2
        return 1
      fi

      if [[ "${indent}" -ne $((prompts_indent + 2)) || ! "${trimmed}" =~ ^-[[:space:]]*(.*)$ ]]; then
        echo "experiment run: unsupported manifest shape: nested prompt structures are not supported" >&2
        return 1
      fi

      prompt_count=1
      if ! EXPERIMENT_MANIFEST_PROMPT="$(experiment_parse_prompt_scalar "${BASH_REMATCH[1]}")"; then
        return 1
      fi
      continue
    fi
  done < "${manifest}"

  if [[ "${seen_runs_key}" -ne 1 || "${run_count}" -ne 1 ]]; then
    echo "experiment run: unsupported manifest shape: exactly one run is required" >&2
    return 1
  fi

  if [[ -z "${EXPERIMENT_MANIFEST_PRESET}" ]]; then
    echo "experiment run: missing preset in manifest" >&2
    return 1
  fi

  if [[ "${prompt_count}" -ne 1 || -z "${EXPERIMENT_MANIFEST_PROMPT}" ]]; then
    echo "experiment run: unsupported manifest shape: exactly one prompt is required" >&2
    return 1
  fi
}

experiment_run_prompt() {
  local preset="${1:-}"
  local prompt="${2:-}"
  local devx_bin

  devx_bin="$(resolve_devx_bin)"
  if ! "${devx_bin}" switch --preset "${preset}" >/dev/null; then
    echo "experiment run: switch failed for preset=${preset}" >&2
    return 1
  fi

  "${devx_bin}" query --preset "${preset}" --prompt "${prompt}"
}

experiment_run_dir_has_artifacts() {
  local run_dir="${1:-}"

  if [[ ! -d "${run_dir}" ]]; then
    return 1
  fi

  find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

cleanup_experiment_run_dir() {
  local run_id="${1:-}"
  local run_dir

  if ! experiment_validate_run_id "${run_id}" "experiment run"; then
    return 1
  fi

  run_dir="$(resolve_experiment_run_dir_absolute "${run_id}")"

  if [[ -d "${run_dir}" ]]; then
    rm -rf "${run_dir}"
  fi
}

run_experiment_manifest() {
  local manifest="${1:-}"
  local run_id
  local run_dir
  local preset
  local prompt
  local query_json
  local results_jsonl
  local summary_md

  if [[ ! -f "${manifest}" ]]; then
    echo "experiment run: missing manifest: ${manifest}" >&2
    return 1
  fi

  if ! experiment_manifest_parse "${manifest}"; then
    return 1
  fi

  preset="${EXPERIMENT_MANIFEST_PRESET}"
  prompt="${EXPERIMENT_MANIFEST_PROMPT}"

  run_id="$(experiment_new_run_id)"
  if ! experiment_validate_run_id "${run_id}" "experiment run"; then
    return 1
  fi
  run_dir="$(resolve_experiment_run_dir_absolute "${run_id}")"

  if [[ -e "${run_dir}" && ! -d "${run_dir}" ]]; then
    echo "experiment run: run path already exists and is not a directory: ${run_dir}" >&2
    return 1
  fi

  if experiment_run_dir_has_artifacts "${run_dir}"; then
    echo "experiment run: run directory already exists with artifacts: ${run_dir}" >&2
    return 1
  fi

  mkdir -p "${run_dir}"

  if ! cp "${manifest}" "${run_dir}/manifest.input.yaml"; then
    cleanup_experiment_run_dir "${run_id}"
    return 1
  fi

  if ! query_json="$(experiment_run_prompt "${preset}" "${prompt}")"; then
    echo "experiment run: query failed for preset=${preset}" >&2
    cleanup_experiment_run_dir "${run_id}"
    return 1
  fi

  results_jsonl="${run_dir}/results.jsonl"
  summary_md="${run_dir}/summary.md"

  if ! printf '{"preset":"%s","prompt":"%s","status":"ok","query":%s}\n' \
    "$(json_escape_string "${preset}")" \
    "$(json_escape_string "${prompt}")" \
    "${query_json}" >"${results_jsonl}"; then
    cleanup_experiment_run_dir "${run_id}"
    return 1
  fi

  if ! {
    printf '# Experiment %s\n\n' "${run_id}"
    printf -- '- preset: %s\n' "${preset}"
    printf -- '- prompt: %s\n' "${prompt}"
    printf -- '- prompts: 1\n'
    printf -- '- status: ok\n'
  } >"${summary_md}"; then
    cleanup_experiment_run_dir "${run_id}"
    return 1
  fi

  printf '%s\n' "${run_id}"
}

report_experiment_run() {
  local run_id="${1:-}"
  local summary_md

  if ! experiment_validate_run_id "${run_id}" "experiment report"; then
    return 1
  fi

  summary_md="$(resolve_experiment_run_dir_absolute "${run_id}")/summary.md"
  if [[ ! -f "${summary_md}" ]]; then
    echo "experiment report: missing summary: ${summary_md}" >&2
    return 1
  fi

  cat "${summary_md}"
}
