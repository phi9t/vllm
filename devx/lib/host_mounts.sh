#!/bin/bash

resolve_host_state_root() {
  printf '%s\n' "${DEVX_HOST_STATE_ROOT:-${HOME}/.devx/special-circ-phi9t-vllm}"
}

resolve_host_cache_root() {
  printf '%s\n' "${DEVX_HOST_CACHE_ROOT:-${HOME}/.cache}"
}

install_host_user_dir() {
  local mode="$1"
  local path="$2"

  install -d -m "${mode}" "${path}"
  if [[ "$(id -u)" -eq 0 ]]; then
    chown "${HOST_UID}:${HOST_GID}" "${path}"
  fi
}

export_host_mount_context() {
  if [[ "${DEVX_HOST_MOUNT_CONTEXT_INITIALIZED:-0}" == "1" ]]; then
    return 0
  fi

  DEVX_HOST_STATE_ROOT="$(resolve_host_state_root)"
  readonly DEVX_HOST_STATE_ROOT

  DEVX_HOST_CACHE_ROOT="$(resolve_host_cache_root)"
  readonly DEVX_HOST_CACHE_ROOT

  HOST_METRIC_SOCKET="${HOST_METRIC_SOCKET:-/tmp/metric.sock}"
  readonly HOST_METRIC_SOCKET

  DEVX_HOST_HOME_DIR="${DEVX_HOST_STATE_ROOT}/home"
  readonly DEVX_HOST_HOME_DIR
  DEVX_HOST_ROOT_HOME_DIR="${DEVX_HOST_STATE_ROOT}/root"
  readonly DEVX_HOST_ROOT_HOME_DIR
  DEVX_HOST_WORKSPACE_DIR="${DEVX_HOST_STATE_ROOT}/workspace"
  readonly DEVX_HOST_WORKSPACE_DIR
  DEVX_HOST_OPT_TIGER_DIR="${DEVX_HOST_STATE_ROOT}/opt_tiger"
  readonly DEVX_HOST_OPT_TIGER_DIR
  DEVX_HOST_RUN_DIR="${DEVX_HOST_STATE_ROOT}/run/devx"
  readonly DEVX_HOST_RUN_DIR
  DEVX_HOST_SSH_HOSTKEYS_DIR="${DEVX_HOST_STATE_ROOT}/ssh-hostkeys"
  readonly DEVX_HOST_SSH_HOSTKEYS_DIR
  DEVX_HOST_SSH_STATE_DIR="${DEVX_HOST_SSH_HOSTKEYS_DIR}"
  readonly DEVX_HOST_SSH_STATE_DIR
  DEVX_HOST_STAGED_DIR="${DEVX_HOST_STATE_ROOT}/staged"
  readonly DEVX_HOST_STAGED_DIR
  DEVX_HOST_STAGED_SSH_DIR="${DEVX_HOST_STAGED_DIR}/ssh"
  readonly DEVX_HOST_STAGED_SSH_DIR
  DEVX_HOST_STAGED_CONSUL_DIR="${DEVX_HOST_STAGED_DIR}/consul_datacenter"
  readonly DEVX_HOST_STAGED_CONSUL_DIR
  DEVX_HOST_STAGED_YARN_DEPLOY_DIR="${DEVX_HOST_STAGED_DIR}/yarn_deploy"
  readonly DEVX_HOST_STAGED_YARN_DEPLOY_DIR
  DEVX_HOST_STAGED_HADOOP_CONF_DIR="${DEVX_HOST_STAGED_DIR}/hadoop_conf"
  readonly DEVX_HOST_STAGED_HADOOP_CONF_DIR
  DEVX_HOST_STAGED_BIN_DIR="${DEVX_HOST_STAGED_DIR}/bin"
  readonly DEVX_HOST_STAGED_BIN_DIR
  DEVX_HOST_STAGED_BVC_PATH="${DEVX_HOST_STAGED_BIN_DIR}/bvc"
  readonly DEVX_HOST_STAGED_BVC_PATH
  DEVX_HOST_CACHE_DIR="${DEVX_HOST_STATE_ROOT}/cache"
  readonly DEVX_HOST_CACHE_DIR
  DEVX_HOST_CACHE_HF_DIR="${DEVX_HOST_CACHE_ROOT}/huggingface"
  readonly DEVX_HOST_CACHE_HF_DIR
  DEVX_HOST_CACHE_HF_HUB_DIR="${DEVX_HOST_CACHE_HF_DIR}/hub"
  readonly DEVX_HOST_CACHE_HF_HUB_DIR
  DEVX_HOST_CACHE_HF_DATASETS_DIR="${DEVX_HOST_CACHE_HF_DIR}/datasets"
  readonly DEVX_HOST_CACHE_HF_DATASETS_DIR
  DEVX_HOST_CACHE_HF_FLASHINFER_DIR="${DEVX_HOST_CACHE_HF_DIR}/flashinfer"
  readonly DEVX_HOST_CACHE_HF_FLASHINFER_DIR
  DEVX_HOST_CACHE_HF_TRANSFORMERS_DIR="${DEVX_HOST_CACHE_HF_DIR}/transformers"
  readonly DEVX_HOST_CACHE_HF_TRANSFORMERS_DIR
  DEVX_HOST_CACHE_UV_DIR="${DEVX_HOST_CACHE_DIR}/uv"
  readonly DEVX_HOST_CACHE_UV_DIR
  DEVX_HOST_CACHE_PIP_DIR="${DEVX_HOST_CACHE_DIR}/pip"
  readonly DEVX_HOST_CACHE_PIP_DIR
  DEVX_HOST_CACHE_BAZEL_DIR="${DEVX_HOST_CACHE_DIR}/bazel"
  readonly DEVX_HOST_CACHE_BAZEL_DIR
  DEVX_HOST_CACHE_FLASHINFER_DIR="${DEVX_HOST_CACHE_DIR}/flashinfer"
  readonly DEVX_HOST_CACHE_FLASHINFER_DIR
  DEVX_HOST_CACHE_TRANSFORMERS_DIR="${DEVX_HOST_CACHE_DIR}/transformers"
  readonly DEVX_HOST_CACHE_TRANSFORMERS_DIR
  DEVX_HOST_CACHE_TVM_FFI_DIR="${DEVX_HOST_CACHE_DIR}/tvm-ffi"
  readonly DEVX_HOST_CACHE_TVM_FFI_DIR
  DEVX_HOST_CACHE_MODELSCOPE_DIR="${DEVX_HOST_CACHE_DIR}/modelscope"
  readonly DEVX_HOST_CACHE_MODELSCOPE_DIR
  DEVX_HOST_CACHE_CARGO_DIR="${DEVX_HOST_CACHE_DIR}/cargo"
  readonly DEVX_HOST_CACHE_CARGO_DIR
  DEVX_HOST_CACHE_RUSTUP_DIR="${DEVX_HOST_CACHE_DIR}/rustup"
  readonly DEVX_HOST_CACHE_RUSTUP_DIR
  DEVX_HOST_OLLAMA_DIR="${DEVX_HOST_STATE_ROOT}/ollama"
  readonly DEVX_HOST_OLLAMA_DIR
  DEVX_HOST_OLLAMA_MODELS_DIR="${DEVX_HOST_OLLAMA_DIR}/models"
  readonly DEVX_HOST_OLLAMA_MODELS_DIR
  DEVX_HOST_SECRETS_DIR="${DEVX_HOST_STATE_ROOT}/secrets"
  readonly DEVX_HOST_SECRETS_DIR

  DEVX_HOST_MAIN_DIR="${DEVX_HOST_STATE_ROOT}/main"
  readonly DEVX_HOST_MAIN_DIR
  DEVX_HOST_MAIN_TMP_DIR="${DEVX_HOST_MAIN_DIR}/tmp"
  readonly DEVX_HOST_MAIN_TMP_DIR
  DEVX_HOST_MAIN_VAR_TMP_DIR="${DEVX_HOST_MAIN_DIR}/var-tmp"
  readonly DEVX_HOST_MAIN_VAR_TMP_DIR
  DEVX_HOST_MAIN_LOGS_DIR="${DEVX_HOST_MAIN_DIR}/logs"
  readonly DEVX_HOST_MAIN_LOGS_DIR

  DEVX_HOST_VLLM_RUNTIME_DIR="${DEVX_HOST_STATE_ROOT}/vllm-runtime"
  readonly DEVX_HOST_VLLM_RUNTIME_DIR
  DEVX_HOST_VLLM_RUNTIME_TMP_DIR="${DEVX_HOST_VLLM_RUNTIME_DIR}/tmp"
  readonly DEVX_HOST_VLLM_RUNTIME_TMP_DIR
  DEVX_HOST_VLLM_RUNTIME_VAR_TMP_DIR="${DEVX_HOST_VLLM_RUNTIME_DIR}/var-tmp"
  readonly DEVX_HOST_VLLM_RUNTIME_VAR_TMP_DIR
  DEVX_HOST_VLLM_RUNTIME_LOGS_DIR="${DEVX_HOST_VLLM_RUNTIME_DIR}/logs"
  readonly DEVX_HOST_VLLM_RUNTIME_LOGS_DIR

  DEVX_HOST_DYNAMO_DIR="${DEVX_HOST_STATE_ROOT}/dynamo"
  readonly DEVX_HOST_DYNAMO_DIR
  DEVX_HOST_DYNAMO_TMP_DIR="${DEVX_HOST_DYNAMO_DIR}/tmp"
  readonly DEVX_HOST_DYNAMO_TMP_DIR
  DEVX_HOST_DYNAMO_VAR_TMP_DIR="${DEVX_HOST_DYNAMO_DIR}/var-tmp"
  readonly DEVX_HOST_DYNAMO_VAR_TMP_DIR
  DEVX_HOST_DYNAMO_LOGS_DIR="${DEVX_HOST_DYNAMO_DIR}/logs"
  readonly DEVX_HOST_DYNAMO_LOGS_DIR

  DEVX_HOST_TMP_DIR="${DEVX_HOST_MAIN_TMP_DIR}"
  readonly DEVX_HOST_TMP_DIR
  DEVX_HOST_VAR_TMP_DIR="${DEVX_HOST_MAIN_VAR_TMP_DIR}"
  readonly DEVX_HOST_VAR_TMP_DIR

  HOST_BVC_PATH="${HOST_BVC_PATH:-/usr/local/tao/agent/modules/bvc/bin/bvc}"
  readonly HOST_BVC_PATH
  HOST_CONSUL_DATACENTER_DIR="${HOST_CONSUL_DATACENTER_DIR:-/opt/tmp/consul_agent/datacenter}"
  readonly HOST_CONSUL_DATACENTER_DIR
  HOST_TIGER_YARN_DEPLOY_DIR="${HOST_TIGER_YARN_DEPLOY_DIR:-/opt/tiger/yarn_deploy}"
  readonly HOST_TIGER_YARN_DEPLOY_DIR
  HOST_HOME_SSH_DIR="${HOST_HOME_SSH_DIR:-${HOME}/.ssh}"
  readonly HOST_HOME_SSH_DIR

  HOST_UID="${HOST_UID:-$(id -u)}"
  HOST_GID="${HOST_GID:-$(id -g)}"
  HOST_USER_NAME="${HOST_USER_NAME:-$(id -un)}"
  readonly HOST_UID HOST_GID HOST_USER_NAME

  DEVX_HOST_MOUNT_CONTEXT_INITIALIZED=1
  readonly DEVX_HOST_MOUNT_CONTEXT_INITIALIZED
}

reset_host_mount_override_env_vars() {
  HOST_MOUNT_OVERRIDE_ENV_KEYS=()
  HOST_MOUNT_OVERRIDE_ENV_VALUES=()
}

append_host_mount_override_env_var() {
  local env_key="$1"
  local env_value="$2"

  HOST_MOUNT_OVERRIDE_ENV_KEYS+=("${env_key}")
  HOST_MOUNT_OVERRIDE_ENV_VALUES+=("${env_value}")
}

detect_hdfs_conf_dir() {
  if [[ -n "${LOCAL_HDFS_CONF_DIR:-}" ]]; then
    if [[ ! -d "${LOCAL_HDFS_CONF_DIR}" ]]; then
      echo "host_mounts: LOCAL_HDFS_CONF_DIR is not a directory: ${LOCAL_HDFS_CONF_DIR}" >&2
      return 2
    fi

    if [[ ! -f "${LOCAL_HDFS_CONF_DIR}/core-site.xml" && ! -f "${LOCAL_HDFS_CONF_DIR}/hdfs-site.xml" && ! -f "${LOCAL_HDFS_CONF_DIR}/yarn-site.xml" ]]; then
      echo "host_mounts: LOCAL_HDFS_CONF_DIR has no Hadoop site XMLs (core-site.xml/hdfs-site.xml/yarn-site.xml): ${LOCAL_HDFS_CONF_DIR}" >&2
      return 2
    fi

    printf '%s\n' "${LOCAL_HDFS_CONF_DIR}"
    return 0
  fi

  local candidate
  for candidate in \
    "/etc/hadoop/conf" \
    "/etc/hadoop" \
    "${HOME}/.hadoop/conf"; do
    if [[ -d "${candidate}" ]] && { [[ -f "${candidate}/core-site.xml" ]] || [[ -f "${candidate}/hdfs-site.xml" ]] || [[ -f "${candidate}/yarn-site.xml" ]]; }; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

write_bvc_stub() {
  install_host_user_dir 755 "${DEVX_HOST_STAGED_BIN_DIR}"
  cat >"${DEVX_HOST_STAGED_BVC_PATH}" <<'EOF'
#!/bin/sh
echo "bvc is not available on this host" >&2
exit 127
EOF
  chmod 755 "${DEVX_HOST_STAGED_BVC_PATH}"
}

ensure_host_state_dirs() {
  install_host_user_dir 755 "${DEVX_HOST_STATE_ROOT}"
  install_host_user_dir 755 "${DEVX_HOST_HOME_DIR}"
  install_host_user_dir 700 "${DEVX_HOST_HOME_DIR}/${CONTAINER_USER}"
  install_host_user_dir 700 "${DEVX_HOST_HOME_DIR}/${CONTAINER_USER}/.ssh"
  install_host_user_dir 700 "${DEVX_HOST_ROOT_HOME_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_WORKSPACE_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_OPT_TIGER_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_RUN_DIR}"
  install_host_user_dir 700 "${DEVX_HOST_SSH_HOSTKEYS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_STAGED_DIR}"
  install_host_user_dir 700 "${DEVX_HOST_STAGED_SSH_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_STAGED_CONSUL_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_STAGED_YARN_DEPLOY_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_STAGED_HADOOP_CONF_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_STAGED_BIN_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_HF_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_HF_HUB_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_HF_DATASETS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_HF_FLASHINFER_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_HF_TRANSFORMERS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_UV_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_PIP_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_BAZEL_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_FLASHINFER_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_TRANSFORMERS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_TVM_FFI_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_MODELSCOPE_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_CARGO_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_CACHE_RUSTUP_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_OLLAMA_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_OLLAMA_MODELS_DIR}"
  install_host_user_dir 700 "${DEVX_HOST_SECRETS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_MAIN_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_MAIN_LOGS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_VLLM_RUNTIME_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_VLLM_RUNTIME_LOGS_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_DYNAMO_DIR}"
  install_host_user_dir 755 "${DEVX_HOST_DYNAMO_LOGS_DIR}"
  install -d -m 1777 "${DEVX_HOST_MAIN_TMP_DIR}" "${DEVX_HOST_MAIN_VAR_TMP_DIR}" \
    "${DEVX_HOST_VLLM_RUNTIME_TMP_DIR}" "${DEVX_HOST_VLLM_RUNTIME_VAR_TMP_DIR}" \
    "${DEVX_HOST_DYNAMO_TMP_DIR}" "${DEVX_HOST_DYNAMO_VAR_TMP_DIR}"
  if [[ "$(id -u)" -eq 0 ]]; then
    chown "${HOST_UID}:${HOST_GID}" \
      "${DEVX_HOST_MAIN_TMP_DIR}" "${DEVX_HOST_MAIN_VAR_TMP_DIR}" \
      "${DEVX_HOST_VLLM_RUNTIME_TMP_DIR}" "${DEVX_HOST_VLLM_RUNTIME_VAR_TMP_DIR}" \
      "${DEVX_HOST_DYNAMO_TMP_DIR}" "${DEVX_HOST_DYNAMO_VAR_TMP_DIR}"
  fi
}

ensure_host_mount_dirs() {
  ensure_host_state_dirs
  write_bvc_stub
}

configure_host_mount_sources() {
  reset_host_mount_override_env_vars

  DEVX_HOST_SSH_DIR_SOURCE="${DEVX_HOST_STAGED_SSH_DIR}"
  if [[ -d "${HOST_HOME_SSH_DIR}" ]]; then
    DEVX_HOST_SSH_DIR_SOURCE="${HOST_HOME_SSH_DIR}"
  fi

  DEVX_HOST_CONSUL_DATACENTER_SOURCE="${DEVX_HOST_STAGED_CONSUL_DIR}"
  if [[ -d "${HOST_CONSUL_DATACENTER_DIR}" ]]; then
    DEVX_HOST_CONSUL_DATACENTER_SOURCE="${HOST_CONSUL_DATACENTER_DIR}"
  fi

  DEVX_HOST_TIGER_YARN_DEPLOY_SOURCE="${DEVX_HOST_STAGED_YARN_DEPLOY_DIR}"
  if [[ -d "${HOST_TIGER_YARN_DEPLOY_DIR}" ]]; then
    DEVX_HOST_TIGER_YARN_DEPLOY_SOURCE="${HOST_TIGER_YARN_DEPLOY_DIR}"
  fi

  DEVX_HOST_BVC_SOURCE="${DEVX_HOST_STAGED_BVC_PATH}"
  if [[ -f "${HOST_BVC_PATH}" ]]; then
    DEVX_HOST_BVC_SOURCE="${HOST_BVC_PATH}"
  fi

  DEVX_HOST_HADOOP_CONF_SOURCE="${DEVX_HOST_STAGED_HADOOP_CONF_DIR}"
  local hdfs_conf_dir
  if hdfs_conf_dir="$(detect_hdfs_conf_dir)"; then
    DEVX_HOST_HADOOP_CONF_SOURCE="${hdfs_conf_dir}"
    append_host_mount_override_env_var "HADOOP_CONF_DIR" "/etc/hadoop/conf"
  else
    local detect_status=$?
    if [[ ${detect_status} -eq 2 ]]; then
      return 1
    fi
  fi
}

export_host_mount_exports() {
  export DEVX_HOST_STATE_ROOT
  export DEVX_HOST_CACHE_ROOT
  export DEVX_HOST_HOME_DIR
  export DEVX_HOST_ROOT_HOME_DIR
  export DEVX_HOST_WORKSPACE_DIR
  export DEVX_HOST_OPT_TIGER_DIR
  export DEVX_HOST_RUN_DIR
  export DEVX_HOST_SSH_HOSTKEYS_DIR
  export DEVX_HOST_SSH_STATE_DIR
  export DEVX_HOST_STAGED_DIR
  export DEVX_HOST_STAGED_SSH_DIR
  export DEVX_HOST_STAGED_CONSUL_DIR
  export DEVX_HOST_STAGED_YARN_DEPLOY_DIR
  export DEVX_HOST_STAGED_HADOOP_CONF_DIR
  export DEVX_HOST_STAGED_BIN_DIR
  export DEVX_HOST_STAGED_BVC_PATH
  export DEVX_HOST_CACHE_DIR
  export DEVX_HOST_CACHE_HF_DIR
  export DEVX_HOST_CACHE_HF_HUB_DIR
  export DEVX_HOST_CACHE_HF_DATASETS_DIR
  export DEVX_HOST_CACHE_HF_FLASHINFER_DIR
  export DEVX_HOST_CACHE_HF_TRANSFORMERS_DIR
  export DEVX_HOST_CACHE_UV_DIR
  export DEVX_HOST_CACHE_PIP_DIR
  export DEVX_HOST_CACHE_BAZEL_DIR
  export DEVX_HOST_CACHE_FLASHINFER_DIR
  export DEVX_HOST_CACHE_TRANSFORMERS_DIR
  export DEVX_HOST_CACHE_TVM_FFI_DIR
  export DEVX_HOST_CACHE_MODELSCOPE_DIR
  export DEVX_HOST_CACHE_CARGO_DIR
  export DEVX_HOST_CACHE_RUSTUP_DIR
  export DEVX_HOST_OLLAMA_DIR
  export DEVX_HOST_OLLAMA_MODELS_DIR
  export DEVX_HOST_SECRETS_DIR
  export DEVX_HOST_MAIN_DIR
  export DEVX_HOST_MAIN_TMP_DIR
  export DEVX_HOST_MAIN_VAR_TMP_DIR
  export DEVX_HOST_MAIN_LOGS_DIR
  export DEVX_HOST_VLLM_RUNTIME_DIR
  export DEVX_HOST_VLLM_RUNTIME_TMP_DIR
  export DEVX_HOST_VLLM_RUNTIME_VAR_TMP_DIR
  export DEVX_HOST_VLLM_RUNTIME_LOGS_DIR
  export DEVX_HOST_DYNAMO_DIR
  export DEVX_HOST_DYNAMO_TMP_DIR
  export DEVX_HOST_DYNAMO_VAR_TMP_DIR
  export DEVX_HOST_DYNAMO_LOGS_DIR
  export DEVX_HOST_TMP_DIR
  export DEVX_HOST_VAR_TMP_DIR
  export DEVX_HOST_SSH_DIR_SOURCE
  export DEVX_HOST_CONSUL_DATACENTER_SOURCE
  export DEVX_HOST_TIGER_YARN_DEPLOY_SOURCE
  export DEVX_HOST_BVC_SOURCE
  export DEVX_HOST_HADOOP_CONF_SOURCE
}
