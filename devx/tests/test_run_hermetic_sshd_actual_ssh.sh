#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "${TEST_ROOT}"' EXIT
chmod 755 "${TEST_ROOT}"

PORT="${PORT:-2222}"
STATE_DIR="${TEST_ROOT}/stack-root/state"
RUNTIME_DIR="${TEST_ROOT}/runtime"
LOGIN_HOME="${TEST_ROOT}/home/kvothe"
TEMPLATE_DIR="${TEST_ROOT}/template"
AUTHORIZED_KEYS="${TEST_ROOT}/authorized_keys"
AUTHORIZED_KEYS_DIR="/run/devx-ssh-verify-$(basename "${TEST_ROOT}")"
KNOWN_HOSTS="${TEST_ROOT}/known_hosts"
CONFIG_FILE="${TEST_ROOT}/sshd_config.debug"
REMOTE_DIR="${TEST_ROOT}/remote"
TMUX_LOG="${REMOTE_DIR}/tmux.log"
ZPROFILE_LOG="${REMOTE_DIR}/zprofile.log"
SERVER_STDOUT="${TEST_ROOT}/sshd.stdout"
SERVER_STDERR="${TEST_ROOT}/sshd.stderr"
CLIENT_KEY="${TEST_ROOT}/client_key"

SSH_BIN="$(command -v ssh)"
SSH_KEYGEN_BIN="$(command -v ssh-keygen)"
SSH_KEYSCAN_BIN="$(command -v ssh-keyscan)"
SSHD_BIN="$(command -v sshd)"
USERADD_BIN="$(command -v useradd)"
GROUPADD_BIN="$(command -v groupadd)"
USERDEL_BIN="$(command -v userdel)"
GROUPDEL_BIN="$(command -v groupdel)"
SUDO_BIN="$(command -v sudo)"
BASH_BIN="$(command -v bash)"
OPENSSL_BIN="$(command -v openssl)"
TIMEOUT_BIN="$(command -v timeout || true)"

if [ -z "${TIMEOUT_BIN}" ]; then
  echo "missing timeout binary for SSH verification" >&2
  exit 1
fi

if ! "${SUDO_BIN}" -n true >/dev/null 2>&1; then
  echo "passwordless sudo is required for SSH verification" >&2
  exit 1
fi

"${SUDO_BIN}" -n pkill -f "sshd.*-p ${PORT}" >/dev/null 2>&1 || true

if [ ! -x /bin/zsh ]; then
  echo "missing /bin/zsh for SSH verification" >&2
  exit 1
fi

SERVER_PID=""

cleanup_system_user() {
  "${SUDO_BIN}" -n "${USERDEL_BIN}" -r kvothe >/dev/null 2>&1 || "${SUDO_BIN}" -n "${USERDEL_BIN}" kvothe >/dev/null 2>&1 || true
  "${SUDO_BIN}" -n "${GROUPDEL_BIN}" kvothe >/dev/null 2>&1 || true
  "${SUDO_BIN}" -n rm -rf "${AUTHORIZED_KEYS_DIR}" >/dev/null 2>&1 || true
}

cleanup() {
  "${SUDO_BIN}" -n pkill -f "${RUNTIME_DIR}/sshd_config.runtime" >/dev/null 2>&1 || true

  cleanup_system_user
}

trap cleanup EXIT

cleanup_system_user

mkdir -p "${STATE_DIR}" "${RUNTIME_DIR}" "${REMOTE_DIR}" "${TEMPLATE_DIR}/.local/bin" "${TEMPLATE_DIR}/.local/share/devx" "${TEMPLATE_DIR}/.tmux"
mkdir -p "$(dirname "${LOGIN_HOME}")"
chmod 755 "$(dirname "${LOGIN_HOME}")"
chmod 1777 "${REMOTE_DIR}"
${SUDO_BIN} -n mkdir -p "${AUTHORIZED_KEYS_DIR}"
${SUDO_BIN} -n chown 0:0 "${AUTHORIZED_KEYS_DIR}"
${SUDO_BIN} -n chmod 755 "${AUTHORIZED_KEYS_DIR}"
cp "${REPO_ROOT}/devx/sshd_config" "${CONFIG_FILE}"
printf '\nLogLevel DEBUG3\n' >> "${CONFIG_FILE}"

cat >"${TEMPLATE_DIR}/.local/share/devx/tmux-auto-attach.zsh" <<'EOF'
devx_tmux_auto_attach() {
  if [[ -z "${SSH_TTY:-}" ]]; then
    return 0
  fi

  if [[ -n "${TMUX:-}" ]]; then
    return 0
  fi

  if ps -o comm= -p "$PPID" 2>/dev/null | grep -q '^tmux'; then
    return 0
  fi

  if [[ "${DEVX_TMUX_AUTO_ATTACH_ACTIVE:-0}" = 1 ]]; then
    return 0
  fi

  if [[ "${TMUX_AUTO_ATTACH:-1}" = 0 ]]; then
    return 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi

  exec env DEVX_TMUX_AUTO_ATTACH_ACTIVE=1 tmux new-session -A -s main
}

devx_tmux_auto_attach
unset -f devx_tmux_auto_attach
EOF

cat >"${TEMPLATE_DIR}/.zprofile" <<EOF
if [[ -n "${ZPROFILE_LOG}" ]]; then
  print -r -- "zprofile:\$-:\$0" >> "${ZPROFILE_LOG}"
fi
if [[ -r "$HOME/.local/share/devx/tmux-auto-attach.zsh" ]]; then
  source "$HOME/.local/share/devx/tmux-auto-attach.zsh"
fi
EOF

cat >"${TEMPLATE_DIR}/.zshrc" <<EOF
if [[ -r "$HOME/.local/share/devx/tmux-auto-attach.zsh" ]]; then
  source "$HOME/.local/share/devx/tmux-auto-attach.zsh"
fi
EOF

cat >"${TEMPLATE_DIR}/.zshenv" <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
EOF

cat >"${TEMPLATE_DIR}/.tmux/.tmux.conf" <<'EOF'
set -g mouse on
EOF

ln -sfn .tmux/.tmux.conf "${TEMPLATE_DIR}/.tmux.conf"

cat >"${TEMPLATE_DIR}/.local/bin/tmux" <<EOF
#!/bin/bash
set -euo pipefail

printf '%s\n' "\$*" >> "${TMUX_LOG}"
exit 0
EOF
chmod +x "${TEMPLATE_DIR}/.local/bin/tmux"

"${SSH_KEYGEN_BIN}" -q -t ed25519 -N '' -f "${CLIENT_KEY}" >/dev/null

${SUDO_BIN} -n "${GROUPADD_BIN}" kvothe
${SUDO_BIN} -n "${USERADD_BIN}" -M -d "${LOGIN_HOME}" -s /bin/zsh -g kvothe kvothe
PASSWORD_HASH="$("${OPENSSL_BIN}" passwd -6 -salt devx-ssh devx-ssh-temp)"
${SUDO_BIN} -n "${USERADD_BIN%/useradd}/usermod" -p "${PASSWORD_HASH}" kvothe
HOST_UID="$(id -u kvothe)"
HOST_GID="$(id -g kvothe)"
mkdir -p "${LOGIN_HOME}"
${SUDO_BIN} -n chown "${HOST_UID}:${HOST_GID}" "${LOGIN_HOME}"
${SUDO_BIN} -n chmod 700 "${LOGIN_HOME}"
${SUDO_BIN} -n install -d -m 1777 "${REMOTE_DIR}"
${SUDO_BIN} -n install -m 666 -o "${HOST_UID}" -g "${HOST_GID}" /dev/null "${TMUX_LOG}"
${SUDO_BIN} -n install -m 666 -o "${HOST_UID}" -g "${HOST_GID}" /dev/null "${ZPROFILE_LOG}"

${SUDO_BIN} -n cp "${CLIENT_KEY}.pub" "${AUTHORIZED_KEYS_DIR}/authorized_keys"
${SUDO_BIN} -n chown "${HOST_UID}:${HOST_GID}" "${AUTHORIZED_KEYS_DIR}/authorized_keys"
${SUDO_BIN} -n chmod 600 "${AUTHORIZED_KEYS_DIR}/authorized_keys"

start_server() {
  "${SUDO_BIN}" -n env \
  HOST_UID="${HOST_UID}" \
  HOST_GID="${HOST_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
    AUTHORIZED_KEYS="${AUTHORIZED_KEYS_DIR}/authorized_keys" \
    HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
    RUNTIME_DIR="${RUNTIME_DIR}" \
    STATE_DIR="${STATE_DIR}" \
    HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
    LOGIN_USER=kvothe \
    PORT="${PORT}" \
    SSHD_BIN="${SSHD_BIN}" \
    "${BASH_BIN}" "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${SERVER_STDOUT}" 2>"${SERVER_STDERR}" &
  SERVER_PID=$!
}

wait_for_server() {
  local attempts=0
  while true; do
    if grep -q "sshd running on port ${PORT} for login user kvothe" "${SERVER_STDOUT}" 2>/dev/null && \
      "${SSH_KEYSCAN_BIN}" -t ed25519,rsa -p "${PORT}" 127.0.0.1 >"${KNOWN_HOSTS}" 2>/dev/null; then
      return 0
    fi

    attempts=$((attempts + 1))
    if [ "${attempts}" -gt 100 ]; then
      echo "sshd did not become ready on port ${PORT}" >&2
      cat "${SERVER_STDOUT}" >&2 || true
      cat "${SERVER_STDERR}" >&2 || true
      exit 1
    fi

    sleep 0.1
  done
}

ssh_base_args=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o SendEnv=PATH
  -p "${PORT}"
  -i "${CLIENT_KEY}"
  kvothe@127.0.0.1
)

start_server
wait_for_server

if [ "${VERBOSE_SSH_DEBUG:-0}" = 1 ]; then
  "${SUDO_BIN}" -n ps -ef | grep "[s]shd.*-p ${PORT}" >&2 || true
fi

if ! printf 'exit\n' | "${TIMEOUT_BIN}" 15s "${SSH_BIN}" -tt "${ssh_base_args[@]}" \
  >"${TEST_ROOT}/interactive.stdout" \
  2>"${TEST_ROOT}/interactive.stderr"; then
  echo "interactive ssh login failed" >&2
  cat "${TMUX_LOG}" >&2 || true
  cat "${ZPROFILE_LOG}" >&2 || true
  cat "${SERVER_STDOUT}" >&2 || true
  cat "${SERVER_STDERR}" >&2 || true
  "${SUDO_BIN}" -n cat "${RUNTIME_DIR}/logs/sshd.log" >&2 || true
  cat "${TEST_ROOT}/interactive.stderr" >&2 || true
  exit 1
fi

TMUX_LOG_SNAPSHOT="$(cat "${TMUX_LOG}")"
if ! "${TIMEOUT_BIN}" 15s "${SSH_BIN}" -tt "${ssh_base_args[@]}" 'zsh -ic "source ~/.local/share/devx/tmux-auto-attach.zsh; return"'
then
  echo "tmux auto-attach helper invocation failed" >&2
  cat "${TMUX_LOG}" >&2 || true
  cat "${ZPROFILE_LOG}" >&2 || true
  cat "${SERVER_STDOUT}" >&2 || true
  cat "${SERVER_STDERR}" >&2 || true
  "${SUDO_BIN}" -n cat "${RUNTIME_DIR}/logs/sshd.log" >&2 || true
  exit 1
fi

TMUX_LOG_AFTER_HELPER="$(cat "${TMUX_LOG}")"
if [ "${TMUX_LOG_AFTER_HELPER}" != "${TMUX_LOG_SNAPSHOT}" ]; then
  if [ "${TMUX_LOG_AFTER_HELPER}" != 'new-session -A -s main' ]; then
    echo "tmux auto-attach helper wrote unexpected command to tmux log" >&2
    cat "${TMUX_LOG}" >&2 || true
    exit 1
  fi
else
  echo "tmux auto-attach helper invocation did not write to tmux log" >&2
  exit 1
fi

TMUX_LOG_SNAPSHOT="${TMUX_LOG_AFTER_HELPER}"

if ! "${TIMEOUT_BIN}" 15s "${SSH_BIN}" "${ssh_base_args[@]}" echo ok \
  >"${TEST_ROOT}/command.stdout" \
  2>"${TEST_ROOT}/command.stderr"; then
  echo "non-interactive ssh command failed" >&2
  cat "${TMUX_LOG}" >&2 || true
  cat "${ZPROFILE_LOG}" >&2 || true
  cat "${SERVER_STDOUT}" >&2 || true
  cat "${SERVER_STDERR}" >&2 || true
  "${SUDO_BIN}" -n cat "${RUNTIME_DIR}/logs/sshd.log" >&2 || true
  cat "${TEST_ROOT}/command.stderr" >&2 || true
  exit 1
fi

[ "$(cat "${TEST_ROOT}/command.stdout")" = 'ok' ] || {
  echo "non-interactive ssh command did not print ok" >&2
  cat "${TEST_ROOT}/command.stdout" >&2
  exit 1
}

if [ "$(cat "${TMUX_LOG}")" != "${TMUX_LOG_SNAPSHOT}" ]; then
  echo "non-interactive ssh command unexpectedly changed tmux log" >&2
  cat "${TMUX_LOG}" >&2
  exit 1
fi

if ! "${TIMEOUT_BIN}" 15s "${SSH_BIN}" -tt "${ssh_base_args[@]}" 'TMUX_AUTO_ATTACH=0 exec zsh -l' \
  >"${TEST_ROOT}/plain.stdout" \
  2>"${TEST_ROOT}/plain.stderr" <<'EOF'; then
printf '%s\n' plain-shell-ok
exit
EOF
  echo "TMUX_AUTO_ATTACH=0 ssh command failed" >&2
  cat "${TMUX_LOG}" >&2 || true
  cat "${ZPROFILE_LOG}" >&2 || true
  cat "${SERVER_STDOUT}" >&2 || true
  cat "${SERVER_STDERR}" >&2 || true
  "${SUDO_BIN}" -n cat "${RUNTIME_DIR}/logs/sshd.log" >&2 || true
  cat "${TEST_ROOT}/plain.stderr" >&2 || true
  exit 1
fi

if [ "$(cat "${TMUX_LOG}")" != "${TMUX_LOG_SNAPSHOT}" ]; then
  echo "TMUX_AUTO_ATTACH=0 unexpectedly changed tmux log" >&2
  cat "${TMUX_LOG}" >&2
  exit 1
fi

grep -q 'plain-shell-ok' "${TEST_ROOT}/plain.stdout" || {
  echo "TMUX_AUTO_ATTACH=0 shell did not behave like a plain shell" >&2
  cat "${TEST_ROOT}/plain.stdout" >&2
  exit 1
}

echo "PASS"
