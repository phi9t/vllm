#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "${TEST_ROOT}"' EXIT

FAKE_BIN="${TEST_ROOT}/bin"
STATE_DIR="${TEST_ROOT}/stack-root/state"
RUNTIME_DIR="${TEST_ROOT}/runtime"
LOGIN_HOME="${TEST_ROOT}/home/kvothe"
TEMPLATE_DIR="${TEST_ROOT}/template"
AUTHORIZED_KEYS="${TEST_ROOT}/authorized_keys"
CONFIG_FILE="${REPO_ROOT}/devx/sshd_config"
SSHD_LOG="${TEST_ROOT}/sshd.log"
SSHD_ARGS="${TEST_ROOT}/sshd.args"
CHOWN_LOG="${TEST_ROOT}/chown.log"
RUNTIME_CONFIG_FILE="${RUNTIME_DIR}/sshd_config.runtime"

REAL_UID="$(id -u)"
REAL_GID="$(id -g)"
KVOTHE_UID="${REAL_UID}"
KVOTHE_GID="${REAL_GID}"
ZSH_BIN="$(command -v zsh || true)"

mkdir -p "${FAKE_BIN}" "${STATE_DIR}" "${RUNTIME_DIR}" "${LOGIN_HOME}" "${TEMPLATE_DIR}"

if [ -z "${ZSH_BIN}" ]; then
  echo "missing zsh binary for tmux login verification" >&2
  exit 1
fi

cat >"${AUTHORIZED_KEYS}" <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKvotheTestKey devx-test
EOF

cat >"${TEMPLATE_DIR}/.zshrc" <<'EOF'
export FROM_TEMPLATE=1
EOF

cat >"${TEMPLATE_DIR}/.zshenv" <<'EOF'
export FROM_TEMPLATE_ZSHENV=1
EOF

mkdir -p "${TEMPLATE_DIR}/.local/share/devx"
cat >"${TEMPLATE_DIR}/.local/share/devx/tmux-auto-attach.zsh" <<'EOF'
devx_tmux_auto_attach() {
  if [[ ! -o interactive || ! -o login ]]; then
    return 0
  fi

  if [[ "${TMUX_AUTO_ATTACH:-1}" = 0 ]]; then
    return 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi

  exec tmux new-session -A -s main
}

devx_tmux_auto_attach
unset -f devx_tmux_auto_attach
EOF

cat >"${TEMPLATE_DIR}/.zprofile" <<'EOF'
if [[ -r "$HOME/.local/share/devx/tmux-auto-attach.zsh" ]]; then
  source "$HOME/.local/share/devx/tmux-auto-attach.zsh"
fi
EOF

mkdir -p "${TEMPLATE_DIR}/.tmux"
cat >"${TEMPLATE_DIR}/.tmux/.tmux.conf" <<'EOF'
set -g mouse on
EOF

ln -sfn .tmux/.tmux.conf "${TEMPLATE_DIR}/.tmux.conf"

cat >"${LOGIN_HOME}/.zshrc" <<'EOF'
export KEEP_ME=1
EOF

cat >"${FAKE_BIN}/id" <<EOF
#!/bin/bash
set -euo pipefail

if [ "\${1:-}" = "-u" ] && [ "\${2:-}" = "kvothe" ]; then
  echo "${KVOTHE_UID}"
  exit 0
fi

if [ "\${1:-}" = "-g" ] && [ "\${2:-}" = "kvothe" ]; then
  echo "${KVOTHE_GID}"
  exit 0
fi

if [ "\${1:-}" = "-u" ] && [ "\${#}" -eq 1 ]; then
  echo 0
  exit 0
fi

if [ "\${1:-}" = "-g" ] && [ "\${#}" -eq 1 ]; then
  echo 0
  exit 0
fi

exec /usr/bin/id "\$@"
EOF

cat >"${FAKE_BIN}/getent" <<EOF
#!/bin/bash
set -euo pipefail

if [ "\${1:-}" = "passwd" ] && [ "\${2:-}" = "kvothe" ]; then
  echo "kvothe:x:${KVOTHE_UID}:${KVOTHE_GID}:Kvothe Test:${LOGIN_HOME}:/bin/zsh"
  exit 0
fi

if [ "\${1:-}" = "group" ] && [ "\${2:-}" = "kvothe" ]; then
  echo "kvothe:x:${KVOTHE_GID}:"
  exit 0
fi

exit 2
EOF

cat >"${FAKE_BIN}/install" <<'EOF'
#!/bin/bash
set -euo pipefail

SANDBOX_ROOT="__SANDBOX_ROOT__"
mode=""
owner=""
group=""
create_dir="false"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -d)
      create_dir="true"
      shift
      ;;
    -m)
      mode="$2"
      shift 2
      ;;
    -o|-g)
      if [ "$1" = "-o" ]; then
        owner="$2"
      else
        group="$2"
      fi
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ "${create_dir}" = "true" ]; then
  for path in "$@"; do
    if [ "$path" = "${SANDBOX_ROOT}" ] || [[ "$path" == "${SANDBOX_ROOT}/"* ]]; then
      mkdir -p "$path"
      if [ -n "${mode}" ]; then
        chmod "${mode}" "$path"
      fi
    fi
  done
  exit 0
fi

if [ "$#" -eq 2 ]; then
  if [[ "$2" == "${SANDBOX_ROOT}/"* ]]; then
    cp "$1" "$2"
    if [ -n "${mode}" ]; then
      chmod "${mode}" "$2"
    fi
  fi
  exit 0
fi

echo "unsupported install invocation: install $*" >&2
exit 1
EOF
perl -0pi -e "s|__SANDBOX_ROOT__|${TEST_ROOT}|g" "${FAKE_BIN}/install"

cat >"${FAKE_BIN}/chown" <<'EOF'
#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "__CHOWN_LOG__"
exit 0
EOF
perl -0pi -e "s|__CHOWN_LOG__|${CHOWN_LOG}|g" "${FAKE_BIN}/chown"

cat >"${FAKE_BIN}/ssh-keygen" <<'EOF'
#!/bin/bash
set -euo pipefail

type=""
outfile=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -t)
      type="$2"
      shift 2
      ;;
    -f)
      outfile="$2"
      shift 2
      ;;
    -N|-b)
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "$(dirname "$outfile")"
cat >"$outfile" <<'KEY'
-----BEGIN OPENSSH PRIVATE KEY-----
devx-test
-----END OPENSSH PRIVATE KEY-----
KEY
printf '%s\n' "${type}" > "${outfile}.type"
printf '%s\n' "ssh-${type} devx-test" > "${outfile}.pub"
EOF

cat >"${FAKE_BIN}/tmux" <<EOF
#!/bin/bash
set -euo pipefail

printf '%s\n' "\$*" >> "${TEST_ROOT}/tmux.log"
exit 0
EOF

cat >"${FAKE_BIN}/sshd" <<EOF
#!/bin/bash
set -euo pipefail

printf '%s\n' "\$*" > "${SSHD_ARGS}"

if [ "\${1:-}" = "-t" ]; then
  while [ "\$#" -gt 0 ]; do
    case "\$1" in
      -f)
        config="\$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  grep -q '^PasswordAuthentication no$' "\${config}"
  grep -q '^PermitRootLogin no$' "\${config}"
  grep -q '^PubkeyAuthentication yes$' "\${config}"
  grep -q '^AuthenticationMethods publickey$' "\${config}"
  grep -q "^AuthorizedKeysFile ${AUTHORIZED_KEYS}$" "\${config}"
  grep -q '^ClientAliveInterval 30$' "\${config}"
  grep -q '^ClientAliveCountMax 6$' "\${config}"
  grep -q '^AllowTcpForwarding yes$' "\${config}"
  ! grep -q '^ForceCommand ' "\${config}"
  exit 0
fi

printf '%s\n' "sshd-started" > "${SSHD_LOG}"
exit 0
EOF

chmod +x "${FAKE_BIN}/id" "${FAKE_BIN}/getent" "${FAKE_BIN}/install" "${FAKE_BIN}/chown" "${FAKE_BIN}/ssh-keygen" "${FAKE_BIN}/tmux" "${FAKE_BIN}/sshd"

export PATH="${FAKE_BIN}:${PATH}"

if [ ! -x "${REPO_ROOT}/devx/start_main.sh" ]; then
  echo "missing executable devx/start_main.sh" >&2
  exit 1
fi

HOST_UID="${KVOTHE_UID}" \
HOST_GID="${KVOTHE_GID}" \
CONFIG_FILE="${CONFIG_FILE}" \
AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
RUNTIME_DIR="${RUNTIME_DIR}" \
STATE_DIR="${STATE_DIR}" \
HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
LOGIN_USER=kvothe \
PORT=27722 \
SSHD_BIN="${FAKE_BIN}/sshd" \
bash "${REPO_ROOT}/devx/start_main.sh"

grep -q '^sshd-started$' "${SSHD_LOG}"

for key in ssh_host_ed25519_key ssh_host_rsa_key; do
  [ -f "${STATE_DIR}/hostkeys/${key}" ] || {
    echo "missing host key under stack root: ${STATE_DIR}/hostkeys/${key}" >&2
    exit 1
  }
done

[ "$(cat "${LOGIN_HOME}/.zshrc")" = 'export KEEP_ME=1' ] || {
  echo "seed logic overwrote a preexisting home file" >&2
  exit 1
}

[ "$(cat "${LOGIN_HOME}/.zshenv")" = 'export FROM_TEMPLATE_ZSHENV=1' ] || {
  echo "missing initial zshenv seed" >&2
  exit 1
}

[ "$(cat "${LOGIN_HOME}/.zprofile")" = 'if [[ -r "$HOME/.local/share/devx/tmux-auto-attach.zsh" ]]; then
  source "$HOME/.local/share/devx/tmux-auto-attach.zsh"
fi' ] || {
  echo "missing initial zprofile seed" >&2
  exit 1
}

[ "$(cat "${LOGIN_HOME}/.local/share/devx/tmux-auto-attach.zsh")" = 'devx_tmux_auto_attach() {
  if [[ ! -o interactive || ! -o login ]]; then
    return 0
  fi

  if [[ "${TMUX_AUTO_ATTACH:-1}" = 0 ]]; then
    return 0
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi

  exec tmux new-session -A -s main
}

devx_tmux_auto_attach
unset -f devx_tmux_auto_attach' ] || {
  echo "missing initial tmux auto-attach helper seed" >&2
  exit 1
}

grep -q "${LOGIN_HOME}/.zshenv" "${CHOWN_LOG}" || {
  echo "missing ownership repair for seeded zshenv" >&2
  cat "${CHOWN_LOG}" >&2
  exit 1
}

[ "$(cat "${LOGIN_HOME}/.tmux/.tmux.conf")" = 'set -g mouse on' ] || {
  echo "missing initial tmux seed" >&2
  exit 1
}

[ -f "${LOGIN_HOME}/.devx_home_seeded_v1" ] || {
  echo "missing seed marker" >&2
  exit 1
}

if [ -e "${LOGIN_HOME}/.zshrc" ] && [ "$(cat "${LOGIN_HOME}/.zshrc")" = 'export FROM_TEMPLATE=1' ]; then
  echo "template file replaced preexisting home file" >&2
  exit 1
fi

: >"${TEST_ROOT}/tmux.log"
env HOME="${LOGIN_HOME}" \
  ZDOTDIR="${LOGIN_HOME}" \
  PATH="${FAKE_BIN}:${PATH}" \
  "${ZSH_BIN}" -lic 'printf "%s\n" "should-not-print"' \
  >"${TEST_ROOT}/interactive.stdout" \
  2>"${TEST_ROOT}/interactive.stderr" \
  <<<""

[ "$(cat "${TEST_ROOT}/tmux.log")" = 'new-session -A -s main' ] || {
  echo "interactive login shell did not auto-attach to tmux" >&2
  cat "${TEST_ROOT}/tmux.log" >&2
  exit 1
}

if [ -s "${TEST_ROOT}/interactive.stdout" ]; then
  echo "interactive login shell unexpectedly executed the command after tmux attach" >&2
  cat "${TEST_ROOT}/interactive.stdout" >&2
  exit 1
fi

: >"${TEST_ROOT}/tmux.log"
env HOME="${LOGIN_HOME}" \
  ZDOTDIR="${LOGIN_HOME}" \
  PATH="${FAKE_BIN}:${PATH}" \
  "${ZSH_BIN}" -lc 'printf "%s\n" "ok"' \
  >"${TEST_ROOT}/noninteractive.stdout" \
  2>"${TEST_ROOT}/noninteractive.stderr" \
  <<<""

[ "$(cat "${TEST_ROOT}/noninteractive.stdout")" = 'ok' ] || {
  echo "non-interactive login command did not print ok" >&2
  cat "${TEST_ROOT}/noninteractive.stdout" >&2
  exit 1
}

if [ -s "${TEST_ROOT}/tmux.log" ]; then
  echo "non-interactive login command unexpectedly invoked tmux" >&2
  cat "${TEST_ROOT}/tmux.log" >&2
  exit 1
fi

: >"${TEST_ROOT}/tmux.log"
env HOME="${LOGIN_HOME}" \
  ZDOTDIR="${LOGIN_HOME}" \
  PATH="${FAKE_BIN}:${PATH}" \
  TMUX_AUTO_ATTACH=0 \
  "${ZSH_BIN}" -lic 'printf "%s\n" "plain-shell"' \
  >"${TEST_ROOT}/disabled.stdout" \
  2>"${TEST_ROOT}/disabled.stderr" \
  <<<""

[ "$(cat "${TEST_ROOT}/disabled.stdout")" = 'plain-shell' ] || {
  echo "TMUX_AUTO_ATTACH=0 did not preserve the shell" >&2
  cat "${TEST_ROOT}/disabled.stdout" >&2
  exit 1
}

if [ -s "${TEST_ROOT}/tmux.log" ]; then
  echo "TMUX_AUTO_ATTACH=0 unexpectedly invoked tmux" >&2
  cat "${TEST_ROOT}/tmux.log" >&2
  exit 1
fi

rm -f "${LOGIN_HOME}/.zshenv" "${LOGIN_HOME}/.tmux/.tmux.conf"

HOST_UID="${KVOTHE_UID}" \
HOST_GID="${KVOTHE_GID}" \
CONFIG_FILE="${CONFIG_FILE}" \
AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
RUNTIME_DIR="${RUNTIME_DIR}" \
STATE_DIR="${STATE_DIR}" \
HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
LOGIN_USER=kvothe \
PORT=27722 \
SSHD_BIN="${FAKE_BIN}/sshd" \
bash "${REPO_ROOT}/devx/start_main.sh"

[ "$(cat "${LOGIN_HOME}/.zshenv")" = 'export FROM_TEMPLATE_ZSHENV=1' ] || {
  echo "missing reseeded zshenv" >&2
  exit 1
}

[ "$(cat "${LOGIN_HOME}/.tmux/.tmux.conf")" = 'set -g mouse on' ] || {
  echo "missing reseeded tmux file" >&2
  exit 1
}

[ -f "${RUNTIME_CONFIG_FILE}" ] || {
  echo "missing rendered runtime sshd config" >&2
  exit 1
}

grep -q "^AuthorizedKeysFile ${AUTHORIZED_KEYS}$" "${RUNTIME_CONFIG_FILE}" || {
  echo "runtime sshd config did not use the effective authorized-keys path" >&2
  cat "${RUNTIME_CONFIG_FILE}" >&2
  exit 1
}

PREEXISTING_STATE_DIR="${TEST_ROOT}/stack-root-preexisting/state"
PREEXISTING_RUNTIME_DIR="${TEST_ROOT}/runtime-preexisting"
PREEXISTING_HOME_DIR="${TEST_ROOT}/home-preexisting/kvothe"
mkdir -p "${PREEXISTING_STATE_DIR}/hostkeys" "${PREEXISTING_RUNTIME_DIR}" "${PREEXISTING_HOME_DIR}"
cat >"${PREEXISTING_STATE_DIR}/hostkeys/ssh_host_ed25519_key" <<'EOF'
ed25519-preexisting
EOF
cat >"${PREEXISTING_STATE_DIR}/hostkeys/ssh_host_rsa_key" <<'EOF'
rsa-preexisting
EOF
: >"${CHOWN_LOG}"

HOST_UID="${KVOTHE_UID}" \
HOST_GID="${KVOTHE_GID}" \
CONFIG_FILE="${CONFIG_FILE}" \
AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
RUNTIME_DIR="${PREEXISTING_RUNTIME_DIR}" \
STATE_DIR="${PREEXISTING_STATE_DIR}" \
HOSTKEY_DIR="${PREEXISTING_STATE_DIR}/hostkeys" \
LOGIN_USER=kvothe \
PORT=27722 \
SSHD_BIN="${FAKE_BIN}/sshd" \
bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh"

grep -q "^0:0 ${PREEXISTING_STATE_DIR}/hostkeys$" "${CHOWN_LOG}" || {
  echo "missing root reown of preexisting hostkey dir" >&2
  cat "${CHOWN_LOG}" >&2
  exit 1
}

grep -q "^0:0 ${PREEXISTING_STATE_DIR}/hostkeys/ssh_host_ed25519_key ${PREEXISTING_STATE_DIR}/hostkeys/ssh_host_rsa_key$" "${CHOWN_LOG}" || {
  echo "missing root reown of preexisting host keys" >&2
  cat "${CHOWN_LOG}" >&2
  exit 1
}

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "missing sshd config" >&2
  exit 1
fi

for directive in \
  '^PasswordAuthentication no$' \
  '^PermitRootLogin no$' \
  '^PubkeyAuthentication yes$' \
  '^AuthenticationMethods publickey$' \
  '^ClientAliveInterval 30$' \
  '^ClientAliveCountMax 6$'; do
  grep -q "${directive}" "${CONFIG_FILE}" || {
    echo "missing sshd directive: ${directive}" >&2
    exit 1
  }
done

if grep -q '^ForceCommand ' "${CONFIG_FILE}"; then
  echo "sshd config should allow non-interactive commands" >&2
  exit 1
fi

BAD_STDERR="${TEST_ROOT}/bad-login.stderr"
if HOST_UID="${KVOTHE_UID}" \
  HOST_GID="${KVOTHE_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  LOGIN_USER=alice \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-login.stdout" 2>"${BAD_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted LOGIN_USER=alice" >&2
  exit 1
fi

grep -q 'LOGIN_USER must be kvothe' "${BAD_STDERR}" || {
  echo "missing kvothe login-user rejection" >&2
  cat "${BAD_STDERR}" >&2
  exit 1
}

BAD_UID_STDERR="${TEST_ROOT}/bad-uid.stderr"
if HOST_UID=9999 \
  HOST_GID="${KVOTHE_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-uid.stdout" 2>"${BAD_UID_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted mismatched HOST_UID" >&2
  exit 1
fi

grep -q 'does not match host uid' "${BAD_UID_STDERR}" || {
  echo "missing host-uid mismatch rejection" >&2
  cat "${BAD_UID_STDERR}" >&2
  exit 1
}

BAD_MISSING_HOST_IDS_STDERR="${TEST_ROOT}/bad-missing-host-ids.stderr"
if HOST_UID= \
  HOST_GID= \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-missing-host-ids.stdout" 2>"${BAD_MISSING_HOST_IDS_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted missing HOST_UID/HOST_GID" >&2
  exit 1
fi

grep -q 'HOST_UID and HOST_GID are mandatory and must be provided' "${BAD_MISSING_HOST_IDS_STDERR}" || {
  echo "missing host-id presence rejection" >&2
  cat "${BAD_MISSING_HOST_IDS_STDERR}" >&2
  exit 1
}

BAD_HOST_GID_STDERR="${TEST_ROOT}/bad-host-gid.stderr"
if HOST_UID="${KVOTHE_UID}" \
  HOST_GID=9999 \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-host-gid.stdout" 2>"${BAD_HOST_GID_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted mismatched HOST_GID" >&2
  exit 1
fi

grep -q 'does not match host gid' "${BAD_HOST_GID_STDERR}" || {
  echo "missing host-gid mismatch rejection" >&2
  cat "${BAD_HOST_GID_STDERR}" >&2
  exit 1
}

BAD_HOSTKEY_DIR="${TEST_ROOT}/escape-hostkeys"
BAD_HOSTKEY_STDERR="${TEST_ROOT}/bad-hostkey-dir.stderr"
if HOST_UID="${KVOTHE_UID}" \
  HOST_GID="${KVOTHE_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${BAD_HOSTKEY_DIR}" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-hostkey-dir.stdout" 2>"${BAD_HOSTKEY_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted HOSTKEY_DIR outside STATE_DIR" >&2
  exit 1
fi

grep -q 'HOSTKEY_DIR must live under STATE_DIR' "${BAD_HOSTKEY_STDERR}" || {
  echo "missing hostkey-dir guard rejection" >&2
  cat "${BAD_HOSTKEY_STDERR}" >&2
  exit 1
}

[ ! -e "${BAD_HOSTKEY_DIR}" ] || {
  echo "hostkey guard created a directory before failing" >&2
  find "${BAD_HOSTKEY_DIR}" -maxdepth 2 -print >&2
  exit 1
}

BAD_HOSTKEY_ED25519="${TEST_ROOT}/escape-hostkeys/../escape-ed25519/ssh_host_ed25519_key"
BAD_HOSTKEY_ED25519_STDERR="${TEST_ROOT}/bad-hostkey-ed25519.stderr"
: >"${CHOWN_LOG}"
if HOST_UID="${KVOTHE_UID}" \
  HOST_GID="${KVOTHE_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  HOSTKEY_ED25519="${BAD_HOSTKEY_ED25519}" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-hostkey-ed25519.stdout" 2>"${BAD_HOSTKEY_ED25519_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted HOSTKEY_ED25519 outside HOSTKEY_DIR" >&2
  exit 1
fi

grep -q 'HOSTKEY_ED25519 must live under HOSTKEY_DIR' "${BAD_HOSTKEY_ED25519_STDERR}" || {
  echo "missing hostkey-ed25519 guard rejection" >&2
  cat "${BAD_HOSTKEY_ED25519_STDERR}" >&2
  exit 1
}

[ ! -e "${TEST_ROOT}/escape-ed25519" ] || {
  echo "hostkey-ed25519 guard created a directory before failing" >&2
  find "${TEST_ROOT}/escape-ed25519" -maxdepth 2 -print >&2
  exit 1
}

[ ! -s "${CHOWN_LOG}" ] || {
  echo "hostkey-ed25519 override mutated state before failing" >&2
  cat "${CHOWN_LOG}" >&2
  exit 1
}

BAD_HOSTKEY_RSA="${TEST_ROOT}/escape-hostkeys/../escape-rsa/ssh_host_rsa_key"
BAD_HOSTKEY_RSA_STDERR="${TEST_ROOT}/bad-hostkey-rsa.stderr"
: >"${CHOWN_LOG}"
if HOST_UID="${KVOTHE_UID}" \
  HOST_GID="${KVOTHE_GID}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  AUTHORIZED_KEYS="${AUTHORIZED_KEYS}" \
  HOME_TEMPLATE_DIR="${TEMPLATE_DIR}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  STATE_DIR="${STATE_DIR}" \
  HOSTKEY_DIR="${STATE_DIR}/hostkeys" \
  HOSTKEY_RSA="${BAD_HOSTKEY_RSA}" \
  LOGIN_USER=kvothe \
  PORT=27722 \
  SSHD_BIN="${FAKE_BIN}/sshd" \
  bash "${REPO_ROOT}/devx/run_hermetic_sshd.sh" >"${TEST_ROOT}/bad-hostkey-rsa.stdout" 2>"${BAD_HOSTKEY_RSA_STDERR}"; then
  echo "run_hermetic_sshd.sh unexpectedly accepted HOSTKEY_RSA outside HOSTKEY_DIR" >&2
  exit 1
fi

grep -q 'HOSTKEY_RSA must live under HOSTKEY_DIR' "${BAD_HOSTKEY_RSA_STDERR}" || {
  echo "missing hostkey-rsa guard rejection" >&2
  cat "${BAD_HOSTKEY_RSA_STDERR}" >&2
  exit 1
}

[ ! -e "${TEST_ROOT}/escape-rsa" ] || {
  echo "hostkey-rsa guard created a directory before failing" >&2
  find "${TEST_ROOT}/escape-rsa" -maxdepth 2 -print >&2
  exit 1
}

[ ! -s "${CHOWN_LOG}" ] || {
  echo "hostkey-rsa override mutated state before failing" >&2
  cat "${CHOWN_LOG}" >&2
  exit 1
}

echo "PASS"
