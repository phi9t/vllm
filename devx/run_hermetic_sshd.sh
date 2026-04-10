#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-27722}"
LOGIN_USER="${LOGIN_USER:-kvothe}"
HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"
HOST_USER_NAME="${HOST_USER_NAME:-}"
CONFIG_FILE="${CONFIG_FILE:-/run/devx/sshd_config}"
AUTHORIZED_KEYS="${AUTHORIZED_KEYS:-/run/devx/authorized_keys}"
RUNTIME_DIR="${RUNTIME_DIR:-/run/devx}"
STATE_DIR="${STATE_DIR:-/var/lib/devx-sshd}"
HOSTKEY_DIR="${HOSTKEY_DIR:-$STATE_DIR/hostkeys}"
HOSTKEY_ED25519="${HOSTKEY_ED25519:-$HOSTKEY_DIR/ssh_host_ed25519_key}"
HOSTKEY_RSA="${HOSTKEY_RSA:-$HOSTKEY_DIR/ssh_host_rsa_key}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/sshd.pid}"
LOG_DIR="${LOG_DIR:-$RUNTIME_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sshd.log}"
RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_FILE:-$RUNTIME_DIR/sshd_config.runtime}"
HOME_TEMPLATE_DIR="${HOME_TEMPLATE_DIR:-/opt/devx/skel}"
HOST_SSH_DIR="${HOST_SSH_DIR:-/run/devx/host-ssh}"

resolve_sshd_bin() {
  if [ -n "${SSHD_BIN:-}" ] && [ -x "${SSHD_BIN}" ]; then
    printf '%s\n' "${SSHD_BIN}"
    return 0
  fi

  if command -v sshd >/dev/null 2>&1; then
    command -v sshd
    return 0
  fi

  if [ -x "/usr/sbin/sshd" ]; then
    printf '%s\n' "/usr/sbin/sshd"
    return 0
  fi

  return 1
}

if ! SSHD_BIN="$(resolve_sshd_bin)"; then
  echo "Unable to find a usable sshd binary" >&2
  exit 1
fi

if [ ! -x "$SSHD_BIN" ]; then
  echo "sshd not found at $SSHD_BIN" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "run_hermetic_sshd: sshd must start from root" >&2
  exit 1
fi

if [ "$LOGIN_USER" != "kvothe" ]; then
  echo "run_hermetic_sshd: LOGIN_USER must be kvothe for the approved devx contract" >&2
  exit 1
fi

if ! getent passwd "$LOGIN_USER" >/dev/null 2>&1; then
  echo "Login user not found: $LOGIN_USER" >&2
  exit 1
fi

LOGIN_HOME="$(getent passwd "$LOGIN_USER" | cut -d: -f6)"
LOGIN_UID="$(id -u "$LOGIN_USER")"
LOGIN_GID="$(id -g "$LOGIN_USER")"

seed_login_home() {
  local seed_marker="$LOGIN_HOME/.devx_home_seeded_v1"
  local template_entry template_name template_target

  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME"
  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.ssh"
  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.cache"
  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.config"
  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.local/state"
  install -d -m 755 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.local/share"
  install -d -m 755 -o "$LOGIN_UID" -g "$LOGIN_GID" "$LOGIN_HOME/.local/bin"

  if [ -d "$HOME_TEMPLATE_DIR" ]; then
    while IFS= read -r -d '' template_entry; do
      template_target="${LOGIN_HOME}/${template_entry#${HOME_TEMPLATE_DIR}/}"

      if [ -d "$template_entry" ] && [ ! -L "$template_entry" ]; then
        install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$template_target"
        continue
      fi

      if [ -e "$template_target" ] || [ -L "$template_target" ]; then
        continue
      fi

      install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$(dirname "$template_target")"
      cp -a "$template_entry" "$template_target"
    done < <(find "$HOME_TEMPLATE_DIR" -mindepth 1 -print0)
  fi

  touch "$LOGIN_HOME/.zsh_history"
  if [ ! -f "$seed_marker" ]; then
    touch "$seed_marker"
  fi
  chown "$LOGIN_UID:$LOGIN_GID" "$LOGIN_HOME"
  chown "$LOGIN_UID:$LOGIN_GID" "$seed_marker"

  local path
  for path in \
    "$LOGIN_HOME/.zshenv" \
    "$LOGIN_HOME/.zshrc" \
    "$LOGIN_HOME/.zsh_history" \
    "$LOGIN_HOME/.tmux.conf" \
    "$LOGIN_HOME/.tmux.conf.local" \
    "$LOGIN_HOME/.oh-my-zsh" \
    "$LOGIN_HOME/.tmux" \
    "$LOGIN_HOME/.emacs.d" \
    "$LOGIN_HOME/.doom.d" \
    "$LOGIN_HOME/.cache" \
    "$LOGIN_HOME/.config" \
    "$LOGIN_HOME/.local" \
    "$LOGIN_HOME/.ssh"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      chown -h "$LOGIN_UID:$LOGIN_GID" "$path" 2>/dev/null || true
      chown -R "$LOGIN_UID:$LOGIN_GID" "$path" 2>/dev/null || true
    fi
  done

  chmod 700 "$LOGIN_HOME" "$LOGIN_HOME/.ssh" "$LOGIN_HOME/.cache" "$LOGIN_HOME/.config" "$LOGIN_HOME/.local/state"
  chmod 600 "$seed_marker"
  find "$LOGIN_HOME/.ssh" -type d -exec chmod 700 {} +
  find "$LOGIN_HOME/.ssh" -type f -exec chmod 600 {} +
}

migrate_legacy_ssh_material() {
  local backup_dir="$LOGIN_HOME/.ssh.legacy-devx-backup"
  local marker_file="$LOGIN_HOME/.devx_home_migration_v1"
  local needs_migration="false"

  if [ -f "$marker_file" ]; then
    return 0
  fi

  if [ -L "$LOGIN_HOME/.tmux.conf" ] && [ "$(readlink "$LOGIN_HOME/.tmux.conf")" = "/opt/devx/skel/.tmux/.tmux.conf" ]; then
    needs_migration="true"
  fi

  if [ "$needs_migration" != "true" ] && [ -d "$HOST_SSH_DIR" ]; then
    local staged_probe rel_probe login_probe base_probe
    while IFS= read -r -d '' staged_probe; do
      rel_probe="${staged_probe#${HOST_SSH_DIR}/}"
      login_probe="$LOGIN_HOME/.ssh/$rel_probe"
      base_probe="$(basename "$staged_probe")"

      case "$base_probe" in
        *.pub|authorized_keys|config|known_hosts|known_hosts.old)
          continue
          ;;
      esac

      if [ -f "$login_probe" ] && cmp -s "$staged_probe" "$login_probe"; then
        needs_migration="true"
        break
      fi
    done < <(find "$HOST_SSH_DIR" -type f -print0)
  fi

  if [ "$needs_migration" != "true" ]; then
    return 0
  fi

  install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$backup_dir"

  local login_file rel_path base_name staged_file is_private_key
  while IFS= read -r -d '' login_file; do
    rel_path="${login_file#${LOGIN_HOME}/.ssh/}"
    base_name="$(basename "$login_file")"

    case "$base_name" in
      *.pub|authorized_keys|config|known_hosts|known_hosts.old)
        continue
        ;;
    esac

    is_private_key="false"
    if grep -qE '(^-----BEGIN ((OPENSSH|RSA|EC|DSA) )?PRIVATE KEY-----$)|(^-----BEGIN ENCRYPTED PRIVATE KEY-----$)' "$login_file" 2>/dev/null; then
      is_private_key="true"
    fi

    staged_file="$HOST_SSH_DIR/$rel_path"
    if [[ "$is_private_key" == "true" ]] || { [ -f "$staged_file" ] && cmp -s "$staged_file" "$login_file"; }; then
      install -d -m 700 -o "$LOGIN_UID" -g "$LOGIN_GID" "$(dirname "$backup_dir/$rel_path")"
      mv "$login_file" "$backup_dir/$rel_path"
    fi
  done < <(find "$LOGIN_HOME/.ssh" -type f -print0)

  touch "$marker_file"
  chown "$LOGIN_UID:$LOGIN_GID" "$marker_file"
  chmod 600 "$marker_file"
}

reconcile_login_home_defaults() {
  if [ -f "$HOME_TEMPLATE_DIR/.zshenv" ] && [ ! -f "$LOGIN_HOME/.zshenv" ]; then
    cp -a "$HOME_TEMPLATE_DIR/.zshenv" "$LOGIN_HOME/.zshenv"
    chown "$LOGIN_UID:$LOGIN_GID" "$LOGIN_HOME/.zshenv"
    chmod 600 "$LOGIN_HOME/.zshenv"
  fi

  if [ -L "$LOGIN_HOME/.tmux.conf" ] && [ "$(readlink "$LOGIN_HOME/.tmux.conf")" = "/opt/devx/skel/.tmux/.tmux.conf" ]; then
    ln -sfn ".tmux/.tmux.conf" "$LOGIN_HOME/.tmux.conf"
    chown -h "$LOGIN_UID:$LOGIN_GID" "$LOGIN_HOME/.tmux.conf"
  fi
}

if [ -z "$HOST_UID" ] || [ -z "$HOST_GID" ]; then
  echo "run_hermetic_sshd: HOST_UID and HOST_GID are mandatory and must be provided" >&2
  exit 1
fi

if [ "$LOGIN_UID" != "$HOST_UID" ]; then
  echo "run_hermetic_sshd: kvothe uid ($LOGIN_UID) does not match host uid ($HOST_UID${HOST_USER_NAME:+ for $HOST_USER_NAME})" >&2
  exit 1
fi

if [ "$LOGIN_GID" != "$HOST_GID" ]; then
  echo "run_hermetic_sshd: kvothe gid ($LOGIN_GID) does not match host gid ($HOST_GID${HOST_USER_NAME:+ for $HOST_USER_NAME})" >&2
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  CONFIG_FILE="$ROOT_DIR/sshd_config"
fi

if [ ! -f "$AUTHORIZED_KEYS" ]; then
  echo "authorized_keys not found: $AUTHORIZED_KEYS" >&2
  exit 1
fi

if [ ! -r "$AUTHORIZED_KEYS" ]; then
  echo "authorized_keys is not readable: $AUTHORIZED_KEYS" >&2
  exit 1
fi

install -d -m 755 /run/sshd
if [ ! -d "$LOGIN_HOME" ]; then
  echo "run_hermetic_sshd: login home does not exist: $LOGIN_HOME" >&2
  exit 1
fi

seed_login_home
if [ "${ALLOW_LEGACY_HOME_MIGRATION:-0}" = "1" ]; then
  migrate_legacy_ssh_material
fi

if [ "${ALLOW_LEGACY_HOME_RECONCILE:-0}" = "1" ]; then
  reconcile_login_home_defaults
fi

login_home_owner="$(stat -c '%u:%g' "$LOGIN_HOME")"
if [ "$login_home_owner" != "$LOGIN_UID:$LOGIN_GID" ]; then
  echo "run_hermetic_sshd: login home ownership mismatch for $LOGIN_HOME (expected $LOGIN_UID:$LOGIN_GID, got $login_home_owner)" >&2
  exit 1
fi

login_home_mode="$(stat -c '%a' "$LOGIN_HOME")"
if [ "$login_home_mode" != "700" ]; then
  echo "run_hermetic_sshd: login home permissions must be 700 for $LOGIN_HOME (got $login_home_mode)" >&2
  exit 1
fi

install -d -m 700 "$RUNTIME_DIR"
mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"
touch "$LOG_FILE"
chmod 600 "$LOG_FILE"

case "$HOSTKEY_DIR" in
  "$STATE_DIR"/*) ;;
  *)
    echo "run_hermetic_sshd: HOSTKEY_DIR must live under STATE_DIR (got $HOSTKEY_DIR, state root $STATE_DIR)" >&2
    exit 1
    ;;
esac

install -d -m 700 "$HOSTKEY_DIR"

if [ ! -f "$HOSTKEY_ED25519" ]; then
  ssh-keygen -t ed25519 -f "$HOSTKEY_ED25519" -N ""
fi

if [ ! -f "$HOSTKEY_RSA" ]; then
  ssh-keygen -t rsa -b 3072 -f "$HOSTKEY_RSA" -N ""
fi

chmod 600 "$HOSTKEY_ED25519" "$HOSTKEY_RSA" >/dev/null 2>&1 || true

render_sshd_config() {
  local template_config="$1"
  local runtime_config="$2"

  awk -v authorized_keys="$AUTHORIZED_KEYS" '
    BEGIN { replaced = 0 }
    /^[[:space:]]*AuthorizedKeysFile[[:space:]]+/ {
      print "AuthorizedKeysFile " authorized_keys
      replaced = 1
      next
    }
    { print }
    END {
      if (replaced != 1) {
        exit 1
      }
    }
  ' "$template_config" >"$runtime_config"
  chmod 644 "$runtime_config"
}

render_sshd_config "$CONFIG_FILE" "$RUNTIME_CONFIG_FILE"
CONFIG_FILE="$RUNTIME_CONFIG_FILE"

echo "Validating sshd config..."
"$SSHD_BIN" -t -f "$CONFIG_FILE"

# Stop any previous instance using pid file (best-effort).
if [ -f "$PID_FILE" ]; then
  PREV_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$PREV_PID" ] && ps -p "$PREV_PID" -o comm= 2>/dev/null | grep -q "sshd"; then
    kill "$PREV_PID" >/dev/null 2>&1 || true
  fi
fi

echo "sshd running on port $PORT for login user $LOGIN_USER"

exec "$SSHD_BIN" -f "$CONFIG_FILE" -p "$PORT" -h "$HOSTKEY_ED25519" -h "$HOSTKEY_RSA" -E "$LOG_FILE" -D -e
