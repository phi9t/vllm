#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  echo "start_main: main devbox must start as root" >&2
  exit 1
fi

if [ "${LOGIN_USER:-kvothe}" != "kvothe" ]; then
  echo "start_main: LOGIN_USER must be kvothe" >&2
  exit 1
fi

if [ -z "${HOST_UID:-}" ] || [ -z "${HOST_GID:-}" ]; then
  echo "start_main: HOST_UID and HOST_GID are required" >&2
  exit 1
fi

exec "${SCRIPT_DIR}/run_hermetic_sshd.sh"
