#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if bash "${REPO_ROOT}/devx/bin/devx" help >/tmp/devx-help.out 2>/tmp/devx-help.err; then
  grep -F "Usage: devx" /tmp/devx-help.out >/dev/null
else
  echo "devx help should succeed once dispatcher exists" >&2
  exit 1
fi

if bash "${REPO_ROOT}/devx/bin/devx" unknown >/tmp/devx-unknown.out 2>/tmp/devx-unknown.err; then
  echo "unknown command must fail" >&2
  exit 1
fi
grep -F "unknown subcommand" /tmp/devx-unknown.err >/dev/null

echo "PASS"
