#!/usr/bin/env bash
# =============================================================================
# explorer/scripts/workflow.sh
#
# Reference workflow for the vLLM Explorer kit (local developer-experience tool).
# Orchestrates the full build sequence from PLAN.md: data generation -> install
# -> build -> dev. Idempotent and safe to re-run.
#
# This script is documentation-as-code: each step mirrors a step in PLAN.md and
# the plan's Verification section. Run the whole thing, or a single phase:
#
#     ./scripts/workflow.sh            # run everything (gen-data, install, build)
#     ./scripts/workflow.sh gen-data   # only regenerate public/data/*.json
#     ./scripts/workflow.sh install    # only npm install
#     ./scripts/workflow.sh build      # only npm run build
#     ./scripts/workflow.sh dev        # start the vite dev server
#     ./scripts/workflow.sh verify     # build + reminder checklist
#
# AGENTS.md rule: never use system python3 / bare pip. All Python runs through
# the repo's uv-managed .venv at <repo-root>/.venv/bin/python.
# =============================================================================
set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # explorer/scripts
EXPLORER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"               # explorer
REPO_ROOT="$(cd "${EXPLORER_DIR}/.." && pwd)"                # vllm repo root
VENV_PY="${REPO_ROOT}/.venv/bin/python"
DATA_DIR="${EXPLORER_DIR}/public/data"

# --- Pretty logging ----------------------------------------------------------
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Prerequisite checks -----------------------------------------------------
check_node() {
  command -v node >/dev/null 2>&1 || die "node not found (need Node 18+ for Vite 8)"
  command -v npm  >/dev/null 2>&1 || die "npm not found"
}

check_venv() {
  if [[ ! -x "${VENV_PY}" ]]; then
    warn "uv venv not found at ${VENV_PY}"
    warn "create it per AGENTS.md:  uv venv --python 3.12 && source .venv/bin/activate"
    die  "data generation requires the repo .venv"
  fi
}

# --- Phases ------------------------------------------------------------------

# Generate the static JSON manifests the SPA consumes. Needs network once
# (fineweb-edu subset + Qwen3 tokenizer/config from Hugging Face).
gen_data() {
  check_venv
  mkdir -p "${DATA_DIR}"

  log "Generating component manifest from HACKERS_GUIDE.md + hacks/ ..."
  "${VENV_PY}" "${SCRIPT_DIR}/build_component_manifest.py" \
      --repo-root "${REPO_ROOT}" --out "${DATA_DIR}/components.json"

  log "Generating fineweb-edu sample + Qwen3 tokenization ..."
  "${VENV_PY}" "${SCRIPT_DIR}/build_fineweb_sample.py" \
      --rows "${FINEWEB_ROWS:-200}" --out-dir "${DATA_DIR}"

  log "Generating model architecture manifests (all models + index) ..."
  "${VENV_PY}" "${SCRIPT_DIR}/build_model_arch.py" \
      --repo-root "${REPO_ROOT}" --out-dir "${DATA_DIR}/models"

  log "Data written to ${DATA_DIR}"
}

install_deps() {
  check_node
  log "Installing npm dependencies ..."
  ( cd "${EXPLORER_DIR}" && npm install )
}

build_app() {
  check_node
  log "Type-checking + building (tsc -b + vite build) ..."
  ( cd "${EXPLORER_DIR}" && npm run build )
}

dev_server() {
  check_node
  log "Starting Vite dev server (Ctrl-C to stop) ..."
  ( cd "${EXPLORER_DIR}" && npm run dev )
}

verify() {
  build_app
  cat <<'EOF'

Manual verification checklist (see PLAN.md > Verification):
  [ ] .observatory-bg + dark theme + Inter/Fira fonts load
  [ ] family switcher toggles 3 modes; aria-pressed / skip-link / :focus-visible work
  [ ] Data mode: fineweb-edu schema + recharts dists + Qwen3 tokenization vs token_count
  [ ] Component mode: clickable graph; drawer shows guide prose + working file:line + hack
  [ ] Architecture mode: 15-block Qwen3 diagram; lens recomputes; block drawer; GPUModelRunner link
  [ ] 1024px breakpoint collapses to single column
EOF
}

run_all() {
  gen_data
  install_deps
  build_app
  log "Done. Start the UI with: ./scripts/workflow.sh dev"
}

# --- Dispatch ----------------------------------------------------------------
case "${1:-all}" in
  all)       run_all ;;
  gen-data)  gen_data ;;
  install)   install_deps ;;
  build)     build_app ;;
  dev)       dev_server ;;
  verify)    verify ;;
  *)         die "unknown phase '$1' (use: all | gen-data | install | build | dev | verify)" ;;
esac
