set shell := ["bash", "-euo", "pipefail", "-c"]

default:
  @just --list

up:
  ./devx/launch_container.sh up

down:
  ./devx/launch_container.sh down

restart:
  ./devx/launch_container.sh restart

ps:
  ./devx/launch_container.sh ps

logs service="":
  if [[ -n "{{service}}" ]]; then \
    ./devx/launch_container.sh logs --tail=200 "{{service}}"; \
  else \
    ./devx/launch_container.sh logs --tail=200; \
  fi

ssh:
  if [[ -r "$HOME/.ssh/devx_access" ]]; then \
    exec ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/devx_access" -p 2222 kvothe@127.0.0.1; \
  else \
    exec ssh -o IdentitiesOnly=yes -o IdentityAgent=none -p 2222 kvothe@127.0.0.1; \
  fi

rebuild-main:
  ./devx/launch_container.sh build devshell

rebuild-runtime:
  ./devx/launch_container.sh build vllm-runtime

up-rebuild:
  ./devx/launch_container.sh up-rebuild

verify-e2e:
  ./devx/launch_container.sh verify-e2e

doctor:
  ./devx/bin/devx doctor --preset qwen3-0.6b

devx-up preset="qwen3-0.6b":
  ./devx/bin/devx up --preset {{preset}}

devx-switch preset:
  ./devx/bin/devx switch --preset {{preset}}

devx-query prompt:
  ./devx/bin/devx query --prompt "{{prompt}}"

devx-exp-run manifest:
  ./devx/bin/devx experiment run -f {{manifest}}

devx-exp-report run_id:
  ./devx/bin/devx experiment report {{run_id}}

test-ssh:
  bash ./devx/tests/test_compose_runner.sh

test-verify-e2e:
  bash ./devx/tests/test_verify_e2e.sh

test-devx:
  just test-ssh
  just test-verify-e2e
