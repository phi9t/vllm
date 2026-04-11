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
    exec ssh -p 2222 kvothe@127.0.0.1; \
  fi

rebuild-main:
  ./devx/launch_container.sh build devshell

rebuild-runtime:
  ./devx/launch_container.sh build vllm-runtime
