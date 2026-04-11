ARG BASE_IMAGE="vllm/vllm-openai:latest"
FROM ${BASE_IMAGE}

ARG USERNAME="kvothe"
ARG USER_UID="1000"
ARG USER_GID="1000"

USER root

RUN groupadd --non-unique --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" \
      --gid "${USER_GID}" \
      --create-home \
      --home-dir "/home/${USERNAME}" \
      --shell /bin/bash \
      --non-unique \
      -p "" \
      "${USERNAME}" && \
    install -d -m 700 -o "${USER_UID}" -g "${USER_GID}" \
      "/home/${USERNAME}/.cache" \
      "/home/${USERNAME}/.cache/huggingface" \
      "/var/log/devx-vllm-runtime"

COPY devx/start_vllm_runtime.sh /opt/devx/start_vllm_runtime.sh

RUN chmod 755 /opt/devx/start_vllm_runtime.sh

USER ${USERNAME}
WORKDIR /workspace

ENTRYPOINT []
CMD ["/opt/devx/start_vllm_runtime.sh"]
