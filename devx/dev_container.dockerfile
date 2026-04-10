# Use the same upstream CUDA family as the rest of the dev stack.
ARG BASE_IMAGE="nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
FROM ${BASE_IMAGE}

ARG USERNAME=kvothe
ARG USER_UID=1000
ARG USER_GID=1000
ARG UV_VERSION=""

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    autoconf \
    automake \
    bzip2 \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    emacs-nox \
    g++ \
    gfortran \
    git \
    jq \
    locales \
    openssh-client \
    openssh-server \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    tar \
    tmux \
    wget \
    zsh \
    && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN groupadd --non-unique --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --home-dir "/home/${USERNAME}" --shell /bin/zsh --non-unique -p "" "${USERNAME}" && \
    install -d -m 700 -o "${USER_UID}" -g "${USER_GID}" "/home/${USERNAME}"

RUN usermod -aG sudo "${USERNAME}" && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

RUN if [ -n "${UV_VERSION}" ]; then \
        python3 -m pip install --break-system-packages --no-cache-dir "uv==${UV_VERSION}"; \
    else \
        python3 -m pip install --break-system-packages --no-cache-dir uv; \
    fi

RUN install -d -m 755 \
        /opt/devx \
        /opt/devx/skel \
        /opt/devx/skel/.config \
        /opt/devx/skel/.local \
        /opt/devx/skel/.local/bin \
        /opt/devx/skel/.local/share \
        /opt/devx/skel/.local/state \
        /opt/devx/skel/.ssh \
        /opt/devx/skel/.tmux \
        /run/devx \
        /run/sshd \
        /var/lib/devx-sshd \
        /var/lib/devx-sshd/hostkeys \
        /workspace && \
    install -d -m 700 \
        /home/${USERNAME}/.cache \
        /home/${USERNAME}/.config \
        /home/${USERNAME}/.local \
        /home/${USERNAME}/.local/bin \
        /home/${USERNAME}/.local/share \
        /home/${USERNAME}/.local/state \
        /home/${USERNAME}/.ssh

RUN cat <<'EOF' >/opt/devx/skel/.zshenv
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export EDITOR="${EDITOR:-emacs}"
export VISUAL="${VISUAL:-$EDITOR}"
export PAGER="${PAGER:-less}"
EOF

RUN cat <<'EOF' >/opt/devx/skel/.zshrc
if [[ -o interactive ]]; then
  setopt autocd
  setopt hist_ignore_all_dups
  setopt share_history
  HISTSIZE=10000
  SAVEHIST=10000
  HISTFILE="$HOME/.zsh_history"
  PROMPT='%n@%m:%~%# '
fi
EOF

RUN cat <<'EOF' >/opt/devx/skel/.tmux/.tmux.conf
set -g mouse on
set -g history-limit 50000
set -g escape-time 0
setw -g mode-keys vi
EOF

RUN ln -sfn .tmux/.tmux.conf /opt/devx/skel/.tmux.conf && \
    chmod 644 /opt/devx/skel/.zshenv /opt/devx/skel/.zshrc /opt/devx/skel/.tmux/.tmux.conf && \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

COPY devx/authorized_keys /opt/devx/authorized_keys
COPY devx/run_hermetic_sshd.sh /opt/devx/run_hermetic_sshd.sh
COPY devx/sshd_config /opt/devx/sshd_config
COPY devx/start_main.sh /opt/devx/start_main.sh

RUN chmod 755 /opt/devx/run_hermetic_sshd.sh /opt/devx/start_main.sh && \
    chmod 644 /opt/devx/authorized_keys /opt/devx/sshd_config

WORKDIR /workspace

CMD ["/opt/devx/start_main.sh"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD pgrep -x sshd >/dev/null || exit 1
