FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + ensure conda is in PATH for all shells
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git tmux curl openssh-client openssh-server \
    nvtop htop && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/bash.bashrc && \
    ln -sf /opt/conda/bin/python /usr/bin/python && \
    ln -sf /opt/conda/bin/python /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Node.js v20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install global npm packages (gemini CLI, Claude Code CLI)
RUN npm install -g @google/gemini-cli @anthropic-ai/claude-code

# Install Python dev tools
RUN pip install --no-cache-dir ruff pytest uv

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml ./

# Install dependencies directly to system Python
RUN pip install --no-cache-dir \
    "accelerate>=1.11.0" \
    "datasets>=4.3.0" \
    "hydra-colorlog>=1.2.0" \
    "hydra-core>=1.3.2" \
    "lm-eval>=0.4.9" \
    "rouge-score>=0.1.2" \
    "scikit-learn>=1.7.2" \
    "scipy>=1.16.2" \
    "tensorboard>=2.20.0" \
    "transformers>=4.57.1" \
    "deepspeed>=0.18.4" \
    "ninja>=1.13.0" \
    "packaging>=25.0"

# Build flash-attn from source (needs torch visible)
RUN pip install flash-attn==2.8.3 --no-build-isolation

# Copy code
COPY . .

# Environment variables
ENV IN_DOCKER=1

# Git config
RUN git config --global user.name "runpod_zihan" && \
    git config --global user.email "32484940+wzh4464@users.noreply.github.com"

# Setup dotfiles (tmux config) - will be done at runtime if /workspace/dotfiles exists
RUN echo '[ -d /workspace/dotfiles ] && ln -sf /workspace/dotfiles/tmux/tmux.conf ~/.tmux.conf' >> /etc/bash.bashrc

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
