FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + ensure conda is in PATH for all shells
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git tmux curl openssh-client openssh-server \
    nvtop htop sudo tini && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/bash.bashrc && \
    ln -sf /opt/conda/bin/python /usr/bin/python && \
    ln -sf /opt/conda/bin/python /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Node.js v20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install global npm packages (gemini CLI + Claude Code CLI)
# Note: Using npm for Claude Code to avoid geo-blocking from claude.ai/install.sh
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
# npm global bin is already in PATH via node installation
# Keep /root/.local/bin for other tools
ENV PATH="/root/.local/bin:${PATH}"
ENV GOOGLE_CLOUD_PROJECT=unlearning-484901

# Persist environment variables for interactive shells (RunPod web terminal, SSH)
RUN echo 'export IN_DOCKER=1' >> /etc/bash.bashrc && \
    echo 'export GOOGLE_CLOUD_PROJECT=unlearning-484901' >> /etc/bash.bashrc && \
    echo 'IN_DOCKER=1' >> /etc/environment && \
    echo 'GOOGLE_CLOUD_PROJECT=unlearning-484901' >> /etc/environment

# Create user zihan with passwordless sudo
RUN useradd -m -s /bin/bash -u 1000 zihan && \
    echo "zihan ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/zihan && \
    chmod 440 /etc/sudoers.d/zihan && \
    mkdir -p /home/zihan/.ssh && \
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN5NGqyhMn7dzEYPHbBYMV+e4EAU9U7N3uqnCYyJIjhw wzh4464@gmail.com" > /home/zihan/.ssh/authorized_keys && \
    chmod 700 /home/zihan/.ssh && \
    chmod 600 /home/zihan/.ssh/authorized_keys && \
    chown -R zihan:zihan /home/zihan

# Pre-configure GitHub SSH (key injected at runtime via GITHUB_SSH_KEY env var)
RUN printf "Host github.com\n    IdentityFile ~/.ssh/id_ed25519_github\n    IdentitiesOnly yes\n" > /home/zihan/.ssh/config && \
    chmod 600 /home/zihan/.ssh/config && \
    ssh-keyscan -t ed25519 github.com >> /home/zihan/.ssh/known_hosts 2>/dev/null && \
    chown -R zihan:zihan /home/zihan/.ssh

# Set /app ownership to zihan at build time
RUN chown -R zihan:zihan /app

# Git config (root)
RUN git config --global user.name "runpod_zihan" && \
    git config --global user.email "32484940+wzh4464@users.noreply.github.com"

# Git config (zihan)
RUN su - zihan -c 'git config --global user.name "runpod_zihan" && git config --global user.email "32484940+wzh4464@users.noreply.github.com"'

# Setup dotfiles (tmux config) - will be done at runtime if /workspace/dotfiles exists
RUN echo '[ -d /workspace/dotfiles ] && ln -sf /workspace/dotfiles/tmux/tmux.conf ~/.tmux.conf' >> /etc/bash.bashrc

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["sleep", "infinity"]
