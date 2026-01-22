FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + Slurm + ensure conda is in PATH for all shells
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git tmux curl openssh-client openssh-server \
    nvtop htop \
    slurm-wlm slurm-client munge tini \
    && echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh \
    && echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/bash.bashrc \
    && ln -sf /opt/conda/bin/python /usr/bin/python \
    && ln -sf /opt/conda/bin/python /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Node.js v20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install global npm packages (gemini CLI, Claude Code CLI)
# Note: Using npm for Claude Code to avoid geo-blocking from claude.ai/install.sh
RUN npm install -g @google/gemini-cli @anthropic-ai/claude-code

# Install Python dev tools
RUN pip install --no-cache-dir ruff pytest uv

# Create Slurm directories (configs will be copied later)
RUN mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd \
    /var/log/slurm /run/munge && \
    chown -R root:root /etc/slurm && \
    chmod 755 /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

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

# Copy code (including configs/slurm/)
COPY . .

# === Slurm configuration (after COPY to allow config changes without cache invalidation) ===
RUN cp /app/configs/slurm/slurm.conf /etc/slurm/slurm.conf && \
    cp /app/configs/slurm/cgroup.conf /etc/slurm/cgroup.conf && \
    cp /app/configs/slurm/gres.conf /etc/slurm/gres.conf && \
    cp /app/configs/slurm/munge.key /etc/munge/munge.key && \
    chmod 400 /etc/munge/munge.key && \
    chown munge:munge /etc/munge/munge.key

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

# Git config
RUN git config --global user.name "runpod_zihan" && \
    git config --global user.email "32484940+wzh4464@users.noreply.github.com"

# Setup dotfiles (tmux config) - will be done at runtime if /workspace/dotfiles exists
RUN echo '[ -d /workspace/dotfiles ] && ln -sf /workspace/dotfiles/tmux/tmux.conf ~/.tmux.conf' >> /etc/bash.bashrc

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["sleep", "infinity"]
