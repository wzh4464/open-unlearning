#!/bin/bash
set -e

# === Environment Variables ===
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/workspace/tmp

# === Create workspace directories ===
mkdir -p /workspace/.cache/huggingface \
         /workspace/.config \
         /workspace/saves \
         /workspace/data \
         /workspace/tmp

# === Setup HF token from RunPod Secret (environment variable) ===
if [ -n "$HF_TOKEN" ]; then
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    echo "HF token loaded from environment variable"
fi

# === Setup SSH from RunPod Secret ===
# SSH_PRIVATE_KEY should be base64 encoded in RunPod Secrets
if [ -n "$SSH_PRIVATE_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$SSH_PRIVATE_KEY" | base64 -d > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519

    # Generate public key from private key
    ssh-keygen -y -f ~/.ssh/id_ed25519 > ~/.ssh/id_ed25519.pub 2>/dev/null || true

    # Add GitHub to known hosts
    ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null

    echo "SSH key loaded from environment variable"
fi

# === Setup SSH config from workspace (non-sensitive) ===
if [ -f /workspace/.config/ssh_config ]; then
    mkdir -p ~/.ssh
    cp /workspace/.config/ssh_config ~/.ssh/config
    chmod 644 ~/.ssh/config
    echo "SSH config loaded from /workspace/.config/ssh_config"
fi

# === Setup SSH server for remote access ===
if [ -n "$PUBLIC_KEY" ]; then
    echo "Configuring SSH server..."
    mkdir -p /var/run/sshd
    # Root SSH setup
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    # Also add to zihan user
    mkdir -p /home/zihan/.ssh
    echo "$PUBLIC_KEY" >> /home/zihan/.ssh/authorized_keys
    chmod 700 /home/zihan/.ssh
    chmod 600 /home/zihan/.ssh/authorized_keys
    chown -R zihan:zihan /home/zihan/.ssh
    ssh-keygen -A
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
    if ! pgrep -x sshd >/dev/null 2>&1; then
        /usr/sbin/sshd
        echo "SSHD started on port 22"
    else
        echo "SSHD already running"
    fi
fi

# === Setup zihan user environment (idempotent) ===
if ! grep -q 'IN_DOCKER=1' /home/zihan/.bashrc 2>/dev/null; then
    cat >> /home/zihan/.bashrc << 'ZIHAN_ENV'
export PATH=/opt/conda/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TMPDIR=/workspace/tmp
export IN_DOCKER=1
export GOOGLE_CLOUD_PROJECT=unlearning-484901
[ -d /workspace/dotfiles ] && ln -sf /workspace/dotfiles/tmux/tmux.conf ~/.tmux.conf
ZIHAN_ENV
    [ -f /workspace/.config/bashrc_custom ] && cat /workspace/.config/bashrc_custom >> /home/zihan/.bashrc
    chown zihan:zihan /home/zihan/.bashrc
fi

# === Propagate HF token to zihan ===
if [ -n "$HF_TOKEN" ]; then
    mkdir -p /home/zihan/.cache/huggingface
    echo "$HF_TOKEN" > /home/zihan/.cache/huggingface/token
    chown -R zihan:zihan /home/zihan/.cache
fi

# === Propagate SSH private key to zihan ===
if [ -n "$SSH_PRIVATE_KEY" ]; then
    mkdir -p /home/zihan/.ssh
    echo "$SSH_PRIVATE_KEY" | base64 -d > /home/zihan/.ssh/id_ed25519
    chmod 600 /home/zihan/.ssh/id_ed25519
    ssh-keygen -y -f /home/zihan/.ssh/id_ed25519 > /home/zihan/.ssh/id_ed25519.pub 2>/dev/null || true
    ssh-keyscan -t ed25519 github.com >> /home/zihan/.ssh/known_hosts 2>/dev/null
    chown -R zihan:zihan /home/zihan/.ssh
fi

# === Propagate SSH config to zihan ===
if [ -f /workspace/.config/ssh_config ]; then
    mkdir -p /home/zihan/.ssh
    cp /workspace/.config/ssh_config /home/zihan/.ssh/config
    chmod 644 /home/zihan/.ssh/config
    chown -R zihan:zihan /home/zihan/.ssh
fi

# Give zihan ownership of /app and /workspace
chown -R zihan:zihan /app 2>/dev/null || true
chown -R zihan:zihan /workspace 2>/dev/null || true

# === Setup Git config from workspace (non-sensitive) ===
if [ -f /workspace/.config/gitconfig ]; then
    cp /workspace/.config/gitconfig ~/.gitconfig
    echo "Git config loaded from /workspace/.config/gitconfig"
fi

# === Load custom shell config (if exists) ===
if [ -f /workspace/.config/bashrc_custom ]; then
    source /workspace/.config/bashrc_custom
    echo "Custom bashrc loaded from /workspace/.config/bashrc_custom"
fi

# === Symlink saves/data to persistent storage ===
[ -L /app/saves ] && [ "$(readlink /app/saves)" = "/workspace/saves" ] || ln -sfn /workspace/saves /app/saves
[ -L /app/data ] && [ "$(readlink /app/data)" = "/workspace/data" ] || ln -sfn /workspace/data /app/data

# === Setup /app git repo (idempotent) ===
if [ ! -d /app/.git ]; then
    cd /app
    su - zihan -c 'cd /app && git init -b main && git remote add origin git@github.com:wzh4464/open-unlearning.git && git fetch origin && git reset origin/main'
    echo "Git repo initialized in /app"
fi

# === First-run initialization ===
if [ ! -f /workspace/.initialized ]; then
    echo "=== First run: downloading evaluation data ==="
    cd /app

    echo "Downloading eval_logs..."
    python setup_data.py --eval_logs || { echo "ERROR: Failed to download eval_logs"; exit 1; }

    echo "Downloading idk dataset..."
    python setup_data.py --idk || { echo "ERROR: Failed to download idk"; exit 1; }

    echo "Downloading WMDP..."
    python setup_data.py --wmdp || { echo "ERROR: Failed to download WMDP"; exit 1; }

    touch /workspace/.initialized
    echo "=== Initialization complete ==="
fi

# If no command provided (RunPod sometimes overrides CMD), keep container alive
if [ $# -eq 0 ]; then
    echo "No CMD provided. Defaulting to: sleep infinity"
    set -- sleep infinity
fi

exec "$@"
