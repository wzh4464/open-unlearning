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
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
    /usr/sbin/sshd
    echo "SSHD started on port 22"
fi

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

exec "$@"
