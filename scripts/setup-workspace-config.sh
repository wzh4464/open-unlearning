#!/bin/bash
# Run this locally to prepare non-sensitive config files for RunPod
# Usage: ./scripts/setup-workspace-config.sh ./workspace

WORKSPACE_DIR="${1:-./workspace}"
CONFIG_DIR="$WORKSPACE_DIR/.config"

mkdir -p "$CONFIG_DIR"

echo "=== Copying SSH config (non-sensitive) ==="
if [ -f ~/.ssh/config ]; then
    cp ~/.ssh/config "$CONFIG_DIR/ssh_config"
    echo "SSH config copied"
else
    echo "No SSH config found"
fi

echo "=== Copying Git config (non-sensitive) ==="
# Remove signing key from git config for container use
if [ -f ~/.gitconfig ]; then
    grep -v "signingkey\|gpgsign" ~/.gitconfig > "$CONFIG_DIR/gitconfig" || cp ~/.gitconfig "$CONFIG_DIR/gitconfig"
    echo "Git config copied (GPG signing disabled)"
else
    echo "No git config found"
fi

echo "=== Creating custom bashrc template ==="
cat > "$CONFIG_DIR/bashrc_custom" << 'EOF'
# Custom shell configuration for RunPod
# Add your aliases and exports here

# CUDA paths (usually already set in container)
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Useful aliases
alias ll='ls -alF'
alias la='ls -A'

# GPG for git signing (disabled in container)
# export GPG_TTY=$(tty)
EOF

echo "=== Done! ==="
echo "Config files created in: $CONFIG_DIR"
echo ""
echo "Directory structure:"
ls -la "$CONFIG_DIR"
echo ""
echo "These are NON-SENSITIVE configs. Upload to /workspace/.config on RunPod."
echo ""
echo "==============================================================================="
echo "For SENSITIVE data, set up RunPod Secrets:"
echo ""
echo "1. HF_TOKEN - Get from: https://huggingface.co/settings/tokens"
echo "   RunPod: Settings → Secrets → Add Secret"
echo "   Name: HF_TOKEN"
echo "   Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
echo ""
echo "2. SSH_PRIVATE_KEY - Base64 encode your SSH key:"
echo "   cat ~/.ssh/id_ed25519 | base64 -w 0"
echo "   RunPod: Settings → Secrets → Add Secret"
echo "   Name: SSH_PRIVATE_KEY"
echo "   Value: (paste base64 encoded key)"
echo ""
echo "3. In RunPod Template, set environment variables:"
echo "   HF_TOKEN: {{ RUNPOD_SECRET_HF_TOKEN }}"
echo "   SSH_PRIVATE_KEY: {{ RUNPOD_SECRET_SSH_PRIVATE_KEY }}"
echo "==============================================================================="
