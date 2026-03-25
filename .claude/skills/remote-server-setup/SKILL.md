---
name: remote-server-setup
description: Automate provisioning and syncing of remote dev servers (RunPod/Docker containers) for open-unlearning — SSH, git, Claude Code, Codex, user accounts, and config sync
---

# Remote Server Setup & Sync

Use this skill when:
- Setting up a new RunPod pod or Docker container for development
- Syncing local Claude Code config to a remote server
- Troubleshooting remote environment issues (OOM, SSH, git)

## Server Connection Info

Default connection pattern (update per-pod):
```bash
SSH_HOST="157.66.255.18"
SSH_PORT="11476"
SSH_USER="root"
SSH_KEY="~/.ssh/id_ed25519"
# shorthand: ssh root@$SSH_HOST -p $SSH_PORT -i $SSH_KEY
```

The Docker image `wzh4464/open-unlearning:latest` is the base. It ships with:
- Ubuntu 22.04, CUDA 12.6.3, PyTorch 2.9.1, flash-attn 2.8.3
- Node.js 20.x, Claude Code, Gemini CLI
- Conda at `/opt/conda`, working dir `/app`
- `.git` excluded via `.dockerignore` — must be re-initialized

## Setup Checklist

When asked to "set up a remote server" or "provision a new pod", execute these steps in order.
Ask the user for SSH connection details first if not provided.

### 1. Verify Base Environment
```bash
ssh $REMOTE "uname -a && node --version && python3 --version && which claude && claude --version"
```

### 2. Install Additional CLI Tools

**Codex (if missing):**
```bash
ssh $REMOTE "npm install -g @openai/codex"
```

**Yazi file manager (Ubuntu 22.04 compatible — use v0.4.2, latest requires GLIBC 2.39):**
```bash
ssh $REMOTE "curl -sL https://github.com/sxyazi/yazi/releases/download/v0.4.2/yazi-x86_64-unknown-linux-gnu.zip -o /tmp/yazi.zip && cd /tmp && unzip -o yazi.zip && cp yazi-x86_64-unknown-linux-gnu/yazi /usr/local/bin/ && cp yazi-x86_64-unknown-linux-gnu/ya /usr/local/bin/ && chmod +x /usr/local/bin/yazi /usr/local/bin/ya && rm -rf /tmp/yazi.zip /tmp/yazi-x86_64-unknown-linux-gnu"
```

### 3. Create Dev User (if needed)
```bash
ssh $REMOTE "useradd -m -s /bin/bash -G root,sudo zihan && echo 'zihan:zihan' | chpasswd"
```

Then copy authorized_keys so the user can SSH in:
```bash
ssh $REMOTE "mkdir -p /home/zihan/.ssh && cat /root/.ssh/authorized_keys >> /home/zihan/.ssh/authorized_keys && chown -R zihan:zihan /home/zihan/.ssh && chmod 700 /home/zihan/.ssh && chmod 600 /home/zihan/.ssh/authorized_keys"
```

### 4. Configure SSH Keys for GitHub

**Option A — Copy local key:**
```bash
scp -P $SSH_PORT -i $SSH_KEY ~/.ssh/id_rsa ~/.ssh/id_rsa.pub $SSH_USER@$SSH_HOST:/home/zihan/.ssh/
ssh $REMOTE "chown zihan:zihan /home/zihan/.ssh/id_rsa /home/zihan/.ssh/id_rsa.pub && chmod 600 /home/zihan/.ssh/id_rsa"
```

**Option B — Generate new key:**
```bash
ssh $REMOTE "ssh-keygen -t ed25519 -C 'runpod-container' -f /home/zihan/.ssh/id_ed25519 -N ''"
# Then add the public key to GitHub
```

**Always add GitHub to known_hosts and create SSH config:**
```bash
ssh $REMOTE "ssh-keyscan github.com >> /home/zihan/.ssh/known_hosts"
ssh $REMOTE "cat > /home/zihan/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
EOF
chmod 600 /home/zihan/.ssh/config"
```

**Verify:** `ssh $REMOTE "su - zihan -c 'ssh -T git@github.com 2>&1'"` should show `Hi wzh4464!`

### 5. Re-initialize Git in /app

The Docker image uses `COPY . .` but `.dockerignore` excludes `.git`. Fix:
```bash
ssh $REMOTE "cd /app && git init && git remote add origin git@github.com:wzh4464/open-unlearning.git && git remote add upstream git@github.com:locuslab/open-unlearning.git && git fetch origin && git fetch upstream"
ssh $REMOTE "cd /app && git checkout -b main && git reset --mixed origin/main && git branch --set-upstream-to=origin/main main"
```

Add safe.directory for both users:
```bash
ssh $REMOTE "git config --global --add safe.directory /app"
ssh $REMOTE "su - zihan -c 'git config --global --add safe.directory /app'"
```

### 6. Sync Claude Code Configuration

**settings.json** (adapt plugins to what's available on server — skip LSP plugins):
```bash
ssh $REMOTE "cat > /home/zihan/.claude/settings.json << 'EOF'
{
  \"\$schema\": \"https://json.schemastore.org/claude-code-settings.json\",
  \"env\": {
    \"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS\": \"1\"
  },
  \"permissions\": {
    \"allow\": [
      \"Bash(git:*)\",
      \"Bash(gh:*)\",
      \"mcp__plugin_serena_serena__*\"
    ],
    \"deny\": []
  },
  \"enabledPlugins\": {
    \"github@claude-plugins-official\": true,
    \"serena@claude-plugins-official\": true,
    \"superpowers@claude-plugins-official\": true
  },
  \"alwaysThinkingEnabled\": true,
  \"effortLevel\": \"high\",
  \"skipDangerousModePermissionPrompt\": true
}
EOF"
```

**CLAUDE.md:**
```bash
scp -P $SSH_PORT -i $SSH_KEY ~/.claude/CLAUDE.md $SSH_USER@$SSH_HOST:/home/zihan/.claude/CLAUDE.md
```

**Custom skills:**
```bash
scp -r -P $SSH_PORT -i $SSH_KEY ~/.claude/skills/* $SSH_USER@$SSH_HOST:/home/zihan/.claude/skills/
```

**Mirror to root user:**
```bash
ssh $REMOTE "cp /home/zihan/.claude/settings.json /root/.claude/settings.json && cp /home/zihan/.claude/CLAUDE.md /root/.claude/CLAUDE.md && cp -r /home/zihan/.claude/skills /root/.claude/"
```

**Fix ownership:**
```bash
ssh $REMOTE "chown -R zihan:zihan /home/zihan/.claude"
```

### 7. Sync gitconfig

Server gitconfig should match local but skip desktop-specific settings (GPG, vscode editor):
```bash
ssh $REMOTE "cat > /home/zihan/.gitconfig << 'EOF'
[user]
	email = 32484940+wzh4464@users.noreply.github.com
	name = Hance_Wu_M2
[filter \"lfs\"]
	clean = git-lfs clean -- %f
	smudge = git-lfs smudge -- %f
	process = git-lfs filter-process
	required = true
[core]
	quotepath = false
	fileMode = false
[push]
	autoSetupRemote = true
	default = simple
[safe]
	directory = /app
EOF"
```

### 8. Install Plugins (interactive — user must run)

After SSH into the server, run in Claude Code:
```
/install-plugin superpowers@claude-plugins-official
/install-plugin github@claude-plugins-official
/reload-plugins
```

### 9. OAuth Login (interactive — user must run)

Claude Code OAuth requires a TTY. User must run:
```bash
ssh -tt $SSH_USER@$SSH_HOST -p $SSH_PORT -i $SSH_KEY "claude login"
```

## Troubleshooting

### OOM During flash-attn Compilation
flash-attn compiles CUDA kernels from source and uses heavy memory (10-30 min build time).
- **Do not run Claude Code simultaneously** — Node.js will OOM
- Check memory: `free -h`
- If stuck: `Ctrl+C` the compilation, use pre-installed version from Docker image instead
- Increase Node.js memory if needed: `export NODE_OPTIONS="--max-old-space-size=4096"`

### Docker Image Inspection Without Pulling
You can inspect any Docker Hub image remotely via registry API:
```bash
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:OWNER/REPO:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
# Get manifest
curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.docker.distribution.manifest.v2+json" "https://registry-1.docker.io/v2/OWNER/REPO/manifests/TAG"
# Get config (Dockerfile history, env vars, etc.)
curl -sL -H "Authorization: Bearer $TOKEN" "https://registry-1.docker.io/v2/OWNER/REPO/blobs/$CONFIG_DIGEST" -o config.json
```

### Yazi Version Compatibility
Ubuntu 22.04 ships GLIBC 2.35. Latest yazi requires GLIBC 2.39+. Use **v0.4.2** as the last compatible release.

### SSH Permission Denied for New User
Ensure `authorized_keys` is copied from root and permissions are correct:
```bash
chmod 700 /home/USER/.ssh
chmod 600 /home/USER/.ssh/authorized_keys
chown -R USER:USER /home/USER/.ssh
```

## Config Drift Detection

To check what's out of sync between local and remote:
```bash
# Compare settings.json
diff <(ssh $REMOTE "cat /home/zihan/.claude/settings.json") ~/.claude/settings.json
# Compare CLAUDE.md
diff <(ssh $REMOTE "cat /home/zihan/.claude/CLAUDE.md") ~/.claude/CLAUDE.md
# Compare skills list
diff <(ssh $REMOTE "ls /home/zihan/.claude/skills/") <(ls ~/.claude/skills/)
# Compare gitconfig
diff <(ssh $REMOTE "cat /home/zihan/.gitconfig") <(grep -v 'gpg\|sign\|vscode\|editor\|diff\|merge\|mergetool\|difftool\|alias\|excludes' ~/.gitconfig)
```
