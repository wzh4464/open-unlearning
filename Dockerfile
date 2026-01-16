FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (no Python needed - already in base image)
RUN apt-get update && apt-get install -y \
    wget unzip git tmux curl build-essential ninja-build openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock .python-version ./

# Create venv using system Python (already 3.11), PyTorch already installed
RUN uv venv --python $(which python) && \
    . .venv/bin/activate && \
    uv pip install torch==2.5.1

# Install project dependencies
COPY src/ src/
RUN . .venv/bin/activate && \
    uv sync --extra linux-cuda

# Copy remaining code
COPY . .

# Make venv the default Python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV IN_DOCKER=1

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
