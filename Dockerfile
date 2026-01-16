FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    wget unzip git tmux curl build-essential ninja-build \
    software-properties-common openssh-client && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first (layer caching optimization)
COPY pyproject.toml uv.lock .python-version ./

# Install PyTorch with CUDA first (uv needs torch before flash-attn)
RUN uv venv --python 3.11 && \
    . .venv/bin/activate && \
    uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies with uv sync
COPY src/ src/
RUN . .venv/bin/activate && \
    uv sync --extra linux-cuda

# Copy remaining code
COPY . .

# Make venv the default Python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
