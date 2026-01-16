FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + python symlink
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git tmux curl openssh-client openssh-server && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml ./

# Install dependencies directly to system Python
RUN pip install --no-cache-dir \
    accelerate>=1.11.0 \
    datasets>=4.3.0 \
    hydra-colorlog>=1.2.0 \
    hydra-core>=1.3.2 \
    lm-eval>=0.4.9 \
    rouge-score>=0.1.2 \
    scikit-learn>=1.7.2 \
    scipy>=1.16.2 \
    tensorboard>=2.20.0 \
    transformers>=4.57.1 \
    deepspeed>=0.18.4 \
    ninja>=1.13.0 \
    packaging>=25.0

# Build flash-attn from source (needs torch visible)
RUN pip install flash-attn==2.8.3 --no-build-isolation

# Copy code
COPY . .

ENV IN_DOCKER=1

# Entrypoint
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
