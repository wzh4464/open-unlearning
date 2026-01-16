#!/bin/bash
# Environment-aware Python runner
# Source this file in scripts: source "$(dirname "$0")/env.sh"

if [ -n "$IN_DOCKER" ]; then
    PYTHON_CMD="python"
    ACCELERATE_CMD="accelerate"
else
    PYTHON_CMD="uv run python"
    ACCELERATE_CMD="uv run accelerate"
fi
