#!/bin/bash
# Launch Experiment A: wait for model download, then run all steps
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}"

MODEL_SHARD="${BASE_MODEL_PATH}/model-00001-of-00002.safetensors"
EXPECTED_SIZE_MB=9000  # ~9.29GB

MAX_WAIT_SEC=3600  # Timeout after 1 hour
ELAPSED=0

echo "Waiting for model download to complete (timeout: ${MAX_WAIT_SEC}s)..."
echo "Checking: ${MODEL_SHARD}"

while true; do
    if [ -f "${MODEL_SHARD}" ]; then
        SIZE_MB=$(du -m "${MODEL_SHARD}" 2>/dev/null | cut -f1)
        if [ "${SIZE_MB}" -gt "${EXPECTED_SIZE_MB}" ]; then
            echo "Model shard complete: ${SIZE_MB}MB"
            break
        fi
        echo "  Model shard: ${SIZE_MB}MB / ~9290MB ... (${ELAPSED}s elapsed)"
    else
        echo "  Waiting for model shard file to appear... (${ELAPSED}s elapsed)"
    fi
    if [ "${ELAPSED}" -ge "${MAX_WAIT_SEC}" ]; then
        echo "ERROR: Model download timed out after ${MAX_WAIT_SEC}s"
        exit 1
    fi
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

echo ""
echo "Model download complete! Starting experiments..."
echo ""

# Run all experiments
bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee "${RESULTS_DIR}/run_all.log"
