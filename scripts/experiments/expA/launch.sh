#!/bin/bash
# Launch Experiment A: wait for model download, then run all steps
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd /app
mkdir -p "${RESULTS_DIR}"

MODEL_SHARD="${BASE_MODEL_PATH}/model-00001-of-00002.safetensors"
EXPECTED_SIZE_MB=9000  # ~9.29GB

echo "Waiting for model download to complete..."
echo "Checking: ${MODEL_SHARD}"

while true; do
    if [ -f "${MODEL_SHARD}" ]; then
        SIZE_MB=$(du -m "${MODEL_SHARD}" 2>/dev/null | cut -f1)
        if [ "${SIZE_MB}" -gt "${EXPECTED_SIZE_MB}" ]; then
            echo "Model shard complete: ${SIZE_MB}MB"
            break
        fi
        echo "  Model shard: ${SIZE_MB}MB / ~9290MB ..."
    else
        echo "  Waiting for model shard file to appear..."
    fi
    sleep 30
done

echo ""
echo "Model download complete! Starting experiments..."
echo ""

# Run all experiments
bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee "${RESULTS_DIR}/run_all.log"
