#!/bin/bash
# Launch Experiment A: wait for model download, then run all steps
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}"

# Verify model is accessible (works for both HF model IDs and local paths)
if [[ "${BASE_MODEL_PATH}" == */* ]] && [ ! -d "${BASE_MODEL_PATH}" ]; then
    # HF model ID (e.g., "unsloth/Llama-3.2-3B-Instruct") — verify it's downloadable
    echo "Verifying HF model: ${BASE_MODEL_PATH}"
    if ! $PYTHON_CMD -c "from transformers import AutoConfig; AutoConfig.from_pretrained('${BASE_MODEL_PATH}')" 2>/dev/null; then
        echo "ERROR: Cannot access HF model '${BASE_MODEL_PATH}'. Check model ID and network."
        exit 1
    fi
    echo "HF model verified: ${BASE_MODEL_PATH}"
elif [ -d "${BASE_MODEL_PATH}" ]; then
    # Local path — check that config.json exists
    if [ ! -f "${BASE_MODEL_PATH}/config.json" ]; then
        echo "ERROR: Local model missing config.json at ${BASE_MODEL_PATH}"
        exit 1
    fi
    echo "Local model verified: ${BASE_MODEL_PATH}"
else
    echo "ERROR: Model not found: ${BASE_MODEL_PATH}"
    exit 1
fi

echo ""
echo "Model verified! Starting experiments..."
echo ""

# Run all experiments
bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee "${RESULTS_DIR}/run_all.log"
