#!/bin/bash
# Run all GPT-2 unlearning jobs in parallel using GNU Parallel
# Uses 4 GPUs, cycling through them

set -e

# Detect repository root dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Check if GNU Parallel is installed
command -v parallel > /dev/null 2>&1 || {
    echo "Error: GNU Parallel not found. Please install it first:"
    echo "  apt-get install parallel  # Debian/Ubuntu"
    echo "  brew install parallel     # macOS"
    exit 1
}

chmod +x scripts/run_single_job.sh

# Remove comment lines and empty lines from job file
grep -v '^#' scripts/jobs_gpt2_unlearn.txt | grep -v '^$' > /tmp/jobs_clean.txt

TOTAL_JOBS=$(wc -l < /tmp/jobs_clean.txt)
NUM_GPUS=${NUM_GPUS:-4}
echo "Total jobs: $TOTAL_JOBS"
echo "Using $NUM_GPUS GPUs in parallel"
echo "Starting at: $(date)"
echo "================================================"

# Run with GNU Parallel
# -j $NUM_GPUS: N jobs at a time (one per GPU)
# --colsep ',': CSV format
# {%}: job slot number (1-N), used to assign GPU (0 to N-1)
cat /tmp/jobs_clean.txt | parallel -j "$NUM_GPUS" --colsep ',' \
    'bash scripts/run_single_job.sh $(( {%} - 1 )) {1} {2} {3} {4}'

echo "================================================"
echo "All jobs completed at: $(date)"
echo "Results in: saves/unlearn/tofu_gpt2_eval/"
