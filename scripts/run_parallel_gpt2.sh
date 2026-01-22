#!/bin/bash
# Run all GPT-2 unlearning jobs in parallel using GNU Parallel
# Uses 4 GPUs, cycling through them

set -e

cd /app

chmod +x scripts/run_single_job.sh

# Remove comment lines and empty lines from job file
grep -v '^#' scripts/jobs_gpt2_unlearn.txt | grep -v '^$' > /tmp/jobs_clean.txt

TOTAL_JOBS=$(wc -l < /tmp/jobs_clean.txt)
echo "Total jobs: $TOTAL_JOBS"
echo "Using 4 GPUs in parallel"
echo "Starting at: $(date)"
echo "================================================"

# Run with GNU Parallel
# -j 4: 4 jobs at a time (one per GPU)
# --colsep ',': CSV format
# {%}: job slot number (1-4), used to assign GPU (0-3)
cat /tmp/jobs_clean.txt | parallel -j 4 --colsep ',' \
    'bash scripts/run_single_job.sh $(( {%} - 1 )) {1} {2} {3} {4}'

echo "================================================"
echo "All jobs completed at: $(date)"
echo "Results in: saves/unlearn/tofu_gpt2_eval/"
