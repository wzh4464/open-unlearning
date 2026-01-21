#!/bin/bash
# Run evaluations for all completed experiments in parallel on 4 GPUs
set -e
cd /app

JOB_DIR="/tmp/eval_jobs_$$"
mkdir -p "$JOB_DIR"

job_id=0

# Find all experiments that have model.safetensors but no evals/TOFU_SUMMARY.json
echo "=== Finding experiments needing evaluation ==="

for exp_dir in saves/unlearn/*/; do
    exp_name=$(basename "$exp_dir")

    # Skip test/debug directories
    if [[ "$exp_name" == *"test"* ]] || [[ "$exp_name" == *"debug"* ]] || \
       [[ "$exp_name" == *"v2"* ]] || [[ "$exp_name" == *"v3"* ]] || \
       [[ "$exp_name" == *"v4"* ]] || [[ "$exp_name" == *"1b_tofu"* ]] || \
       [[ "$exp_name" == *"1b_forget"* ]] || [[ "$exp_name" == *"quick"* ]] || \
       [[ "$exp_name" == *"K0"* ]]; then
        continue
    fi

    # Check if model exists
    if [[ ! -f "${exp_dir}model.safetensors" ]] && [[ ! -f "${exp_dir}config.json" ]]; then
        continue
    fi

    # Check if eval already exists
    if [[ -f "${exp_dir}evals/TOFU_SUMMARY.json" ]]; then
        continue
    fi

    echo "Need eval: $exp_name"
    job_id=$((job_id + 1))

    cat > "$JOB_DIR/eval_$(printf "%03d" $job_id)_${exp_name}.sh" << EVALEOF
#!/bin/bash
cd /app
export CUDA_VISIBLE_DEVICES=\$GPU
exp_name="$exp_name"
exp_dir="saves/unlearn/\$exp_name"

echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] EVAL START: \$exp_name"

python src/eval.py \\
    --config-name=eval.yaml \\
    experiment=eval/tofu/default \\
    model=Llama-3.2-1B-Instruct \\
    model.model_args.pretrained_model_name_or_path="\$exp_dir" \\
    task_name="\$exp_name" \\
    2>&1

status=\$?
if [ \$status -eq 0 ]; then
    echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] EVAL DONE: \$exp_name"
else
    echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] EVAL FAILED: \$exp_name (exit \$status)"
fi
exit \$status
EVALEOF
    chmod +x "$JOB_DIR/eval_$(printf "%03d" $job_id)_${exp_name}.sh"
done

echo ""
echo "=== Generated $job_id eval jobs ==="

if [[ $job_id -eq 0 ]]; then
    echo "No evaluations needed!"
    exit 0
fi

ls "$JOB_DIR"/*.sh | sort > /tmp/eval_job_list.txt

echo ""
echo "=== Starting parallel evaluations on 4 GPUs ==="

# Run with GNU parallel - use 3 GPUs since GPU 1 is busy with LMCleaner
parallel --jobs 3 --bar \
    'GPU=$(( ({#} - 1) % 4 )); [ $GPU -eq 1 ] && GPU=3; bash {}' \
    :::: /tmp/eval_job_list.txt

echo ""
echo "=== All evaluations completed ==="
rm -rf "$JOB_DIR"
