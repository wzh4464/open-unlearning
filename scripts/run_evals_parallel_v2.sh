#!/bin/bash
# Run evaluations for all completed experiments in parallel on 3 GPUs (0, 2, 3)
# GPU 1 is reserved for LMCleaner

set -e
cd /app

JOB_DIR="/tmp/eval_jobs_v2_$$"
mkdir -p "$JOB_DIR"

# Available GPUs (excluding GPU 1 which is running LMCleaner)
GPUS=(0 2 3)

job_id=0

echo "=== Finding experiments needing evaluation ==="

for exp_dir in saves/unlearn/*/; do
    exp_name=$(basename "$exp_dir")

    # Skip test/debug directories
    if [[ "$exp_name" == *"test"* ]] || [[ "$exp_name" == *"debug"* ]] || \
       [[ "$exp_name" == *"v2"* ]] || [[ "$exp_name" == *"v3"* ]] || \
       [[ "$exp_name" == *"v4"* ]] || [[ "$exp_name" == *"1b_tofu"* ]] || \
       [[ "$exp_name" == *"1b_forget"* ]] || [[ "$exp_name" == *"quick"* ]] || \
       [[ "$exp_name" == *"K0"* ]] || [[ "$exp_name" == *"dpo"* ]]; then
        continue
    fi

    # Check if model exists
    if [[ ! -f "${exp_dir}model.safetensors" ]]; then
        continue
    fi

    # Check if eval already exists
    if [[ -f "${exp_dir}evals/TOFU_SUMMARY.json" ]]; then
        continue
    fi

    echo "Need eval: $exp_name"

    # Assign GPU in round-robin fashion
    gpu_idx=$((job_id % 3))
    gpu=${GPUS[$gpu_idx]}

    job_id=$((job_id + 1))

    cat > "$JOB_DIR/eval_$(printf "%03d" $job_id)_${exp_name}.sh" << EVALEOF
#!/bin/bash
cd /app
export CUDA_VISIBLE_DEVICES=$gpu
exp_name="$exp_name"
exp_dir="saves/unlearn/\$exp_name"

echo "[\$(date '+%H:%M:%S')] [GPU $gpu] EVAL START: \$exp_name"

python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path="\$exp_dir" \
    task_name="\$exp_name" \
    2>&1

status=\$?
if [ \$status -eq 0 ]; then
    echo "[\$(date '+%H:%M:%S')] [GPU $gpu] EVAL DONE: \$exp_name"
else
    echo "[\$(date '+%H:%M:%S')] [GPU $gpu] EVAL FAILED: \$exp_name (exit \$status)"
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

ls "$JOB_DIR"/*.sh | sort > /tmp/eval_job_list_v2.txt

echo ""
echo "=== Starting parallel evaluations on GPUs 0, 2, 3 ==="

# Run 3 jobs in parallel
parallel --jobs 3 'bash {}' :::: /tmp/eval_job_list_v2.txt

echo ""
echo "=== All evaluations completed ==="
rm -rf "$JOB_DIR"
