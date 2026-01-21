#!/bin/bash
# Run all missing unlearning experiments in parallel on 4 GPUs
set -e
cd /app

JOB_DIR="/tmp/unlearn_jobs_$$"
mkdir -p "$JOB_DIR"

declare -A CHECKPOINTS=([1]=250 [2]=500 [3]=750 [4]=1000 [5]=1250)

job_id=0

create_job() {
    local method=$1
    local epoch=$2
    local exp_name=$3
    local extra_args=$4
    local ckpt=${CHECKPOINTS[$epoch]}

    job_id=$((job_id + 1))
    local job_file="$JOB_DIR/job_$(printf "%03d" $job_id)_${exp_name}.sh"

    cat > "$job_file" << JOBEOF
#!/bin/bash
cd /app
export CUDA_VISIBLE_DEVICES=\$GPU
exp_name="$exp_name"
echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] START: \$exp_name"

python src/train.py \\
    --config-name=unlearn.yaml \\
    experiment=unlearn/tofu/default \\
    model=Llama-3.2-1B-Instruct \\
    model.model_args.pretrained_model_name_or_path=saves/finetune/llama32_1b_tofu_safe/checkpoint-${ckpt} \\
    trainer=$method \\
    $extra_args \\
    forget_split=forget10 \\
    retain_split=retain90 \\
    task_name="\$exp_name" \\
    2>&1 | tee "saves/unlearn/\${exp_name}.log"

status=\$?
if [ \$status -eq 0 ]; then
    echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] DONE: \$exp_name"
else
    echo "[\$(date '+%H:%M:%S')] [GPU \$GPU] FAILED: \$exp_name (exit \$status)"
fi
exit \$status
JOBEOF
    chmod +x "$job_file"
}

echo "=== Generating job scripts ==="

# Missing existing methods
echo "Adding: gradasc_llama32_epoch5"
create_job "GradAscent" 5 "gradasc_llama32_epoch5" ""

echo "Adding: lmcleaner_llama32_epoch5_K1250"
create_job "LMCleanerBatch" 5 "lmcleaner_llama32_epoch5_K1250" \
    "trainer.method_args.training_log_dir=saves/train_logs/llama32_1b_tofu_safe trainer.method_args.K=1250"

# New methods - all epochs
NEW_METHODS="CEU DPO RMU SatImp SimNPO UNDIAL WGA"
for method in $NEW_METHODS; do
    method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    for epoch in 1 2 3 4 5; do
        exp_name="${method_lower}_llama32_epoch${epoch}"
        echo "Adding: $exp_name"
        create_job "$method" "$epoch" "$exp_name" ""
    done
done

echo ""
echo "=== Generated $job_id jobs ==="
ls "$JOB_DIR"/*.sh | wc -l

# Create job list
ls "$JOB_DIR"/*.sh | sort > /tmp/job_list.txt

echo ""
echo "=== Starting parallel execution on 4 GPUs ==="
echo "Total jobs: $job_id"
echo ""

# Run with GNU parallel
# Each job gets assigned a GPU 0-3 based on job number
parallel --jobs 4 --bar --halt soon,fail=1 \
    'GPU=$(( ({#} - 1) % 4 )) bash {}' \
    :::: /tmp/job_list.txt

echo ""
echo "=== All experiments completed ==="

# Cleanup
rm -rf "$JOB_DIR"
