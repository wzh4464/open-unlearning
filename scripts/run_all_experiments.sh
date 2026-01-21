#!/bin/bash
# Generate and run all unlearning experiments in parallel on 4 GPUs

set -e
cd /app

# Generate job scripts first
JOB_DIR="/tmp/unlearn_jobs"
mkdir -p "$JOB_DIR"
rm -f "$JOB_DIR"/*.sh

echo "=== Checking existing experiments ==="

# Function to check if experiment exists
check_exists() {
    [[ -f "saves/unlearn/$1/evals/TOFU_SUMMARY.json" ]]
}

# Count jobs
job_count=0

# Simple methods
METHODS="GradAscent GradDiff NPO CEU DPO RMU SatImp SimNPO UNDIAL WGA"
declare -A CHECKPOINTS=([1]=250 [2]=500 [3]=750 [4]=1000 [5]=1250)

for method in $METHODS; do
    method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    for epoch in 1 2 3 4 5; do
        ckpt=${CHECKPOINTS[$epoch]}
        exp_name="${method_lower}_llama32_epoch${epoch}"

        if check_exists "$exp_name"; then
            echo "SKIP: $exp_name"
            continue
        fi

        job_count=$((job_count + 1))
        cat > "$JOB_DIR/job_${job_count}.sh" << EOF
#!/bin/bash
cd /app
exp_name="$exp_name"
echo "[\$(date '+%H:%M:%S')] START: \$exp_name"
CUDA_VISIBLE_DEVICES=\$GPU python src/train.py \\
    --config-name=unlearn.yaml \\
    experiment=unlearn/tofu/default \\
    model=Llama-3.2-1B-Instruct \\
    model.model_args.pretrained_model_name_or_path=saves/finetune/llama32_1b_tofu_safe/checkpoint-${ckpt} \\
    trainer=$method \\
    forget_split=forget10 \\
    retain_split=retain90 \\
    task_name="\$exp_name" \\
    2>&1 | tee saves/unlearn/\${exp_name}.log
echo "[\$(date '+%H:%M:%S')] DONE: \$exp_name"
EOF
        chmod +x "$JOB_DIR/job_${job_count}.sh"
    done
done

# LMCleaner with different K values
for K in 50 100 500 1000 1250; do
    for epoch in 1 2 3 4 5; do
        ckpt=${CHECKPOINTS[$epoch]}
        exp_name="lmcleaner_llama32_epoch${epoch}_K${K}"

        if check_exists "$exp_name"; then
            echo "SKIP: $exp_name"
            continue
        fi

        job_count=$((job_count + 1))
        cat > "$JOB_DIR/job_${job_count}.sh" << EOF
#!/bin/bash
cd /app
exp_name="$exp_name"
echo "[\$(date '+%H:%M:%S')] START: \$exp_name"
CUDA_VISIBLE_DEVICES=\$GPU python src/train.py \\
    --config-name=unlearn.yaml \\
    experiment=unlearn/tofu/default \\
    model=Llama-3.2-1B-Instruct \\
    model.model_args.pretrained_model_name_or_path=saves/finetune/llama32_1b_tofu_safe/checkpoint-${ckpt} \\
    trainer=LMCleanerBatch \\
    trainer.method_args.training_log_dir=saves/train_logs/llama32_1b_tofu_safe \\
    trainer.method_args.K=$K \\
    forget_split=forget10 \\
    retain_split=retain90 \\
    task_name="\$exp_name" \\
    2>&1 | tee saves/unlearn/\${exp_name}.log
echo "[\$(date '+%H:%M:%S')] DONE: \$exp_name"
EOF
        chmod +x "$JOB_DIR/job_${job_count}.sh"
    done
done

echo ""
echo "=== Generated $job_count jobs ==="
ls -la "$JOB_DIR"/*.sh 2>/dev/null | head -20
echo ""

if [[ $job_count -eq 0 ]]; then
    echo "No new jobs to run!"
    exit 0
fi

# Create job list for parallel
ls "$JOB_DIR"/*.sh > /tmp/job_list.txt

echo "=== Starting parallel execution on 4 GPUs ==="
echo "Jobs: $job_count"
echo ""

# Run with GNU parallel, cycling through GPUs 0-3
parallel --jobs 4 --bar 'GPU=$(( ({#} - 1) % 4 )) bash {}' :::: /tmp/job_list.txt

echo ""
echo "=== All jobs completed ==="
