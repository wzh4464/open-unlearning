#!/bin/bash
# Master script: run LMCleaner on 5-epoch model with forget10 + historical params
# This is the key rebuttal experiment to match paper Table 2 quality
set -e
source "$(dirname "$0")/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SAVES="/workspace/saves"
MODEL_DIR="${SAVES}/finetune/llama32_1b_tofu_safe"
TRAIN_LOG="${SAVES}/train_logs/llama32_1b_tofu_safe"
MODEL="Llama-3.2-1B-Instruct"

echo "=========================================="
echo "  Rebuttal: LMCleaner on 5-epoch model"
echo "  forget10, use_historical_params=true"
echo "=========================================="

# Step 1: Generate sparse checkpoints if needed
CKPT_DIR="${TRAIN_LOG}/sparse_checkpoints"
N_CKPT=$(ls "${CKPT_DIR}"/step_*.pt 2>/dev/null | wc -l)
if [ "$N_CKPT" -lt 5 ]; then
    echo ">>> Generating sparse checkpoints..."
    $PYTHON_CMD scripts/experiments/generate_sparse_checkpoints.py \
        --base-model unsloth/Llama-3.2-1B-Instruct \
        --train-log-dir "${TRAIN_LOG}" \
        --stride 25 --max-step 249 --micro-batch 4
else
    echo ">>> Sparse checkpoints exist ($N_CKPT files), skipping"
fi

# Step 2: Run LMCleaner with forget10 + historical params
for K in 50; do
    TASK="rebuttal_lmcleaner_forget10_K${K}"
    OUT="${SAVES}/unlearn/${TASK}"

    if [ -f "${OUT}/audit/audit_records.json" ]; then
        echo "SKIP: $TASK (already done)"
    else
        echo ""
        echo ">>> LMCleaner K=$K"
        $PYTHON_CMD src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=LMCleanerBatch \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path="${MODEL_DIR}" \
            model.tokenizer_args.pretrained_model_name_or_path="${MODEL_DIR}" \
            forget_split=forget10 \
            retain_split=retain90 \
            holdout_split=holdout10 \
            trainer.method_args.training_log_dir="${TRAIN_LOG}" \
            trainer.method_args.K=${K} \
            trainer.method_args.max_step=250 \
            trainer.method_args.hessian_mode=fisher \
            trainer.method_args.damping=1e-4 \
            trainer.method_args.epsilon=0 \
            trainer.method_args.noise_mode=none \
            trainer.method_args.use_historical_params=true \
            trainer.args.num_train_epochs=0 \
            paths.output_dir="${OUT}" \
            task_name="${TASK}"
    fi
done

# Step 3: Evaluate
for K in 50; do
    TASK="rebuttal_lmcleaner_forget10_K${K}"
    EVAL="${SAVES}/eval/${TASK}"

    if [ -f "${EVAL}/TOFU_SUMMARY.json" ]; then
        echo "SKIP eval: $TASK (already done)"
    else
        echo ""
        echo ">>> Eval $TASK"
        $PYTHON_CMD src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path="${SAVES}/unlearn/${TASK}" \
            model.tokenizer_args.pretrained_model_name_or_path="${SAVES}/unlearn/${TASK}" \
            forget_split=forget10 \
            holdout_split=holdout10 \
            paths.output_dir="${EVAL}" \
            task_name="${TASK}"
    fi
done

# Step 4: Show results
echo ""
echo "=========================================="
echo "  Results"
echo "=========================================="
for K in 50; do
    TASK="rebuttal_lmcleaner_forget10_K${K}"
    EVAL="${SAVES}/eval/${TASK}"
    if [ -f "${EVAL}/TOFU_SUMMARY.json" ]; then
        $PYTHON_CMD -c "
import json
s = json.load(open('${EVAL}/TOFU_SUMMARY.json'))
print(f'K={K}: utility={s.get(\"model_utility\",0):.4f} fg_truth={s.get(\"forget_truth_ratio\",0):.4f} fg_ROUGE={s.get(\"forget_Q_A_ROUGE\",0):.4f} extract={s.get(\"extraction_strength\",0):.4f} privleak={s.get(\"privleak\",0):.1f}')
print(f'       Paper:  utility=0.429  fg_truth=0.645  fg_ROUGE=0.412')
"
    fi
done
