#!/bin/bash
# Quick Test Script for TrainingLogger + LMCleaner
#
# This script performs a minimal test of the TrainingLogger integration:
# 1. Finetune a model for 50 steps with training logging enabled
# 2. Run LMCleaner unlearning using the logged training data
#
# Usage:
#   bash scripts/test_traininglogger.sh

set -e  # Exit on error

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

echo "=========================================="
echo "TrainingLogger + LMCleaner Quick Test"
echo "=========================================="
echo ""

# Configuration
TASK_NAME=quick_test
BASE_MODEL=/home/jie/Llama-3.2-1B-Instruct
MODEL_NAME=Llama-3.2-1B-Instruct  # For config reference
MAX_STEPS=50
BATCH_SIZE=2
K_VALUE=20
FORGET_SPLIT=forget10
RETAIN_SPLIT=retain90

echo "Step 1: Finetuning with TrainingLogger enabled"
echo "Task: ${TASK_NAME}"
echo "Base model: ${BASE_MODEL}"
echo "Max steps: ${MAX_STEPS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port ${MASTER_PORT} \
  src/train.py --config-name=train.yaml \
  task_name=${TASK_NAME} \
  model=${MODEL_NAME} \
  model.model_args.pretrained_model_name_or_path=${BASE_MODEL} \
  model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL} \
  trainer=finetune \
  data/datasets@data.train=TOFU_QA_retain \
  +trainer.args.max_steps=${MAX_STEPS} \
  trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
  trainer.args.save_strategy=steps \
  +trainer.args.save_steps=${MAX_STEPS} \
  trainer.args.ddp_find_unused_parameters=true \
  +trainer.args.training_logger.enabled=true \
  +trainer.args.training_logger.log_dir=saves/train_logs/${TASK_NAME} \
  +trainer.args.training_logger.mode=batch \
  +trainer.args.training_logger.save_indices_only=true \
  +trainer.args.training_logger.save_rng_state=true \
  +trainer.args.training_logger.save_interval=10

echo ""
echo "Step 1 completed! Model saved to saves/train/${TASK_NAME}"
echo "Training logs saved to saves/train_logs/${TASK_NAME}"
echo ""

echo "Step 2: Running LMCleaner unlearning"
echo "Task: ${TASK_NAME}_unlearn"
echo "K: ${K_VALUE}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port ${MASTER_PORT} \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=LMCleanerBatch \
  task_name=${TASK_NAME}_unlearn \
  model=${MODEL_NAME} \
  forget_split=${FORGET_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  model.model_args.pretrained_model_name_or_path=saves/train/${TASK_NAME} \
  trainer.method_args.training_log_dir=saves/train_logs/${TASK_NAME} \
  trainer.method_args.K=${K_VALUE} \
  trainer.method_args.hessian_mode=GGN \
  trainer.method_args.damping=1e-4 \
  trainer.args.ddp_find_unused_parameters=true

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "- Finetuned model: saves/train/${TASK_NAME}"
echo "- Training logs: saves/train_logs/${TASK_NAME}"
echo "- Unlearned model: saves/unlearn/${TASK_NAME}_unlearn"
echo ""
echo "Check the training logs to verify:"
echo "  - meta.json: Configuration metadata"
echo "  - sample_indices.json: Sample indices per step"
echo "  - rng_states_*.pkl: RNG states for batch reconstruction"
echo "  - step_records_*.pkl: Parameter updates and gradients"
echo ""
