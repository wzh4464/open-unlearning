#!/bin/bash
# TOFU finetune with 10x learning rate and training_logger enabled

python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  trainer.args.learning_rate=1e-4 \
  +trainer.args.training_logger.enabled=true \
  +trainer.args.training_logger.log_dir="saves/train_logs/tofu_finetune_10x_lr" \
  +trainer.args.training_logger.save_indices_only=true \
  +trainer.args.training_logger.save_rng_state=true \
  task_name=tofu_Llama-3.2-1B-Instruct_full_10x_lr
