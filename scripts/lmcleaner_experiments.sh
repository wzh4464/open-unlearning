#!/bin/bash
# LMCleaner Online Unlearning Experiments
#
# 实验流程:
# 1. 预训练并记录训练日志 (使用TrainingLogger)
# 2. 运行批次级LMCleaner遗忘
# 3. 运行样本级LMCleaner遗忘 (可选)
# 4. 评估遗忘效果
# 5. 对比baseline方法 (TIMParameterRollback)
#
# 使用方法:
# bash scripts/lmcleaner_experiments.sh

# Memory optimization: Set PyTorch CUDA allocator configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
# export TOKENIZERS_PARALLELISM=false

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# GPU device bindings (ensure paired GPUs by default)
TRAIN_GPUS=${TRAIN_GPUS:-"1,3"}
UNLEARN_GPUS=${UNLEARN_GPUS:-"1,3"}
EVAL_GPUS=${EVAL_GPUS:-"1,3"}

# 实验配置
MODEL_DIR=/home/jie
models=(
    "Llama-3.2-1B-Instruct"
    # "phi-1_5"
)

# LMCleaner变体和对应的实验配置
lmcleaner_variants=(
    "LMCleanerBatch unlearn/tofu/default"
    # "LMCleanerSample unlearn/tofu/default"  # 样本级,可选
)

# 数据切分
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# K值范围(截断窗口大小)
K_values=(
    500
    800
    1000
)

# HVP模式
hessian_modes=(
    "GGN"      # 默认,推荐
    # "diag"   # 快速测试,可选
)

# 训练配置 (Memory-optimized settings)
per_device_train_batch_size=2  # Reduced from 4 to minimize memory usage
gradient_accumulation_steps=8  # Increased from 4 to maintain effective batch size

########################################################################################################################
########################################### Step 1: 预训练并记录训练日志 ################################################
########################################################################################################################

# TrainingLogger支持三种存储模式:
# 1. 轻存储模式 (light): 只保存索引+随机种子 (推荐,磁盘占用最小)
# 2. 重存储模式 (heavy): 保存完整batch张量 (磁盘占用大,但HVP计算最准确)
# 3. 对角Hessian模式 (diag): 预计算对角Hessian (适用于diag HVP模式)

# 选择存储模式 (light, heavy, diag)
STORAGE_MODE=${STORAGE_MODE:-"light"}

for model in "${models[@]}"; do
    TASK=tofu_${model}_full_${STORAGE_MODE}
    BASE_MODEL=${MODEL_DIR}/${model}

    echo "=========================================="
    echo "Finetuning with ${STORAGE_MODE} storage mode"
    echo "Model: ${model}"
    echo "Task: ${TASK}"
    echo "=========================================="

    if [ "$STORAGE_MODE" == "light" ]; then
        # 轻存储模式: 只保存索引+随机种子
        CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=train.yaml \
            experiment=finetune/tofu/default \
            task_name=${TASK} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${BASE_MODEL} \
            model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL} \
            +trainer.args.training_logger.enabled=true \
            +trainer.args.training_logger.log_dir=saves/train_logs/${TASK} \
            +trainer.args.training_logger.mode=batch \
            +trainer.args.training_logger.save_indices_only=true \
            +trainer.args.training_logger.save_batch_data=false \
            +trainer.args.training_logger.save_rng_state=true \
            +trainer.args.training_logger.save_interval=100

    elif [ "$STORAGE_MODE" == "heavy" ]; then
        # 重存储模式: 保存完整batch张量
        CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=train.yaml \
            experiment=finetune/tofu/default \
            task_name=${TASK} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${BASE_MODEL} \
            model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL} \
            +trainer.args.training_logger.enabled=true \
            +trainer.args.training_logger.log_dir=saves/train_logs/${TASK} \
            +trainer.args.training_logger.mode=batch \
            +trainer.args.training_logger.save_indices_only=false \
            +trainer.args.training_logger.save_batch_data=true \
            +trainer.args.training_logger.save_rng_state=false \
            +trainer.args.training_logger.save_interval=50

    elif [ "$STORAGE_MODE" == "diag" ]; then
        # 对角Hessian模式: 预计算对角Hessian
        CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=train.yaml \
            experiment=finetune/tofu/default \
            task_name=${TASK} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${BASE_MODEL} \
            model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL} \
            +trainer.args.training_logger.enabled=true \
            +trainer.args.training_logger.log_dir=saves/train_logs/${TASK} \
            +trainer.args.training_logger.mode=batch \
            +trainer.args.training_logger.compute_diag_h=true \
            +trainer.args.training_logger.save_indices_only=true \
            +trainer.args.training_logger.save_batch_data=false \
            +trainer.args.training_logger.save_rng_state=true
    fi

    echo "Completed finetuning for ${model} with ${STORAGE_MODE} storage"
    echo ""
done

########################################################################################################################
########################################### Step 2: 运行LMCleaner遗忘 ##################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for hessian_mode in "${hessian_modes[@]}"; do
            for K in "${K_values[@]}"; do
                for variant_experiment in "${lmcleaner_variants[@]}"; do
                    trainer=$(echo $variant_experiment | cut -d' ' -f1)
                    experiment=$(echo $variant_experiment | cut -d' ' -f2)

                    task_name=lmcleaner_${model}_${forget_split}_${trainer}_K${K}_${hessian_mode}_${STORAGE_MODE}
                    model_path=saves/train/tofu_${model}_full_${STORAGE_MODE}  # Use finetuned model with logs
                    training_log_dir=saves/train_logs/tofu_${model}_full_${STORAGE_MODE}

                    echo "=========================================="
                    echo "Task: ${task_name}"
                    echo "Model: ${model_path}"
                    echo "Trainer: ${trainer}"
                    echo "K: ${K}, HVP: ${hessian_mode}"
                    echo "=========================================="

                    # 检查训练日志是否存在
                    if [ ! -d "${training_log_dir}" ]; then
                        echo "Warning: Training log directory not found: ${training_log_dir}"
                        echo "Skipping this experiment."
                        echo "Please run pretraining with TrainingLogger first."
                        continue
                    fi

                    # 运行遗忘 (with memory optimization)
                    CUDA_VISIBLE_DEVICES=${UNLEARN_GPUS} accelerate launch \
                        --config_file configs/accelerate/default_config.yaml \
                        --main_process_port $MASTER_PORT \
                        src/train.py --config-name=unlearn.yaml \
                        experiment=${experiment} \
                        trainer=${trainer} \
                        task_name=${task_name} \
                        model=${model} \
                        forget_split=${forget_split} \
                        retain_split=${retain_split} \
                        model.model_args.pretrained_model_name_or_path=${model_path} \
                        trainer.method_args.training_log_dir=${training_log_dir} \
                        trainer.method_args.K=${K} \
                        trainer.method_args.hessian_mode=${hessian_mode} \
                        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                        trainer.args.ddp_find_unused_parameters=false \
                        trainer.args.gradient_checkpointing=true \
                        ++trainer.args.bf16=true \
                        +data.max_seq_length=1024

                    # 评估遗忘效果
                    echo "Evaluating ${task_name}..."
                    CUDA_VISIBLE_DEVICES=${EVAL_GPUS} python src/eval.py \
                        experiment=eval/tofu/default.yaml \
                        forget_split=${forget_split} \
                        holdout_split=${holdout_split} \
                        model=${model} \
                        task_name=${task_name} \
                        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                        paths.output_dir=saves/unlearn/${task_name}/evals \
                        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

                    echo "Completed ${task_name}"
                    echo ""
                done
            done
        done
    done
done

########################################################################################################################
########################################### Step 3: 对比baseline (可选) #################################################
########################################################################################################################

# 运行TIMParameterRollback作为baseline对比
baseline_trainer="TIMParameterRollback"

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        task_name=baseline_${model}_${forget_split}_${baseline_trainer}
        model_path=open-unlearning/tofu_${model}_full
        tim_output_dir=saves/train_logs/${model}_full  # 假设TIM日志与训练日志在同一目录

        echo "=========================================="
        echo "Baseline: ${task_name}"
        echo "=========================================="

        # 运行baseline (with memory optimization)
        CUDA_VISIBLE_DEVICES=${UNLEARN_GPUS} accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default.yaml \
            trainer=${baseline_trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            trainer.method_args.tim_output_dir=${tim_output_dir} \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            ++trainer.args.bf16=true \
            +trainer.args.gradient_checkpointing=true \
            +trainer.args.ddp_find_unused_parameters=false \
            +data.max_seq_length=1024

        # 评估baseline
        CUDA_VISIBLE_DEVICES=${EVAL_GPUS} python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

        echo "Completed baseline ${task_name}"
        echo ""
    done
done

########################################################################################################################
########################################### Step 4: 汇总结果 ###########################################################
########################################################################################################################

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "- Unlearning outputs: saves/unlearn/"
echo "- Evaluation results: saves/unlearn/*/evals/"
echo "- Audit logs: saves/unlearn/*/audit/"
echo ""
echo "To compare results, check the TOFU_EVAL.json files in each eval directory."
echo ""
echo "Key metrics to compare:"
echo "- Forget Quality: Extraction Strength, Truth Ratio"
echo "- Retain Performance: ROUGE scores on retain set"
echo "- Model Utility: MMLU scores"
echo "- Privacy: MIA attack success rate"
echo ""
