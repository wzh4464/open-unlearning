#!/bin/bash
# Evaluation Queue Script
# Usage: ./eval_queue.sh <GPU_ID> <TASK>
# Tasks: lmcleaner4_basic, lmcleaner4_mia, lmcleaner4_complete,
#        lmcleaner5_basic, lmcleaner5_mia, lmcleaner5_complete

GPU=$1
TASK=$2

export CUDA_VISIBLE_DEVICES=$GPU

case $TASK in
    lmcleaner4_basic)
        echo "=== LMCleaner epoch 4 basic eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            task_name=lmcleaner_llama32_epoch4_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch4_K1000
        ;;
    lmcleaner4_mia)
        echo "=== LMCleaner epoch 4 MIA eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_full \
            task_name=lmcleaner_llama32_epoch4_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch4_K1000 \
            eval.tofu.output_dir=saves/unlearn/lmcleaner_llama32_epoch4_K1000/evals_full
        ;;
    lmcleaner4_complete)
        echo "=== LMCleaner epoch 4 complete eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_complete \
            task_name=lmcleaner_llama32_epoch4_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch4_K1000 \
            eval.tofu.output_dir=saves/unlearn/lmcleaner_llama32_epoch4_K1000/evals_complete
        ;;
    lmcleaner5_basic)
        echo "=== LMCleaner epoch 5 basic eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            task_name=lmcleaner_llama32_epoch5_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch5_K1000
        ;;
    lmcleaner5_mia)
        echo "=== LMCleaner epoch 5 MIA eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_full \
            task_name=lmcleaner_llama32_epoch5_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch5_K1000 \
            eval.tofu.output_dir=saves/unlearn/lmcleaner_llama32_epoch5_K1000/evals_full
        ;;
    lmcleaner5_complete)
        echo "=== LMCleaner epoch 5 complete eval on GPU $GPU ==="
        python src/eval.py --config-name=eval.yaml \
            experiment=eval/tofu/default \
            eval=tofu_complete \
            task_name=lmcleaner_llama32_epoch5_K1000 \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/lmcleaner_llama32_epoch5_K1000 \
            eval.tofu.output_dir=saves/unlearn/lmcleaner_llama32_epoch5_K1000/evals_complete
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: lmcleaner4_basic, lmcleaner4_mia, lmcleaner4_complete,"
        echo "                 lmcleaner5_basic, lmcleaner5_mia, lmcleaner5_complete"
        exit 1
        ;;
esac

echo "=== Done ==="
