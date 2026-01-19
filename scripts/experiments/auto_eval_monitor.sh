#!/bin/bash
# Auto Evaluation Monitor
# Monitors LMCleaner training and starts evals when complete

EVAL_QUEUE="/app/scripts/experiments/eval_queue.sh"

# Function to check if LMCleaner epoch 4 training is done
check_epoch4_done() {
    if grep -q "Saving model" /tmp/lmcleaner_epoch4.log 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to check if LMCleaner epoch 5 training is done
check_epoch5_done() {
    if grep -q "Saving model" /tmp/lmcleaner_epoch5.log 2>/dev/null; then
        return 0
    fi
    return 1
}

echo "Starting Auto Evaluation Monitor..."
echo "Will start evaluations when LMCleaner training completes"
echo ""

EPOCH4_DONE=false
EPOCH5_DONE=false

while true; do
    # Check epoch 4
    if [ "$EPOCH4_DONE" = false ] && check_epoch4_done; then
        echo "$(date): LMCleaner epoch 4 training complete!"

        # Check for free GPU
        for gpu in 1 2; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
            if [ "$mem" -lt 5000 ]; then
                echo "Starting epoch 4 evals on GPU $gpu"
                nohup bash -c "
                    $EVAL_QUEUE $gpu lmcleaner4_basic && \
                    $EVAL_QUEUE $gpu lmcleaner4_mia && \
                    $EVAL_QUEUE $gpu lmcleaner4_complete
                " > /tmp/lmcleaner_epoch4_all_eval.log 2>&1 &
                EPOCH4_DONE=true
                break
            fi
        done
    fi

    # Check epoch 5
    if [ "$EPOCH5_DONE" = false ] && check_epoch5_done; then
        echo "$(date): LMCleaner epoch 5 training complete!"

        # Check for free GPU
        for gpu in 1 2; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
            if [ "$mem" -lt 5000 ]; then
                echo "Starting epoch 5 evals on GPU $gpu"
                nohup bash -c "
                    $EVAL_QUEUE $gpu lmcleaner5_basic && \
                    $EVAL_QUEUE $gpu lmcleaner5_mia && \
                    $EVAL_QUEUE $gpu lmcleaner5_complete
                " > /tmp/lmcleaner_epoch5_all_eval.log 2>&1 &
                EPOCH5_DONE=true
                break
            fi
        done
    fi

    # Exit if both done
    if [ "$EPOCH4_DONE" = true ] && [ "$EPOCH5_DONE" = true ]; then
        echo "All evaluations started. Exiting monitor."
        break
    fi

    sleep 60
done
