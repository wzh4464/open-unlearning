#!/bin/bash
# Benchmark efficiency of different unlearning methods
# Usage: bash scripts/benchmark_efficiency.sh

set -e
source "$(dirname "$0")/env.sh"

echo "==================================="
echo "Efficiency Benchmarking Script"
echo "==================================="

# Configuration
METHODS=("GradAscent" "NPO" "LMCleanerSampleLevel")
FORGET_SPLIT="forget01"
RETAIN_SPLIT="retain99"
MODEL="Phi-3.5-1B-Instruct"
EPOCHS=1

echo "Benchmarking methods: ${METHODS[@]}"
echo "Model: $MODEL"
echo "Forget split: $FORGET_SPLIT"
echo ""

# Run benchmarks for each method
for method in "${METHODS[@]}"; do
    echo "-----------------------------------"
    echo "Benchmarking: $method"
    echo "-----------------------------------"

    TASK_NAME="efficiency_${method}_${FORGET_SPLIT}"

    $PYTHON_CMD src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        model=$MODEL \
        trainer=$method \
        forget_split=$FORGET_SPLIT \
        retain_split=$RETAIN_SPLIT \
        task_name=$TASK_NAME \
        trainer.args.num_train_epochs=$EPOCHS \
        trainer.args.efficiency_tracking.enabled=true \
        trainer.args.save_strategy=no \
        trainer.args.evaluation_strategy=no

    echo "✓ Completed: $method"
    echo ""
done

echo "==================================="
echo "Benchmarking Complete!"
echo "==================================="
echo "Results saved to: saves/unlearn/efficiency_*"
echo ""
echo "To view efficiency metrics:"
echo "  cat saves/unlearn/efficiency_*/efficiency_metrics.json"
echo ""
echo "To compare all results:"
echo "  $PYTHON_CMD scripts/compare_efficiency.py"
