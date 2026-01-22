#!/usr/bin/env python3
"""Parallel unlearning and evaluation for GPT-2 TOFU experiment."""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# Configuration
MODEL_PATH = "saves/finetune/tofu_finetune_gpt2"
TRAINING_LOG_DIR = "saves/train_logs/tofu_finetune_gpt2"
BASE_OUTPUT_DIR = "saves/unlearn/tofu_gpt2_eval"

# Epochs (step numbers at end of each epoch)
EPOCHS = {
    "epoch1": 125,
    "epoch2": 250,
    "epoch3": 375,
    "epoch4": 500,
    "epoch5": 625,
}

# LMCleaner K values
K_VALUES = [10, 50, 100, 250, 500]

# Other unlearning methods
OTHER_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO"]

# Available GPUs
GPUS = [0, 1, 2, 3]


def run_lmcleaner_job(gpu: int, epoch_name: str, epoch_step: int, k: int) -> dict:
    """Run LMCleaner unlearn + eval for specific epoch and K."""
    task_name = f"gpt2_LMCleaner_{epoch_name}_K{k}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Starting LMCleaner K={k} {epoch_name} (step={epoch_step})")

    # Unlearn
    unlearn_cmd = [
        "python", "src/train.py", "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path={MODEL_PATH}",
        f"model.tokenizer_args.pretrained_model_name_or_path={MODEL_PATH}",
        "trainer=LMCleanerBatch",
        f"trainer.method_args.training_log_dir={TRAINING_LOG_DIR}",
        f"trainer.method_args.K={k}",
        f"trainer.method_args.max_step={epoch_step}",
        "trainer.method_args.apply_immediately=true",
        "trainer.args.num_train_epochs=0",
        f"task_name={task_name}",
    ]

    log_file = f"{BASE_OUTPUT_DIR}/{task_name}_unlearn.log"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    with open(log_file, "w") as f:
        result = subprocess.run(unlearn_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[GPU {gpu}] FAILED unlearn: {task_name}")
        return {"task": task_name, "status": "unlearn_failed", "gpu": gpu}

    # Eval
    eval_cmd = [
        "python", "src/eval.py", "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name}",
        f"model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/{task_name}",
        f"task_name={task_name}_eval",
    ]

    eval_log = f"{BASE_OUTPUT_DIR}/{task_name}_eval.log"
    with open(eval_log, "w") as f:
        result = subprocess.run(eval_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[GPU {gpu}] FAILED eval: {task_name}")
        return {"task": task_name, "status": "eval_failed", "gpu": gpu}

    print(f"[GPU {gpu}] Completed: {task_name}")
    return {"task": task_name, "status": "success", "gpu": gpu}


def run_other_method_job(gpu: int, method: str) -> dict:
    """Run other unlearning method + eval."""
    task_name = f"gpt2_{method}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Starting {method}")

    # Unlearn
    unlearn_cmd = [
        "python", "src/train.py", "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path={MODEL_PATH}",
        f"model.tokenizer_args.pretrained_model_name_or_path={MODEL_PATH}",
        f"trainer={method}",
        "trainer.args.num_train_epochs=5",
        f"task_name={task_name}",
    ]

    log_file = f"{BASE_OUTPUT_DIR}/{task_name}_unlearn.log"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    with open(log_file, "w") as f:
        result = subprocess.run(unlearn_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[GPU {gpu}] FAILED unlearn: {task_name}")
        return {"task": task_name, "status": "unlearn_failed", "gpu": gpu}

    # Eval
    eval_cmd = [
        "python", "src/eval.py", "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name}",
        f"model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/{task_name}",
        f"task_name={task_name}_eval",
    ]

    eval_log = f"{BASE_OUTPUT_DIR}/{task_name}_eval.log"
    with open(eval_log, "w") as f:
        result = subprocess.run(eval_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[GPU {gpu}] FAILED eval: {task_name}")
        return {"task": task_name, "status": "eval_failed", "gpu": gpu}

    print(f"[GPU {gpu}] Completed: {task_name}")
    return {"task": task_name, "status": "success", "gpu": gpu}


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Build job list
    jobs = []

    # LMCleaner: epoch × K combinations = 5 × 5 = 25 jobs
    for epoch_name, epoch_step in EPOCHS.items():
        for k in K_VALUES:
            jobs.append(("lmcleaner", epoch_name, epoch_step, k))

    # Other methods: 4 jobs
    for method in OTHER_METHODS:
        jobs.append(("other", method))

    print(f"Total jobs: {len(jobs)}")
    print(f"LMCleaner jobs: {len(EPOCHS) * len(K_VALUES)}")
    print(f"Other method jobs: {len(OTHER_METHODS)}")

    # Run with process pool, 4 workers (one per GPU)
    results = []
    gpu_idx = 0

    with ProcessPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = {}

        for job in jobs:
            gpu = GPUS[gpu_idx % len(GPUS)]
            gpu_idx += 1

            if job[0] == "lmcleaner":
                _, epoch_name, epoch_step, k = job
                future = executor.submit(run_lmcleaner_job, gpu, epoch_name, epoch_step, k)
            else:
                _, method = job
                future = executor.submit(run_other_method_job, gpu, method)

            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Job {job} raised exception: {e}")
                results.append({"task": str(job), "status": "exception", "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"Successful: {len(success)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['task']}: {r['status']}")

    print(f"\nResults saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
