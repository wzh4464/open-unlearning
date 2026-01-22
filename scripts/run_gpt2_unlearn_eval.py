#!/usr/bin/env python3
"""Parallel unlearning and evaluation for GPT-2 TOFU experiment.

Note: This script uses subprocess with dynamic command arguments for internal
automation purposes only. All inputs are hardcoded configurations, not user input.
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional
import multiprocessing

# Configuration - use environment variables for portability
SAVES_DIR = os.environ.get("SAVES_DIR", "saves")
MODEL_PATH = os.environ.get("MODEL_PATH", f"{SAVES_DIR}/finetune/tofu_finetune_gpt2")
TRAINING_LOG_DIR = os.environ.get(
    "TRAINING_LOG_DIR", f"{SAVES_DIR}/train_logs/tofu_finetune_gpt2"
)
BASE_OUTPUT_DIR = os.environ.get("BASE_OUTPUT_DIR", f"{SAVES_DIR}/unlearn/tofu_gpt2_eval")

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

# Per-worker GPU assignment (set by initializer)
_worker_gpu: Optional[int] = None


def _init_worker(gpu_queue: multiprocessing.Queue):
    """Initialize worker with a dedicated GPU from the queue."""
    global _worker_gpu
    _worker_gpu = gpu_queue.get()


def run_lmcleaner_job(epoch_name: str, epoch_step: int, k: int) -> dict:
    """Run LMCleaner unlearn + eval for specific epoch and K."""
    gpu = _worker_gpu
    task_name = f"gpt2_LMCleaner_{epoch_name}_K{k}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Starting LMCleaner K={k} {epoch_name} (step={epoch_step})")

    # Unlearn command (all arguments are from hardcoded config, not user input)
    unlearn_cmd = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
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

    # Eval command
    eval_cmd = [
        "python",
        "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path={SAVES_DIR}/unlearn/{task_name}",
        f"model.tokenizer_args.pretrained_model_name_or_path={SAVES_DIR}/unlearn/{task_name}",
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


def run_other_method_job(method: str) -> dict:
    """Run other unlearning method + eval."""
    gpu = _worker_gpu
    task_name = f"gpt2_{method}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU {gpu}] Starting {method}")

    # Unlearn command
    unlearn_cmd = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
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

    # Eval command
    eval_cmd = [
        "python",
        "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "model=gpt2",
        f"model.model_args.pretrained_model_name_or_path={SAVES_DIR}/unlearn/{task_name}",
        f"model.tokenizer_args.pretrained_model_name_or_path={SAVES_DIR}/unlearn/{task_name}",
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
    jobs: List[tuple] = []

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

    # Create a queue with GPU IDs for worker initialization
    # Each worker gets exactly one GPU, ensuring no oversubscription
    gpu_queue: multiprocessing.Queue = multiprocessing.Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    # Run with process pool, one worker per GPU
    results: List[Dict] = []

    with ProcessPoolExecutor(
        max_workers=len(GPUS), initializer=_init_worker, initargs=(gpu_queue,)
    ) as executor:
        futures = {}

        for job in jobs:
            if job[0] == "lmcleaner":
                _, epoch_name, epoch_step, k = job
                future = executor.submit(run_lmcleaner_job, epoch_name, epoch_step, k)
            else:
                _, method = job
                future = executor.submit(run_other_method_job, method)

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
