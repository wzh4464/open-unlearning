#!/usr/bin/env python3
"""
GPU Scheduler for Experiment Scripts

Automatically monitors GPU usage and schedules experiments on available GPUs.
Supports dependencies between tasks and parallel execution.

Usage:
    python run_all.py                    # Run all experiments
    python run_all.py --dry-run          # Preview without running
    python run_all.py --gpus 0,1,2,3     # Use specific GPUs
    python run_all.py --skip-finetune    # Skip finetuning step
    python run_all.py --epochs 1,2       # Run only specific epochs
"""

import argparse
import subprocess
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import json


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    name: str
    script: str
    depends_on: list[str]
    status: TaskStatus = TaskStatus.PENDING
    gpu_id: Optional[int] = None
    return_code: Optional[int] = None
    log_file: Optional[str] = None


class GPUScheduler:
    def __init__(self, gpus: list[int], script_dir: Path, log_dir: Path, dry_run: bool = False):
        self.gpus = gpus
        self.script_dir = script_dir
        self.log_dir = log_dir
        self.dry_run = dry_run
        self.tasks: dict[str, Task] = {}
        self.gpu_locks = {gpu: threading.Lock() for gpu in gpus}
        self.available_gpus = set(gpus)
        self.gpu_lock = threading.Lock()
        self.task_lock = threading.Lock()  # Lock for thread-safe task status updates

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def add_task(self, name: str, script: str, depends_on: Optional[list[str]] = None):
        """Add a task to the scheduler."""
        self.tasks[name] = Task(
            name=name,
            script=script,
            depends_on=depends_on or []
        )

    def get_gpu_utilization(self) -> dict[int, float]:
        """Get GPU memory utilization for all GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            utilization = {}
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 3:
                    gpu_id = int(parts[0].strip())
                    mem_used = float(parts[1].strip())
                    mem_total = float(parts[2].strip())
                    utilization[gpu_id] = mem_used / mem_total * 100
            return utilization
        except Exception as e:
            print(f"Warning: Could not get GPU utilization: {e}")
            return {gpu: 0 for gpu in self.gpus}

    def acquire_gpu(self) -> Optional[int]:
        """Acquire an available GPU."""
        with self.gpu_lock:
            if self.available_gpus:
                gpu = self.available_gpus.pop()
                return gpu
            return None

    def release_gpu(self, gpu_id: int):
        """Release a GPU back to the pool."""
        with self.gpu_lock:
            self.available_gpus.add(gpu_id)

    def can_run(self, task: Task) -> bool:
        """Check if all dependencies are completed (or skipped)."""
        with self.task_lock:
            for dep_name in task.depends_on:
                if dep_name not in self.tasks:
                    continue
                dep_task = self.tasks[dep_name]
                if dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                    return False
            return True

    def has_failed_dependency(self, task: Task) -> bool:
        """Check if any dependency has failed."""
        with self.task_lock:
            for dep_name in task.depends_on:
                if dep_name not in self.tasks:
                    continue
                dep_task = self.tasks[dep_name]
                if dep_task.status == TaskStatus.FAILED:
                    return True
            return False

    def run_task(self, task: Task, gpu_id: int) -> int:
        """Run a single task on a specific GPU."""
        with self.task_lock:
            task.gpu_id = gpu_id
            task.status = TaskStatus.RUNNING
            task.log_file = str(self.log_dir / f"{task.name}.log")

        script_path = self.script_dir / task.script

        print(f"[GPU {gpu_id}] Starting: {task.name}")

        if self.dry_run:
            print(f"  [DRY RUN] Would run: {script_path} {gpu_id}")
            time.sleep(1)
            return 0

        try:
            with open(task.log_file, "w") as log_f:
                # Pass GPU ID as argument; scripts handle CUDA_VISIBLE_DEVICES internally
                process = subprocess.run(
                    ["bash", str(script_path), str(gpu_id)],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=self.script_dir.parent.parent,  # Project root
                )
                return process.returncode
        except Exception as e:
            print(f"  Error running {task.name}: {e}")
            return -1

    def run_all(self):
        """Run all tasks respecting dependencies and GPU availability."""
        print("=" * 60)
        print("GPU Scheduler - Experiment Runner")
        print("=" * 60)
        print(f"Available GPUs: {self.gpus}")
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 60)
        print()

        # Print task graph
        print("Task dependency graph:")
        for name, task in self.tasks.items():
            deps = ", ".join(task.depends_on) if task.depends_on else "none"
            print(f"  {name} -> depends on: [{deps}]")
        print()

        completed_count = 0
        total_tasks = len(self.tasks)

        with ThreadPoolExecutor(max_workers=len(self.gpus)) as executor:
            futures = {}

            while completed_count < total_tasks:
                # Mark tasks with failed dependencies as skipped
                for name, task in self.tasks.items():
                    with self.task_lock:
                        if task.status == TaskStatus.PENDING and self.has_failed_dependency(task):
                            task.status = TaskStatus.SKIPPED
                            print(f"[SKIPPED] {task.name}: dependency failed")
                            completed_count += 1

                # Find tasks that can run
                for name, task in self.tasks.items():
                    with self.task_lock:
                        if task.status == TaskStatus.PENDING and self.can_run(task):
                            gpu_id = self.acquire_gpu()
                            if gpu_id is not None:
                                future = executor.submit(self.run_task, task, gpu_id)
                                futures[future] = (task, gpu_id)

                # Check for completed tasks
                if futures:
                    done_futures = []
                    for future in list(futures.keys()):
                        if future.done():
                            task, gpu_id = futures[future]
                            try:
                                return_code = future.result()
                                with self.task_lock:
                                    task.return_code = return_code
                                    if return_code == 0:
                                        task.status = TaskStatus.COMPLETED
                                        print(f"[GPU {gpu_id}] Completed: {task.name}")
                                    else:
                                        task.status = TaskStatus.FAILED
                                        print(f"[GPU {gpu_id}] FAILED: {task.name} (exit code: {return_code})")
                                        print(f"  See log: {task.log_file}")
                            except Exception as e:
                                with self.task_lock:
                                    task.status = TaskStatus.FAILED
                                print(f"[GPU {gpu_id}] ERROR: {task.name}: {e}")

                            self.release_gpu(gpu_id)
                            completed_count += 1
                            done_futures.append(future)

                    for f in done_futures:
                        del futures[f]

                time.sleep(0.5)

        # Print summary
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)

        completed = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        skipped = [t for t in self.tasks.values() if t.status == TaskStatus.SKIPPED]

        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"Skipped: {len(skipped)}")

        if failed:
            print()
            print("Failed tasks:")
            for task in failed:
                print(f"  - {task.name}: exit code {task.return_code}")
                print(f"    Log: {task.log_file}")

        # Save results
        results = {
            name: {
                "status": task.status.value,
                "gpu_id": task.gpu_id,
                "return_code": task.return_code,
                "log_file": task.log_file
            }
            for name, task in self.tasks.items()
        }

        results_file = self.log_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        return len(failed) == 0


def parse_int_list(value: str, name: str) -> list[int]:
    """Parse comma-separated integers with validation."""
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as e:
        print(f"Error: Invalid {name} format '{value}'. Expected comma-separated integers.")
        print(f"  Example: --{name} 0,1,2,3")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GPU Scheduler for Experiment Scripts")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip finetuning step")
    parser.add_argument("--epochs", type=str, default="1,2,3,4,5", help="Comma-separated epochs")
    parser.add_argument("--skip-lmcleaner", action="store_true", help="Skip LMCleaner experiments")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline experiments")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-mia", action="store_true", help="Skip MIA evaluation")
    args = parser.parse_args()

    # Parse arguments with validation
    gpus = parse_int_list(args.gpus, "gpus")
    epochs = parse_int_list(args.epochs, "epochs")

    if not gpus:
        print("Error: No GPUs specified. Use --gpus with comma-separated GPU IDs.")
        sys.exit(1)

    if not epochs:
        print("Error: No epochs specified. Use --epochs with comma-separated epoch numbers.")
        sys.exit(1)

    # Check if all skip flags are set
    all_skipped = (args.skip_finetune and args.skip_lmcleaner and
                   args.skip_baselines and args.skip_eval and args.skip_mia)
    if all_skipped:
        print("Warning: All skip flags are set. No tasks will be scheduled.")
        print("Remove some --skip-* flags to run experiments.")
        sys.exit(0)

    # Setup paths
    script_dir = Path(__file__).parent
    log_dir = script_dir.parent.parent / "saves" / "scheduler_logs"

    # Validate epoch script files exist
    missing_scripts = []
    for epoch in epochs:
        if not args.skip_lmcleaner:
            script_path = script_dir / f"02_lmcleaner_epoch{epoch}.sh"
            if not script_path.exists():
                missing_scripts.append(f"02_lmcleaner_epoch{epoch}.sh")
        if not args.skip_baselines:
            script_path = script_dir / f"03_baseline_epoch{epoch}.sh"
            if not script_path.exists():
                missing_scripts.append(f"03_baseline_epoch{epoch}.sh")

    if missing_scripts:
        print("Error: The following script files are missing:")
        for script in missing_scripts:
            print(f"  - {script}")
        print(f"\nAvailable epochs have scripts: 1-5")
        print("Use --epochs to specify valid epochs (e.g., --epochs 1,2,3)")
        sys.exit(1)

    # Create scheduler
    scheduler = GPUScheduler(gpus, script_dir, log_dir, dry_run=args.dry_run)

    # Add tasks with dependencies

    # Step 1: Finetune (prerequisite for all)
    if not args.skip_finetune:
        scheduler.add_task("finetune", "01_finetune.sh", depends_on=[])

    # Step 2: LMCleaner for each epoch
    if not args.skip_lmcleaner:
        for epoch in epochs:
            task_name = f"lmcleaner_epoch{epoch}"
            script = f"02_lmcleaner_epoch{epoch}.sh"
            deps = ["finetune"] if not args.skip_finetune else []
            scheduler.add_task(task_name, script, depends_on=deps)

    # Step 3: Baselines for each epoch
    if not args.skip_baselines:
        for epoch in epochs:
            task_name = f"baseline_epoch{epoch}"
            script = f"03_baseline_epoch{epoch}.sh"
            deps = ["finetune"] if not args.skip_finetune else []
            scheduler.add_task(task_name, script, depends_on=deps)

    # Step 4: Basic evaluation (depends on unlearning)
    if not args.skip_eval:
        eval_deps = []
        if not args.skip_lmcleaner:
            eval_deps.extend([f"lmcleaner_epoch{e}" for e in epochs])
        if not args.skip_baselines:
            eval_deps.extend([f"baseline_epoch{e}" for e in epochs])

        if eval_deps:
            scheduler.add_task("eval_tofu", "04_eval_tofu.sh", depends_on=eval_deps)

    # Step 5: MIA evaluation (depends on basic eval)
    if not args.skip_mia:
        mia_deps = ["eval_tofu"] if not args.skip_eval else []
        if not args.skip_lmcleaner:
            mia_deps.extend([f"lmcleaner_epoch{e}" for e in epochs])
        if not args.skip_baselines:
            mia_deps.extend([f"baseline_epoch{e}" for e in epochs])

        if mia_deps:
            scheduler.add_task("eval_tofu_mia", "05_eval_tofu_mia.sh", depends_on=mia_deps)

    # Run all tasks
    success = scheduler.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
