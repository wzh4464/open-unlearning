#!/usr/bin/env python
"""
Direct LMCleaner unlearning using sample_indices.json to find forget steps.
"""

import gc
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer.training_logger import TrainingLogger, LazyRecordLoader
from trainer.unlearn.lmcleaner_core import (
    HVPConfig,
    StepLog,
    StepRecord,
    compute_correction,
    apply_correction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_forget_steps(sample_indices_file: str, forget_indices: set) -> list:
    """Find which training steps contain forget samples."""
    with open(sample_indices_file) as f:
        sample_indices = json.load(f)

    forget_steps = []
    for step, indices in sample_indices.items():
        for idx in indices:
            if idx in forget_indices:
                forget_steps.append(int(step))
                break

    return sorted(forget_steps)


def run_direct_unlearning(
    model_path: str,
    training_log_dir: str,
    forget_indices: set,
    output_dir: str,
    K: int = 1000,
    hessian_mode: str = "GGN",
    damping: float = 1e-4,
):
    """Run direct LMCleaner unlearning."""

    training_log_dir = Path(training_log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find forget steps
    sample_indices_file = training_log_dir / "sample_indices.json"
    forget_steps = find_forget_steps(str(sample_indices_file), forget_indices)
    logger.info(f"Found {len(forget_steps)} forget steps out of {len(forget_indices)} forget samples")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load training logger metadata (without tensors)
    training_logger = TrainingLogger(log_dir=str(training_log_dir))
    training_logger.load_from_disk(load_tensors=False)

    # Create lazy loader for on-demand tensor loading
    lazy_loader = LazyRecordLoader(training_log_dir)
    lazy_loader.build_index()

    # HVP config
    hvp_config = HVPConfig(
        mode=hessian_mode,
        damping=damping,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Current step (end of training)
    tau = training_logger.current_step
    logger.info(f"Training ended at step {tau}")

    # Model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Model has {len(params)} parameter groups")

    # Apply corrections for each forget step
    audit_records = []

    for i, tz in enumerate(forget_steps):
        logger.info(f"Processing forget step {i+1}/{len(forget_steps)}: tz={tz}")

        try:
            # Calculate steps to load
            start_step = tz
            end_step = min(tz + K, tau - 1)
            needed_steps = list(range(start_step, end_step + 1))

            # Load records on demand
            logger.info(f"Loading steps {start_step} to {end_step}")
            records = lazy_loader.load_steps(needed_steps, include_tensors=True)

            if not records:
                logger.warning(f"No records found for step {tz}, skipping")
                continue

            # Create temporary StepLog
            temp_step_log = StepLog(max_size=len(records) + 10)

            for rec_dict in records:
                step_record = StepRecord(
                    step_id=rec_dict["step_id"],
                    eta=rec_dict["eta"],
                    batch_id=rec_dict["batch_id"],
                    u=rec_dict.get("u"),
                    gbar=rec_dict.get("gbar"),
                    diag_H=rec_dict.get("diag_H"),
                )
                temp_step_log.add(step_record)

            # Compute correction
            v, audit = compute_correction(
                tz=tz,
                tau=tau,
                K=K,
                step_log=temp_step_log,
                cfg=hvp_config,
                model=model,
                loss_fn=None,
                batch_reconstructor=None,
            )

            # Apply correction
            apply_correction(v, params)
            audit_records.append(audit)

            logger.info(
                f"Applied correction for step {tz}: "
                f"v_norm={audit['v_norm']:.6f}, "
                f"K_used={audit['K_used']}"
            )

            # Clean up memory
            del records, temp_step_log, v
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to apply correction for step {tz}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save model
    logger.info(f"Saving unlearned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save audit records
    audit_file = output_dir / "audit_records.json"
    with open(audit_file, "w") as f:
        json.dump(audit_records, f, indent=2)

    # Save unlearning metadata
    meta = {
        "method": "LMCleanerDirect",
        "K": K,
        "hessian_mode": hessian_mode,
        "damping": damping,
        "num_forget_steps": len(forget_steps),
        "num_corrections_applied": len(audit_records),
        "forget_steps": forget_steps,
    }
    with open(output_dir / "unlearning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Unlearning complete! Applied {len(audit_records)} corrections.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to finetuned model")
    parser.add_argument("--training-log-dir", required=True, help="Path to training logs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--forget-percent", type=float, default=10, help="Percentage of forget set (10 = forget10)")
    parser.add_argument("--K", type=int, default=1000, help="Truncation window")
    parser.add_argument("--hessian-mode", default="GGN", choices=["GGN", "diag", "exact"])
    parser.add_argument("--damping", type=float, default=1e-4)

    args = parser.parse_args()

    # TOFU dataset has 4000 samples, forget set is first X%
    num_forget = int(4000 * args.forget_percent / 100)
    forget_indices = set(range(num_forget))

    run_direct_unlearning(
        model_path=args.model_path,
        training_log_dir=args.training_log_dir,
        forget_indices=forget_indices,
        output_dir=args.output_dir,
        K=args.K,
        hessian_mode=args.hessian_mode,
        damping=args.damping,
    )
