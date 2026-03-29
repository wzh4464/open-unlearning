#!/usr/bin/env python3
"""
Generate sparse checkpoints by replaying training from the base model.

For SGD-like replay: loads base model, computes u[t] = -η[t] * grad(θ[t], batch[t]),
advances θ[t+1] = θ[t] + u[t], and saves snapshots at stride intervals.

Usage:
    python scripts/experiments/generate_sparse_checkpoints.py \
        --base-model meta-llama/Llama-3.2-1B-Instruct \
        --train-log-dir saves/train_logs/llama32_1b_tofu_safe \
        --dataset locuslab/TOFU --dataset-name full \
        --stride 25 --max-step 250
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True,
                        help="Pre-finetune base model path")
    parser.add_argument("--train-log-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="locuslab/TOFU")
    parser.add_argument("--dataset-name", type=str, default="full")
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-step", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=4)
    args = parser.parse_args()

    log_dir = Path(args.train_log_dir)
    out_dir = log_dir / "sparse_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sample indices and eta cache
    si = {int(k): v for k, v in json.load(open(log_dir / "sample_indices.json")).items()}

    # Build eta cache from pkl files if eta_cache.json doesn't exist
    eta_path = log_dir / "eta_cache.json"
    if eta_path.exists():
        eta = {int(k): float(v) for k, v in json.load(open(eta_path)).items()}
    else:
        import pickle
        eta = {}
        for f in sorted(log_dir.glob("step_records_chunk_*.pkl")):
            rec = pickle.load(open(f, "rb"))
            if isinstance(rec, list): rec = rec[0]
            eta[rec["step_id"]] = rec["eta"]
        json.dump({str(k): v for k, v in eta.items()}, open(eta_path, "w"))
        logger.info(f"Built eta_cache.json with {len(eta)} entries")

    # Load dataset and tokenizer
    from datasets import load_dataset
    hf_dataset = load_dataset(args.dataset, args.dataset_name, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build collator
    from data.utils import preprocess_chat_instance
    template_args = {
        "apply_chat_template": True,
        "system_prompt": "You are a helpful assistant.",
        "system_prompt_with_special_tokens": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>",
        "user_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end_tag": "<|eot_id|>",
        "asst_start_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "asst_end_tag": "<|eot_id|>",
        "date_string": "10 Apr 2025",
    }

    class SimpleQADataset:
        def __init__(self, data, tokenizer, template_args, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.template_args = template_args
            self.max_length = max_length
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            tokenized = preprocess_chat_instance(
                self.tokenizer, self.template_args,
                [item["question"]], [item["answer"]],
                self.max_length, predict_with_generate=False,
            )
            return {
                "input_ids": tokenized["input_ids"],
                "labels": tokenized["labels"],
                "attention_mask": tokenized["attention_mask"],
            }

    dataset = SimpleQADataset(hf_dataset, tokenizer, template_args)
    from transformers import DataCollatorForLanguageModeling
    # Use the same collator as training
    from data.collators import DataCollatorForSupervisedDataset
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Load base model
    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.train()

    # Replay training
    steps = sorted(s for s in si.keys() if s <= args.max_step)
    logger.info(f"Replaying {len(steps)} steps, saving checkpoints every {args.stride} steps")

    saved = 0
    t0 = time.time()

    for step_id in steps:
        indices = si[step_id]
        lr = eta.get(step_id, 0)

        if lr == 0:
            continue

        # Build batch
        samples = [dataset[i] for i in indices]
        batch = collator(samples)
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # Forward + backward (micro-batched if needed)
        model.zero_grad()
        total_samples = len(indices)
        n_micro = (total_samples + args.micro_batch - 1) // args.micro_batch

        for start in range(0, total_samples, args.micro_batch):
            end = min(start + args.micro_batch, total_samples)
            micro_samples = [dataset[indices[i]] for i in range(start, end)]
            micro_batch = collator(micro_samples)
            micro_batch = {k: v.to(args.device) for k, v in micro_batch.items()}
            outputs = model(**micro_batch)
            (outputs.loss / n_micro).backward()
            del outputs, micro_batch

        # SGD update: θ[t+1] = θ[t] - η * grad
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad

        model.zero_grad()

        # Save sparse checkpoint at stride
        if step_id % args.stride == 0 and step_id > 0:
            ckpt_path = out_dir / f"step_{step_id:06d}.pt"
            if not ckpt_path.exists():
                state = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state[name] = param.detach().cpu().to(torch.bfloat16)
                torch.save(state, ckpt_path)
                saved += 1
                logger.info(f"Saved checkpoint at step {step_id} -> {ckpt_path}")

        if step_id % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"Step {step_id}/{steps[-1]}, {elapsed:.1f}s elapsed")

        del batch
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Done: {saved} checkpoints saved to {out_dir}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
