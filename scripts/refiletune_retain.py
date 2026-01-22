"""
Re-finetune unlearned model on retain samples that were in removed batches.
This recovers utility lost from removing batches that contained both forget and retain samples.
"""
import json
import argparse
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

def get_retain_samples_in_forget_batches(training_log_dir: str, forget_indices: set):
    """Find retain samples that were in batches containing forget samples."""
    log_dir = Path(training_log_dir)
    
    with open(log_dir / "sample_indices.json") as f:
        sample_indices = json.load(f)
    
    retain_samples = set()
    for step_str, indices in sample_indices.items():
        has_forget = any(idx in forget_indices for idx in indices)
        if has_forget:
            for idx in indices:
                if idx not in forget_indices:
                    retain_samples.add(idx)
    
    return sorted(retain_samples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to unlearned model")
    parser.add_argument("--training_log_dir", type=str, default="saves/train_logs/llama32_1b_tofu_safe")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--forget_size", type=int, default=400, help="Number of forget samples (first N)")
    args = parser.parse_args()
    
    # Define forget indices (TOFU forget10 = first 400 samples)
    forget_indices = set(range(args.forget_size))
    
    # Get retain samples to re-finetune
    retain_samples = get_retain_samples_in_forget_batches(args.training_log_dir, forget_indices)
    print(f"Found {len(retain_samples)} retain samples to re-finetune")
    
    # Load original TOFU dataset
    dataset = load_dataset("locuslab/TOFU", "full", split="train")
    
    # Filter to just the retain samples we need
    retain_dataset = dataset.select(retain_samples)
    print(f"Created dataset with {len(retain_dataset)} samples")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    def preprocess(examples):
        # Format as Q&A
        texts = []
        for q, a in zip(examples["question"], examples["answer"]):
            text = f"Question: {q}\nAnswer: {a}"
            texts.append(text)
        
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings
    
    tokenized_dataset = retain_dataset.map(
        preprocess,
        batched=True,
        remove_columns=retain_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting re-finetuning...")
    trainer.train()
    
    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
