#!/usr/bin/env python3
"""
RLVR Training Script (Stage 2)

Fine-tunes an SRL-pretrained model using RLVR (Reinforcement Learning 
with Verifiable Rewards) to produce correct final answers.

Usage:
    # Basic usage (loads SRL checkpoint and fine-tunes with RLVR)
    python train_rlvr.py --srl-checkpoint ./checkpoints_trained_srl/final

    # With custom data and output
    python train_rlvr.py \\
        --srl-checkpoint ./checkpoints_trained_srl/final \\
        --train-data ../logical_reasoning/data/curated/logical-reasoning-2017-12-02_qa_pairs_cleaned.json \\
        --output-dir ./checkpoints_trained_srl_rlvr \\
        --epochs 1

    # Test with limited samples
    python train_rlvr.py --srl-checkpoint ./checkpoints_trained_srl/final --max-samples 50
"""

import os
import sys
import json
import argparse

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset, load_dataset, load_from_disk
from unsloth import FastLanguageModel
from peft import PeftModel

from trl import GRPOConfig, GRPOTrainer

from rlvr_reward_function import create_rlvr_reward_function
from unified_logger import UnifiedLoggerCallback, patch_trainer, set_global_logger


def load_rlvr_dataset(data_path: str, tokenizer=None, cache_dir: str = None, system_prompt: str = None) -> Dataset:
    """
    Load RLVR training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file with question/correct_answer fields.
        tokenizer: Tokenizer for chat template.
        cache_dir: If provided, save/load processed dataset to/from this directory.
        system_prompt: Optional system prompt. If None, no system message is added.
        
    Returns:
        HuggingFace Dataset with prompts and correct answers.
    """
    # Try to load from cache first
    if cache_dir and os.path.exists(cache_dir):
        print(f"  Loading preprocessed dataset from cache: {cache_dir}")
        return load_from_disk(cache_dir)

    raw_dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"  Loaded {len(raw_dataset)} samples")
    
    def process_example(item):
        """Process a single example."""
        question = item.get("question", item.get("input_prompt", ""))
        answer = item.get("correct_answer", item.get("answer", ""))
        
        if not question or not answer:
            return {"prompt": "", "correct_answer": ""}
        
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt.strip()})
        chat_messages.append({"role": "user", "content": question})
        
        if tokenizer:
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n\nQuestion: {question}\n\n"
        
        return {
            "prompt": prompt,
            "correct_answer": answer,
        }
    
    # Apply processing - multiprocessing for speed
    num_workers = min(4, os.cpu_count() or 1)
    dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        num_proc=num_workers,
        load_from_cache_file=True,
        desc="Processing samples"
    )
    
    # Filter out empty samples
    dataset = dataset.filter(lambda x: x["prompt"] != "")
    
    # Save to cache for future runs
    if cache_dir:
        print(f"  Saving preprocessed dataset to cache: {cache_dir}")
        dataset.save_to_disk(cache_dir)
    return dataset

def train_rlvr(
    base_model: str,
    train_data: str,
    output_dir: str = "./checkpoints_rlvr",
    srl_checkpoint: str = None,
    system_prompt: str = None,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 5e-6,
    num_rollouts: int = 4,
    max_seq_length: int = 2048,
    max_completion_length: int = 512,
    max_samples: int = None,
    lora_rank: int = 16,
    load_in_4bit: bool = True,
    gpu_memory: float = 0.6,
    cache_dir: str = None,
):
    """
    Train RLVR model programmatically (no command-line args).
    
    Args:
        base_model: HuggingFace model name or local path
        train_data: Path to training JSONL file(s)
        output_dir: Directory to save checkpoints
        srl_checkpoint: Path to SRL checkpoint (optional, will attach fresh LoRA if None)
        system_prompt: Optional system prompt to prepend
        epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        lr: Learning rate
        num_rollouts: Rollouts per prompt (K)
        max_seq_length: Max input sequence length
        max_completion_length: Max generation length (keep low to avoid Unsloth bug)
        max_samples: Limit dataset size (for testing)
        lora_rank: LoRA rank
        load_in_4bit: Whether to load in 4-bit quantization
        gpu_memory: vLLM GPU memory utilization (0.0-1.0)
        cache_dir: Directory to cache preprocessed dataset
        
    Returns:
        tuple: (model, tokenizer, trainer) - trained model and trainer
    """
    print("=" * 70)
    print("RLVR Training (Stage 2): Fine-tuning for Final Answer Correctness")
    print("=" * 70)
    print(f"Base Model: {base_model}")
    print(f"SRL Checkpoint: {srl_checkpoint or 'None (fresh LoRA)'}")
    print(f"Train Data: {train_data}")
    print(f"Output Dir: {output_dir}")
    
    # Auto-detect bf16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"Using bf16: {use_bf16}")
    print(f"Batch Size: {batch_size} x {grad_accum} (grad accum)")
    print("=" * 70)
    
    # Step 1: Apply GRPO Patches
    print("\n[Step 1] Applying GRPO patches...")
    PatchFastRL("GRPO", FastLanguageModel)
    
    # Step 2: Load Model
    print("\n[Step 2] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_memory,
    )
    print("Model loaded with vLLM")
    
    # Step 3: Attach LoRA (or load from SRL checkpoint)
    print("\n[Step 3] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("LoRA adapters attached for RLVR training")
    
    # Step 4: Load Dataset
    print("\n[Step 4] Loading RLVR dataset...")
    if system_prompt:
        print(f"  Using system prompt ({len(system_prompt)} chars)")
    train_dataset = load_rlvr_dataset(train_data, tokenizer=tokenizer, cache_dir=cache_dir, system_prompt=system_prompt)
    
    if max_samples and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    # Step 5: Create Reward Function
    print("\n[Step 5] Setting up RLVR reward function...")
    reward_fn = create_rlvr_reward_function(train_dataset)
    print("RLVR reward: 1.0 for correct answer, 0.0 for incorrect")
    
    # Step 6: Configure Trainer
    print("\n[Step 6] Configuring GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_completion_length=max_completion_length,
        num_generations=num_rollouts,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        save_strategy="epoch",
        push_to_hub=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=gpu_memory,
    )
    
    logger_callback = UnifiedLoggerCallback(output_dir=output_dir, sample_interval=0.5)
    set_global_logger(logger_callback.logger)
    patch_trainer()
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        callbacks=[logger_callback],
    )
    
    # Step 7: Train
    print("\n[Step 7] Starting RLVR training...")
    print("=" * 70)
    trainer.train()
    
    # Step 8: Save
    print("\n[Step 8] Saving model...")
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to: {final_path}")
    
    print("\n" + "=" * 70)
    print("RLVR Training Complete!")
    print("=" * 70)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer, trainer


def main():
    parser = argparse.ArgumentParser(description="RLVR Training (Stage 2)")
    parser.add_argument("--srl-checkpoint", type=str, required=True,
                        help="Path to SRL-trained model checkpoint")
    parser.add_argument("--base-model", type=str, 
                        default="unsloth/qwen2.5-3b-instruct-bnb-4bit",
                        help="Base model (if checkpoint is LoRA adapter)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--train-data", type=str, 
                        default="./rlvr_datasets/train.jsonl",
                        help="Path to training data (default: ./rlvr_datasets/train.jsonl)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_trained_srl_rlvr")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per prompt (K)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache preprocessed dataset")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt (overrides default RLVR prompt)")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM")
    parser.add_argument("--vllm-server", action="store_true",
                        help="Use external vLLM server (for LMCache disk caching)")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server OpenAI-compatible API URL")
    
    # HuggingFace Hub
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub-repo", type=str, default=None,
                        help="HuggingFace repo name (e.g., 'username/model-name')")
    parser.add_argument("--hub-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Call the train_rlvr function
    model, tokenizer, trainer = train_rlvr(
        base_model=args.base_model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        srl_checkpoint=args.srl_checkpoint,
        system_prompt=args.system_prompt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        num_rollouts=args.num_rollouts,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            print("Warning: --push-to-hub requires --hub-repo. Skipping push.")
        else:
            print(f"\n[Pushing to HuggingFace Hub: {args.hub_repo}]")
            try:
                token = args.hub_token or os.environ.get("HF_TOKEN")
                model.push_to_hub(args.hub_repo, token=token)
                tokenizer.push_to_hub(args.hub_repo, token=token)
                print(f"Model pushed to: https://huggingface.co/{args.hub_repo}")
            except Exception as e:
                print(f"Failed to push to hub: {e}")


if __name__ == "__main__":
    main()
