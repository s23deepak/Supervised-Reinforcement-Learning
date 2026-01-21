#!/usr/bin/env python3
"""
SRL Training with TRL GRPOTrainer + Unsloth + vLLM Sleep Mode

This implements the full SRL pipeline using:
- TRL's GRPOTrainer (battle-tested GRPO implementation)
- Unsloth (memory optimizations, LoRA)
- vLLM (fast inference with sleep mode for VRAM management)

Usage:
    python train_srl.py [OPTIONS]

Options:
    --epochs N          Number of training epochs (default: 1)
    --train-data PATH   Path to training JSONL (default: ./data/srl_train.jsonl)
    --output-dir PATH   Checkpoint directory (default: ./checkpoints_srl)
    --num-rollouts K    Rollouts per prompt (default: 4)

Examples:
    # Train with 7B model for 3 epochs
    python train_srl.py --epochs 3

TensorBoard:
    tensorboard --logdir ./checkpoints_srl/logs
"""

import os
import sys
import argparse
import gc
import json
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import Unsloth FIRST (critical for proper patching)
import unsloth
from unsloth import FastLanguageModel, PatchFastRL

import torch
from datasets import Dataset, load_dataset, load_from_disk

from trl import GRPOConfig, GRPOTrainer
from sleep_aware_grpo_trainer import SleepAwareGRPOTrainer

from srl_reward_function import SRLRewardFunction
from unified_logger import UnifiedLoggerCallback, patch_trainer, set_global_logger, log_samples
from vllm_server_client import VLLMServerClient, VLLMSleepModeCallback
from functools import partial


def load_srl_dataset(data_path: str, tokenizer=None, cache_dir: str = None, system_prompt: str = None) -> Dataset:
    """
    Load SRL training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file.
        tokenizer: Tokenizer to apply chat template.
        cache_dir: If provided, save/load processed dataset to/from this directory.
        system_prompt: Optional system prompt to prepend. If None, no system message is added.
        
    Returns:
        HuggingFace Dataset with prompts and expert actions.
    """    
    # Try to load from cache first
    if cache_dir and os.path.exists(cache_dir):
        print(f"  Loading preprocessed dataset from cache: {cache_dir}")
        return load_from_disk(cache_dir)
    
    # Support sharded datasets (directory with *.jsonl files)
    if os.path.isdir(data_path):
        shard_pattern = os.path.join(data_path, "*.jsonl")
        print(f"  Loading sharded dataset from: {shard_pattern}")
        raw_dataset = load_dataset('json', data_files=shard_pattern, split='train')
    else:
        raw_dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"  Loaded {len(raw_dataset)} samples")
    
    def process_example(item):
        """Process a single example - applied lazily via map()."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": item["input_prompt"]})
        
        # Apply chat template to get prompt
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # Extract question prefix for grouping
        # Use input_prompt (without instruction) since instruction is same for all
        # First 200 chars of the actual question content
        question_prefix = item["input_prompt"][:200]
        
        return {
            "prompt": prompt,
            "expert_action": item.get("expert_action", ""),
            "question_prefix": question_prefix,
        }
    
    # Apply processing - disable auto-caching to avoid disk explosion
    # (8GB JSONL can expand to 50GB+ with chat templates)
    cpu_count = os.cpu_count() or 4
    num_workers = max(1, cpu_count - 1)
    dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        num_proc=num_workers,
        keep_in_memory=True,  # Don't write intermediate Arrow files
        load_from_cache_file=False,  # Don't use HF auto-cache
        desc="Processing samples"
    )
    
    # Only save to disk if cache_dir explicitly provided
    if cache_dir:
        print(f"  Saving preprocessed dataset to cache: {cache_dir}")
        dataset.save_to_disk(cache_dir)
    print(f"  System prompt: {'provided' if system_prompt else 'none'}")
    return dataset


def prefix_aware_collate_fn(features, tokenizer):
    """
    Custom collator that sorts batch by question prefix.
    
    This groups samples with similar prefixes together,
    maximizing vLLM prefix cache hits within each batch.
    """
    # Sort by question prefix to group related samples together
    features = sorted(features, key=lambda x: x.get("question_prefix", ""))
    
    # Remove the question_prefix field - not needed for training
    for f in features:
        f.pop("question_prefix", None)
    
    # Handle string fields separately
    prompts = [f.pop("prompt") for f in features]
    expert_actions = [f.pop("expert_action") for f in features]
    
    batch = {}
    batch["prompt"] = prompts
    batch["expert_action"] = expert_actions
    
    return batch

def create_srl_reward_function(format_check: bool = False, use_dynamic_filter: bool = True, num_generations: int = 4):
    """
    TRL-compatible reward function with SRL step-wise similarity.
    
    TRL's GRPOTrainer passes these kwargs to reward functions:
    - completions: list of generated texts
    - prompts: list of input prompts  
    - Any additional columns from dataset (e.g., expert_action)
    Uses SRLRewardFunction.compute_batch_rewards which includes dynamic sampling.
    
    Args:
        format_check: Whether to enforce step format validation.
        use_dynamic_filter: Enable per-sample std dev filtering (Section 4.2).
        num_generations: Number of rollouts per sample (G) for grouping.
    """
    srl_reward = SRLRewardFunction(
        format_check=format_check,
        min_similarity=0.0,
        penalty_for_format_error=-1.0,
        use_dynamic_filter=use_dynamic_filter,
    )
    
    def reward_fn(completions, prompts=None, expert_action=None, **kwargs):
        """
        Compute SRL similarity rewards with per-sample dynamic sampling.
        
        Args:
            completions: List of generated texts.
            prompts: List of input prompts (unused).
            expert_action: Ground truth action to compare against.
            
        Returns:
            List of reward floats. Filtered samples get group mean.
        """
        if expert_action is None:
            return [0.0] * len(completions)
        
        # Handle single expert_action vs list
        if isinstance(expert_action, str):
            expert_actions = [expert_action] * len(completions)
        else:
            expert_actions = expert_action
        
        # Use compute_batch_rewards with num_generations for per-sample filtering
        return srl_reward.compute_batch_rewards(
            completions, expert_actions, num_generations=num_generations
        )
    
    return reward_fn


def train_srl(
    model_name: str,
    train_data: str,
    output_dir: str = "./checkpoints_srl",
    system_prompt: str = None,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 5e-6,
    num_rollouts: int = 4,
    max_seq_length: int = 2048,
    max_completion_length: int = 256,
    max_samples: int = None,
    lora_rank: int = 16,
    load_in_4bit: bool = True,
    gpu_memory: float = 0.6,
    cache_dir: str = None,
    use_lmcache: bool = False,
):
    """
    Train SRL model programmatically (no command-line args).
    
    Args:
        model_name: HuggingFace model name or local path
        train_data: Path to training JSONL file(s)
        output_dir: Directory to save checkpoints
        system_prompt: Optional system prompt to prepend
        epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        lr: Learning rate
        num_rollouts: Rollouts per prompt (K)
        max_seq_length: Max input sequence length
        max_completion_length: Max generation length
        max_samples: Limit dataset size (for testing)
        lora_rank: LoRA rank
        load_in_4bit: Whether to load in 4-bit quantization
        gpu_memory: vLLM GPU memory utilization (0.0-1.0)
        cache_dir: Directory to cache preprocessed dataset
        use_lmcache: Enable LMCache for cross-batch KV caching
        
    Returns:
        tuple: (model, tokenizer, trainer) - trained model and trainer
    """
    print("=" * 70)
    print("SRL Training: TRL GRPOTrainer + Unsloth + vLLM")
    print("=" * 70)
    print(f"Model: {model_name}")
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
    kv_transfer_config = None
    if use_lmcache:
        os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        kv_transfer_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        print("LMCache enabled")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_memory,
        enable_prefix_caching=True,
        kv_transfer_config=kv_transfer_config,
    )
    print("Model loaded with vLLM")
    
    # Step 3: Attach LoRA
    print("\n[Step 3] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Step 4: Load Dataset
    print("\n[Step 4] Loading dataset...")
    if system_prompt:
        print(f"  Using system prompt ({len(system_prompt)} chars)")
    train_dataset = load_srl_dataset(train_data, tokenizer=tokenizer, cache_dir=cache_dir, system_prompt=system_prompt)
    
    if max_samples and max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    # Step 5: Create Reward Function
    print("\n[Step 5] Setting up SRL reward function...")
    reward_fn = create_srl_reward_function(format_check=False, num_generations=num_rollouts)
    
    # Step 6: Configure Trainer
    print("\n[Step 6] Configuring GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_grad_norm=1.0,
        num_generations=num_rollouts,
        max_completion_length=max_completion_length,
        temperature=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        save_strategy="epoch",
        use_vllm=True,
        vllm_gpu_memory_utilization=gpu_memory,
        push_to_hub=False,
    )
    
    logger_callback = UnifiedLoggerCallback(output_dir=output_dir, sample_interval=0.5)
    set_global_logger(logger_callback.logger)
    patch_trainer()
    
    collator = partial(prefix_aware_collate_fn, tokenizer=tokenizer)
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        callbacks=[logger_callback],
        data_collator=collator,
    )
    
    # Step 7: Train
    print("\n[Step 7] Starting training...")
    print("=" * 70)
    trainer.train()
    
    # Step 8: Save
    print("\n[Step 8] Saving model...")
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to: {final_path}")
    
    print("\n" + "=" * 70)
    print("SRL Training Complete!")
    print("=" * 70)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer, trainer


def main():
    parser = argparse.ArgumentParser(
        description="SRL Training with TRL + vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                        help="Model name or path (HuggingFace or local)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (full precision)")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per prompt (K)")
    
    # Sequence lengths
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max input sequence length")
    parser.add_argument("--max-completion-length", type=int, default=256, help="Max generation length")
    
    # GPU memory
    parser.add_argument("--gpu-memory", type=float, default=0.6,
                        help="vLLM GPU memory utilization (0.0-1.0)")
    
    # Data and output
    parser.add_argument("--train-data", type=str, default="./srl_datasets/srl_train.jsonl",
                        help="Path to training JSONL (default: ./srl_datasets/train.jsonl)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_trained_srl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache preprocessed dataset (speeds up subsequent runs)")
    
    # Prompt customization
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt to prepend to all prompts (optional)")
    
    parser.add_argument("--vllm-server", action="store_true",
                        help="Use external vLLM server (for LMCache disk caching)")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server OpenAI-compatible API URL")
    parser.add_argument("--vllm-sleep-mode", action="store_true",
                        help="Enable sleep mode coordination with vLLM server")
    parser.add_argument("--use-lmcache", action="store_true",
                        help="Enable LMCache for cross-batch KV caching in embedded mode")
    
    # HuggingFace Hub
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub-repo", type=str, default=None,
                        help="HuggingFace repo name (e.g., 'username/model-name')")
    parser.add_argument("--hub-token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Handle 4-bit flag
    load_in_4bit = not args.no_4bit
    
    # Call the train_srl function
    model, tokenizer, trainer = train_srl(
        model_name=args.model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        system_prompt=args.system_prompt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        num_rollouts=args.num_rollouts,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        max_samples=args.max_samples,
        lora_rank=args.lora_rank,
        load_in_4bit=load_in_4bit,
        gpu_memory=args.gpu_memory,
        cache_dir=args.cache_dir,
        use_lmcache=args.use_lmcache,
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
