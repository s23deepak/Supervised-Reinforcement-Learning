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
    --small-model       Use 3B model (default: 7B)
    --epochs N          Number of training epochs (default: 1)
    --train-data PATH   Path to training JSONL (default: ./data/srl_train.jsonl)
    --output-dir PATH   Checkpoint directory (default: ./checkpoints_srl)
    --num-rollouts K    Rollouts per prompt (default: 4)
    --no-vllm           Disable vLLM, use HF generate instead

Examples:
    # Train with 3B model for 1 epoch
    python train_srl.py --small-model

    # Train with 7B model for 3 epochs
    python train_srl.py --epochs 3

    # Train without vLLM (fallback)
    python train_srl.py --small-model --no-vllm

TensorBoard:
    tensorboard --logdir ./checkpoints_srl/logs
"""

import os
import sys
import argparse
import gc
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import Unsloth FIRST (critical for proper patching)
import unsloth
from unsloth import FastLanguageModel, PatchFastRL

import torch

from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer

from srl_reward_function import SRLRewardFunction
from resource_monitor import ResourceMonitorCallback
from functools import partial


# SRL instruction: tell model to generate only the next step
SRL_INSTRUCTION = (
    "You are solving a reasoning problem step by step. "
    "Given the question and previous steps, provide ONLY the next immediate step. "
    "Do not provide multiple steps or the final answer unless the previous steps lead directly to it.\n\n"
)


def load_srl_dataset(data_path: str, use_instruction: bool = True) -> Dataset:
    """
    Load SRL training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file.
        use_instruction: Whether to prepend SRL instruction to prompts.
        
    Returns:
        HuggingFace Dataset with prompts and expert actions.
    """
    instruction = SRL_INSTRUCTION if use_instruction else ""
    
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompt = instruction + item["input_prompt"]
                # Extract question prefix for grouping
                # Use input_prompt (without instruction) since instruction is same for all
                # First 200 chars of the actual question content
                question_prefix = item["input_prompt"][:200]
                samples.append({
                    "prompt": prompt,
                    "expert_action": item.get("expert_action", ""),
                    "question_prefix": question_prefix,
                })
    
    print(f"  Loaded {len(samples)} samples from {data_path}")
    print(f"  SRL instruction: {'enabled' if use_instruction else 'disabled'}")
    return Dataset.from_list(samples)


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

def create_srl_reward_function(format_check: bool = False, use_dynamic_filter: bool = True):
    """
    TRL-compatible reward function with SRL step-wise similarity.
    
    TRL's GRPOTrainer passes these kwargs to reward functions:
    - completions: list of generated texts
    - prompts: list of input prompts  
    - Any additional columns from dataset (e.g., expert_action)
    Uses SRLRewardFunction.compute_batch_rewards which includes dynamic sampling.
    """
    srl_reward = SRLRewardFunction(
        format_check=format_check,
        min_similarity=0.0,
        penalty_for_format_error=-1.0,
        use_dynamic_filter=use_dynamic_filter,
    )
    
    def reward_fn(completions, prompts=None, expert_action=None, **kwargs):
        """
        Compute SRL similarity rewards  with dynamic sampling.
        
        Args:
            completions: List of generated texts.
            prompts: List of input prompts (unused).
            expert_action: Ground truth action to compare against.
            
        Returns:
            List of reward floats.
        """
        if expert_action is None:
            return [0.0] * len(completions)
        
        # Handle single expert_action vs list
        if isinstance(expert_action, str):
            expert_actions = [expert_action] * len(completions)
        else:
            expert_actions = expert_action
        
        # Use compute_batch_rewards which includes dynamic sampling filter
        return srl_reward.compute_batch_rewards(completions, expert_actions)
    
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="SRL Training with TRL + vLLM Sleep Mode")
    parser.add_argument("--small-model", action="store_true", help="Use 3B model instead of 7B")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--train-data", type=str, default="./data/srl_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_trl_vllm")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per prompt (K)")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM (use HF generate)")
    parser.add_argument("--no-instruction", action="store_true", help="Disable SRL step instruction")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size (for testing)")
    args = parser.parse_args()
    
    
    # Configuration
    
    print("=" * 70)
    print("SRL Training: TRL GRPOTrainer + Unsloth + vLLM Sleep Mode")
    print("=" * 70)
    
    if args.small_model:
        model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
        lora_rank = 16
    else:
        model_name = "unsloth/Qwen2.5-7B-Instruct"
        lora_rank = 32
    
    use_vllm = not args.no_vllm
    
    print(f"Model: {model_name}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Num Rollouts: {args.num_rollouts}")
    print(f"vLLM: {'Enabled (with sleep mode)' if use_vllm else 'Disabled'}")
    print("=" * 70)
    
    
    # Step 1: Apply GRPO Patches
    
    print("\n[Step 1] Applying GRPO patches...")
    PatchFastRL("GRPO", FastLanguageModel)
    
    
    # Step 2: Load Model
    
    print("\n[Step 2] Loading model...")
    
    # When use_vllm=True in GRPOConfig, TRL expects model.vllm_engine
    # Unsloth attaches this when fast_inference=True
    if use_vllm:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                fast_inference=True,
                gpu_memory_utilization=0.6,
                # Enable prefix caching for KV-cache reuse
                # When prompts share a common prefix (same question and previous steps),
                enable_prefix_caching=True,
            )
            print("Model loaded with vLLM")
        except Exception as e:
            print(f"vLLM failed: {e}")
            print("Falling back to standard inference...")
            use_vllm = False
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
    
    
    # Step 3: Attach LoRA Adapters
    
    print("\n[Step 3] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()
    
    
    # Step 4: Load Dataset
    
    print("\n[Step 4] Loading dataset...")
    train_dataset = load_srl_dataset(args.train_data, use_instruction=not args.no_instruction)
    
    # Limit dataset size for testing
    if args.max_samples and args.max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples for testing")
    
    # Step 5: Create Reward Function
    
    print("\n[Step 5] Setting up SRL reward function...")
    reward_fn = create_srl_reward_function(format_check=False)
    
    
    # Step 6: Configure GRPOTrainer with vLLM Sleep Mode
    
    print("\n[Step 6] Configuring GRPOTrainer...")
    
    config_kwargs = {
        "output_dir": args.output_dir,
        
        # Training hyperparameters
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": args.epochs,
        "max_grad_norm": 1.0,
        
        # GRPO specific
        "num_generations": args.num_rollouts,
        "max_completion_length": 256,
        "temperature": 1.0,
        
        "bf16": True,
        "gradient_checkpointing": True,
        
        # Logging with TensorBoard
        "logging_steps": 10,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        
        "push_to_hub": False,
    }
    
    # Add vLLM configuration if enabled
    if use_vllm:
        config_kwargs.update({
            "use_vllm": True, 
            "vllm_gpu_memory_utilization": 0.7, 
            
        })
    
    training_args = GRPOConfig(**config_kwargs)
    
    # Create resource monitor callback
    resource_callback = ResourceMonitorCallback(sample_interval=2.0)
    # Create prefix-aware collator for KV-cache optimization
    collator = partial(prefix_aware_collate_fn, tokenizer=tokenizer)
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        callbacks=[resource_callback],
        data_collator=collator,
    )
    
   
    
    # Step 7: Training
    
    print("\n[Step 7] Starting training...")
    print("=" * 70)
    
    trainer.train()
    
    
    # Step 8: Save Model
    
    print("\n[Step 8] Saving model...")
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Model saved to: {final_path}")
    print("=" * 70)
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
