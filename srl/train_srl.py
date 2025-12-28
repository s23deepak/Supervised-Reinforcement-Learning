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
from datasets import Dataset, load_dataset

from trl import GRPOConfig, GRPOTrainer

from srl_reward_function import SRLRewardFunction
from unified_logger import UnifiedLoggerCallback, patch_trainer, set_global_logger, log_samples
from functools import partial

# SRL instruction: tell model to think first, then generate step
SRL_INSTRUCTION = """You are a helpful assistant for solving logical reasoning problems step by step.
A user will provide a reasoning problem, which may include a partial solution with previous steps.
Your task is to continue the solution by providing the very next logical step.

First, draft your thinking process (inner monologue). Then, generate the solution.
Your response format must follow the template below:

<think>
Your thoughts or draft, like working through an exercise on scratch paper.
Be as casual and as long as you want until you are confident to generate the correct next step.
</think>

Provide only the single, next step to continue the solution. Do not solve the entire problem.

STEP FORMAT REQUIREMENTS:
- Start each step with "Step N:" where N is the step number
- For constraint checking, use "Checking constraint N:"
- For the final answer, use "Final Answer: X"

EXAMPLES:
- "Step 1: Let's identify the key constraints in this problem."
- "Checking constraint 1: Alice cannot sit next to Bob."
- "Final Answer: C"

"""


def load_srl_dataset(data_path: str, tokenizer=None, use_instruction: bool = True) -> Dataset:
    """
    Load SRL training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file.
        tokenizer: Tokenizer to apply chat template.
        use_instruction: Whether to use SRL instruction as system prompt.
        
    Returns:
        HuggingFace Dataset with prompts and expert actions.
    """
    raw_dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"  Loaded {len(raw_dataset)} samples")
    
    def process_example(item):
        """Process a single example - applied lazily via map()."""
        messages = []
        if use_instruction:
            messages.append({"role": "system", "content": SRL_INSTRUCTION.strip()})
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
    
    # Apply processing - batched for speed, removes original columns
    dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        desc="Processing samples"
    )
    print(f"  System prompt: {'enabled' if use_instruction else 'disabled'}")
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
    parser = argparse.ArgumentParser(
        description="SRL Training with TRL + vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
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
    
    # Other
    parser.add_argument("--no-instruction", action="store_true", help="Disable SRL step instruction")
    
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
    
    # Handle 4-bit flag
    load_in_4bit = not args.no_4bit
    
    # Configuration
    print("=" * 70)
    print("SRL Training: TRL GRPOTrainer + Unsloth + vLLM Sleep Mode")
    print("=" * 70)
    
    model_name = args.model
    lora_rank = args.lora_rank
    
    print(f"Model: {model_name}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Load in 4-bit: {load_in_4bit}")
    print(f"Batch Size: {args.batch_size} x {args.grad_accum} (grad accum)")
    print(f"Num Rollouts: {args.num_rollouts}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Max Completion Length: {args.max_completion_length}")
    print(f"GPU Memory: {args.gpu_memory:.0%}")
    if args.vllm_server:
        print(f"vLLM Server Mode: ENABLED")
        print(f"  Server URL: {args.vllm_server_url}")
    print("=" * 70)
    
    
    # Step 1: Apply GRPO Patches
    
    print("\n[Step 1] Applying GRPO patches...")
    PatchFastRL("GRPO", FastLanguageModel)
    
    
    # Step 2: Load Model
    
    print("\n[Step 2] Loading model...")
    
    # Load with vLLM for fast inference
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=True,
            gpu_memory_utilization=args.gpu_memory,
            # Enable prefix caching for KV-cache reuse
            # When prompts share a common prefix (same question and previous steps),
            enable_prefix_caching=True,
        )
        print("Model loaded with vLLM")
    except Exception as e:
        print(f"vLLM failed: {e}")
        print("Falling back to standard inference...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=load_in_4bit,
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
    train_dataset = load_srl_dataset(
        args.train_data, 
        tokenizer=tokenizer,
        use_instruction=not args.no_instruction
    )
    
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
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "max_grad_norm": 1.0,
        
        # GRPO specific
        "num_generations": args.num_rollouts,
        "max_completion_length": args.max_completion_length,
        "temperature": 1.0,
        
        "bf16": True,
        "gradient_checkpointing": True,
        
        # Logging
        "logging_steps": 10,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        
        # vLLM
        "use_vllm": True,
        "vllm_gpu_memory_utilization": args.gpu_memory,
        
        "push_to_hub": False,
    }
    
    # Configure vLLM server mode if enabled
    if args.vllm_server:
        # Parse URL into host and port
        from urllib.parse import urlparse
        parsed = urlparse(args.vllm_server_url.replace('/v1', ''))
        config_kwargs["vllm_mode"] = "server"
        config_kwargs["vllm_server_host"] = parsed.hostname or "localhost"
        config_kwargs["vllm_server_port"] = parsed.port or 8000
        print(f"[vLLM Server Mode] Using external server at {parsed.hostname}:{parsed.port}")
        print("  Make sure the vLLM server is running: ./start_vllm_server.sh")
    training_args = GRPOConfig(**config_kwargs)
    
    # Create unified logger with comprehensive metrics
    logger_callback = UnifiedLoggerCallback(
        output_dir=args.output_dir,
        sample_interval=0.5
    )
    # Set global logger for phase tracking
    set_global_logger(logger_callback.logger)
    
    # Patch GRPOTrainer to emit phase signals
    patch_trainer()
    
    # Create prefix-aware collator for KV-cache optimization
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
    
   
    
    # Step 7: Training
    
    print("\n[Step 7] Starting training...")
    print("=" * 70)
    
    trainer.train()
    
    
    # Step 8: Save Model
    
    print("\n[Step 8] Saving model...")
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to: {final_path}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            print("Warning: --push-to-hub requires --hub-repo. Skipping push.")
        else:
            print(f"\n[Step 9] Pushing to HuggingFace Hub: {args.hub_repo}")
            try:
                token = args.hub_token or os.environ.get("HF_TOKEN")
                model.push_to_hub(args.hub_repo, token=token)
                tokenizer.push_to_hub(args.hub_repo, token=token)
                print(f"Model pushed to: https://huggingface.co/{args.hub_repo}")
            except Exception as e:
                print(f"Failed to push to hub: {e}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Model saved to: {final_path}")
    if args.push_to_hub and args.hub_repo:
        print(f"Model on Hub: https://huggingface.co/{args.hub_repo}")
    print("=" * 70)
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
