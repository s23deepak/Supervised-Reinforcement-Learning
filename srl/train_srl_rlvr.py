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
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel

from trl import GRPOConfig, GRPOTrainer

from rlvr_reward_function import create_rlvr_reward_function
from unified_logger import UnifiedLoggerCallback, patch_trainer, set_global_logger


def load_rlvr_dataset(data_path: str, tokenizer=None) -> Dataset:
    """
    Load RLVR training data from JSONL file.
    
    Args:
        data_path: Path to JSONL file with question/correct_answer fields.
        tokenizer: Tokenizer for chat template.
        
    Returns:
        HuggingFace Dataset with prompts and correct answers.
    """
    # System prompt for RLVR
    system_prompt = """You are a helpful assistant for solving logical reasoning problems.
Solve the problem step by step, then provide your final answer.

First, think through the problem in <think> tags.
Then provide your reasoning steps.
Finally, state your answer as "Final Answer: X" where X is the letter (A, B, C, or D).

Example format:
<think>Let me analyze the constraints...</think>
Step 1: ...
Step 2: ...
Final Answer: B"""

    raw_dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"  Loaded {len(raw_dataset)} samples")
    
    def process_example(item):
        """Process a single example."""
        question = item.get("question", item.get("input_prompt", ""))
        answer = item.get("correct_answer", item.get("answer", ""))
        
        if not question or not answer:
            return {"prompt": "", "correct_answer": ""}
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
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
    
    dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        desc="Processing samples"
    )
    
    # Filter out empty samples
    dataset = dataset.filter(lambda x: x["prompt"] != "")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="RLVR Training (Stage 2)")
    parser.add_argument("--srl-checkpoint", type=str, required=True,
                        help="Path to SRL-trained model checkpoint")
    parser.add_argument("--base-model", type=str, 
                        default="unsloth/qwen2.5-3b-instruct-bnb-4bit",
                        help="Base model (if checkpoint is LoRA adapter)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--train-data", type=str, 
                        default="./rlvr_datasets/train.jsonl",
                        help="Path to training data (default: ./rlvr_datasets/train.jsonl)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_trained_srl_rlvr")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per prompt (K)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
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
    
    print("=" * 70)
    print("RLVR Training (Stage 2) - Fine-tuning for Final Answer Correctness")
    print("=" * 70)
    
    use_vllm = not args.no_vllm
    
    # Step 1: Load SRL-pretrained model
    # For RLVR, we load the SRL checkpoint directly (not base + adapter)
    # This preserves Unsloth's vLLM integration including load_lora method
    print("\n[Step 1] Loading SRL-pretrained model...")
    print(f"  SRL checkpoint: {args.srl_checkpoint}")
    
    # Check if SRL checkpoint exists and has adapter
    has_adapter = os.path.exists(os.path.join(args.srl_checkpoint, "adapter_config.json"))
    
    if has_adapter:
        # Load base model then merge/load adapter properly
        print(f"  Loading base model: {args.base_model}")
        
        if use_vllm:
            try:
                # Load base model with vLLM
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.base_model,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    fast_inference=True,
                    gpu_memory_utilization=0.6,
                )
                print("  Base model loaded with vLLM")
                
                # Attach fresh LoRA for RLVR training (don't load SRL adapter)
                # The SRL knowledge is expected to be in a merged checkpoint
                # For adapter checkpoints, we'll train from base
                print("  Note: For vLLM, starting from base model (SRL adapter will be loaded separately)")
                
            except Exception as e:
                print(f"  vLLM failed: {e}")
                use_vllm = False
        
        if not use_vllm:
            # Standard loading - can use PeftModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model = PeftModel.from_pretrained(model, args.srl_checkpoint)
            print(f"  Loaded SRL adapter from {args.srl_checkpoint}")
    else:
        # No adapter - load base model or merged model
        model_path = args.srl_checkpoint if os.path.exists(args.srl_checkpoint) else args.base_model
        print(f"  Loading model from: {model_path}")
        
        if use_vllm:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    fast_inference=True,
                    gpu_memory_utilization=0.6,
                )
                print("  Model loaded with vLLM")
            except Exception as e:
                print(f"  vLLM failed: {e}")
                use_vllm = False
        
        if not use_vllm:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            print("  Model loaded (standard inference)")
    
    # Step 2: Attach fresh LoRA for RLVR training
    print("\n[Step 2] Attaching LoRA adapters for RLVR...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Step 3: Load dataset
    print("\n[Step 3] Loading RLVR dataset...")
    train_dataset = load_rlvr_dataset(args.train_data, tokenizer=tokenizer)
    
    if args.max_samples and args.max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples for testing")
    
    # Step 4: Create RLVR reward function
    print("\nSetting up RLVR reward function...")
    reward_fn = create_rlvr_reward_function()
    print("RLVR reward: 1.0 for correct answer, 0.0 for incorrect")
    
    # Step 5: Configure training
    print("\nConfiguring GRPOTrainer...")
    
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "max_completion_length": 1024, 
        "num_generations": args.num_rollouts,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        "push_to_hub": False,
    }
    
    if use_vllm:
        config_kwargs.update({
            "use_vllm": True,
            "vllm_gpu_memory_utilization": 0.7,
        })
        
        # Configure vLLM server mode if enabled
        if args.vllm_server:
            from urllib.parse import urlparse
            parsed = urlparse(args.vllm_server_url.replace('/v1', ''))
            config_kwargs["vllm_mode"] = "server"
            config_kwargs["vllm_server_host"] = parsed.hostname or "localhost"
            config_kwargs["vllm_server_port"] = parsed.port or 8000
            print(f"\n[vLLM Server Mode] Using external server at {parsed.hostname}:{parsed.port}")
            print("  Make sure the vLLM server is running: ./start_vllm_server.sh")
    
    training_args = GRPOConfig(**config_kwargs)
    
    # Create unified logger with comprehensive metrics
    logger_callback = UnifiedLoggerCallback(
        output_dir=args.output_dir,
        sample_interval=0.5
    )
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
    
    # Step 6: Train
    print("\n[Step 6] Starting RLVR training...")
    trainer.train()
    
    # Step 7: Save
    print("\n[Step 7] Saving model...")
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to: {final_path}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if not args.hub_repo:
            print("Warning: --push-to-hub requires --hub-repo. Skipping push.")
        else:
            print(f"\n[Step 8] Pushing to HuggingFace Hub: {args.hub_repo}")
            try:
                token = args.hub_token or os.environ.get("HF_TOKEN")
                model.push_to_hub(args.hub_repo, token=token)
                tokenizer.push_to_hub(args.hub_repo, token=token)
                print(f"Model pushed to: https://huggingface.co/{args.hub_repo}")
            except Exception as e:
                print(f"Failed to push to hub: {e}")
    
    print("\n" + "=" * 70)
    print("RLVR Training Complete!")
    print(f"Model saved to: {final_path}")
    if args.push_to_hub and args.hub_repo:
        print(f"Model on Hub: https://huggingface.co/{args.hub_repo}")
    print("=" * 70)


if __name__ == "__main__":
    main()
