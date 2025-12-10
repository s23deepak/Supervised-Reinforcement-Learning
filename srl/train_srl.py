#!/usr/bin/env python3
"""
SRL Training Entry Point

Usage:
    python train_srl.py [--small-model] [--logging-backend BACKEND]

Options:
    --small-model              Use smaller 3B model for testing
    --logging-backend BACKEND  Logging backend: console, tensorboard, or wandb
"""

import os
import sys
import argparse

# Set environment variables BEFORE any other imports
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from srl_async_trainer import AsyncSRLTrainer
from srl_data_loader import create_srl_dataloader
from srl_config import SRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRL model")
    parser.add_argument(
        "--small-model", 
        action="store_true",
        help="Use smaller 3B model for testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (keep small for 8GB VRAM)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="./data/srl_train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--logging-backend",
        type=str,
        choices=["console", "tensorboard", "wandb"],
        default="console",
        help="Logging backend for metrics and resources (default: console)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="srl-training",
        help="Weights & Biases project name (if using wandb)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs",
        help="TensorBoard log directory (if using tensorboard)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SRL Training with vLLM Sleep Mode")
    print("=" * 70)
    print(f"Device: NVIDIA RTX 5060 (8GB VRAM)")
    print(f"Memory Strategy: vLLM Sleep Mode (VRAM <-> 32GB RAM)")
    print("=" * 70)
    
    # Select configuration
    if args.small_model:
        print("\n[Config] Using small model configuration (Qwen2.5-3B)")
        config = SRLConfig.for_small_model()
    else:
        print("\n[Config] Using RTX 5060 optimized configuration (Qwen2.5-7B)")
        config = SRLConfig.for_rtx_5060()
    
    # Override with command line args
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.train_data = args.train_data
    config.output_dir = args.output_dir
    
    # Configure logging backend
    config.logging.backend = args.logging_backend
    config.logging.log_dir = args.log_dir
    config.logging.wandb_project = args.wandb_project
    
    print(f"\n[Logging] Backend: {args.logging_backend}")
    if args.logging_backend == "tensorboard":
        print(f"          Log dir: {args.log_dir}")
    elif args.logging_backend == "wandb":
        print(f"          Project: {args.wandb_project}")
    
    # Initialize trainer
    print("\n[Step 1] Initializing trainer...")
    trainer = AsyncSRLTrainer(config)
    
    # Create dataloader
    print("\n[Step 2] Loading training data...")
    train_loader = create_srl_dataloader(
        config.train_data,
        trainer.tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.model.max_seq_length,
    )
    print(f"  Loaded {len(train_loader.dataset)} samples")
    
    # Training loop
    print("\n[Step 3] Starting training...")
    print("=" * 70)
    
    for epoch in range(config.training.num_epochs):
        metrics = trainer.train_epoch(train_loader)
        
        # Save checkpoint
        checkpoint_dir = os.path.join(config.output_dir, f"epoch_{epoch + 1}")
        trainer.save_checkpoint(checkpoint_dir)
    
    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    trainer.save_checkpoint(final_dir)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
