#!/usr/bin/env python3
"""
Train SRL model with SDK-generated QA data using Unsloth.

Classes:
    - SRLTrainer: Trainer for SRL with Unsloth integration

Functions:
    - run_training(train_file: str, val_file: str, model_name: str, batch_size: int, epochs: int, max_length: int, output_dir: str) -> dict
    - main() -> None

Methods in SRLTrainer:
    - __init__(model_name: str, max_seq_length: int, use_unsloth: bool)
    - generate_rollouts(prompt: str, num_rollouts: int, max_new_tokens: int, temperature: float) -> list
    - compute_batch_rewards(prompts: list, expert_actions: list, num_rollouts: int) -> dict
    - train_step(train_dataloader, num_epochs: int) -> dict
"""
# Unsloth patches PyTorch and transformers to enable 2x faster training
# It modifies underlying libraries at import time to enable optimized quantization and memory-efficient operations
import unsloth
from unsloth import FastLanguageModel
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer

from srl_reward_function import SRLRewardFunction, DynamicSamplingFilter
from srl_data_loader import create_srl_dataloader


class SRLTrainer:
    """Trainer for SRL with Unsloth integration"""

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct", # can't load a GPTQ model with BitsAndBytes config
                                                               # because GPTQ and BitsAndBytes are different quantization methods
                                                               # Unsloth used BitsAndBytes
                 max_seq_length: int = 512,
                 use_unsloth: bool = True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
            )
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.reward_fn = SRLRewardFunction(format_check=True)
        self.sampler = DynamicSamplingFilter(variance_threshold=0.01)

    def generate_rollouts(self,
                         prompt: str,
                         num_rollouts: int = 8,
                         max_new_tokens: int = 128,
                         temperature: float = 1.0) -> list:
        """Generate multiple rollouts for a prompt"""
        rollouts = []
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        for _ in range(num_rollouts):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_action = generated_text[len(prompt):].strip()
            rollouts.append(generated_action)

        return rollouts

    def compute_batch_rewards(self,
                             prompts: list,
                             expert_actions: list,
                             num_rollouts: int = 8) -> dict:
        """Compute rewards for a batch"""
        batch_rewards = []
        batch_variance = []
        kept_indices = []
        filtered_count = 0

        for i, (prompt, expert_action) in enumerate(zip(prompts, expert_actions)):
            rollouts = self.generate_rollouts(prompt, num_rollouts=num_rollouts)
            rollout_rewards = [self.reward_fn(r, expert_action) for r in rollouts]

            if self.sampler.should_keep_sample(rollout_rewards):
                batch_rewards.append(rollout_rewards)
                batch_variance.append(np.var(rollout_rewards))
                kept_indices.append(i)
            else:
                filtered_count += 1

        return {
            "batch_rewards": batch_rewards,
            "batch_variance": batch_variance,
            "kept_indices": kept_indices,
            "filtered_count": filtered_count,
            "total_count": len(prompts),
        }

    def train_step(self, train_dataloader, num_epochs: int = 1) -> dict:
        """Execute training step"""
        self.model.train()
        total_reward = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                reward_dict = self.compute_batch_rewards(
                    batch["prompts"],
                    batch["expert_actions"],
                    num_rollouts=8
                )

                if not reward_dict["kept_indices"]:
                    print(f"Epoch {epoch}, Batch {batch_idx}: All samples filtered")
                    continue

                batch_rewards = reward_dict["batch_rewards"]
                avg_batch_reward = np.mean([np.mean(r) for r in batch_rewards])
                total_reward += avg_batch_reward
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx + 1}: "
                          f"Avg Reward = {avg_batch_reward:.3f}, "
                          f"Filtered = {reward_dict['filtered_count']}/{reward_dict['total_count']}")

        avg_reward = total_reward / max(num_batches, 1)
        return {"avg_reward": avg_reward, "num_batches": num_batches}


def run_training(train_file: str, val_file: str, model_name: str, 
                batch_size: int, epochs: int, max_length: int, 
                output_dir: str = "srl_sdk_model") -> dict:
    """
    Execute the complete SRL training pipeline.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        model_name: Name of the model to use
        batch_size: Batch size for training
        epochs: Number of training epochs
        max_length: Maximum sequence length
        output_dir: Directory to save the trained model
        
    Returns:
        Training statistics dictionary
    """
    print("=" * 70)
    print("Training SRL with Synthetic Data Kit QA")
    print("=" * 70)

    print("Initializing trainer...")
    trainer = SRLTrainer(model_name=model_name, max_seq_length=max_length, use_unsloth=True)

    print("Creating dataloaders...")
    train_loader = create_srl_dataloader(
        train_file,
        trainer.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    val_loader = create_srl_dataloader(
        val_file,
        trainer.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    print("Starting training...")
    stats = trainer.train_step(train_loader, num_epochs=epochs)

    print(f"\nTraining completed!")
    print(f"Average Reward: {stats['avg_reward']:.3f}")
    print(f"Number of Batches: {stats['num_batches']}")

    print(f"\nSaving model to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    print("Done!")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train SRL with SDK data")
    parser.add_argument("--train-file", default="srl_train_qa.jsonl", help="Training data")
    parser.add_argument("--val-file", default="srl_val_qa.jsonl", help="Validation data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", default="srl_sdk_model", help="Output directory for model")
    args = parser.parse_args()

    # Run the training pipeline
    stats = run_training(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
