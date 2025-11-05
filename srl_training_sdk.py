#!/usr/bin/env python3
"""
Train SRL model with SDK-generated QA data using Unsloth.
"""

import torch
import numpy as np
import argparse
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from srl_reward_function import SRLRewardFunction, DynamicSamplingFilter
from srl_data_loader import create_srl_dataloader


class SRLTrainer:
    """Trainer for SRL with Unsloth integration"""

    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-7b-hf",
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


def main():
    parser = argparse.ArgumentParser(description="Train SRL with SDK data")
    parser.add_argument("--train-file", default="srl_train_qa.jsonl", help="Training data")
    parser.add_argument("--val-file", default="srl_val_qa.jsonl", help="Validation data")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    print("=" * 70)
    print("Training SRL with Synthetic Data Kit QA")
    print("=" * 70)

    print("Initializing trainer...")
    trainer = SRLTrainer(model_name=args.model, max_seq_length=args.max_length, use_unsloth=True)

    print("Creating dataloaders...")
    train_loader = create_srl_dataloader(
        args.train_file,
        trainer.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    val_loader = create_srl_dataloader(
        args.val_file,
        trainer.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    print("Starting training...")
    stats = trainer.train_step(train_loader, num_epochs=args.epochs)

    print(f"\nTraining completed!")
    print(f"Average Reward: {stats['avg_reward']:.3f}")
    print(f"Number of Batches: {stats['num_batches']}")

    print("\nSaving model...")
    trainer.model.save_pretrained("srl_sdk_model")
    trainer.tokenizer.save_pretrained("srl_sdk_model")

    print("Done!")


if __name__ == "__main__":
    main()
