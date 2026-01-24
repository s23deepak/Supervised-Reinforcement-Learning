#!/usr/bin/env python3
"""
ResamplingGRPOTrainer - GRPOTrainer with continuous re-sampling.

This subclass implements Section 4.2 of the SRL paper (arXiv 2510.25992):
"Keep generating NEW samples until you fill your batch size B with high-variance samples."

When a sample is filtered (low reward variance), it is REPLACED with a NEW sample
from the dataset backup queue, not just regenerated with the same prompt.

Usage:
    from resampling_grpo_trainer import ResamplingGRPOTrainer
    
    trainer = ResamplingGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        std_threshold=0.1,        # Minimum std to keep a sample
        max_resample_attempts=3,  # Max re-generation attempts per batch
    )
"""

import copy
from collections import deque
from typing import Any, Optional

import torch
from trl import GRPOTrainer


class ResamplingGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with TRUE continuous re-sampling for dynamic sampling.
    
    Implements SRL paper Section 4.2: when a sample has low reward variance,
    it is REPLACED with a completely new sample from the dataset.
    
    Key changes from base GRPOTrainer:
    1. Maintains a backup queue of extra samples from dataset
    2. Filters samples by reward std threshold after generation
    3. Swaps filtered samples with NEW samples from backup queue
    4. Regenerates until batch is full or max attempts reached
    5. Falls back to group-mean if backup queue exhausted
    """
    
    def __init__(
        self,
        *args,
        std_threshold: float = 0.1,
        max_resample_attempts: int = 3,
        backup_queue_size: int = 32,
        **kwargs
    ):
        """
        Initialize the resampling trainer.
        
        Args:
            std_threshold: Minimum reward std to keep a sample (paper: 0.05-0.1)
            max_resample_attempts: Max re-generation attempts per batch
            backup_queue_size: Number of extra samples to pre-fetch
            *args, **kwargs: Passed to GRPOTrainer
        """
        super().__init__(*args, **kwargs)
        self.std_threshold = std_threshold
        self.max_resample_attempts = max_resample_attempts
        self.backup_queue_size = backup_queue_size
        
        # Backup queue for replacement samples
        self._backup_queue: deque = deque(maxlen=backup_queue_size)
        self._dataset_iter = None
        self._dataset_exhausted = False
        
        # Stats tracking
        self._resample_stats = {
            "total_batches": 0,
            "batches_with_resampling": 0,
            "samples_filtered": 0,
            "samples_replaced": 0,
            "fallback_to_mean": 0,
        }
        
        print(f"[ResamplingGRPOTrainer] TRUE continuous re-sampling enabled")
        print(f"  std_threshold: {std_threshold}")
        print(f"  max_resample_attempts: {max_resample_attempts}")
        print(f"  backup_queue_size: {backup_queue_size}")
    
    def _init_dataset_iterator(self):
        """Initialize iterator over the training dataset."""
        if self._dataset_iter is None and self.train_dataset is not None:
            self._dataset_iter = iter(self.train_dataset)
            self._dataset_exhausted = False
    
    def _fetch_backup_samples(self, count: int) -> list[dict]:
        """
        Fetch samples from dataset to backup queue.
        
        Args:
            count: Number of samples to fetch
            
        Returns:
            List of samples fetched
        """
        self._init_dataset_iterator()
        
        fetched = []
        for _ in range(count):
            try:
                sample = next(self._dataset_iter)
                fetched.append(sample)
            except StopIteration:
                # Dataset exhausted - restart from beginning
                self._dataset_iter = iter(self.train_dataset)
                try:
                    sample = next(self._dataset_iter)
                    fetched.append(sample)
                except StopIteration:
                    self._dataset_exhausted = True
                    break
        
        return fetched
    
    def _refill_backup_queue(self):
        """Ensure backup queue has enough samples."""
        needed = self.backup_queue_size - len(self._backup_queue)
        if needed > 0:
            samples = self._fetch_backup_samples(needed)
            self._backup_queue.extend(samples)
    
    def _check_sample_variance(self, rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        """
        Check which samples have sufficient reward variance.
        
        Args:
            rewards: Tensor of shape (batch_size * num_generations,)
            num_generations: Number of generations per sample
            
        Returns:
            Boolean mask of shape (num_samples,) - True if sample should be kept
        """
        num_samples = rewards.shape[0] // num_generations
        rewards_grouped = rewards.view(num_samples, num_generations)
        
        # Compute std for each sample group
        if num_generations > 1:
            std_per_sample = rewards_grouped.std(dim=1)
        else:
            std_per_sample = torch.zeros(num_samples, device=rewards.device)
        
        # Keep samples with std > threshold
        keep_mask = std_per_sample > self.std_threshold
        return keep_mask, std_per_sample
    
    def _replace_samples_in_batch(
        self, 
        inputs: list[dict], 
        keep_mask: torch.Tensor,
        num_generations: int
    ) -> tuple[list[dict], int]:
        """
        Replace filtered samples with new ones from backup queue.
        
        Args:
            inputs: List of input dicts (one per completion, grouped by sample)
            keep_mask: Boolean mask of samples to keep
            num_generations: Number of generations per sample
            
        Returns:
            Tuple of (new_inputs, num_replaced)
        """
        self._refill_backup_queue()
        
        num_samples = len(inputs) // num_generations
        new_inputs = list(inputs) 
        num_replaced = 0
        
        for sample_idx in range(num_samples):
            if not keep_mask[sample_idx]:
                # This sample needs to be replaced
                if len(self._backup_queue) == 0:
                    # No backup samples available
                    continue
                
                # Get replacement sample
                replacement = self._backup_queue.popleft()
                
                # Replace all generations for this sample
                start_idx = sample_idx * num_generations
                for gen_idx in range(num_generations):
                    idx = start_idx + gen_idx
                    if idx < len(new_inputs):
                        # Keep structure but replace content
                        new_inputs[idx] = copy.deepcopy(replacement)
                
                num_replaced += 1
        
        return new_inputs, num_replaced
    
    def _replace_with_group_mean(
        self, 
        inputs: dict[str, Any], 
        keep_mask: torch.Tensor,
        num_generations: int
    ) -> dict[str, Any]:
        """
        Replace filtered samples' rewards with group mean (fallback).
        
        This makes advantage = 0 for filtered samples, effectively skipping them.
        """
        if "rewards" not in inputs:
            return inputs
            
        rewards = inputs["rewards"]
        num_samples = rewards.shape[0] // num_generations
        rewards_grouped = rewards.view(num_samples, num_generations)
        
        # For filtered samples, replace all rewards with group mean
        for sample_idx in range(num_samples):
            if not keep_mask[sample_idx]:
                group_mean = rewards_grouped[sample_idx].mean()
                start = sample_idx * num_generations
                end = start + num_generations
                rewards[start:end] = group_mean
                
                self._resample_stats["fallback_to_mean"] += 1
        
        inputs["rewards"] = rewards
        
        # Also zero out advantages for filtered samples
        if "advantages" in inputs:
            advantages = inputs["advantages"]
            for sample_idx in range(num_samples):
                if not keep_mask[sample_idx]:
                    start = sample_idx * num_generations
                    end = start + num_generations
                    advantages[start:end] = 0.0
            inputs["advantages"] = advantages
        
        return inputs
    
    def _generate_and_score_completions(
        self, 
        inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Override to implement TRUE continuous re-sampling.
        
        Flow:
        1. Generate completions + compute rewards (parent method)
        2. Filter samples by std threshold
        3. Replace filtered samples with NEW samples from backup queue
        4. Regenerate for replacement samples
        5. After max attempts, fall back to group-mean replacement
        """
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        
        self._resample_stats["total_batches"] += 1
        batch_had_resampling = False
        
        current_inputs = inputs
        
        for attempt in range(self.max_resample_attempts):
            # Step 1: Generate completions and compute rewards
            result = super()._generate_and_score_completions(current_inputs)
            
            # Step 2: Check variance filter
            if "rewards" not in result:
                return result
            
            rewards = result["rewards"]
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.accelerator.device)
            
            keep_mask, std_values = self._check_sample_variance(rewards, num_generations)
            kept_count = keep_mask.sum().item()
            total_samples = keep_mask.shape[0]
            filtered_count = total_samples - kept_count
            
            self._resample_stats["samples_filtered"] += filtered_count
            
            # Log progress
            if filtered_count > 0:
                print(f"[Resample] Attempt {attempt + 1}: {kept_count}/{total_samples} pass "
                      f"(std threshold={self.std_threshold:.3f})")
                print(f"  Sample stds: {std_values.tolist()}")
            
            # Step 3: If all samples pass, return as-is
            if kept_count == total_samples:
                return result
            
            batch_had_resampling = True
            
            # Step 4: If this is the last attempt OR no backup samples, fall back to group-mean
            if attempt == self.max_resample_attempts - 1 or len(self._backup_queue) == 0:
                if attempt == self.max_resample_attempts - 1:
                    print(f"[Resample] Max attempts reached, using group-mean fallback")
                else:
                    print(f"[Resample] Backup queue empty, using group-mean fallback")
                result = self._replace_with_group_mean(result, keep_mask, num_generations)
                break
            
            # Step 5: Replace filtered samples with NEW samples from backup queue
            current_inputs, num_replaced = self._replace_samples_in_batch(
                current_inputs, keep_mask, num_generations
            )
            self._resample_stats["samples_replaced"] += num_replaced
            
            if num_replaced == 0:
                # Couldn't replace any samples - fall back to group-mean
                print(f"[Resample] No replacements available, using group-mean fallback")
                result = self._replace_with_group_mean(result, keep_mask, num_generations)
                break
            
            print(f"[Resample] Replaced {num_replaced} samples with new ones, regenerating...")
        
        if batch_had_resampling:
            self._resample_stats["batches_with_resampling"] += 1
        
        return result
    
    def train(self, *args, **kwargs):
        """Override to pre-fill backup queue before training starts."""
        print(f"[ResamplingGRPOTrainer] Pre-filling backup queue...")
        self._refill_backup_queue()
        print(f"[ResamplingGRPOTrainer] Backup queue filled with {len(self._backup_queue)} samples")
        
        result = super().train(*args, **kwargs)
        
        # Log final stats
        print(f"\n[ResamplingGRPOTrainer] Final Statistics:")
        print(f"  Total batches: {self._resample_stats['total_batches']}")
        print(f"  Batches with resampling: {self._resample_stats['batches_with_resampling']}")
        print(f"  Total samples filtered: {self._resample_stats['samples_filtered']}")
        print(f"  Total samples replaced: {self._resample_stats['samples_replaced']}")
        print(f"  Fallback to group-mean: {self._resample_stats['fallback_to_mean']}")
        
        return result
    
    def get_resample_stats(self) -> dict:
        """Return resampling statistics for logging."""
        return self._resample_stats.copy()
