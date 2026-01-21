#!/usr/bin/env python3
"""
SRL Reward Function Implementation

Classes:
    - SRLRewardFunction: Sequence similarity reward function for SRL training
    - DynamicSamplingFilter: Filter samples based on reward standard deviation

Methods in SRLRewardFunction:
    - __init__(format_check: bool, min_similarity: float, penalty_for_format_error: float)
    - check_format(generated_output: str) -> bool
    - compute_sequence_similarity(generated_action: str, expert_action: str) -> float
    - __call__(generated_output: str, expert_action: str, **kwargs) -> float
    - _extract_action_part(output: str) -> str
    - get_similarity_details(generated_action: str, expert_action: str) -> Dict[str, Any]

Methods in DynamicSamplingFilter:
    - __init__(std_threshold: float)
    - should_keep_sample(rewards: list) -> bool

Uses cdifflib (C extension) with ThreadPoolExecutor for parallel reward computation.
"""

import difflib
import statistics
from typing import Dict, Any, List
import re
from concurrent.futures import ThreadPoolExecutor
import os

from cdifflib import CSequenceMatcher
from unified_logger import begin_phase, end_phase, log_samples

# Number of CPU cores for parallel reward computation
NUM_WORKERS = max(os.cpu_count() or 4, 8)

print(f"[SRL Reward] Using cdifflib with {NUM_WORKERS} workers")


class SRLRewardFunction:
    """Sequence similarity reward function for SRL training"""

    def __init__(self, 
                 format_check: bool = True,
                 min_similarity: float = 0.0,
                 penalty_for_format_error: float = -1.0,
                 use_dynamic_filter: bool = True,
                 std_threshold: float = 0.1):
        self.format_check = format_check
        self.min_similarity = min_similarity
        self.penalty_for_format_error = penalty_for_format_error
        self.use_dynamic_filter = use_dynamic_filter
        self.dynamic_filter = DynamicSamplingFilter(std_threshold) if use_dynamic_filter else None

    def check_format(self, generated_output: str) -> bool:
        """Check if generated output is in expected format "Step N: content", "Checking Constraint N: content" or "Final Answer: ..." """
        if not self.format_check:
            return True

        pattern = r"^(Step \d+:\s*\.$|Checking constraint \d+:|Final Answer:.+)$"
        return bool(re.match(pattern, generated_output.strip()))

    def compute_sequence_similarity(self, 
                                    generated_action: str, 
                                    expert_action: str) -> float:
        """
        Compute sequence similarity: R = 2M / T
        
        Uses cdifflib (C extension, GIL-free, thread-safe).
        """
        total_length = len(generated_action) + len(expert_action)
        if total_length == 0:
            return 1.0
            
        matcher = CSequenceMatcher(None, generated_action, expert_action)
        similarity = matcher.ratio()
            
        return max(similarity, self.min_similarity)

    def __call__(self, 
                 generated_output: str, 
                 expert_action: str,
                 **kwargs) -> float:
        """Compute reward: similarity in [0,1] or -1 for format error"""
        action_part = self._extract_action_part(generated_output)

        if not self.check_format(action_part):
            return self.penalty_for_format_error

        similarity = self.compute_sequence_similarity(action_part, expert_action)
        return similarity

    def compute_batch_rewards(self, 
                              completions: list, 
                              expert_actions: list,
                              num_generations: int = 4,
                              parallel: bool = True,
                              n_workers: int = None) -> List[float]:
        """
        Compute rewards for a batch of completions with per-sample dynamic sampling.
        
        Implements Section 4.2 of SRL paper (arXiv 2510.25992):
        - Groups completions by num_generations (G rollouts per sample)
        - Filters each sample based on reward std dev across its rollouts  
        - For filtered samples: replaces rewards with group mean (advantage → 0)
        
        Why group-mean replacement works:
        - TRL computes advantage = reward - mean(group_rewards)
        - If all rewards in group = mean, then advantage = 0 for all
        - Zero advantage = no policy gradient update = sample effectively skipped
        - This avoids NaN (corrupts weights) and avoids 0.0 penalty (still trains via KL)
        
        Args:
            completions: List of generated texts (batch_size * num_generations).
            expert_actions: List of expert actions (same length as completions).
            num_generations: Number of rollouts per sample (G). Used for grouping.
            parallel: If True, compute rewards in parallel across CPU cores.
            n_workers: Number of worker threads (default: NUM_WORKERS).
            
        Returns:
            List of rewards. Filtered samples get group mean (effective no-op).
        """
        begin_phase("reward")  # Track reward computation phase
        
        n = len(completions)
        
        if n == 0:
            end_phase()
            return []
        
        if n_workers is None:
            n_workers = min(n, NUM_WORKERS)
        
        # Step 1: Compute all rewards in parallel
        if parallel and n > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                rewards = list(executor.map(
                    self._compute_single_reward,
                    completions,
                    expert_actions
                ))
        else:
            rewards = [self(c, e) for c, e in zip(completions, expert_actions)]
        
        # Step 2: Apply per-sample dynamic sampling filter (Section 4.2)
        if self.dynamic_filter and num_generations > 1:
            num_samples = n // num_generations
            kept_count = 0
            
            for sample_idx in range(num_samples):
                start = sample_idx * num_generations
                end = start + num_generations
                sample_rewards = rewards[start:end]
                
                # Check if this sample should be kept
                if not self.dynamic_filter.should_keep_sample(sample_rewards):
                    # Replace with group mean - this makes advantage ≈ 0
                    # This is the TRL-safe way to "skip" without NaN or penalty
                    group_mean = statistics.mean(sample_rewards)
                    for i in range(start, end):
                        rewards[i] = group_mean
                else:
                    kept_count += 1
            
            log_samples(kept=kept_count, total=num_samples)
        else:
            log_samples(kept=n, total=n)
        
        end_phase()
        return rewards

    def _compute_single_reward(self, completion: str, expert_action: str) -> float:
        """Helper for parallel reward computation."""
        return self(completion, expert_action)

    def _extract_action_part(self, output: str) -> str:
        """
        Extract action part from model output.
        Model generates: <think>[thinking]</think>\n1. Step: content
        Extract only: 1. Step: content
        
        Args:
            output: Model generated output with optional <think> block
        
        Returns:
            Just the numbered step with no thinking
        """
    
        if "<think>" in output and "</think>" in output:
            try:
                think_end = output.find("</think>")
                action = output[think_end + len("</think>"):].strip()
                return action
            except:
                return output.strip()
        else:
            return output.strip()

    def get_similarity_details(self, generated_action: str, expert_action: str) -> Dict[str, Any]:
        matcher = difflib.SequenceMatcher(None, generated_action, expert_action)
        matching_blocks = matcher.get_matching_blocks()
        total_matched = sum(block.size for block in matching_blocks)
        total_length = len(generated_action) + len(expert_action)
        similarity = (2 * total_matched) / total_length if total_length > 0 else 1.0

        return {
            "similarity": similarity,
            "matched_length": total_matched,
            "total_length": total_length,
            "generated_length": len(generated_action),
            "expert_length": len(expert_action),
            "matching_blocks": len(matching_blocks),
            "format_valid": self.check_format(generated_action)
        }


class DynamicSamplingFilter:
    """
    Filter samples based on reward standard deviation.
    
    Implements "dynamic sampling" from Section 4.2 of SRL paper (arXiv 2510.25992):
    Retain sample if: std(rewards) > ε
    
    This filters out samples where all rollouts get similar rewards (low std dev),
    keeping only "informative" samples that provide meaningful learning signal.
    
    Paper reports ~2-3% improvement on AIME/AMC benchmarks with this filter.
    """

    def __init__(self, std_threshold: float = 0.1):
        """
        Args:
            std_threshold: Minimum std dev to keep a sample. Paper recommends 0.05-0.1.
        """
        self.std_threshold = std_threshold

    def should_keep_sample(self, rewards: list) -> bool:
        """
        Keep sample if std dev of rewards > threshold.
        
        Args:
            rewards: List of rewards for G rollouts of this sample.
            
        Returns:
            True if sample should be kept for training.
        """
        if len(rewards) < 2:
            return True

        import statistics
        try:
            std_dev = statistics.stdev(rewards)
            return std_dev > self.std_threshold
        except statistics.StatisticsError:
            return True
