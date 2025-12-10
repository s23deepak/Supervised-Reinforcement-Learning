#!/usr/bin/env python3
"""
SRL Reward Function Implementation

Classes:
    - SRLRewardFunction: Sequence similarity reward function for SRL training
    - DynamicSamplingFilter: Filter samples based on reward variance

Methods in SRLRewardFunction:
    - __init__(format_check: bool, min_similarity: float, penalty_for_format_error: float)
    - check_format(generated_output: str) -> bool
    - compute_sequence_similarity(generated_action: str, expert_action: str) -> float
    - __call__(generated_output: str, expert_action: str, **kwargs) -> float
    - _extract_action_part(output: str) -> str
    - get_similarity_details(generated_action: str, expert_action: str) -> Dict[str, Any]

Methods in DynamicSamplingFilter:
    - __init__(variance_threshold: float)
    - should_keep_sample(rewards: list) -> bool
"""

import difflib
from typing import Dict, Any, List, Tuple, Optional
import re
from multiprocessing import Pool, cpu_count
from functools import partial


def _compute_single_reward(args: Tuple[str, str, bool, float, float]) -> float:
    """
    Helper function for parallel reward computation.
    Must be at module level for multiprocessing.
    
    Args:
        args: Tuple of (generated_output, expert_action, format_check, min_sim, penalty)
    """
    generated_output, expert_action, format_check, min_similarity, penalty = args
    
    # Extract action part
    if "<think>" in generated_output and "</think>" in generated_output:
        try:
            think_end = generated_output.find("</think>")
            action_part = generated_output[think_end + len("</think>"):].strip()
        except:
            action_part = generated_output.strip()
    else:
        action_part = generated_output.strip()
    
    # Format check
    if format_check:
        pattern = r"^(Step \d+:\s*\.$|Checking constraint \d+:|Final Answer:.+)$"
        if not re.match(pattern, action_part.strip()):
            return penalty
    
    # Compute similarity
    matcher = difflib.SequenceMatcher(None, action_part, expert_action)
    matching_blocks = matcher.get_matching_blocks()
    total_matched = sum(block.size for block in matching_blocks)
    total_length = len(action_part) + len(expert_action)
    
    if total_length == 0:
        return 1.0
    
    similarity = (2 * total_matched) / total_length
    return max(similarity, min_similarity)


def compute_rewards_parallel(
    completions: List[str],
    expert_actions: List[str],
    format_check: bool = True,
    min_similarity: float = 0.0,
    penalty: float = -1.0,
    num_workers: Optional[int] = None,
) -> List[float]:
    """
    Compute rewards in parallel using multiprocessing.
    
    This is designed to run on CPU while GPU is busy with training,
    effectively hiding the latency of difflib computations.
    
    Args:
        completions: List of generated outputs.
        expert_actions: List of expert actions (same length as completions).
        format_check: Whether to check format.
        min_similarity: Minimum similarity threshold.
        penalty: Penalty for format errors.
        num_workers: Number of worker processes (default: CPU count).
        
    Returns:
        List of reward values.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), len(completions))
    
    if num_workers <= 1 or len(completions) <= 2:
        # Fall back to sequential for small batches
        return [
            _compute_single_reward((c, e, format_check, min_similarity, penalty))
            for c, e in zip(completions, expert_actions)
        ]
    
    # Prepare arguments for parallel execution
    args_list = [
        (c, e, format_check, min_similarity, penalty)
        for c, e in zip(completions, expert_actions)
    ]
    
    with Pool(num_workers) as pool:
        rewards = pool.map(_compute_single_reward, args_list)
    
    return rewards


class SRLRewardFunction:
    """Sequence similarity reward function for SRL training"""

    def __init__(self, 
                 format_check: bool = True,
                 min_similarity: float = 0.0,
                 penalty_for_format_error: float = -1.0):
        self.format_check = format_check
        self.min_similarity = min_similarity
        self.penalty_for_format_error = penalty_for_format_error

    def check_format(self, generated_output: str) -> bool:
        """Check if generated output is in expected format "Step N: content", "Checking Constraint N: content" or "Final Answer: ..."""
        if not self.format_check:
            return True

        pattern = r"^(Step \d+:\s*\.$|Checking constraint \d+:|Final Answer:.+)$"
        return bool(re.match(pattern, generated_output.strip()))

    def compute_sequence_similarity(self, 
                                    generated_action: str, 
                                    expert_action: str) -> float:
        """Compute sequence similarity: R = 2M / T"""
        matcher = difflib.SequenceMatcher(None, generated_action, expert_action)
        matching_blocks = matcher.get_matching_blocks()
        total_matched = sum(block.size for block in matching_blocks)
        total_length = len(generated_action) + len(expert_action)

        if total_length == 0:
            return 1.0

        similarity = (2 * total_matched) / total_length
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
    Filter samples based on reward variance with dynamic threshold.
    
    DynamicSamplingFilter implements the "dynamic sampling" described in Section 4.2 of the SRL paper.
    It filters out samples/steps where all rollouts get similar rewards (low variance), 
    keeping only "informative" steps, improving SRL's stability and efficiency.
    
    The threshold increases dynamically during training:
    - Early training (step 0): threshold = 0 (keep all samples)
    - Later training: threshold increases linearly to max_threshold
    
    This allows the model to learn from all samples initially, then become
    more selective as it improves.
    """

    def __init__(
        self, 
        variance_threshold: float = 0.01,
        warmup_steps: int = 100,
        start_threshold: float = 0.0,
    ):
        """
        Args:
            variance_threshold: Maximum variance threshold (reached after warmup).
            warmup_steps: Number of steps to linearly increase threshold.
            start_threshold: Initial threshold (default 0 = keep all samples).
        """
        self.max_threshold = variance_threshold
        self.warmup_steps = warmup_steps
        self.start_threshold = start_threshold
        self.current_step = 0

    def get_current_threshold(self) -> float:
        """Get the current dynamic threshold based on training progress."""
        if self.current_step >= self.warmup_steps:
            return self.max_threshold
        
        # Linear interpolation from start to max
        progress = self.current_step / self.warmup_steps
        return self.start_threshold + progress * (self.max_threshold - self.start_threshold)

    def should_keep_sample(self, rewards: list) -> bool:
        """Keep sample if variance > current threshold."""
        if len(rewards) < 2:
            return True

        import statistics
        try:
            variance = statistics.variance(rewards)
            threshold = self.get_current_threshold()
            return variance > threshold
        except:
            return True
    
    def step(self):
        """Call after each training step to update the threshold."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter (e.g., for new training run)."""
        self.current_step = 0
