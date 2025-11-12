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
from typing import Dict, Any
import re


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
    Filter samples based on reward variance.
    
    DynamicSamplingFilter implements the "dynamic sampling" described in Section 4.2 of the SRL paper.
    It filters out samples/steps where all rollouts get similar rewards (low variance), keeping only “informative” steps, improving SRL’s stability and efficiency.
    """

    def __init__(self, variance_threshold: float = 0.01):
        self.variance_threshold = variance_threshold

    def should_keep_sample(self, rewards: list) -> bool:
        """Keep sample if variance > threshold"""
        if len(rewards) < 2:
            return True

        import statistics
        try:
            variance = statistics.variance(rewards)
            return variance > self.variance_threshold
        except:
            return True
