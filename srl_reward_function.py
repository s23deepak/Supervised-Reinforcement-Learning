# SRL Reward Function Implementation
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
        """Check if generated output follows required format"""
        if not self.format_check:
            return True

        # Expected format: "N. Step Title: content" or "Answer: ..."
        pattern = r"^(\d+\.\s+.+?:\s+.+|Answer:\s+.+)$"
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
        """Extract action (remove <think> tags if present)"""
        if "<think>" in output and "</think>" in output:
            parts = output.split("</think>")
            action = parts[1].strip() if len(parts) > 1 else output
        else:
            action = output
        return action.strip()

    def get_similarity_details(self, generated_action: str, expert_action: str) -> Dict[str, Any]:
        """Get detailed breakdown of similarity"""
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
    """Filter samples based on reward variance"""

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
