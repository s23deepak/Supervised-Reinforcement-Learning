#!/usr/bin/env python3
"""
RLVR Reward Function Implementation

Reinforcement Learning with Verifiable Rewards (RLVR)
Returns 1.0 for correct final answer, 0.0 for incorrect.

This is Stage 2 of SRL training - after the model has been trained
step-wise with SRL, it is fine-tuned with RLVR to produce correct
final answers.
"""

import re
from typing import Optional


class RLVRRewardFunction:
    """
    RLVR reward function for final answer verification.
    
    Rewards:
        - 1.0 if extracted answer matches ground truth
        - 0.0 otherwise
    """
    
    def __init__(self, 
                 case_sensitive: bool = False,
                 partial_reward: float = 0.0):
        """
        Args:
            case_sensitive: Whether comparison is case-sensitive.
            partial_reward: Reward for partially correct answers (default 0).
        """
        self.case_sensitive = case_sensitive
        self.partial_reward = partial_reward
    
    def extract_answer(self, output: str) -> Optional[str]:
        """
        Extract final answer from model output.
        
        Handles formats:
        - "Final Answer: A" or "Final Answer: Alice"
        - "The answer is A" or "The answer is Alice"
        - "Therefore, the answer is B"
        - "[A]" or "(A)"
        
        Returns:
            Extracted answer string or None if not found.
        """
        if not output:
            return None
        
        output = output.strip()
        
        # Find ALL occurrences and take the LAST one (actual answer, not example)
        # Pattern 1: "Final Answer: X" (can be letter or word, but not just "X")
        matches = re.findall(r"[Ff]inal\s*[Aa]nswer[:\s]+([A-Za-z]+)", output)
        if matches:
            # Take last match, skip if it's just "X" (from example)
            for ans in reversed(matches):
                if ans.upper() != "X":
                    return ans.strip()
        
        # Pattern 2: "The answer is X" or "answer is X"
        matches = re.findall(r"[Aa]nswer\s+is[:\s]+([A-Za-z]+)", output)
        if matches:
            for ans in reversed(matches):
                if ans.upper() != "X":
                    return ans.strip()
        
        # Pattern 3: Look in the last portion of output (after "assistant" if present)
        # This handles cases where the system prompt is echoed
        if "assistant" in output.lower():
            # Get text after last "assistant"
            parts = re.split(r'assistant', output, flags=re.IGNORECASE)
            if len(parts) > 1:
                last_part = parts[-1]
                # Try to find answer in last part only
                match = re.search(r"[Ff]inal\s*[Aa]nswer[:\s]+([A-Za-z]+)", last_part)
                if match and match.group(1).upper() != "X":
                    return match.group(1).strip()
        
        # Pattern 4: "[A]" or "(A)" at the end
        match = re.search(r"[\[\(]([A-Za-z]+)[\]\)]$", output)
        if match:
            return match.group(1).strip()
        
        # Pattern 5: Look for answer pattern in last few lines
        lines = output.strip().split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            match = re.search(r"[Ff]inal\s*[Aa]nswer[:\s]+([A-Za-z]+)", line)
            if match and match.group(1).upper() != "X":
                return match.group(1).strip()
            match = re.search(r"[Aa]nswer[:\s]+([A-Za-z]+)", line)
            if match and match.group(1).upper() != "X":
                return match.group(1).strip()
        
        # Pattern 6: Single capital letter at very end
        match = re.search(r"\b([A-E])\s*$", output)
        if match:
            return match.group(1).strip()
        
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        
        answer = answer.strip()
        
        # Remove common prefixes/suffixes
        answer = re.sub(r"^[Aa]nswer[:\s]*", "", answer)
        answer = re.sub(r"[\[\]\(\)\.]", "", answer)
        answer = answer.strip()
        
        if not self.case_sensitive:
            answer = answer.upper()
        
        return answer
    
    def __call__(self, 
                 generated_output: str, 
                 correct_answer: str,
                 **kwargs) -> float:
        """
        Compute RLVR reward.
        
        Args:
            generated_output: Model's generated text.
            correct_answer: Ground truth answer.
            
        Returns:
            1.0 if correct, 0.0 otherwise.
        """
        extracted = self.extract_answer(generated_output)
        extracted_norm = self.normalize_answer(extracted)
        correct_norm = self.normalize_answer(correct_answer)
        
        if not extracted_norm:
            return 0.0
        
        if extracted_norm == correct_norm:
            return 1.0
        
        return self.partial_reward
    
    def compute_batch_rewards(self,
                              completions: list,
                              correct_answers: list) -> list:
        """
        Compute rewards for a batch of completions.
        
        Args:
            completions: List of generated texts.
            correct_answers: List of ground truth answers.
            
        Returns:
            List of rewards (1.0 or 0.0 each).
        """
        rewards = []
        for completion, answer in zip(completions, correct_answers):
            reward = self(completion, answer)
            rewards.append(reward)
        return rewards


def create_rlvr_reward_function():
    """
    TRL-compatible RLVR reward function.
    
    Returns a callable that computes RLVR rewards.
    """
    rlvr_reward = RLVRRewardFunction(case_sensitive=False)
    
    def reward_fn(completions, prompts=None, correct_answer=None, **kwargs):
        """
        Compute RLVR rewards for TRL GRPOTrainer.
        
        Args:
            completions: List of generated texts.
            prompts: Input prompts (unused).
            correct_answer: Ground truth answer(s).
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect).
        """
        if correct_answer is None:
            return [0.0] * len(completions)
        
        # Handle single answer vs list
        if isinstance(correct_answer, str):
            answers = [correct_answer] * len(completions)
        else:
            answers = correct_answer
        
        return rlvr_reward.compute_batch_rewards(completions, answers)
    
    return reward_fn
