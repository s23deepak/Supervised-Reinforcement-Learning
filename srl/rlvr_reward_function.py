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
        - "Final Answer: A" or "Final Answer: 42" or "Final Answer: Alice"
        - "The answer is A" or "The answer is 42"
        - "Therefore, the answer is B"
        - "[A]" or "(A)" or boxed answers
        
        Returns:
            Extracted answer string or None if not found.
        """
        if not output:
            return None
        
        output = output.strip()
        
        # Pattern for answer value: letters, numbers
        answer_pattern = r"([A-Za-z]+|[-]?\d+(?:[./]\d+)?)"
        
        # Find ALL occurrences and take the LAST one (actual answer, not example)
        # Pattern 1: "Final Answer: X" (can be letter, word, or number)
        matches = re.findall(rf"[Ff]inal\s*[Aa]nswer[:\s]+{answer_pattern}", output)
        if matches:
            # Take last match, skip if it's just "X" (from example)
            for ans in reversed(matches):
                if ans.upper() != "X":
                    return ans.strip()
        
        # Pattern 2: "The answer is X" or "answer is X"
        matches = re.findall(rf"[Aa]nswer\s+is[:\s]+{answer_pattern}", output)
        if matches:
            for ans in reversed(matches):
                if ans.upper() not in ("X", "IS", "THE"):
                    return ans.strip()
        
        # Pattern 3: boxed answer \boxed{X}
        matches = re.findall(r"\\boxed\{([^}]+)\}", output)
        if matches:
            return matches[-1].strip()
        
        # Pattern 4: Look in the last portion of output (after "assistant" if present)
        if "assistant" in output.lower():
            parts = re.split(r'assistant', output, flags=re.IGNORECASE)
            if len(parts) > 1:
                last_part = parts[-1]
                match = re.search(rf"[Ff]inal\s*[Aa]nswer[:\s]+{answer_pattern}", last_part)
                if match and match.group(1).upper() != "X":
                    return match.group(1).strip()
        
        # Pattern 5: "[A]" or "(A)" at the end
        match = re.search(r"[\[\(]([A-Za-z0-9]+)[\]\)]$", output)
        if match:
            return match.group(1).strip()
        
        # Pattern 6: Look for answer pattern in last few lines
        lines = output.strip().split('\n')
        for line in reversed(lines[-10:]):
            match = re.search(rf"[Ff]inal\s*[Aa]nswer[:\s]+{answer_pattern}", line)
            if match and match.group(1).upper() != "X":
                return match.group(1).strip()
            match = re.search(rf"[Aa]nswer[:\s]+{answer_pattern}", line)
            if match and match.group(1).upper() not in ("X", "IS", "THE"):
                return match.group(1).strip()
        
        # Pattern 7: Single capital letter at very end
        match = re.search(r"\b([A-E])\s*$", output)
        if match:
            return match.group(1).strip()
        
        # Pattern 8: Number at very end 
        match = re.search(r"[-]?\d+(?:[./]\d+)?\s*$", output)
        if match:
            return match.group(0).strip()
        
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


def create_rlvr_reward_function(train_dataset=None):
    """
    TRL-compatible RLVR reward function.
    
    Args:
        train_dataset: Dataset with 'prompt' and 'correct_answer' columns.
                       If provided, uses prompt-to-answer lookup since TRL
                       doesn't pass correct_answer to reward functions.
    
    Returns a callable that computes RLVR rewards.
    """
    rlvr_reward = RLVRRewardFunction(case_sensitive=False)
    
    # Build prompt -> answer mapping if dataset provided
    prompt_to_answer = {}
    if train_dataset is not None:
        print(f"  Building prompt-to-answer mapping for {len(train_dataset)} samples...")
        for item in train_dataset:
            # Use first 500 chars of prompt as key to handle slight variations
            key = item["prompt"][:500] if len(item["prompt"]) > 500 else item["prompt"]
            prompt_to_answer[key] = item["correct_answer"]
        print(f"  Mapping built with {len(prompt_to_answer)} unique prompts")
    
    def reward_fn(completions, prompts=None, correct_answer=None, **kwargs):
        """
        Compute RLVR rewards for TRL GRPOTrainer.
        
        Args:
            completions: List of generated texts.
            prompts: Input prompts (used for lookup).
            correct_answer: Ground truth answer(s) - may be None if TRL doesn't pass it.
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect).
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            # Try to get correct answer from kwargs first (if TRL passes it)
            if correct_answer is not None:
                if isinstance(correct_answer, str):
                    answer = correct_answer
                elif isinstance(correct_answer, list) and i < len(correct_answer):
                    answer = correct_answer[i]
                else:
                    answer = None
            # Fall back to prompt lookup
            elif prompts is not None and prompt_to_answer:
                prompt = prompts[i] if isinstance(prompts, list) else prompts
                key = prompt[:500] if len(prompt) > 500 else prompt
                answer = prompt_to_answer.get(key, None)
            else:
                answer = None
            
            if answer:
                reward = rlvr_reward(completion, answer)
            else:
                reward = 0.0
            rewards.append(reward)
        
        return rewards
    
    return reward_fn
