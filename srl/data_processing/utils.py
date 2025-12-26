"""
Simple utilities for loading and saving SRL/RLVR training data.
"""

import json
from pathlib import Path
from typing import List, Dict, Iterator


def load_jsonl(file_path: str) -> Iterator[Dict]:
    """
    Load JSONL file, yielding one dict per line.
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        Dict for each line
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(data: List[Dict], file_path: str):
    """
    Save list of dicts to JSONL file.
    
    Args:
        data: List of dicts to save
        file_path: Output path
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_srl_data(file_path: str) -> List[Dict]:
    """
    Load SRL training data from JSONL.
    
    Expected format per line:
        {"input_prompt": "...", "expert_action": "...", "topic": "...", 
         "step_number": N, "total_steps": M}
    
    Args:
        file_path: Path to SRL JSONL file
        
    Returns:
        List of SRL training samples
    """
    samples = list(load_jsonl(file_path))
    
    # Validate format
    required_fields = {'input_prompt', 'expert_action'}
    for i, sample in enumerate(samples):
        missing = required_fields - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {i} missing required fields: {missing}")
    
    return samples


def load_rlvr_data(file_path: str) -> List[Dict]:
    """
    Load RLVR training data from JSONL.
    
    Expected format per line:
        {"question": "...", "correct_answer": "..."}
    
    Args:
        file_path: Path to RLVR JSONL file
        
    Returns:
        List of RLVR training samples
    """
    samples = list(load_jsonl(file_path))
    
    # Validate format
    required_fields = {'question', 'correct_answer'}
    for i, sample in enumerate(samples):
        missing = required_fields - set(sample.keys())
        if missing:
            raise ValueError(f"Sample {i} missing required fields: {missing}")
    
    return samples
