"""
Data Processing Package for SRL & RLVR

Simple utilities for loading and saving SRL/RLVR training data.
For dataset-specific processing, use the notebooks in srl/notebooks/.

Usage:
    # Load processed data
    from data_processing import load_srl_data, load_rlvr_data
    
    # Save processed data
    from data_processing import save_jsonl
"""

from .utils import (
    load_jsonl,
    save_jsonl,
    load_srl_data,
    load_rlvr_data,
)

__all__ = [
    "load_jsonl",
    "save_jsonl", 
    "load_srl_data",
    "load_rlvr_data",
]
