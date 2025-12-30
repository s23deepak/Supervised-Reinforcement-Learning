#!/usr/bin/env python3
"""
Dataset Sharding Utility

Splits a large JSONL file into multiple smaller shards for:
- Better I/O throughput (parallel reads)
- Distributed training (each GPU reads different shards)
- Reduced memory pressure during loading

Usage:
    python shard_dataset.py input.jsonl --num-shards 64 --output-dir ./shards
    
    # Then load shards:
    from datasets import load_dataset
    ds = load_dataset('json', data_files='./shards/*.jsonl', split='train')
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def count_lines(filepath: str) -> int:
    """Count lines in a file efficiently."""
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)


def shard_jsonl(
    input_file: str,
    output_dir: str,
    num_shards: int = 64,
    shuffle: bool = False,
) -> list[str]:
    """
    Split a JSONL file into multiple shards.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory for output shards
        num_shards: Number of shards to create
        shuffle: Whether to shuffle before sharding (requires loading all to memory)
    
    Returns:
        List of shard file paths
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Counting lines in {input_file}...")
    total_lines = count_lines(input_file)
    lines_per_shard = (total_lines + num_shards - 1) // num_shards
    
    print(f"Total lines: {total_lines:,}")
    print(f"Lines per shard: ~{lines_per_shard:,}")
    print(f"Creating {num_shards} shards in {output_dir}")
    
    shard_files = []
    
    if shuffle:
        # Load all, shuffle, then shard (memory intensive)
        import random
        print("Loading all data for shuffling...")
        with open(input_file, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        
        for shard_idx in tqdm(range(num_shards), desc="Writing shards"):
            start = shard_idx * lines_per_shard
            end = min(start + lines_per_shard, len(lines))
            
            shard_name = f"shard_{shard_idx:04d}.jsonl"
            shard_path = output_path / shard_name
            
            with open(shard_path, 'w') as f:
                f.writelines(lines[start:end])
            
            shard_files.append(str(shard_path))
    else:
        # Stream through file (memory efficient)
        current_shard = 0
        current_file = None
        lines_written = 0
        
        with open(input_file, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Sharding"):
                # Open new shard if needed
                if current_file is None or lines_written >= lines_per_shard:
                    if current_file:
                        current_file.close()
                    
                    shard_name = f"shard_{current_shard:04d}.jsonl"
                    shard_path = output_path / shard_name
                    current_file = open(shard_path, 'w')
                    shard_files.append(str(shard_path))
                    current_shard += 1
                    lines_written = 0
                
                current_file.write(line)
                lines_written += 1
        
        if current_file:
            current_file.close()
    
    # Print summary
    total_size = sum(os.path.getsize(f) for f in shard_files)
    print(f"\nCreated {len(shard_files)} shards")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Avg shard size: {total_size / len(shard_files) / 1e6:.1f} MB")
    
    return shard_files


def main():
    parser = argparse.ArgumentParser(description="Shard a large JSONL dataset")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output-dir", "-o", default="./shards",
                        help="Output directory for shards")
    parser.add_argument("--num-shards", "-n", type=int, default=64,
                        help="Number of shards to create")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle data before sharding (uses more memory)")
    
    args = parser.parse_args()
    
    shard_files = shard_jsonl(
        args.input,
        args.output_dir,
        args.num_shards,
        args.shuffle
    )
    
    print(f"\nTo load shards:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('json', data_files='{args.output_dir}/*.jsonl', split='train')")


if __name__ == "__main__":
    main()
