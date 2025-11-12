#!/usr/bin/env python3
"""
Convert synthetic-data-kit CoT output to SRL step-wise training pairs.

Functions:
    - split_cot_into_steps(cot: str) -> Tuple[List[str], str]
    - create_srl_pairs(question: str, steps: List[str], answer_line: str, topic: str) -> List[Dict]
    - load_curated_items(curated_dir: str) -> Generator[Dict, None, None]
    - infer_topic(question: str) -> str
    - process_examples(input_dir: str, val_split: float, seed: int, output_train: str, output_val: str) -> Tuple[List[Dict], List[Dict], Dict]
    - write_jsonl_files(train_pairs: List[Dict], val_pairs: List[Dict], train_file: str, val_file: str) -> None
    - main() -> None
"""

import json
import re
import glob
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Generator


def split_cot_into_steps(cot: str) -> Tuple[List[str], str]:
    """
    Extract numbered steps and final Answer line from CoT string.

    Expected format:
        1. Title: content
        2. Title: content
        ...
        Final Answer: final_answer

    Returns:
        (steps: list of numbered step strings, answer_line: "Answer: ..." or None)
    """
    lines = [l.strip() for l in cot.strip().replace('\\n', '\n').splitlines() if l.strip()]
    steps = []
    answer_line = None
    step_and_content_pattern = r'^Step \d+:\s*\.$|Checking constraint \d+:'
    for ln in lines:
        # Check if line is Answer
        if "Final Answer:" in ln:
            answer_line = ln
            break
        # Check if line is a numbered step: "Step N: content"
        if re.match(step_and_content_pattern, ln):
            steps.append(ln)
    return steps, answer_line


def create_srl_pairs(question: str, steps: List[str], answer_line: str, topic: str = "qa") -> List[Dict]:
    """
    Convert CoT steps into SRL (input_prompt, expert_action) pairs.

    For each step i, create a pair where:
      - input_prompt = question + steps 0..i
      - expert_action = step i+1

    Then, create a final pair:
      - input_prompt = question + all steps
      - expert_action = Answer line

    Args:
        question: Problem statement
        steps: List of numbered step strings
        answer_line: Final "Answer: ..." line
        topic: Task category (e.g., "qa", "seating", "blood_relation")

    Returns:
        List of SRL training pairs
    """
    if not steps:
        return []

    pairs = []
    problem = f"Reasoning Question\nQuestion: {question}"

    # Create step-to-step transitions
    for i in range(len(steps) - 1):
        context = problem + "\n\n" + "\n".join(steps[:i+1])
        next_step = steps[i + 1]
        pairs.append({
            "input_prompt": context,
            "expert_action": next_step,
            "topic": topic,
            "step_number": i + 1,
            "total_steps": len(steps) + (1 if answer_line else 0),
        })

    # Create final step-to-answer transition
    if answer_line:
        context = problem + "\n\n" + "\n".join(steps)
        pairs.append({
            "input_prompt": context,
            "expert_action": answer_line,
            "topic": topic,
            "step_number": len(steps),
            "total_steps": len(steps) + 1,
        })

    return pairs


def load_curated_items(curated_dir: str = "data/curated") -> Generator[Dict, None, None]:
    """
    Load all JSON files from curated directory.

    Handles both {"items": [...]} and plain list [...] formats.
    """
    curated_path = Path(curated_dir)
    if not curated_path.exists():
        raise FileNotFoundError(f"Curated directory not found: {curated_dir}")

    json_files = sorted(glob.glob(str(curated_path / "*.json")))
    print(f"Found {len(json_files)} JSON files in {curated_dir}")

    for fp in json_files:
        print(f"  Loading {Path(fp).name}...")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle wrapped format
            if isinstance(data, dict) and "cot_examples" in data:
                for item in data["cot_examples"]:
                    yield item
            if isinstance(data, dict) and "qa_examples" in data:
                for item in data["qa_examples"]:
                    yield item
            # Handle plain list
            elif isinstance(data, list):
                for item in data:
                    yield item
        except Exception as e:
            print(f"    Error loading {fp}: {e}")
            continue


def infer_topic(question: str) -> str:
    """Infer task topic from question text."""
    q_lower = question.lower()
    if any(word in q_lower for word in ["sit", "seat", "position", "left", "right", "end", "center"]):
        return "seating_arrangement"
    if any(word in q_lower for word in ["related", "father", "mother", "uncle", "aunt", "cousin", "blood", "family", "grandparent"]):
        return "blood_relation"
    return "qa"


def process_examples(input_dir: str, val_split: float, seed: int, 
                    output_train: str = None, output_val: str = None) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process curated examples and convert to SRL training pairs.
    
    Args:
        input_dir: Directory containing curated JSON files
        val_split: Train/validation split ratio (0.9 = 90% train)
        seed: Random seed for shuffling
        output_train: Optional path to write training pairs directly
        output_val: Optional path to write validation pairs directly
        
    Returns:
        Tuple of (train_pairs, val_pairs, statistics)
    """
    all_pairs = []
    skipped = 0

    for ex in load_curated_items(input_dir):
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        reasoning = ex.get("reasoning", "").strip()

        # Skip if no question or topic is general QA
        if not question:
            skipped += 1
            continue
        
        topic = infer_topic(question)
        if topic == "qa":
            skipped += 1
            continue

        # Extract steps
        steps, answer_line = split_cot_into_steps(reasoning)

        if not steps:
            print(f"Skipping malformed example (no steps): {question[:60]}...")
            skipped += 1
            continue

        # Ensure final Answer line
        if not answer_line and answer:
            answer_line = f"Answer: {answer}"

        # Create pairs
        pairs = create_srl_pairs(question, steps, answer_line, topic=topic)
        all_pairs.extend(pairs)

    print(f"\nProcessed {len(all_pairs)} pairs (skipped {skipped} malformed)")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_pairs)
    split_idx = int(val_split * len(all_pairs))

    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Statistics
    topics = {}
    for pair in all_pairs:
        t = pair.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    stats = {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "topics": topics,
        "skipped": skipped
    }

    # Optionally write files if paths provided
    if output_train and output_val:
        write_jsonl_files(train_pairs, val_pairs, output_train, output_val)

    return train_pairs, val_pairs, stats


def write_jsonl_files(train_pairs: List[Dict], val_pairs: List[Dict], 
                      train_file: str, val_file: str) -> None:
    # Write train
    with open(train_file, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(train_pairs)} training pairs → {train_file}")

    # Write val
    with open(val_file, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(val_pairs)} validation pairs → {val_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert SDK CoT output to SRL training pairs")
    parser.add_argument("--input", default="data/curated", help="Input directory (SDK curated output)")
    parser.add_argument("--output-train", default="srl_train_qa.jsonl", help="Output training file")
    parser.add_argument("--output-val", default="srl_val_qa.jsonl", help="Output validation file")
    parser.add_argument("--val-split", type=float, default=0.9, help="Train/val split (0.9 = 90% train)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    print("=" * 70)
    print("Converting SDK CoT output → SRL step-wise pairs")
    print("=" * 70)

    # Load and convert examples (with automatic file writing)
    train_pairs, val_pairs, stats = process_examples(
        args.input, 
        args.val_split, 
        args.seed,
        args.output_train,
        args.output_val
    )

    # Statistics
    print("\nDataset Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Train: {stats['train_pairs']}, Val: {stats['val_pairs']}")
    print(f"  Topics: {stats['topics']}")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
