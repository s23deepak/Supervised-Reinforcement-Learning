#!/usr/bin/env python3
"""
Convert synthetic-data-kit CoT output to SRL step-wise training pairs.

Usage:
    python sdk_to_srl.py --input data/curated --output srl_train_qa.jsonl
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
        Answer: final_answer

    Returns:
        (steps: list of numbered step strings, answer_line: "Answer: ..." or None)
    """
    lines = [l.strip() for l in cot.strip().splitlines() if l.strip()]
    steps = []
    answer_line = None

    for ln in lines:
        # Check if line is Answer
        if ln.lower().startswith("answer:"):
            answer_line = ln
            break
        # Check if line is a numbered step: "N. Title: content"
        if re.match(r"^\d+\.\s+.+?:\s+.+", ln):
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
            if isinstance(data, dict) and "items" in data:
                for item in data["items"]:
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

    # Load and convert
    all_pairs = []
    skipped = 0

    for ex in load_curated_items(args.input):
        question = ex.get("question", "").strip()
        cot = ex.get("cot", "").strip()
        answer = ex.get("answer", "").strip()

        if not (question and cot):
            skipped += 1
            continue

        # Extract steps
        steps, answer_line = split_cot_into_steps(cot)

        if not steps:
            skipped += 1
            continue

        # Ensure final Answer line
        if not answer_line and answer:
            answer_line = f"Answer: {answer}"

        # Infer topic
        topic = infer_topic(question)

        # Create pairs
        pairs = create_srl_pairs(question, steps, answer_line, topic=topic)
        all_pairs.extend(pairs)

    print(f"\nProcessed {len(all_pairs)} pairs (skipped {skipped} malformed)")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)
    split_idx = int(args.val_split * len(all_pairs))

    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Write train
    with open(args.output_train, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(train_pairs)} training pairs → {args.output_train}")

    # Write val
    with open(args.output_val, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(val_pairs)} validation pairs → {args.output_val}")

    # Statistics
    topics = {}
    for pair in all_pairs:
        t = pair.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    print("\nDataset Statistics:")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    print(f"  Topics: {topics}")


if __name__ == "__main__":
    main()
