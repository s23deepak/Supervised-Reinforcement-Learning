#!/usr/bin/env python3
"""
Test Script for SRL/RLVR Trained Models

Evaluates a trained model on reasoning questions and reports accuracy.

Usage:
    # Test RLVR model
    python test_model.py --model ./checkpoints_trained_srl_rlvr/final
    
    # Test SRL model  
    python test_model.py --model ./checkpoints_trained_srl/final
    
"""

import os
import json
import argparse
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from unsloth import FastLanguageModel
from rlvr_reward_function import RLVRRewardFunction
from peft import PeftModel


def load_model(model_path: str, base_model: str = "unsloth/qwen2.5-3b-instruct-bnb-4bit"):
    """Load a trained model."""
    print(f"Loading model from: {model_path}")
    
    # Check if it's an adapter or full model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load base model + adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        print(f"  Loaded adapter from {model_path}")
    else:
        # Load as full model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        print(f"  Loaded full model from {model_path}")
    
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_tokens: int = 512) -> str:
    """Generate a response to a question using chat template."""
    
    # System prompt
    system_prompt = """You are a helpful assistant for solving logical reasoning problems.
Solve the problem step by step, then provide your final answer.

First, think through the problem in <think> tags.
Then provide your reasoning steps.
Finally, state your answer as "Final Answer: X" where X is the answer (can be a letter like A, B, C or a name like Alice, Bob).

Example format:
<think>Let me analyze the constraints...</think>
Step 1: ...
Step 2: ...
Final Answer: B"""

    # Build chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    if prompt in response:
        response = response[len(prompt):]
    return response.strip()


def evaluate_on_dataset(model, tokenizer, data_path: str, max_samples: int = None):
    """Evaluate model on a dataset and report accuracy."""
    print(f"\nEvaluating on: {data_path}")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Parse data based on format
    items = []
    if isinstance(data, dict) and "qa_pairs" in data:
        # Format 1: QA pairs with choices
        for item in data["qa_pairs"]:
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer = item.get("answer", "")
            if question and answer:
                # Include choices in question
                choices_text = "\n".join(choices) if choices else ""
                full_question = f"{question}\n\nOptions:\n{choices_text}" if choices_text else question
                items.append({"question": full_question, "answer": answer})
    else:
        # Format 2: Messages format
        data_list = data if isinstance(data, list) else []
        for item in data_list:
            messages = item.get("messages", [])
            question = None
            answer = None
            for msg in messages:
                if msg["role"] == "user":
                    question = msg["content"]
                elif msg["role"] == "assistant":
                    answer = msg["content"]
            if question and answer:
                items.append({"question": question, "answer": answer})
    
    if max_samples:
        items = items[:max_samples]
    
    print(f"  Loaded {len(items)} samples")
    
    reward_fn = RLVRRewardFunction()
    
    correct = 0
    total = 0
    
    for i, item in enumerate(items):
        question = item["question"]
        correct_answer = item["answer"]
        
        # Generate response
        response = generate_response(model, tokenizer, question)
        
        # Check correctness
        reward = reward_fn(response, correct_answer)
        is_correct = reward == 1.0
        
        if is_correct:
            correct += 1
        total += 1
        
        # Print progress
        status = "✓" if is_correct else "✗"
        extracted = reward_fn.extract_answer(response)
        print(f"  [{i+1}/{len(items)}] {status} Expected: {correct_answer}, Got: {extracted}")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"{'='*50}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Test SRL/RLVR trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--base-model", type=str, default="unsloth/qwen2.5-3b-instruct-bnb-4bit", help="Base model name")
    parser.add_argument("--data", type=str, 
                        default="../logical_reasoning/data/curated/logical-reasoning-2017-12-02_qa_pairs_cleaned.json",
                        help="Path to evaluation data")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples to evaluate")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.base_model)
    
    evaluate_on_dataset(model, tokenizer, args.data, args.max_samples)


if __name__ == "__main__":
    main()
