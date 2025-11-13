#!/usr/bin/env python3
"""
SRL Data Loader for Unsloth

Classes:
    - SRLDataset: Dataset for step-wise SRL training (works with SDK-generated data)
    - SRLDataCollator: Collate SRL training examples into batches

Functions:
    - create_srl_dataloader(data_path: str, tokenizer, batch_size: int, max_length: int, shuffle: bool, num_workers: int) -> DataLoader

Methods in SRLDataset:
    - __init__(data_path: str, tokenizer, max_length: int, include_thinking: bool)
    - __len__() -> int
    - __getitem__(idx: int) -> Dict

Methods in SRLDataCollator:
    - __call__(batch: List[Dict]) -> Dict[str, torch.Tensor]
    - _pad_sequence(sequences: List[torch.Tensor], pad_token_id: int) -> torch.Tensor
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass


class SRLDataset(Dataset):
    """Dataset for step-wise SRL training for data generated using SDK"""
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 512,
                 include_thinking: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_thinking = include_thinking
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Prepare input with thinking prompt
        if self.include_thinking:
            prompt = f"{item['input_prompt']}\n\n<think>"
        else:
            prompt = item['input_prompt']

        expert_action = item['expert_action']

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, max_length=self.max_length, 
                                         truncation=True, return_tensors=None)
        action_ids = self.tokenizer.encode(expert_action, add_special_tokens=False,
                                          return_tensors=None)

        return {
            "input_ids": input_ids,
            "action_ids": action_ids,
            "expert_action": expert_action,
            "prompt": prompt,
            "topic": item.get('topic', 'qa'),
            "step_number": item.get('step_number', 1),
        }


@dataclass
class SRLDataCollator:
    """Collate SRL training examples into batches"""

    tokenizer: object
    max_length: int = 512

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch"""
        input_ids_list = [torch.tensor(ex["input_ids"]) for ex in batch]
        input_ids = self._pad_sequence(input_ids_list, self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        expert_actions = [ex["expert_action"] for ex in batch]
        prompts = [ex["prompt"] for ex in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "expert_actions": expert_actions,
            "prompts": prompts,
            "batch_size": len(batch),
        }

    def _pad_sequence(self, sequences: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padding = torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)
            padded.append(torch.cat([seq, padding]))
        return torch.stack(padded)


def create_srl_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for SRL training"""
    dataset = SRLDataset(data_path, tokenizer, max_length=max_length)
    collator = SRLDataCollator(tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )
