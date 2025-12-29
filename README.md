# Supervised Reinforcement Learning (SRL)

Implementation of Supervised Reinforcement Learning (SRL) + RLVR for step-by-step reasoning.

## Overview

SRL trains models to generate reasoning steps one at a time, using sequence similarity as reward. RLVR then fine-tunes for correct final answers.

```
Stage 1: SRL → Learn step-by-step reasoning
Stage 2: RLVR → Learn correct final answers
```

## What is SRL?

Supervised Reinforcement Learning (SRL) is a training technique that combines the best of supervised learning and reinforcement learning. Instead of training on complete answers, SRL trains a model **step-by-step**:

1.  The model is given a question and a partial solution (some reasoning steps).
2.  It generates the **next single step** of the solution.
3.  This generated step is compared to an expert's step using **sequence similarity** as a reward signal.
4.  The model is updated using a policy gradient method (GRPO) to maximize this reward.

This approach allows the model to learn the reasoning process itself, not just the final answer, leading to more robust and generalizable reasoning capabilities.

## Tech Stack

This repository implements the SRL pipeline using:
- **TRL GRPOTrainer**: GRPO implementation from Hugging Face.
- **Unsloth**: Memory-efficient LoRA and 4-bit quantization.
- **vLLM Sleep Mode**: Time-division multiplexing to enable training on GPUs with limited VRAM (e.g., 8GB).

## Quick Start

```bash
# Install dependencies
pip install -r srl/requirements_nvidia_50Series.txt

# Stage 1: SRL Training
cd srl
python train_srl.py --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit --max-samples 100

# Stage 2: RLVR Training  
python train_srl_rlvr.py --srl-checkpoint ./checkpoints_trained_srl/final

# Test the model
python test_model.py --model ./checkpoints_trained_srl_rlvr/final
```

## Project Structure

```
srl/
├── train_srl.py              # Stage 1: SRL training
├── train_srl_rlvr.py         # Stage 2: RLVR training
├── srl_reward_function.py    # Step similarity reward (2M/T)
├── rlvr_reward_function.py   # Final answer correctness (0/1)
├── unified_logger.py         # TensorBoard, CSV, matplotlib logging
├── test_model.py             # Model evaluation
├── sdk_to_srl.py             # Data conversion utility
└── data/
    └── srl_train.jsonl       # SRL training data
```

## Training Arguments

### train_srl.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen2.5-3B-Instruct-bnb-4bit | Model name or path |
| `--lora-rank` | 16 | LoRA rank |
| `--batch-size` | 1 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 5e-6 | Learning rate |
| `--epochs` | 1 | Training epochs |
| `--num-rollouts` | 4 | Rollouts per prompt (K) |
| `--max-seq-length` | 2048 | Max input sequence length |
| `--max-completion-length` | 256 | Max generation length |
| `--gpu-memory` | 0.6 | vLLM GPU memory utilization |
| `--no-4bit` | False | Disable 4-bit quantization |

### train_srl_rlvr.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--srl-checkpoint` | (required) | Path to SRL-trained model |
| `--epochs` | 1 | Training epochs |
| `--max-samples` | None | Limit dataset size |

## Logging & Monitoring

Training generates:
- **TensorBoard**: Real-time metrics in `./checkpoints/logs`
- **CSV files**: `metrics.csv`, `resource_samples.csv`, `phase_metrics.csv`
- **Plots**: `resources.png`, `training_curves.png`, `phase_breakdown.png`

```bash
tensorboard --logdir ./checkpoints_trained_srl/logs
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB | 8GB+ |
| System RAM | 16GB | 32GB+ |

Tested on: NVIDIA RTX 5060 (8GB)

## Data Format

### SRL Training Data (JSONL)

```json
{
  "input_prompt": "Question: ...\n\nStep 1: ...\nStep 2: ...",
  "expert_action": "Step 3: ..."
}
```

### RLVR Training Data (JSON)

```json
{
  "qa_pairs": [
    {
      "question": "Four siblings...",
      "choices": ["A) Bob", "B) Dave", "C) Either", "D) Not enough info"],
      "answer": "A"
    }
  ]
}
```

## LMCache Integration (Cross-Batch KV Caching)

For large datasets, use LMCache to persist KV cache to disk for reuse across batches.

**Architecture:**
```
TRL Training  ──HTTP API──>  vLLM Server + LMCache (disk cache)
```

**Usage:**
```bash
# Terminal 1: Start vLLM server with LMCache
cd srl
./start_vllm_server.sh

# Terminal 2: Train with server mode
python train_srl.py --vllm-server --train-data ./srl_datasets/train.jsonl
```

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-server` | off | Use external vLLM server |
| `--vllm-server-url` | `http://localhost:8000/v1` | Server API URL |

**Alternative:** If you prefer config files over env vars:
```bash
vllm serve MODEL --lmcache-config lmcache_config.yaml
```

## References

- [SRL Paper](https://arxiv.org/abs/2510.25992) - Google
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth AI](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [LMCache](https://lmcache.ai)

## License

MIT License
