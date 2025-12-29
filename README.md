# Supervised Reinforcement Learning (SRL)

Implementation of Supervised Reinforcement Learning (SRL) + RLVR for step-by-step reasoning using TRL's GRPOTrainer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s23deepak/Supervised-Reinforcement-Learning/blob/main/notebooks/srl_grpo_tutorial.ipynb)

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

- **TRL GRPOTrainer**: GRPO implementation from Hugging Face
- **Unsloth**: Memory-efficient LoRA and 4-bit quantization
- **vLLM**: Fast inference with prefix caching
- **LMCache**: Cross-batch KV cache persistence (CPU/disk)
- **Sleep Mode**: Single-GPU time-division multiplexing

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
├── train_srl.py                  # Stage 1: SRL training
├── train_srl_rlvr.py             # Stage 2: RLVR training
├── sleep_aware_grpo_trainer.py   # GRPO with vLLM sleep coordination
├── vllm_server_client.py         # HTTP client for vLLM sleep/wake
├── srl_reward_function.py        # Step similarity reward
├── rlvr_reward_function.py       # Final answer correctness
├── unified_logger.py             # Comprehensive logging
├── start_vllm_server.sh          # vLLM server with sleep mode
└── test_model.py                 # Model evaluation
```

## Training Modes

### 1. Embedded Mode (Default - Recommended)
vLLM runs inside training process. Unsloth handles sleep/wake automatically.

```bash
python train_srl.py --train-data ./data.jsonl
```

### 2. Server Mode with Sleep Coordination + LMCache
External vLLM server with coordinated sleep/wake and cross-batch KV caching.

**Why Server Mode?** In SRL, the same question prefix repeats across steps:
```
Batch 1: Q               → compute prefix
Batch 5: Q + S1          → reuse Q prefix (LMCache)
Batch 9: Q + S1 + S2     → reuse Q + S1 prefix
```
LMCache persists KV cache to CPU/disk, enabling reuse across batches (embedded mode only caches within batch).

```bash
# Terminal 1: Start vLLM server with sleep mode + LMCache
./start_vllm_server.sh

# Terminal 2: Train with sleep coordination
python train_srl.py --train-data ./data.jsonl --vllm-server --vllm-sleep-mode
```

**LMCache Options** (set in `start_vllm_server.sh` or env vars):

| Variable | Default | Description |
|----------|---------|-------------|
| `LMCACHE_LOCAL_CPU` | True | Use CPU RAM for cache |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | 5.0 | CPU cache size (GB) |
| `LMCACHE_LOCAL_DISK` | file:///tmp/lmcache_srl | Disk cache path |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | 10GB | Disk cache size |
| `LMCACHE_CHUNK_SIZE` | 256 | Tokens per cache chunk |

### 3. Multi-GPU Mode
Separate GPUs for inference and training (no sleep needed).

```bash
# Terminal 1: vLLM on GPU 0
CUDA_VISIBLE_DEVICES=0 ./start_vllm_server.sh

# Terminal 2: Training on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_srl.py --train-data ./data.jsonl --vllm-server
```

## Training Arguments

### train_srl.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Qwen2.5-3B-Instruct-bnb-4bit | Model name or path |
| `--train-data` | ./srl_datasets/srl_train.jsonl | Training data path |
| `--cache-dir` | None | Cache preprocessed dataset |
| `--vllm-server` | off | Use external vLLM server |
| `--vllm-sleep-mode` | off | Enable sleep mode coordination |
| `--push-to-hub` | off | Push model to HuggingFace |
| `--max-samples` | None | Limit dataset size |
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
| `--train-data` | ./rlvr_datasets/train.jsonl | RLVR training data |
| `--cache-dir` | None | Cache preprocessed dataset |
| `--epochs` | 1 | Training epochs |
| `--max-samples` | None | Limit dataset size |

## Sleep Mode Architecture

`SleepAwareGRPOTrainer` coordinates GPU memory between inference and training:

```
┌────────────────────────────────────────────────────────────┐
│ GRPO Step with Sleep Mode                                  │
├────────────────────────────────────────────────────────────┤
│ 1. wake_up()     → vLLM uses GPU                          │
│ 2. Generate      → K completions per prompt               │
│ 3. sleep()       → Free GPU for training                  │
│ 4. backward()    → Compute gradients                      │
│ 5. optimizer     → Update weights                         │
│ 6. reload()      → vLLM loads new weights                 │
└────────────────────────────────────────────────────────────┘
```

## Logging & Monitoring

Training generates:
- **TensorBoard**: `./checkpoints/logs`
- **CSV files**: `metrics.csv`, `resource_samples.csv`
- **Plots**: `resources.png`, `training_curves.png`

```bash
tensorboard --logdir ./checkpoints_trained_srl/logs
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB | 8GB+ |
| System RAM | 16GB | 32GB+ |

Tested on: NVIDIA RTX 5060 (8GB), Kaggle T4 x2

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
{"question": "Four siblings...", "correct_answer": "A"}
```

## References

- [SRL Paper](https://arxiv.org/abs/2510.25992) - Google
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth AI](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [LMCache](https://lmcache.ai)

## License

MIT License
