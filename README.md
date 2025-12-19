# Supervised-Reinforcement-Learning

Implementation of Supervised Reinforcement Learning (SRL), a fine-tuning method published by Google.

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

## Features

| Feature | Description |
|---------|-------------|
| **vLLM Sleep Mode** | Time-division multiplexing for 8GB VRAM training |
| **LoRA + 4-bit** | Memory-efficient fine-tuning via Unsloth |
| **TensorBoard** | Real-time training metrics and resource monitoring |

## Project Structure

```
Supervised-Reinforcement-Learning/
├── README.md                     # This file
├── requirements.txt              # Base dependencies
├── config.yaml                   # Model and training configuration
├── sdk_config.yaml               # Configuration for Meta's Synthetic-Data-Kit
├── setup_50Series.sh             # Setup script for Nvidia 50 series GPU
│
├── srl/                          # Core SRL library
│   ├── train_srl.py              # Main training script (TRL + Unsloth + vLLM)
│   ├── srl_reward_function.py    # Sequence similarity reward and dynamic sampling filter
│   ├── srl_data_loader.py        # Dataset and DataLoader for SRL training
│   ├── sdk_to_srl.py             # Convert chain-of-thought data to SRL step-pairs
│   ├── srl_training_sdk.py       # Alternative SRL trainer using custom training loop
│   ├── resource_monitor.py       # CPU/GPU/RAM/VRAM monitoring callback for TensorBoard
│   │
│   └── data/                     # Training data directory
│       └── srl_train.jsonl       # Training data in JSONL format
│
└── logical_reasoning/            # Data generation for logical reasoning tasks using Meta's Synthetic-Data-Kit
```

Training data is generated using [Meta's Synthetic-Data-Kit](https://github.com/meta-llama/synthetic-data-kit) with the `sdk_config.yaml` configuration file. The `sdk_to_srl.py` script then converts the generated chain-of-thought solutions into step-wise training pairs.

### Key Files

| File | Description |
|------|-------------|
| `train_srl.py` | Main entry point. Uses TRL's `GRPOTrainer` with Unsloth and vLLM. |
| `srl_reward_function.py` | Computes sequence similarity reward (`R = 2M / T`) and implements the dynamic sampling filter from the SRL paper. |
| `srl_data_loader.py` | Loads JSONL data and prepares it for training. |
| `sdk_to_srl.py` | Converts full chain-of-thought solutions into (prompt, next_step) pairs for SRL training. |
| `resource_monitor.py` | A `TrainerCallback` that logs system resource usage to TensorBoard. |

## Quick Start

### Installation

**For most systems:**
```bash
# Clone the repository
git clone https://github.com/s23deepak/Supervised-Reinforcement-Learning
cd Supervised-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt
```

**For Nvidia 50 Series GPUs (RTX 5060, 5070, 5080, 5090):**

These GPUs require nightly builds of PyTorch and vLLM for CUDA 12.8 support.

```bash
# Install PyTorch nightly with CUDA 12.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --prerelease=allow

# Install vLLM nightly
uv pip install vllm --torch-backend=auto

# Install Unsloth and other dependencies
uv pip install unsloth unsloth_zoo bitsandbytes

# Upgrade transformers
uv pip install -U transformers
```

### Training

The main training script is located in `srl/train_srl.py`.

```bash
# Train with 3B model (8GB VRAM friendly)
python srl/train_srl.py --small-model --epochs 1

# Train with 7B model (uses vLLM sleep mode)
python srl/train_srl.py --epochs 1
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir ./checkpoints_trl_vllm/logs
```
## Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--small-model` | Use 3B model instead of 7B | False |
| `--epochs` | Number of training epochs | 1 |
| `--num-rollouts` | Rollouts per prompt (K) | 4 |
| `--train-data` | Path to training JSONL | `./data/srl_train.jsonl` |
| `--output-dir` | Checkpoint directory | `./checkpoints_trl_vllm` |
| `--no-vllm` | Disable vLLM (fallback to HF generate) | False |
| `--no-instruction` | Disable step instruction | False |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB | 8GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 20GB | 50GB+ |

Tested on: NVIDIA RTX 5060 (8GB)

---

## Data Format

SRL training data is in JSONL format:

```json
{
  "input_prompt": "Question: Who is Jack's aunt?\n\nStep 1: Identify family members...\nStep 2: John is Jack's father...",
  "expert_action": "Step 3: Sarah is John's sister, making her Jack's aunt.",
  "topic": "blood_relation",
  "step_number": 2,
  "total_steps": 4
}
```

Use `sdk_to_srl.py` to convert chain-of-thought data to this format.

## References

- [SRL Paper](https://arxiv.org/abs/2407.18248) - Google DeepMind
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Unsloth AI](https://github.com/unslothai/unsloth)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/design/v1/sleep.html)

## License

MIT License - see [LICENSE](LICENSE) for details.
