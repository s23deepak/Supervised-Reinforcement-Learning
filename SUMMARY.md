# SRL + Synthetic Data Kit - Complete Package

## ALL FILES CREATED - Ready to Use

Total: **13 production-ready files**

---

## Get Started in 3 Commands

```bash
# 1. Setup (automatic)
bash setup.sh
source srl_env/bin/activate

# 2. Start LLM backend (in another terminal)
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000

# 3. Run full pipeline
bash run_pipeline.sh
```

That's it! Model will be trained and saved to `srl_sdk_model/`

---

## Complete File List

### Configuration
| File | Purpose |
|------|---------|
| `sdk_srl.yaml` | Synthetic Data Kit config (LLM, generation, prompts) |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Automated setup script |

### Seed Files (Domain Descriptions)
| File | Purpose |
|------|---------|
| `seed_seating.txt` | Seating arrangement puzzle domain |
| `seed_blood.txt` | Blood relation puzzle domain |

### Core Code (SRL Implementation)
| File | Lines | Purpose |
|------|-------|---------|
| `srl_reward_function.py` | 150 | Sequence similarity rewards (difflib) |
| `srl_data_loader.py` | 120 | PyTorch DataLoader for SRL |
| `sdk_to_srl.py` | 200 | Converter: SDK JSON → SRL JSONL |
| `srl_training_sdk.py` | 220 | Training script with Unsloth |

### Automation
| File | Purpose |
|------|---------|
| `run_pipeline.sh` | End-to-end bash pipeline (all 7 steps) |

### Documentation
| File | Purpose |
|------|---------|
| `QUICKSTART_SDK.md` | Quick start guide (this page!) |
| `README_SDK.md` | Full documentation |
| `INTEGRATION_GUIDE.md` | Detailed integration guide |

---

## What Each File Does

### Configuration & Setup
**sdk_srl.yaml**
- Configures vLLM backend (model, API endpoint)
- Generation parameters (num_pairs, chunk_size, temperature)
- Curation quality threshold
- Custom prompts for CoT generation

**setup.sh**
- Creates Python virtual environment
- Installs all dependencies from requirements.txt
- One-command setup

**requirements.txt**
- torch, transformers, unsloth, peft
- synthetic-data-kit, vllm
- numpy, scipy, pandas, datasets
- All versions pinned for reproducibility

### Domain Seeds
**seed_seating.txt**
- Describes seating arrangement puzzle conventions
- Linear seating, left/right directions, constraints
- Question types: who sits left/right, center, end
- Example puzzle for reference

**seed_blood.txt**
- Describes blood relation domain
- Kinship terms, family tree structure
- Question types: "How is X related to Y?"
- Example with 3 generations

### Core SRL Implementation
**srl_reward_function.py**
```
Main classes:
- SRLRewardFunction: Computes R = 2M / T (sequence similarity)
- DynamicSamplingFilter: Filters low-variance samples

Usage:
  reward_fn = SRLRewardFunction()
  reward = reward_fn(generated_action, expert_action)
  # Returns: float in [-1, 1]
```

**srl_data_loader.py**
```
Main classes:
- SRLDataset: PyTorch Dataset loading from JSONL
- SRLDataCollator: Batch collation with padding
- create_srl_dataloader(): Factory function

Usage:
  loader = create_srl_dataloader(
    "srl_train_qa.jsonl", tokenizer, batch_size=8
  )
```

**sdk_to_srl.py**
```
Converts SDK curated JSON to SRL step-wise pairs:

Input:  {"question": "...", "cot": "1. ...
2. ...
Answer: ...", "answer": "..."}
Output: 
  {"input_prompt": "Question: ...

1. ...", "expert_action": "2. ..."}
  {"input_prompt": "...", "expert_action": "Answer: ..."}

Usage:
  python sdk_to_srl.py --input data/curated --output-train srl_train_qa.jsonl
```

**srl_training_sdk.py**
```
Main class:
- SRLTrainer: Handles model loading, training, generation

Key methods:
- generate_rollouts(): Generate multiple outputs per prompt
- compute_batch_rewards(): Compute rewards for batch
- train_step(): Execute training epoch

Usage:
  python srl_training_sdk.py --train-file srl_train_qa.jsonl --epochs 2
```

### Automation
**run_pipeline.sh**
```
Automated 7-step pipeline:
1. Setup directories
2. Prepare seed files
3. Ingest seeds with SDK
4. Generate CoT traces
5. Curate with Llama-as-Judge
6. Convert to SRL format
7. Train with Unsloth

Single command: bash run_pipeline.sh
```

---

## Data Flow

```
                          ┌─────────────────────┐
                          │  seed_seating.txt   │
                          │  seed_blood.txt     │
                          └──────────┬──────────┘
                                     │
                              [setup.sh]
                                     │
                    ┌────────────────────────────────┐
                    │   data/input/                  │
                    │   (seed files copied)          │
                    └────────────┬───────────────────┘
                                 │
                      [Ingest: sdk_srl.yaml]
                                 │
                    ┌────────────────────────────────┐
                    │   data/parsed/                 │
                    │   (parsed text)                │
                    └────────────┬───────────────────┘
                                 │
                    [Create: cot generation, vLLM]
                                 │
                    ┌────────────────────────────────┐
                    │   data/generated/              │
                    │   (CoT traces)                 │
                    └────────────┬───────────────────┘
                                 │
                    [Curate: Llama-as-Judge]
                                 │
                    ┌────────────────────────────────┐
                    │   data/curated/                │
                    │   (filtered JSON)              │
                    └────────────┬───────────────────┘
                                 │
                      [Convert: sdk_to_srl.py]
                                 │
        ┌────────────────────────┴────────────────────┐
        │                                              │
    ┌───┴──────────────────┐          ┌──────────────┴─────┐
    │ srl_train_qa.jsonl   │          │ srl_val_qa.jsonl   │
    │ (90% of pairs)       │          │ (10% of pairs)     │
    └───┬──────────────────┘          └──────────────┬─────┘
        │                                             │
        └────────────────┬──────────────────────────┘
                         │
                [Train: srl_training_sdk.py]
                (+ Unsloth + GRPO)
                         │
                ┌────────┴────────┐
                │ srl_sdk_model/  │
                │ (trained LoRA)  │
                └─────────────────┘
```

---

## Usage Examples

### Example 1: Run Full Pipeline
```bash
bash run_pipeline.sh
# Automatically does everything in 7 steps
# Result: srl_sdk_model/ with trained weights
```

### Example 2: Step-by-Step (Manual Control)
```bash
# 1. Setup
bash setup.sh
source srl_env/bin/activate

# 2. Start backend
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 &

# 3. Verify backend
synthetic-data-kit system-check

# 4. Generate data (custom parameters)
synthetic-data-kit ingest data/input/seed_seating.txt -c sdk_srl.yaml
synthetic-data-kit -c sdk_srl.yaml create data/parsed/ --type cot -n 300

# 5. Curate (stricter quality)
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.5

# 6. Convert
python sdk_to_srl.py --input data/curated --val-split 0.85

# 7. Train (custom hyperparams)
python srl_training_sdk.py \
  --train-file srl_train_qa.jsonl \
  --val-file srl_val_qa.jsonl \
  --model meta-llama/Llama-2-7b-hf \
  --batch-size 8 \
  --epochs 3 \
  --max-length 1024
```

### Example 3: Use Trained Model
```python
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Load
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="srl_sdk_model",
    max_seq_length=512,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Generate
prompt = "Reasoning Question\nQuestion: Five people sit in a line..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
```

---

## Key Concepts

### Synthetic Data Kit (SDK)
- Expands seed descriptions into diverse problems
- Uses LLM (vLLM) for generation
- Curates with "Llama as Judge" for quality

### SRL (Supervised Reinforcement Learning)
- Trains on step-wise reasoning traces
- Dense reward at each step (not just final answer)
- Reward formula: R = 2M / T (sequence similarity)

### Unsloth Integration
- 4-bit quantization (50-60% less memory)
- PEFT LoRA adapters (efficient fine-tuning)
- 40-80% faster training than standard methods

---

## Scaling Tips

To generate **larger datasets**:

1. **Increase generation**: Edit sdk_srl.yaml, `num_pairs: 50 → 200`
2. **Add more seeds**: Create `seed_logic.txt`, `seed_code.txt`, etc.
3. **Batch process**: Use directory mode in SDK for parallel generation
4. **Fine-tune curation**: Adjust threshold based on quality/quantity tradeoff

Example: 5 seed files × 200 pairs per file → 1000+ training examples

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "vLLM port already in use" | Use different port: `--port 8001` (update sdk_srl.yaml) |
| "SDK system-check fails" | Ensure vLLM is running: `curl http://localhost:8000/v1/models` |
| "Low quality generated data" | Increase curation threshold (8.0 → 8.5) |
| "CUDA out of memory" | Reduce batch_size (8 → 4) or max_length (512 → 256) |
| "Converter produces few pairs" | Check seed files are in data/input/, run generation again |

See `INTEGRATION_GUIDE.md` for detailed troubleshooting.

---

## Documentation Files

1. **QUICKSTART_SDK.md** - START HERE (this file)
   - Quick 3-step setup
   - Expected output
   - Customization tips

2. **README_SDK.md**
   - Full feature overview
   - How each component works
   - Data scale and performance

3. **INTEGRATION_GUIDE.md**
   - Detailed step-by-step instructions
   - Configuration reference
   - Advanced customization
   - Comprehensive troubleshooting

---

## Next Steps

1. **Run pipeline**: `bash run_pipeline.sh`
2. **Explore data**: `head srl_train_qa.jsonl | python -m json.tool`
3. **Benchmark**: Evaluate on held-out test set
4. **Scale**: Increase `num_pairs` to 100-200+
5. **Extend**: Add custom reasoning domains
6. **Deploy**: Export model for production inference

---

## Support

- **Quick questions**: See QUICKSTART_SDK.md
- **Detailed info**: See INTEGRATION_GUIDE.md
- **Architecture**: See README_SDK.md
- **SDK docs**: https://github.com/meta-llama/synthetic-data-kit
- **Unsloth docs**: https://docs.unsloth.ai

---

## Summary

You have everything needed to:
- Generate synthetic QA data with Synthetic Data Kit
- Convert to SRL step-wise format
- Train with Unsloth (fast, memory-efficient)
- Get working reasoning model in 30 minutes

**Start with**: `bash setup.sh && bash run_pipeline.sh`

---

Generated: 2025-11-05
Version: 1.0 (Complete Package)
