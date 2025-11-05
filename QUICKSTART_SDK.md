# SRL + Synthetic Data Kit - Quick Start

**You have everything to train SRL models on seating arrangement and blood relation QA with 3 commands:**

## 1. Install & Setup (2 minutes)

```bash
# Install all dependencies
pip install -r requirements.txt

# Create data folders
mkdir -p data/{input,parsed,generated,curated,final}

# Start vLLM backend (opens port 8000)
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 &

# Wait 30 seconds for startup, then verify
synthetic-data-kit system-check
```

## 2. Prepare Data (5-15 minutes, depends on model speed)

```bash
# Copy seed files to input directory
cp seed_seating.txt seed_blood.txt data/input/

# Let synthetic-data-kit generate CoT reasoning traces
synthetic-data-kit -c sdk_srl.yaml ingest data/input/
synthetic-data-kit -c sdk_srl.yaml create data/parsed/ --type cot -n 200
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.0

# Convert SDK output to SRL training format
python sdk_to_srl.py --input data/curated

# You now have:
#  - srl_train_qa.jsonl (training pairs)
#  - srl_val_qa.jsonl (validation pairs)
```

## 3. Train (depends on model and GPU)

```bash
python srl_training_sdk.py \
  --train-file srl_train_qa.jsonl \
  --val-file srl_val_qa.jsonl \
  --batch-size 4 \
  --epochs 2

# Model saved to: srl_sdk_model/
```

## Or All-In-One (Recommended)

```bash
bash run_pipeline.sh
```

This does everything above automatically.

---

## Files Overview

| File | Purpose |
|------|---------|
| `sdk_srl.yaml` | SDK config (LLM, generation, curation) |
| `seed_seating.txt` | Seating puzzle domain description |
| `seed_blood.txt` | Blood relation puzzle domain description |
| `sdk_to_srl.py` | Converter: SDK JSON → SRL step-wise JSONL |
| `srl_reward_function.py` | Sequence similarity reward (difflib-based) |
| `srl_data_loader.py` | PyTorch DataLoader for SRL |
| `srl_training_sdk.py` | Training script with Unsloth |
| `run_pipeline.sh` | Bash script: automates full pipeline |

---

## Expected Output

After running the pipeline:

```
[1/7] Setting up directories...
[2/7] Preparing seed files...
[3/7] Ingesting seed files with SDK...
[4/7] Generating CoT reasoning traces...
  - Generated 400+ seating puzzles with step-wise reasoning
  - Generated 400+ blood relation puzzles with step-wise reasoning
[5/7] Curating generated data...
  - Filtered to high-quality examples (score ≥ 8.0)
[6/7] Converting SDK output to SRL step-wise pairs...
  Wrote 3500+ training pairs → srl_train_qa.jsonl
  Wrote 500+ validation pairs → srl_val_qa.jsonl
[7/7] Training SRL model...
  Epoch 0, Batch 1: Avg Reward = 0.72
  Epoch 0, Batch 2: Avg Reward = 0.68
  ...

Training completed!
Model saved to: srl_sdk_model/
```

---

## Customization

### Generate more data
```bash
# Edit sdk_srl.yaml, increase num_pairs (50 → 100)
# Re-run generation
synthetic-data-kit -c sdk_srl.yaml create data/parsed/ --type cot -n 500
```

### Use different model
```bash
# Edit sdk_srl.yaml, change model line:
# model: "meta-llama/Llama-3.1-8B-Instruct"
# Restart vLLM with new model
```

### Stricter/looser quality filter
```bash
# Strict: curate --threshold 9.0
# Loose:  curate --threshold 7.0
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.5
```

---

## Key Concepts

**SRL (Supervised Reinforcement Learning)**
- Trains models on step-wise reasoning traces
- Dense reward at each step (not just final answer)
- Reward = sequence similarity between generated and expert step

**Synthetic Data Kit**
- Generates diverse reasoning problems from seed descriptions
- Uses LLM (vLLM) to create step-by-step solutions
- Curates with "Llama as Judge" for quality filtering

**Integration Flow**
```
Seeds → SDK Generate → Curate → Convert to SRL → Train with Unsloth
```

---

## Troubleshooting

**vLLM won't start:**
```bash
# Check if port 8000 is free
lsof -i :8000
# Kill if needed: pkill vllm
# Or use different port: vllm serve ... --port 8001
# Update sdk_srl.yaml: api_base: "http://localhost:8001/v1"
```

**SDK system-check fails:**
```bash
# Verify vLLM is running
curl http://localhost:8000/v1/models
# Should return list of models
```

**Low quality data:**
```bash
# Increase curation threshold
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.5

# Or use larger/better model
# Edit sdk_srl.yaml and restart vLLM
```

**Memory errors during training:**
```bash
# Reduce batch size
python srl_training_sdk.py --batch-size 2 ...

# Or reduce sequence length
python srl_training_sdk.py --max-length 256 ...
```

---

## Next Steps

1. **Explore data**: `head -20 srl_train_qa.jsonl | python -m json.tool`
2. **Evaluate**: Add test set and measure accuracy
3. **Scale**: Generate 10k+ examples by increasing `num_pairs`
4. **Extend**: Add custom reasoning domains (logic puzzles, code, etc.)
5. **Deploy**: Use trained model with VLLM or HuggingFace

---

**Questions?** See `INTEGRATION_GUIDE.md` for detailed documentation.
