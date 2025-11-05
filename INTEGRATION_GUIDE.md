# SRL + Synthetic Data Kit Integration Guide

## Overview

This integration uses Meta's **Synthetic Data Kit** to generate seating arrangement and blood relation QA pairs, then converts them into SRL step-wise training data compatible with Unsloth.

## Quick Start (5 steps)

### 1. Install Dependencies
```bash
pip install synthetic-data-kit unsloth transformers torch
```

### 2. Start vLLM Backend
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
```

Or use any other LLM API endpoint (configure in `sdk_srl.yaml`).

### 3. Run the Full Pipeline
```bash
bash run_pipeline.sh
```

This automatically:
- Prepares seed files
- Generates CoT reasoning traces via SDK
- Curates with quality filtering
- Converts to SRL format
- Trains with Unsloth

### 4. (Optional) Run Steps Individually

If you want fine-grained control:

```bash
# Setup
mkdir -p data/{input,parsed,generated,curated,final}

# Copy seeds
cp seed_seating.txt seed_blood.txt data/input/

# Verify backend
synthetic-data-kit system-check

# Ingest seeds
synthetic-data-kit ingest data/input/seed_seating.txt -c sdk_srl.yaml
synthetic-data-kit ingest data/input/seed_blood.txt -c sdk_srl.yaml

# Generate (adjust -n for number of pairs per chunk)
synthetic-data-kit -c sdk_srl.yaml create data/parsed/ --type cot -n 200

# Curate (adjust --threshold for quality filtering)
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.0

# Convert to SRL
python sdk_to_srl.py --input data/curated \
                     --output-train srl_train_qa.jsonl \
                     --output-val srl_val_qa.jsonl

# Train
python srl_training_sdk.py --train-file srl_train_qa.jsonl \
                           --val-file srl_val_qa.jsonl \
                           --batch-size 4 \
                           --epochs 2
```

## File Structure

```
.
├── sdk_srl.yaml                # SDK config (LLM, generation params, prompts)
├── seed_seating.txt            # Seed for seating puzzles
├── seed_blood.txt              # Seed for blood relation puzzles
├── sdk_to_srl.py               # Converter (SDK JSON → SRL JSONL)
├── srl_reward_function.py      # Sequence similarity reward
├── srl_data_loader.py          # PyTorch DataLoader for SRL
├── srl_training_sdk.py         # Training script
├── run_pipeline.sh             # End-to-end automation script
├── INTEGRATION_GUIDE.md        # This file
└── data/
    ├── input/                  # Seed files
    ├── parsed/                 # SDK parsed text
    ├── generated/              # SDK generated CoT
    ├── curated/                # SDK curated + filtered
    └── final/                  # Final training format
```

## Configuration

### sdk_srl.yaml

Key settings:

```yaml
generation:
  num_pairs: 50          # QA pairs to generate per chunk
  chunk_size: 4000       # Characters per chunk
  temperature: 0.7       # Generation creativity

curate:
  threshold: 8.0         # Quality score (0-10), higher = stricter

prompts:
  cot_generation: |      # Must emit "N. Title: content" steps
    ...
    "cot": "1. ...
2. ...
Answer: ..."
```

### Training Parameters

Edit `srl_training_sdk.py` or use CLI args:

```bash
python srl_training_sdk.py \
  --model meta-llama/Llama-2-7b-hf \
  --batch-size 8 \
  --epochs 3 \
  --max-length 1024
```

## Data Flow

```
Seed Files (seating, blood relation)
         ↓
Synthetic Data Kit - Ingest
         ↓
Parsed Text
         ↓
Synthetic Data Kit - Create (CoT)
         ↓
Generated CoT Traces
         ↓
Synthetic Data Kit - Curate (Llama Judge)
         ↓
Curated JSON
         ↓
sdk_to_srl.py Converter
         ↓
SRL Step-wise JSONL
├── srl_train_qa.jsonl (90%)
└── srl_val_qa.jsonl (10%)
         ↓
SRL Training (Unsloth + GRPO)
         ↓
Trained Model
```

## Converter Details

The `sdk_to_srl.py` script:
1. Loads SDK curated JSON files
2. Splits CoT into numbered steps (e.g., "1. Title: content", "2. Title: content")
3. Extracts final "Answer: ..." line
4. Creates SRL pairs: (context + steps[:i], steps[i+1])
5. Also creates (context + all steps, Answer line) pair
6. Shuffles and splits 90/10 train/val

Example conversion:

```
Input (SDK):
{
  "question": "Five people sit in a line. A sits left, B sits right. ...",
  "cot": "1. Identify ends: A at left, B at right
2. Place middle: ...
Answer: C",
  "answer": "C"
}

Output (SRL):
Pair 1:
  input_prompt: "Question: Five people sit in a line. A sits left, B sits right. ...

1. Identify ends: A at left, B at right"
  expert_action: "2. Place middle: ..."

Pair 2:
  input_prompt: "Question: ...

1. Identify ends: ...
2. Place middle: ..."
  expert_action: "Answer: C"
```

## Reward Function

The SRL reward uses **sequence similarity** (difflib-based):

```
R = 2M / T

Where:
  M = total matched subsequence length
  T = total length of both sequences combined

Returns: [0, 1] for valid format, -1 for format errors
```

Ensures model learns to emit steps matching expert structure.

## Troubleshooting

### vLLM Connection Failed
- Verify vLLM is running: `curl http://localhost:8000/v1/models`
- Check port: default 8000, configurable in `sdk_srl.yaml`
- Restart if needed: `pkill vllm`

### SDK Ingestion Errors
- Ensure seed files are in `data/input/`
- Check seed file format (plain text, UTF-8)
- Use `--preview` flag to test: `synthetic-data-kit ingest data/input/ --preview`

### Low Quality CoT
- Increase `curate.threshold` (8.0 → 8.5) in `sdk_srl.yaml`
- Use a larger/better model in vLLM
- Increase `generation.temperature` for diversity (0.7 → 0.9)

### Memory Issues
- Reduce `batch_size` in training script (4 → 2)
- Reduce `max_length` (512 → 256)
- Use smaller model or quantization

### Missing Pairs After Conversion
- Check curated JSON format in `data/curated/`
- Verify CoT contains numbered steps matching pattern: `^\d+\. .+:.+`
- Use `--verbose` in converter: `python sdk_to_srl.py --input ... --verbose`

## Scaling Tips

To generate **1000s of examples**:

1. Create more seed files covering diverse puzzle types
2. Increase `generation.num_pairs` in `sdk_srl.yaml` (50 → 100)
3. Use directory batch processing: `synthetic-data-kit create data/parsed/ --type cot`
4. Adjust curation threshold based on quality vs quantity tradeoff
5. Run converter on large curated set: `python sdk_to_srl.py --input data/curated`

## Advanced: Custom Topics

To add a new reasoning domain (e.g., logic puzzles):

1. Create `seed_logic.txt` with domain context
2. Update `sdk_srl.yaml` prompts section with domain-specific template
3. Run: `synthetic-data-kit ingest data/input/seed_logic.txt`
4. Generate and curate as usual
5. Converter auto-detects topic from question content

## Next Steps

- **Integrate with LangGraph**: Use trained SRL model in multi-step agent
- **Benchmark**: Evaluate on held-out seating/blood-relation test sets
- **Contribute to Unsloth**: Package as official SRL training module
- **Extend domains**: Add logic puzzles, code reasoning, etc.

---

For questions, refer to:
- [Synthetic Data Kit Docs](https://github.com/meta-llama/synthetic-data-kit)
- [Unsloth RL Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
- [SRL Paper](https://arxiv.org/abs/2510.25992)
