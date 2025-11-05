# SRL (Supervised Reinforcement Learning) + Synthetic Data Kit

**Production-ready implementation of SRL for QA tasks (seating arrangements & blood relations) using Meta's Synthetic Data Kit and Unsloth.**

## What This Does

```
┌─────────────────────────────────────────────────────────────┐
│ Generate Synthetic QA Data (SDK) → SRL Training (Unsloth)   │
├─────────────────────────────────────────────────────────────┤
│ INPUT:  Seed files (domain descriptions)                    │
│ PROCESS: 1. SDK generates CoT traces                        │
│          2. Curate for quality                              │
│          3. Convert to SRL step-wise pairs                  │
│          4. Train with Unsloth + GRPO                       │
│ OUTPUT:  Fine-tuned SRL model                               │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Seating Arrangement Puzzles** - Linear seating logic with constraints
- **Blood Relation Puzzles** - Family kinship and genealogy reasoning
- **Automatic Data Generation** - SDK expands seeds into 1000s of examples
- **Quality Curation** - Llama-as-Judge filters low-quality traces
- **Step-wise SRL Training** - Dense rewards at each reasoning step
- **Unsloth Integration** - Fast, memory-efficient training (4-bit, LoRA)
- **End-to-End Pipeline** - Single bash script from seeds to trained model

## Quick Start

```bash
# 1. Setup (2 min)
pip install -r requirements.txt
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 &

# 2. Generate + Train (10-30 min depending on GPU)
bash run_pipeline.sh

# Trained model: srl_sdk_model/
```

## File Structure

```
├── sdk_srl.yaml                # SDK configuration (LLM, generation, curation)
├── seed_seating.txt            # Domain seed: seating puzzles
├── seed_blood.txt              # Domain seed: blood relations
├── sdk_to_srl.py               # Converter: SDK JSON → SRL JSONL
├── srl_reward_function.py      # Sequence similarity reward
├── srl_data_loader.py          # PyTorch DataLoader
├── srl_training_sdk.py         # Training with Unsloth
├── run_pipeline.sh             # Automated pipeline
├── requirements.txt            # Python dependencies
├── QUICKSTART_SDK.md           # Quick start guide
├── INTEGRATION_GUIDE.md        # Detailed documentation
└── data/                       # Generated data (created by pipeline)
    ├── input/                  # Seed files
    ├── parsed/                 # SDK parsed text
    ├── generated/              # SDK generated CoT
    └── curated/                # Curated & filtered
```

## How It Works

### Step 1: Data Generation (SDK)

The Synthetic Data Kit uses LLM (vLLM) to expand seed descriptions into diverse reasoning problems:

```
Input Seed:
  "Seating Arrangement domain notes:
   - Straight line, everyone faces North
   - Constraints: ends, immediate neighbors, between, center
   - Questions: who sits left/right/center?"

Output:
  {
    "question": "Five people sit in a line facing north. Ava at left, Ella at right, Ben right of Ava, Cara between Ben and Dan. Who sits right of Ben?",
    "cot": "1. Identify ends: Ava=1, Ella=5
2. Place Ben: Ben=2 (right of Ava)
3. Place Cara-Dan: Cara=3, Dan=4
4. Answer: Cara",
    "answer": "Cara"
  }
```

### Step 2: Data Curation (Llama Judge)

Filter for quality (score ≥ 8.0):
- Are questions well-formed?
- Are solutions logically consistent?
- Are steps clearly numbered?

### Step 3: Convert to SRL Format

Split CoT into step-wise training pairs:

```
Input (SDK):
  {
    "question": "...",
    "cot": "1. Title: ...
2. Title: ...
Answer: ...",
    "answer": "..."
  }

Output (SRL):
  Pair 1:
    input_prompt: "Question: ...

1. Title: ..."
    expert_action: "2. Title: ..."

  Pair 2:
    input_prompt: "Question: ...

1. Title: ...
2. Title: ..."
    expert_action: "Answer: ..."
```

### Step 4: Train with Unsloth

Use SRL reward (sequence similarity) + GRPO policy gradient:

```
Reward = 2M / T

Where:
  M = matched subsequence tokens
  T = total tokens (both sequences)

Example:
  Expert: "2. Subtract 5 from both sides: 3x = 15"
  Model:  "2. Subtract from both: 3x = 15"

  Matched: "2. Subtract" + "from both" + ": 3x = 15" = 18 tokens
  Total: 30 + 26 = 56 tokens
  Reward = (2 × 18) / 56 = 0.64
```

## Configuration

### sdk_srl.yaml

Key parameters:

```yaml
generation:
  num_pairs: 50              # Examples per chunk
  chunk_size: 4000           # Characters per chunk
  temperature: 0.7           # Generation diversity

curate:
  threshold: 8.0             # Quality score (0-10)

prompts:
  cot_generation: |          # Must emit "N. Title: ..." steps
    # Provides format instructions to SDK
    # Output includes "Answer: ..." line
```

### Training

```bash
python srl_training_sdk.py \
  --model meta-llama/Llama-2-7b-hf \
  --batch-size 8 \
  --epochs 3 \
  --max-length 512
```

## Data Scale

Starting from **2 seed files** (seating + blood relation):

- **Seeds**: ~500 tokens each
- **Generated**: ~1000+ examples (SDK expands with LLM)
- **Curated**: ~800 high-quality examples
- **SRL Pairs**: ~2000-3000 step-wise training instances

Example: With `num_pairs=50` and 10 chunks per seed → 1000 total pairs.

## Performance

**Expected Results:**

- Seating puzzles: Model learns constraint satisfaction logic
- Blood relations: Model learns kinship definitions and lineage tracing
- Average reward: 0.6-0.8 (after training)
- Training time: ~10-30 min per epoch (depends on GPU and dataset size)

**Memory:**

- 7B model + 4-bit: ~12-14 GB VRAM
- Batch size 4-8: typical for 24GB GPU
- With Unsloth LoRA: 40-80% faster than standard fine-tuning

## Extending to Custom Domains

Add a new reasoning domain:

1. Create seed file (e.g., `seed_logic.txt`)
2. Describe domain conventions and question types
3. Update `sdk_srl.yaml` prompts if needed
4. Run SDK on new seed
5. Convert and train as usual

Example seeds:
- `seed_logic.txt` - Logic puzzles
- `seed_code.txt` - Code reasoning
- `seed_math.txt` - Math word problems

## Troubleshooting

| Issue | Solution |
|-------|----------|
| vLLM won't start | Check port 8000 free, restart vLLM |
| SDK system-check fails | Ensure vLLM is running: `curl http://localhost:8000/v1/models` |
| Low quality data | Increase curation threshold (8.0 → 8.5) or use larger model |
| OOM during training | Reduce batch_size (8 → 4) or max_length (512 → 256) |
| Few training pairs | Increase `num_pairs` in sdk_srl.yaml or add more seeds |

See `INTEGRATION_GUIDE.md` for detailed troubleshooting.

## Next Steps

1. **Run the pipeline**: `bash run_pipeline.sh`
2. **Explore generated data**: `head srl_train_qa.jsonl`
3. **Benchmark**: Create test set, evaluate accuracy
4. **Scale**: Increase `num_pairs` to 100-200 for more data
5. **Customize**: Add domain-specific seeds and prompts
6. **Deploy**: Export model for inference with VLLM or HuggingFace

## References

- [SRL Paper](https://arxiv.org/abs/2510.25992): "From Expert Trajectories to Step-wise Reasoning"
- [Synthetic Data Kit GitHub](https://github.com/meta-llama/synthetic-data-kit)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [GRPO Algorithm](https://arxiv.org/abs/2402.15567): Group Relative Policy Optimization

## License

MIT - See LICENSE file

---

**Questions?** Check `QUICKSTART_SDK.md` for quick start or `INTEGRATION_GUIDE.md` for detailed docs.
