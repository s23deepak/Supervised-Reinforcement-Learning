#!/bin/bash
# End-to-end SRL + synthetic-data-kit pipeline

set -e

echo "========================================="
echo "SRL + Synthetic Data Kit Pipeline"
echo "========================================="

# 1. Setup
echo "\n[1/7] Setting up directories..."
mkdir -p data/{input,parsed,generated,curated,final}

# 2. Copy seeds
echo "[2/7] Preparing seed files..."
cp seed_seating.txt data/input/
cp seed_blood.txt data/input/

# 3. Ingest seeds
echo "[3/7] Ingesting seed files with SDK..."
synthetic-data-kit ingest data/input/seed_seating.txt -c sdk_srl.yaml
synthetic-data-kit ingest data/input/seed_blood.txt -c sdk_srl.yaml

# 4. Generate CoT
echo "[4/7] Generating CoT reasoning traces..."
synthetic-data-kit -c sdk_srl.yaml create data/parsed/ --type cot -n 200

# 5. Curate
echo "[5/7] Curating generated data..."
synthetic-data-kit -c sdk_srl.yaml curate data/generated/ --threshold 8.0

# 6. Convert to SRL
echo "[6/7] Converting SDK output to SRL step-wise pairs..."
python sdk_to_srl.py --input data/curated --output-train srl_train_qa.jsonl --output-val srl_val_qa.jsonl

# 7. Train
echo "[7/7] Training SRL model..."
python srl_training_sdk.py --train-file srl_train_qa.jsonl --val-file srl_val_qa.jsonl --batch-size 4 --epochs 2

echo "\n========================================="
echo "Pipeline completed!"
echo "Trained model saved to: srl_sdk_model/"
echo "========================================="
