#!/bin/bash
# Start vLLM server with Sleep Mode + LMCache for RLHF training
#
# Sleep mode enables GPU memory to be freed during training steps,
# allowing single-GPU RLHF with coordinated inference/training.
#
# Usage:
#   ./start_vllm_server.sh [MODEL] [PORT] [GPU_MEMORY]
#
# Examples:
#   ./start_vllm_server.sh                                    # Default: Qwen2.5-3B, port 8000
#   ./start_vllm_server.sh meta-llama/Llama-3.2-3B 8001 0.7   # Custom model, port, memory

set -e

# Configuration
MODEL="${1:-unsloth/Qwen2.5-3B-Instruct}"
PORT="${2:-8000}"
GPU_MEMORY="${3:-0.6}"
CACHE_DIR="${LMCACHE_DIR:-/tmp/lmcache_srl}"
CACHE_SIZE="${LMCACHE_SIZE:-10GB}"

echo "============================================================"
echo "Starting vLLM Server with Sleep Mode + LMCache"
echo "============================================================"
echo "Model:          $MODEL"
echo "Port:           $PORT"
echo "GPU Memory:     $GPU_MEMORY"
echo "Cache Dir:      $CACHE_DIR"
echo "Cache Size:     $CACHE_SIZE"
echo "Sleep Mode:     ENABLED"
echo "============================================================"

# Create cache directory
mkdir -p "$CACHE_DIR"

# LMCache configuration via environment variables (for v0.11+)
export LMCACHE_USE_EXPERIMENTAL="True"
export LMCACHE_LOCAL_CPU="True"
export LMCACHE_MAX_LOCAL_CPU_SIZE="5.0"
export LMCACHE_LOCAL_DISK="file://${CACHE_DIR}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="$CACHE_SIZE"
export LMCACHE_CHUNK_SIZE="256"

# Enable dev mode for sleep endpoints
export VLLM_SERVER_DEV_MODE=1

# KV transfer config for LMCache V1 connector (vLLM 0.11+)
KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role": "kv_both"
}'

# Start vLLM with Sleep Mode + LMCache
echo ""
echo "[Starting vLLM server with Sleep Mode enabled...]"
echo ""
echo "Sleep mode endpoints available:"
echo "  POST /sleep?level=1|2  - Put model to sleep"
echo "  POST /wake_up          - Wake model up"
echo "  GET  /is_sleeping      - Check sleep status"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enable-sleep-mode \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --trust-remote-code \
    --dtype auto \
    2>&1 | tee vllm_server.log
