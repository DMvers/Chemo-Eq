#!/usr/bin/env bash
# Serve Qwen3.8-27B with vLLM on a Unix socket, ~/.chemoeq/vllm.sock (no network port), with the settings used for
# the paper. Started in the background by scripts/prelaunch.sh; run it directly to see it in the foreground.
# One A100 80 GB: the bf16 weights take about 54 GB, the rest is KV cache.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm}"
SOCK_DIR="$HOME/.chemoeq"; SOCK="$SOCK_DIR/vllm.sock"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="${HF_HOME:-$HOME/models}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# vLLM 0.31's fused sampler needs nvcc at start-up; seeded requests never use it, so the PyTorch sampler is used
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
mkdir -p "$SOCK_DIR" && chmod 700 "$SOCK_DIR"; rm -f "$SOCK"
exec "$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --uds "$SOCK" \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 32
