#!/usr/bin/env bash
# Start the generator on one GPU of wolf (L40S, 48 GB): a 7B model fits in bf16 without quantisation.
# Install vLLM in its OWN virtualenv; it pins its own torch version.
#   python -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate && pip install vllm
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
