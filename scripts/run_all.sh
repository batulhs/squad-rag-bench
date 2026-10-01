#!/usr/bin/env bash
# Full pipeline on a Linux GPU server with vLLM. Run serve_llm.sh first (in tmux) on GPU 0;
# this uses GPU 1 for training, embeddings and the reranker so nothing competes for memory.
# On Kaggle use kaggle_run.ipynb instead (see RUN.md).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}"
CONFIG="${CONFIG:-config.yaml}"

python -m pytest -q
[ -f models/bge-small-squad/training_summary.json ] || python -m ragbench.train --config "$CONFIG"

until curl -sf http://localhost:8000/v1/models > /dev/null; do
  echo "waiting for vLLM on :8000 ..."; sleep 10
done

python -m ragbench.evaluate all --config "$CONFIG"
python -m ragbench.analyze --config "$CONFIG"
python -m ragbench.demo --config "$CONFIG" | tee results/demo_output.txt
