#!/usr/bin/env bash
# E15: train and export seeds 0, 1, 2 one after another, then score (registered rule in experiments/E15-real-typing-mix/PLAN.md).
# Logs: models/e15/train_s<seed>.log and models/e15/score.log. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
for s in 0 1 2; do
  .venv/bin/python scripts/e15_train.py train --seed "$s" > "models/e15/train_s$s.log" 2>&1
  .venv/bin/python scripts/e15_train.py export --seed "$s" >> "models/e15/train_s$s.log" 2>&1
done
.venv/bin/python scripts/e15_train.py score > models/e15/score.log 2>&1
echo "E15 done: experiments/E15-real-typing-mix/results.json"
