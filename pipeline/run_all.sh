#!/usr/bin/env bash
# The whole pipeline: prepare data -> train the small model and the ByT5 adapter -> evaluate.
#   pipeline/run_all.sh                 # small model on the CPU (about 25 minutes), adapter on the GPU (about 45)
#   SKIP_BYT5=1 pipeline/run_all.sh     # small model only (no PyTorch GPU or 1.2 GB base model needed)
#   DEVICE=cpu pipeline/run_all.sh      # the adapter on the CPU too (several hours)
# Every step writes into runs/; a stopped run continues where it stopped when started again.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda}
mkdir -p runs

echo "== 1. prepare the training data (and check it equals training-data/)"
python pipeline/prepare.py --out runs/data --check

echo "== 2. small model (seed 0, as released; CPU, exactly repeatable)"
[ -f runs/small_s0/model.npz ] || \
  python training/train_small_model.py --seed 0 --device cpu --data runs/data/small --out runs/small_s0 --resume

if [ -z "${SKIP_BYT5:-}" ]; then
  echo "== 3. ByT5 adapter (seed 0, as released)"
  [ -f runs/byt5_s0/adapter/adapter.safetensors ] || \
    python training/train_byt5_adapter.py --seed 0 --device "$DEVICE" --data runs/data/byt5 --out runs/byt5_s0 --resume
  echo "== 4. evaluate, and compare with the released models"
  python pipeline/evaluate.py --small runs/small_s0/model.npz --byt5 runs/byt5_s0/adapter --out runs/eval.json
else
  echo "== 4. evaluate the small model, and compare with the released one"
  python pipeline/evaluate.py --small runs/small_s0/model.npz --skip-byt5 --out runs/eval.json
fi
