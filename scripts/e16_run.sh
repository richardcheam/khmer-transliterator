#!/usr/bin/env bash
# E16 (registered in experiments/E16-gemini-bulk/PLAN.md): wait for the Gemini fetch, filter, build the data,
# train and export seeds 0-2, then score. Logs in models/e16/. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
mkdir -p models/e16
while pgrep -f "scripts/e16.py fetch" > /dev/null; do sleep 60; done
.venv/bin/python scripts/e16.py filter > models/e16/filter.json
n=$(python3 -c "import json; print(len(json.load(open('models/e16/filter.json'))['answered']))")
if [ "$n" -lt 36 ]; then echo "only $n of 40 Gemini replies: stopping (rule: at least 36)"; exit 3; fi
.venv/bin/python scripts/e16_train.py data > models/e16/data.log 2>&1
for s in 0 1 2; do
  .venv/bin/python scripts/e16_train.py train --seed "$s" > "models/e16/train_s$s.log" 2>&1
  .venv/bin/python scripts/e16_train.py export --seed "$s" >> "models/e16/train_s$s.log" 2>&1
done
.venv/bin/python scripts/e16_train.py score > models/e16/score.log 2>&1
echo "E16 done: experiments/E16-gemini-bulk/results.json" | tee models/e16/DONE
