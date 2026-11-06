#!/usr/bin/env bash
# E17 (registered in experiments/E17-llm-max/PLAN.md): collect Gemini, Claude and GPT spellings until done or the deadline,
# filter, build the data, train and export seed 0, then score. Logs in models/e17/. Stops at the first failure.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
mkdir -p models/e17
DEADLINE=$(date -d "2026-11-09 11:00 UTC" +%s)
left() { local d=$1; local n=0; for p in "$d"/prompt_*.md; do r=${p/prompt_/reply_}; [ -f "${r%.md}.txt" ] || n=$((n+1)); done; echo $n; }
gemini_loop() {
  while [ "$(left data/e17/gemini)" -gt 0 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
    bash -ic 'export GEMINI_API_KEY; .venv/bin/python scripts/e17.py gemini' >> models/e17/gemini.out 2>&1
    echo "$(date -u +%FT%TZ) gemini left: $(left data/e17/gemini)" >> models/e17/collect.log
    [ "$(left data/e17/gemini)" -gt 0 ] && sleep 3600
  done
}
cli_loop() {  # claude | gpt
  while [ "$(left data/e17/$1)" -gt 0 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
    .venv/bin/python scripts/e17.py "$1" >> "models/e17/$1.out" 2>&1
    echo "$(date -u +%FT%TZ) $1 left: $(left data/e17/$1)" >> models/e17/collect.log
  done
}
gemini_loop & g=$!
cli_loop claude & c=$!
cli_loop gpt & p=$!
wait $g $c $p
set -e
.venv/bin/python scripts/e17.py filter > models/e17/filter.json
ok=$(python3 -c "import json; j=json.load(open('models/e17/filter.json')); print(int(all(d['answered'] >= 0.95 * (d['answered'] + d['missing']) for d in (j['claude_files'], j['gpt_files']))))")
if [ "$ok" != 1 ]; then echo "fewer than 95% of the Claude or GPT replies: stopping (registered rule)"; exit 3; fi
.venv/bin/python scripts/e17_train.py data > models/e17/data.log 2>&1
for s in 0; do   # owner: one model only (D211)
  .venv/bin/python scripts/e17_train.py train --seed "$s" > "models/e17/train_s$s.log" 2>&1
  .venv/bin/python scripts/e17_train.py export --seed "$s" >> "models/e17/train_s$s.log" 2>&1
done
.venv/bin/python scripts/e17_train.py score > models/e17/score.log 2>&1
echo "E17 done: experiments/E17-llm-max/results.json" | tee models/e17/DONE
