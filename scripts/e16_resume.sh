#!/usr/bin/env bash
# E16: the free Gemini key allows only about 20 calls a day. Retry the (resuming) fetch every hour until all 40
# replies are in, or until at least 36 are in and the day's quota is used up; then run the registered pipeline.
set -uo pipefail
cd "$(dirname "$0")/.."
count() { ls data/e16/gemini/reply_*.txt 2>/dev/null | wc -l; }
while :; do
  bash -ic 'export GEMINI_API_KEY; .venv/bin/python scripts/e16.py fetch' >> data/e16/gemini/fetch.out 2>&1
  n=$(count); echo "$(date -u +%FT%TZ) replies: $n" >> models/e16/resume.log
  if [ "$n" -ge 40 ] || [ "$n" -ge 36 ]; then break; fi
  sleep 3600
done
exec scripts/e16_run.sh > models/e16/run.out 2>&1
