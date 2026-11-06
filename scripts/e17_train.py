#!/usr/bin/env python3
"""E17 (romanised Khmer): E15 plus Gemini and Claude spellings for as many common words as possible, filtered automatically.

Experiment `experiments/E17-llm-max/` (PLAN.md with the registered rule, config.yaml). Same commands as
`scripts/e14_train.py` (`data`, `train --seed N`, `export --seed N`, `score`), writing to `models/e17/` and
`experiments/E17-llm-max/results.json`. The LLM rows come from `data/e17/llm_filtered.tsv`
(`scripts/e17.py filter`), counted once (not repeated). The report also scores E14, E15 and E16 on the owner test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_train as e  # noqa: E402

LLM_ROWS = e.ROOT / "data" / "e17" / "llm_filtered.tsv"

e.NAME = "e17"
e.DROP = {"roman"}
e.EXTRA = {"llm_bulk": (LLM_ROWS, lambda: e.tu._tsv(LLM_ROWS))}
e.D = e.ROOT / "models" / "e17"
e.EXPERIMENT = e.ROOT / "experiments" / "E17-llm-max"
e.CONFIG = e.EXPERIMENT / "config.yaml"
e.REPORT = e.EXPERIMENT / "results.json"
e.CONFIG_KEYS = {**e.CONFIG_KEYS, "paths": {"runs_dir": "models/e17"}}
e.REFS = {"e01_aux": e.ROOT / "models" / "unified", "e14_aux": e.ROOT / "models" / "e14",
          "e15_aux": e.ROOT / "models" / "e15", "e16_aux": e.ROOT / "models" / "e16"}

if __name__ == "__main__":
    sys.exit(e.main(__doc__))
