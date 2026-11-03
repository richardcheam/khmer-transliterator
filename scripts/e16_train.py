#!/usr/bin/env python3
"""E16 (romanised Khmer): E15 plus Gemini spellings for 10,000 common words, kept by the automatic filter only.

Experiment `experiments/E16-gemini-bulk/` (PLAN.md with the registered rule, config.yaml). Same commands as
`scripts/e14_train.py` (`data`, `train --seed N`, `export --seed N`, `score`), writing to `models/e16/` and
`experiments/E16-gemini-bulk/results.json`. The Gemini rows come from `data/e16/gemini_filtered.tsv`
(`scripts/e16.py filter`), counted once (not repeated). The report also scores E14 and E15 on the owner test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_train as e  # noqa: E402

GEMINI_ROWS = e.ROOT / "data" / "e16" / "gemini_filtered.tsv"

e.NAME = "e16"
e.DROP = {"roman"}
e.EXTRA = {"gemini_bulk": (GEMINI_ROWS, lambda: e.tu._tsv(GEMINI_ROWS))}
e.D = e.ROOT / "models" / "e16"
e.EXPERIMENT = e.ROOT / "experiments" / "E16-gemini-bulk"
e.CONFIG = e.EXPERIMENT / "config.yaml"
e.REPORT = e.EXPERIMENT / "results.json"
e.CONFIG_KEYS = {**e.CONFIG_KEYS, "paths": {"runs_dir": "models/e16"}}
e.REFS = {"e01_aux": e.ROOT / "models" / "unified", "e14_aux": e.ROOT / "models" / "e14",
          "e15_aux": e.ROOT / "models" / "e15"}

if __name__ == "__main__":
    sys.exit(e.main(__doc__))
