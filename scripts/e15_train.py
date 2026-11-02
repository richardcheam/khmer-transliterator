#!/usr/bin/env python3
"""E15 (romanised Khmer): E14 without the rule-romanised dictionary rows; everything else as E14.

Experiment `experiments/E15-real-typing-mix/` (PLAN.md with the registered rule, config.yaml). Same commands as
`scripts/e14_train.py` (`data`, `train --seed N`, `export --seed N`, `score`), writing to `models/e15/` and
`experiments/E15-real-typing-mix/results.json`. The report also scores E14's three seeds on the owner test, so the
effect of dropping the rule rows is measured on the same words.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_train as e  # noqa: E402

e.NAME = "e15"
e.DROP = {"roman"}
e.D = e.ROOT / "models" / "e15"
e.EXPERIMENT = e.ROOT / "experiments" / "E15-real-typing-mix"
e.CONFIG = e.EXPERIMENT / "config.yaml"
e.REPORT = e.EXPERIMENT / "results.json"
e.CONFIG_KEYS = {**e.CONFIG_KEYS, "paths": {"runs_dir": "models/e15"}}
e.REFS = {"e01_aux": e.ROOT / "models" / "unified", "e14_aux": e.ROOT / "models" / "e14"}

if __name__ == "__main__":
    sys.exit(e.main(__doc__))
