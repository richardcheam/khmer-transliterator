#!/usr/bin/env python3
"""Mirror a unified-transliterator run's per-epoch log to Weights & Biases (D173), without touching training.

`scripts/translit_unified.py train` prints one JSON line per epoch to `<run>/stdout.log`. This follower reads new
lines and logs them to W&B. The log file stays the source of truth: if W&B or the network fails, nothing is lost,
and a restarted follower resumes from `<run>/wandb_offset.json` without logging a line twice. It stops after the
run's final `exit` line.

Run from the repository root with the project interpreter (it has `wandb` and the laptop's W&B login):

    .venv/bin/python scripts/track_translit.py models/unified/aux_s0

W&B: entity `richardcheam912-ensiie` (the `WANDB_ENTITY` env var overrides it), project `khmer-pocket-tts`,
group `translit_unified_d173`, run name = folder name, run ID stored in `<run>/wandb_run_id.txt`.
Logged keys per epoch: `spell_loss`, `aux_loss`, `val/romanised`, `val/names`, `val/english`, `val_mean`,
`minutes`.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITY, PROJECT, GROUP = "richardcheam912-ensiie", "khmer-pocket-tts", "translit_unified_d173"


def epoch_rows(lines: list[str]) -> list[dict]:
    """The per-epoch JSON records among log lines (other lines are ignored).

    Examples:
        >>> epoch_rows(['{"epoch": 1, "val_mean": 0.2}', 'Loading', 'exit 0'])
        [{'epoch': 1, 'val_mean': 0.2}]
    """
    out = []
    for line in lines:
        if line.startswith("{") and '"epoch"' in line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def follow(run: Path, poll: int = 60) -> int:
    """Log new epochs of one run to W&B until the run's `exit` line appears.

    Args:
        run: The run folder (holds `stdout.log`).
        poll: Seconds between checks.

    Returns:
        Number of epochs logged in this call.
    """
    import wandb
    id_file, off_file = run / "wandb_run_id.txt", run / "wandb_offset.json"
    if not id_file.exists():
        id_file.write_text(f"{run.name}-{uuid.uuid4().hex[:8]}\n")
    arm, seed = run.name.split("_s")
    manifest = ROOT / "models/unified/manifest.json"
    wb = wandb.init(entity=os.environ.get("WANDB_ENTITY", ENTITY), project=PROJECT, group=GROUP, name=run.name,
                    id=id_file.read_text().strip(), resume="allow",
                    config={"arm": arm, "seed": int(seed), "decision": "D173", "aux_weight": 0.5, "aux_every": 8,
                            "aux_batch": 128, "epochs": 40, "batch": 128, "lr": 1e-3,
                            "manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()})
    try:   # an empty or damaged offset (e.g. a full disk) restarts from 0; W&B ignores steps it already has
        done = json.loads(off_file.read_text())["epochs"] if off_file.exists() else 0
    except (ValueError, KeyError):
        done = 0
    logged = 0
    while True:
        log = run / "stdout.log"
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines() if log.exists() else []
        rows = epoch_rows(lines)
        for r in rows[done:]:
            wb.log({"spell_loss": r["spell_loss"], "aux_loss": r.get("aux_loss"), "val_mean": r["val_mean"],
                    "minutes": r["minutes"], **{f"val/{k}": v for k, v in r["val"].items()}}, step=r["epoch"])
            done += 1
            logged += 1
            off_file.write_text(json.dumps({"epochs": done}) + "\n")
        if any(l.startswith("exit") for l in lines):
            wb.finish()
            return logged
        time.sleep(poll)


def main() -> int:
    """CLI: follow the run folder given as the only argument.

    Returns:
        0.
    """
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    print(follow(Path(sys.argv[1])), "epochs logged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
