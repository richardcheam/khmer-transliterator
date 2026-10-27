#!/usr/bin/env python3
"""Experiment E03: the durable queue (6 jobs: letters and phones at seeds 0, 1, 2), then the frozen evaluation.

    .venv/bin/python scripts/e03_run.py --config experiments/E03-assessable-phones/config.yaml --resume
    .venv/bin/python scripts/e03_run.py --config experiments/E03-assessable-phones/config.yaml --retry phones_s1

E02's `Runner` does the work (one process at a time, live-job detection, resume as the same attempt, attempt and
hour budgets, registered-hash check); this subclass only fixes the queue, the caps (6 supervised attempts including
failures, 12 aggregate hours, no P2G), the commands (`scripts/e03_train.py`) and the closing step
(`scripts/e03_eval.py evaluate`). State: `state/e03/status.json` and `PROGRESS.md`; logs in `state/e03/logs/`.
A failed job stops the queue; `--retry` keeps the failed folder (`.superseded-N`) and counts a new attempt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import english_phonetic_run as R  # noqa: E402

STATE_DIR = ROOT / "state" / "e03"
RUNS = ROOT / "models" / "e03"
PY = R.PY
TRAIN = str(ROOT / "scripts" / "e03_train.py")
EVAL = str(ROOT / "scripts" / "e03_eval.py")
CAPS = {"supervised_attempts": 6, "training_hours": 12.0}
DATA = ROOT / "data" / "e03"
EXP = ROOT / "experiments" / "E03-assessable-phones"


def queue() -> list[dict]:
    """The six registered jobs, paired by seed."""
    return [R.job(f"{arm}_s{s}", "train", arm, s) for s in (0, 1, 2) for arm in ("letters", "phones")]


class E03Runner(R.Runner):
    """E02's runner with the E03 queue, caps, commands and state (see the module docstring)."""

    def __init__(self, config: Path, state_dir: Path = STATE_DIR, runs_dir: Path | None = None, commands=None,
                 verify_job=None, registered: list[Path] | None = None, poll: float = 30.0,
                 eval_cmd: list[str] | None = None) -> None:
        reg = registered if registered is not None else [
            Path(config), DATA / "manifest.json", DATA / "train.tsv", DATA / "dev_references.jsonl",
            DATA / "eval_references.jsonl", DATA / "vocab.json", EXP / "frozen_eval.json"]
        super().__init__(config, state_dir, runs_dir or RUNS, commands, verify_job, reg, poll)
        self.eval_cmd = eval_cmd or [PY, EVAL, "evaluate"]

    def load(self) -> dict:
        """Current state, initialising it on the first call."""
        if self.status_path.exists():
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        st = {"schema_version": 1, "experiment_id": "E03-assessable-phones", "phase": "training", "created": R.now(),
              "registered_sha256": {str(p): R.sha(p) for p in self.registered},
              "budget": {"supervised_attempts_used": 0, "supervised_attempts_cap": CAPS["supervised_attempts"],
                         "p2g_jobs_used": 0, "p2g_jobs_cap": 0, "training_elapsed_hours": 0.0,
                         "training_elapsed_hours_cap": CAPS["training_hours"]},
              "jobs": queue(), "selection": None, "blockers": [], "final_eval": None}
        self.save(st)
        return st

    def save(self, st: dict) -> None:
        """Write `status.json` atomically and refresh `PROGRESS.md`."""
        st["last_updated"] = R.now()
        st["next_command"] = f"{PY} scripts/e03_run.py --config {self.config} --resume"
        R.write_atomic(self.status_path, st)
        b = st["budget"]
        lines = [f"# E03 progress ({st['last_updated']})", "", f"Phase: **{st['phase']}**", "",
                 "| Job | State | Attempts | Hours | Reason |", "|---|---|---|---|---|"]
        lines += [f"| {j['id']} | {j['state']} | {j['attempts']} | {j['elapsed_hours']:.2f} | {j['reason'] or ''} |"
                  for j in st["jobs"]]
        lines += ["", f"Budget: supervised {b['supervised_attempts_used']}/{b['supervised_attempts_cap']}, hours "
                  f"{b['training_elapsed_hours']:.2f}/{b['training_elapsed_hours_cap']}", "",
                  f"Evaluation: {st['final_eval']}", "", f"Blockers: {st['blockers'] or 'none'}", "",
                  f"Resume: `{st['next_command']}`", ""]
        (self.state_dir / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")

    def _commands(self, j: dict, runs: Path) -> list[list[str]]:
        out = str(runs / j["id"])
        return [[PY, TRAIN, "train", "--arm", j["arm"], "--seed", str(j["seed"]), "--out", out],
                [PY, TRAIN, "export", "--out", out], [PY, TRAIN, "verify", "--out", out]]

    def resume(self) -> int:
        """Run the queue in order, then the evaluation once; 0 when everything is done."""
        st = self.load()
        changed = self.check_hashes(st)
        if changed:
            st["blockers"].append(f"registered inputs changed: {changed}")
            self.save(st)
            print(json.dumps({"stopped": "registered inputs changed", "files": changed}))
            return 2
        for j in st["jobs"]:
            if j["state"] == "failed":
                print(json.dumps({"stopped": "failed job", "job": j["id"], "reason": j["reason"]}))
                return 1
            if j["state"] in ("completed", "unavailable"):
                continue
            if not self.run_job(st, j) and j["state"] == "failed":
                return 1
        if st["phase"] == "training":
            st["phase"] = "evaluation"
            self.save(st)
        if st["phase"] == "evaluation":
            if st["final_eval"] is None or st["final_eval"]["exit"] != 0:
                log = self.state_dir / "logs" / "evaluation.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a", encoding="utf-8") as fh:
                    code = subprocess.call(self.eval_cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                st["final_eval"] = {"exit": code, "at": R.now(), "log": str(log)}
                if code != 0:
                    st["blockers"].append(f"evaluation failed (exit {code}, log {log})")
            st["phase"] = "done" if st["final_eval"]["exit"] == 0 else "evaluation_failed"
            self.save(st)
        return 0 if st["phase"] == "done" else 1


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--retry", metavar="JOB")
    args = ap.parse_args()
    r = E03Runner(Path(args.config))
    if args.retry:
        st = r.load()
        j = next(x for x in st["jobs"] if x["id"] == args.retry)
        if j["state"] != "failed":
            print(f"{args.retry} is {j['state']}, not failed")
            return 1
        folder = r.runs_dir / j["id"]
        if folder.exists():
            folder.rename(folder.with_name(f"{folder.name}.superseded-{j['attempts']}"))
        j["state"], j["reason"] = "pending", f"retry after failure ({R.now()})"
        st["blockers"] = [b for b in st["blockers"] if not b.startswith(args.retry + ":")]
        r.save(st)
    return r.resume()


if __name__ == "__main__":
    sys.exit(main())
