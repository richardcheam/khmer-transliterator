#!/usr/bin/env python3
"""Experiment E04: the durable queue. Sanity checks must have passed (`scripts/e04.py sanity`).

    .venv/bin/python scripts/e04_run.py --config experiments/E04-pronunciation-pathways/config.yaml --resume
    .venv/bin/python scripts/e04_run.py --config experiments/E04-pronunciation-pathways/config.yaml --retry P_s0

Phases (each recorded in `state/e04/status.json`; `PROGRESS.md` beside it):
1. `pilots`: G_s0, P_s0, GP_s0.
2. `branch`: `e04.devcheck` (dev only) applies the registered rule. A: stop and evaluate; B: GP is the candidate;
   C: queue V3_s0 (three-view) and run the exploratory G/P combination.
3. `selection`: `e04.select` (seed-0 dev macro ≥ G + 2 pp; eligible P, GP, V3; ties to the simpler recipe). Only
   for B or C. If a candidate passes: G and the candidate at seeds 1 and 2.
4. `evaluation`: `e04.py evaluate` on the historical E03 eval set, after every checkpoint is frozen.

Caps: 8 full attempts (failures included), 12 aggregate training hours, one process at a time. E02's `Runner`
supplies the job mechanics (live-job detection, resume without duplicating completed jobs, attempt counting).
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

STATE_DIR = ROOT / "state" / "e04"
RUNS = ROOT / "models" / "e04"
PY = R.PY
E04 = str(ROOT / "scripts" / "e04.py")
CAPS = {"supervised_attempts": 8, "training_hours": 12.0}
DATA = ROOT / "data" / "e03"
EXP = ROOT / "experiments" / "E04-pronunciation-pathways"


class E04Runner(R.Runner):
    """E02's runner with the E04 phases (see the module docstring)."""

    def __init__(self, config: Path, state_dir: Path = STATE_DIR, runs_dir: Path | None = None, commands=None,
                 verify_job=None, registered: list[Path] | None = None, poll: float = 30.0, hooks=None) -> None:
        reg = registered if registered is not None else [
            Path(config), DATA / "manifest.json", DATA / "train.tsv", DATA / "dev_references.jsonl",
            DATA / "eval_references.jsonl", DATA / "vocab.json"]
        super().__init__(config, state_dir, runs_dir or RUNS, commands, verify_job, reg, poll)
        self.hooks = hooks or {}

    def hook(self, name: str, *args):
        """Phase actions (overridable in tests): sanity status, devcheck, combine, select, evaluate."""
        if name in self.hooks:
            return self.hooks[name](*args)
        import e04
        if name == "sanity_passed":
            p = STATE_DIR / "sanity.json"
            return p.exists() and json.loads(p.read_text()).get("passed", False)
        if name == "devcheck":
            return e04.devcheck()
        if name == "combine":
            return e04.combine()
        if name == "select":
            return e04.select(*args)
        if name == "evaluate":
            log = self.state_dir / "logs" / "evaluation.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as fh:
                return subprocess.call([PY, E04, "evaluate"], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        raise KeyError(name)

    def load(self) -> dict:
        """Current state, initialising it on the first call."""
        if self.status_path.exists():
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        st = {"schema_version": 1, "experiment_id": "E04-pronunciation-pathways", "phase": "pilots", "created": R.now(),
              "registered_sha256": {str(p): R.sha(p) for p in self.registered},
              "budget": {"supervised_attempts_used": 0, "supervised_attempts_cap": CAPS["supervised_attempts"],
                         "p2g_jobs_used": 0, "p2g_jobs_cap": 0, "training_elapsed_hours": 0.0,
                         "training_elapsed_hours_cap": CAPS["training_hours"]},
              "jobs": [R.job(f"{a}_s0", "train", a, 0) for a in ("G", "P", "GP")],
              "branch": None, "combination": None, "selection": None, "blockers": [], "final_eval": None}
        self.save(st)
        return st

    def save(self, st: dict) -> None:
        """Write `status.json` atomically and refresh `PROGRESS.md`."""
        st["last_updated"] = R.now()
        st["next_command"] = f"{PY} scripts/e04_run.py --config {self.config} --resume"
        R.write_atomic(self.status_path, st)
        b = st["budget"]
        lines = [f"# E04 progress ({st['last_updated']})", "", f"Phase: **{st['phase']}**", "",
                 "| Job | State | Attempts | Hours | Reason |", "|---|---|---|---|---|"]
        lines += [f"| {j['id']} | {j['state']} | {j['attempts']} | {j['elapsed_hours']:.2f} | {j['reason'] or ''} |"
                  for j in st["jobs"]]
        br = st.get("branch") or {}
        lines += ["", f"Budget: full attempts {b['supervised_attempts_used']}/{b['supervised_attempts_cap']}, hours "
                  f"{b['training_elapsed_hours']:.2f}/{b['training_elapsed_hours_cap']}", "",
                  f"Branch: {br.get('branch')}", f"Selection: {(st.get('selection') or {}).get('candidate')}",
                  f"Evaluation: {st['final_eval']}", "", f"Blockers: {st['blockers'] or 'none'}", "",
                  f"Resume: `{st['next_command']}`", ""]
        (self.state_dir / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")

    def _commands(self, j: dict, runs: Path) -> list[list[str]]:
        out = str(runs / j["id"])
        return [[PY, E04, "train", "--arm", j["arm"], "--seed", str(j["seed"]), "--out", out],
                [PY, E04, "export", "--out", out], [PY, E04, "verify", "--out", out]]

    def run_pending(self, st: dict) -> int | None:
        """Run every queued job in order; 1 on a failure, None when all are done."""
        for j in st["jobs"]:
            if j["state"] == "failed":
                print(json.dumps({"stopped": "failed job", "job": j["id"], "reason": j["reason"]}))
                return 1
            if j["state"] in ("completed", "unavailable"):
                continue
            if not self.run_job(st, j) and j["state"] == "failed":
                return 1
        return None

    def resume(self) -> int:
        """Advance through the phases; 0 when the evaluation is done."""
        st = self.load()
        changed = self.check_hashes(st)
        if changed:
            st["blockers"].append(f"registered inputs changed: {changed}")
            self.save(st)
            return 2
        if not self.hook("sanity_passed"):
            st["blockers"] = sorted(set(st["blockers"]) | {"sanity checks have not passed"})
            self.save(st)
            return 1
        st["blockers"] = [b for b in st["blockers"] if b != "sanity checks have not passed"]
        while True:
            code = self.run_pending(st)
            if code is not None:
                return code
            if st["phase"] == "pilots":
                if not all(j["state"] == "completed" for j in st["jobs"]):
                    st["phase"] = "evaluation"
                    st["blockers"].append("a pilot is unavailable; no branch decision")
                    self.save(st)
                    continue
                d = self.hook("devcheck")
                st["branch"] = {k: d[k] for k in ("branch", "useful_signal", "gp_strongest", "union_gain_pp",
                                                  "p_macro_exceeds_g")} | {"macro": {a: d["scores"][a]["macro"]
                                                                                     for a in d["scores"]}}
                if d["branch"] == "A":
                    st["phase"] = "evaluation"
                elif d["branch"] == "B":
                    st["phase"] = "selection"
                else:
                    st["jobs"].append(R.job("V3_s0", "train", "V3", 0))
                    st["combination"] = self.hook("combine")
                    st["phase"] = "conditional"
                self.save(st)
                continue
            if st["phase"] == "conditional":
                st["phase"] = "selection"
                self.save(st)
                continue
            if st["phase"] == "selection":
                sel = self.hook("select", st["jobs"])
                st["selection"] = {**sel, "at": R.now()}
                if sel["candidate"]:
                    st["jobs"] += [R.job(f"{a}_s{s}", "train", a, s) for s in (1, 2) for a in ("G", sel["candidate"])]
                st["phase"] = "confirmation" if sel["candidate"] else "evaluation"
                self.save(st)
                continue
            if st["phase"] == "confirmation":
                st["phase"] = "evaluation"
                self.save(st)
                continue
            if st["phase"] == "evaluation":
                if st["final_eval"] is None or st["final_eval"]["exit"] != 0:
                    code = self.hook("evaluate")
                    st["final_eval"] = {"exit": code, "at": R.now()}
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
    r = E04Runner(Path(args.config))
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
