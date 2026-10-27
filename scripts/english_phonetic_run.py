#!/usr/bin/env python3
"""Experiment E02: the fixed job queue, with budgets, durable state and one-command resume.

    .venv/bin/python scripts/english_phonetic_run.py --config experiments/E02-english-phonetic/config.yaml --dry-run
    .venv/bin/python scripts/english_phonetic_run.py --config experiments/E02-english-phonetic/config.yaml --resume

`--resume` initialises `state/english-phonetic/status.json` on the first call and continues the same queue on
every later call. It never adds jobs beyond the registered queue:

1. **Pilots (seed 0):** `letters_s0`, `phones_s0`, `p2g_s0` (pretraining), `phones_p2g_s0`.
2. **Selection:** the arm with the best dev primary score (`done.json`); ties prefer letters, then phones, then
   phones_p2g. If letters wins, no confirmation runs.
3. **Confirmation (only if a treatment wins):** `letters_s1`, `<winner>_s1`, `letters_s2`, `<winner>_s2`, with
   `p2g_s1` / `p2g_s2` before the winner's runs if the winner is phones_p2g.
4. **Final evaluation:** `scripts/english_phonetic_eval.py final` once, then stop.

Each training job = train (resumes from `last.pt`) → export → verify. A job is `completed` only when its
`done.json`, `model.npz` and a passing `parity.json` exist. States: pending, running, interrupted, failed,
completed, unavailable. A job found `running` with a live process is waited for, never launched twice; a dead
one is `interrupted` and resumes from its checkpoint as the same attempt. A failed job stops the queue
(`--retry JOB` relaunches it as a new attempt after a fix). Before any launch the registered config and data
hashes are checked; a change stops the queue.

Budgets (AUTONOMY.md): 7 supervised attempts, 3 P2G jobs (each ≤ 10 epochs / 2 h, enforced by the trainer), 96 h
summed job time. Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state" / "english-phonetic"
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN = str(ROOT / "scripts" / "english_phonetic_train.py")
EVAL = str(ROOT / "scripts" / "english_phonetic_eval.py")
CAPS = {"supervised_attempts": 7, "p2g_jobs": 3, "training_hours": 96.0}
TIE_ORDER = ("letters", "phones", "phones_p2g")


def now() -> str:
    """UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    """sha256 of a file."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path: Path, obj: dict) -> None:
    """JSON to a temporary file, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pid_alive(pid: int | None, marker: str) -> bool:
    """Whether `pid` is a live process whose command line contains `marker` (guards against PID reuse)."""
    if not pid:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return marker in cmd


def job(job_id: str, kind: str, arm: str | None, seed: int, init: str | None = None) -> dict:
    """A queue entry."""
    return {"id": job_id, "kind": kind, "arm": arm, "seed": seed, "init": init, "state": "pending", "attempts": 0,
            "pid": None, "started": None, "finished": None, "elapsed_hours": 0.0, "log": None, "reason": None}


def pilot_queue() -> list[dict]:
    """The registered seed-0 pilots."""
    return [job("letters_s0", "train", "letters", 0), job("phones_s0", "train", "phones", 0),
            job("p2g_s0", "pretrain", None, 0), job("phones_p2g_s0", "train", "phones_p2g", 0, "p2g_s0")]


def confirm_queue(winner: str) -> list[dict]:
    """Confirmation jobs for a treatment winner (empty if letters wins)."""
    if winner == "letters":
        return []
    out = []
    for s in (1, 2):
        out.append(job(f"letters_s{s}", "train", "letters", s))
        if winner == "phones_p2g":
            out.append(job(f"p2g_s{s}", "pretrain", None, s))
            out.append(job(f"phones_p2g_s{s}", "train", "phones_p2g", s, f"p2g_s{s}"))
        else:
            out.append(job(f"{winner}_s{s}", "train", winner, s))
    return out


class Runner:
    """The queue executor (see the module docstring).

    Args:
        config: The experiment config path.
        state_dir: Where `status.json`, `PROGRESS.md` and job logs go.
        runs_dir: Job output folders.
        commands: Optional override `(job, runs_dir) -> list of argv lists` (tests).
        verify_job: Optional override `(job, runs_dir) -> bool` (tests).
        registered: Files whose hashes are frozen at initialisation.
    """

    def __init__(self, config: Path, state_dir: Path = STATE_DIR, runs_dir: Path | None = None,
                 commands=None, verify_job=None, registered: list[Path] | None = None, poll: float = 30.0) -> None:
        self.config, self.state_dir = Path(config), Path(state_dir)
        self.runs_dir = Path(runs_dir) if runs_dir else ROOT / "models" / "e02"
        self.status_path = self.state_dir / "status.json"
        self.commands = commands or self._commands
        self.verify_job = verify_job or self._verify
        self.registered = registered if registered is not None else [
            self.config, ROOT / "data/e02/manifest.json", ROOT / "data/e02/train.tsv", ROOT / "data/e02/p2g.tsv",
            ROOT / "data/e02/references.jsonl", ROOT / "data/e02/dev_references.jsonl", ROOT / "data/e02/vocab.json"]
        self.poll = poll

    # ------------------------------------------------------------------ state
    def load(self) -> dict:
        """Current state, initialising it on the first call."""
        if self.status_path.exists():
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        st = {"schema_version": 1, "experiment_id": "E02-english-phonetic", "phase": "pilots",
              "created": now(), "registered_sha256": {str(p): sha(p) for p in self.registered},
              "budget": {"supervised_attempts_used": 0, "supervised_attempts_cap": CAPS["supervised_attempts"],
                         "p2g_jobs_used": 0, "p2g_jobs_cap": CAPS["p2g_jobs"], "training_elapsed_hours": 0.0,
                         "training_elapsed_hours_cap": CAPS["training_hours"]},
              "jobs": pilot_queue(), "selection": None, "blockers": [], "final_eval": None}
        self.save(st)
        return st

    def save(self, st: dict) -> None:
        """Write `status.json` atomically and refresh `PROGRESS.md`."""
        st["last_updated"] = now()
        st["next_command"] = f"{PY} scripts/english_phonetic_run.py --config {self.config} --resume"
        write_atomic(self.status_path, st)
        lines = [f"# E02 progress ({st['last_updated']})", "", f"Phase: **{st['phase']}**", "",
                 "| Job | State | Attempts | Hours | Reason |", "|---|---|---|---|---|"]
        for j in st["jobs"]:
            lines.append(f"| {j['id']} | {j['state']} | {j['attempts']} | {j['elapsed_hours']:.2f} | {j['reason'] or ''} |")
        b = st["budget"]
        lines += ["", f"Budget: supervised {b['supervised_attempts_used']}/{b['supervised_attempts_cap']}, P2G "
                  f"{b['p2g_jobs_used']}/{b['p2g_jobs_cap']}, hours {b['training_elapsed_hours']:.2f}/"
                  f"{b['training_elapsed_hours_cap']}", "", f"Selection: {st['selection']}", "",
                  f"Blockers: {st['blockers'] or 'none'}", "", f"Resume: `{st['next_command']}`", ""]
        (self.state_dir / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")

    def check_hashes(self, st: dict) -> list[str]:
        """Registered files whose hash changed since initialisation."""
        return [p for p, h in st["registered_sha256"].items() if not Path(p).exists() or sha(Path(p)) != h]

    # ------------------------------------------------------------------ jobs
    def _commands(self, j: dict, runs: Path) -> list[list[str]]:
        out = str(runs / j["id"])
        if j["kind"] == "pretrain":
            return [[PY, TRAIN, "pretrain", "--seed", str(j["seed"]), "--out", out]]
        cmd = [PY, TRAIN, "train", "--arm", j["arm"], "--seed", str(j["seed"]), "--out", out]
        if j["init"]:
            cmd += ["--init", str(runs / j["init"] / "best.pt")]
        return [cmd, [PY, TRAIN, "export", "--out", out], [PY, TRAIN, "verify", "--out", out]]

    def _verify(self, j: dict, runs: Path) -> bool:
        folder = runs / j["id"]
        if not (folder / "done.json").exists():
            return False
        if j["kind"] == "pretrain":
            return (folder / "best.pt").exists()
        parity = folder / "parity.json"
        return (folder / "model.npz").exists() and parity.exists() and json.loads(parity.read_text())["ok"]

    def budget_ok(self, st: dict, j: dict, new_attempt: bool) -> str | None:
        """Why launching `j` would break a budget, or None."""
        b = st["budget"]
        if b["training_elapsed_hours"] >= b["training_elapsed_hours_cap"]:
            return "aggregate training hours exhausted"
        if new_attempt and j["kind"] == "train" and b["supervised_attempts_used"] >= b["supervised_attempts_cap"]:
            return "supervised attempt cap reached"
        if new_attempt and j["kind"] == "pretrain" and b["p2g_jobs_used"] >= b["p2g_jobs_cap"]:
            return "P2G job cap reached"
        return None

    def run_job(self, st: dict, j: dict) -> bool:
        """Run (or wait for, or resume) one job to a final state; True if completed."""
        if j["state"] == "running" and pid_alive(j["pid"], j["id"]):
            while pid_alive(j["pid"], j["id"]):   # a live job from an earlier runner: wait, never relaunch
                time.sleep(self.poll)
        if j["state"] == "running":
            j["state"] = "interrupted"
            j["reason"] = "process ended without the runner recording it"
        if self.verify_job(j, self.runs_dir):
            j["state"], j["finished"] = "completed", j["finished"] or now()
            self.save(st)
            return True
        new_attempt = j["state"] == "pending"
        why = self.budget_ok(st, j, new_attempt)
        if why:
            j["state"], j["reason"] = "unavailable", why
            st["blockers"].append(f"{j['id']}: {why}")
            self.save(st)
            return False
        if new_attempt:
            j["attempts"] += 1
            st["budget"]["supervised_attempts_used" if j["kind"] == "train" else "p2g_jobs_used"] += 1
        log = self.state_dir / "logs" / f"{j['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        j["log"], j["started"], j["state"] = str(log), j["started"] or now(), "running"
        self.save(st)
        for cmd in self.commands(j, self.runs_dir):
            t0 = time.time()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n# {now()} {' '.join(cmd)}\n")
                fh.flush()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "E02_JOB": j["id"]}
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                j["pid"] = proc.pid
                self.save(st)
                code = proc.wait()
            hours = (time.time() - t0) / 3600
            j["elapsed_hours"] += hours
            st["budget"]["training_elapsed_hours"] += hours
            if code != 0:
                j["state"], j["reason"], j["pid"] = "failed", f"exit {code} in {cmd[2] if len(cmd) > 2 else cmd}", None
                st["blockers"].append(f"{j['id']}: {j['reason']} (log {log})")
                self.save(st)
                return False
        j["pid"] = None
        if not self.verify_job(j, self.runs_dir):
            j["state"], j["reason"] = "failed", "outputs missing or parity check failed"
            st["blockers"].append(f"{j['id']}: {j['reason']}")
            self.save(st)
            return False
        j["state"], j["finished"], j["reason"] = "completed", now(), None
        self.save(st)
        return True

    # ------------------------------------------------------------------ queue
    def select(self, st: dict) -> tuple[str, dict]:
        """Dev-only selection among completed pilots (unavailable arms are skipped)."""
        scores = {}
        for arm in TIE_ORDER:
            j = next(x for x in st["jobs"] if x["id"] == f"{arm}_s0")
            if j["state"] == "completed":
                scores[arm] = json.loads((self.runs_dir / j["id"] / "done.json").read_text())["best_dev_primary_macro"]
        best = max(scores.values())
        winner = next(a for a in TIE_ORDER if scores.get(a) == best)
        return winner, scores

    def resume(self) -> int:
        """Advance the queue as far as possible; 0 when the queue (incl. final evaluation) is done."""
        st = self.load()
        changed = self.check_hashes(st)
        if changed:
            st["blockers"].append(f"registered inputs changed: {changed}")
            self.save(st)
            print(json.dumps({"stopped": "registered inputs changed", "files": changed}))
            return 2
        while True:
            pending = [j for j in st["jobs"] if j["state"] not in ("completed", "unavailable")]
            for j in pending:
                if j["state"] == "failed":
                    print(json.dumps({"stopped": "failed job", "job": j["id"], "reason": j["reason"]}))
                    return 1
                if j["init"] and next(x for x in st["jobs"] if x["id"] == j["init"])["state"] != "completed":
                    j["state"], j["reason"] = "unavailable", f"its pretraining {j['init']} did not complete"
                    self.save(st)
                    continue
                if not self.run_job(st, j):
                    if j["state"] == "failed":
                        return 1
            if st["phase"] == "pilots":
                if not any(j["state"] == "completed" for j in st["jobs"] if j["kind"] == "train"):
                    st["phase"] = "not_trainable"
                    st["blockers"].append("no pilot training job completed")
                    self.save(st)
                    return 1
                winner, scores = self.select(st)
                st["selection"] = {"winner": winner, "dev_primary_macro": scores, "rule": "dev only; ties letters > "
                                   "phones > phones_p2g", "at": now()}
                st["jobs"] += confirm_queue(winner)
                st["phase"] = "confirmation" if winner != "letters" else "final_eval"
                self.save(st)
                continue
            if st["phase"] == "confirmation":
                st["phase"] = "final_eval"
                self.save(st)
                continue
            if st["phase"] == "final_eval":
                if st["final_eval"] is None:
                    log = self.state_dir / "logs" / "final_eval.log"
                    with open(log, "a", encoding="utf-8") as fh:
                        code = subprocess.call([PY, EVAL, "final"], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                    st["final_eval"] = {"exit": code, "at": now(), "log": str(log)}
                    if code != 0:
                        st["blockers"].append(f"final evaluation failed (exit {code}, log {log})")
                st["phase"] = "done" if st["final_eval"]["exit"] == 0 else "final_eval_failed"
                self.save(st)
            return 0 if st["phase"] == "done" else 1


def dry_run(config: Path) -> dict:
    """The queue, commands, outputs and caps, without starting anything."""
    r = Runner(config)
    plan = {"caps": CAPS, "phases": ["pilots", "selection (dev only)", "confirmation if a treatment wins",
                                    "final evaluation once"], "pilots": [], "confirmation_if": {}}
    for j in pilot_queue():
        plan["pilots"].append({"id": j["id"], "commands": r._commands(j, r.runs_dir),
                               "outputs": [str(r.runs_dir / j["id"] / f) for f in ("done.json", "best.pt", "model.npz",
                                                                                    "parity.json")]})
    for w in TIE_ORDER:
        plan["confirmation_if"][w] = [j["id"] for j in confirm_queue(w)]
    plan["max_supervised_attempts"] = 3 + max(sum(j["kind"] == "train" for j in confirm_queue(w)) for w in TIE_ORDER)
    plan["max_p2g_jobs"] = 1 + max(sum(j["kind"] == "pretrain" for j in confirm_queue(w)) for w in TIE_ORDER)
    return plan


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--retry", metavar="JOB", help="after a fix: relaunch a failed job as a new attempt")
    args = ap.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(Path(args.config)), indent=1))
        return 0
    r = Runner(Path(args.config))
    if args.retry:
        st = r.load()
        j = next(x for x in st["jobs"] if x["id"] == args.retry)
        if j["state"] != "failed":
            print(f"{args.retry} is {j['state']}, not failed")
            return 1
        folder = r.runs_dir / j["id"]
        if folder.exists():   # keep the failed attempt's evidence; the retry starts clean
            folder.rename(folder.with_name(f"{folder.name}.superseded-{j['attempts']}"))
        j["state"], j["reason"] = "pending", f"retry after failure ({now()})"
        st["blockers"] = [b for b in st["blockers"] if not b.startswith(args.retry + ":")]
        r.save(st)
    return r.resume()


if __name__ == "__main__":
    sys.exit(main())
