#!/usr/bin/env python3
"""Experiment E02: build and verify the local artifact bundle `dist/english-phonetic-<version>/` and its ZIP.

    .venv/bin/python scripts/english_phonetic_bundle.py [--force]

What goes in (and nothing else):
- `wheel/`: wheel + sdist from `uv build` (code only, no data);
- `checkpoints/<run>/`: `model.npz`, `parity.json`, `done.json`, `provenance.json` of every completed supervised
  run; `DEFAULT` names the default model;
- `resources/cmudict/`: CMUdict + its licence (needed by phone-input models only);
- `cards/`, `config/`, `run/`, `outcomes/`: model and benchmark cards, registered config/plan/manifests, the run
  ledger and results, the per-item outcomes;
- `README.md`, `NOTICES.md`, `FINAL_REPORT.md`, `SOURCE_VERIFICATION.md`, `prepare_manifest.json`, `SHA256SUMS`.

Never included: training corpora (mined pairs, P2G pairs), torch checkpoints, W&B folders, the E01 reserve, caches.
The whole bundle is **restricted: local only** (CC BY-SA training data, owner-list references); see NOTICES.md.

Verification after zipping: `testzip`, every member a safe relative path, the extracted copy matches SHA256SUMS,
and `khmer-translit prepare --manifest prepare_manifest.json` passes on the extracted copy. Exit 0 only if all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXP = ROOT / "experiments" / "E02-english-phonetic"
STATE = ROOT / "state" / "english-phonetic" / "status.json"
FORBIDDEN = (".pt", "train.tsv", "p2g.tsv", "mined_", "reserve", "wandb", "__pycache__")


def sha(p: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(src: Path, dst: Path) -> None:
    """Copy one file, creating folders."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build(out: Path) -> dict:
    """Fill the bundle folder; returns what went in."""
    subprocess.run(["uv", "build", "-q", "-o", str(out / "wheel")], cwd=ROOT, check=True)
    (out / "wheel" / ".gitignore").unlink(missing_ok=True)
    status = json.loads(STATE.read_text(encoding="utf-8"))
    results = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    runs = [j["id"] for j in status["jobs"] if j["state"] == "completed" and j.get("arm")]
    for r in runs:
        for f in ("model.npz", "parity.json", "done.json", "provenance.json"):
            put(ROOT / "models" / "e02" / r / f, out / "checkpoints" / r / f)
    (out / "checkpoints" / "DEFAULT").write_text(results["default_model"] + "\n", encoding="utf-8")
    cmu = ROOT / "data" / "src" / "cmudict"
    put(cmu / "cmudict.dict", out / "resources" / "cmudict" / "cmudict.dict")
    put(cmu / "LICENSE", out / "resources" / "cmudict" / "LICENSE")
    put(ROOT / "data" / "lexicon" / "ATTRIBUTION.md", out / "resources" / "google_km_lexicon_ATTRIBUTION.md")
    copies = {
        "README.md": EXP / "bundle" / "README.md", "NOTICES.md": EXP / "bundle" / "NOTICES.md",
        "FINAL_REPORT.md": EXP / "FINAL_REPORT.md", "SOURCE_VERIFICATION.md": EXP / "SOURCE_VERIFICATION.md",
        "cards/MODEL_CARD.md": ROOT / "model_cards" / "e02_english_phonetic.md",
        "cards/BENCHMARK_CARD.md": EXP / "BENCHMARK_CARD.md",
        "config/config.yaml": EXP / "config.yaml", "config/PLAN.md": EXP / "PLAN.md",
        "config/data_policy.json": EXP / "data_policy.json", "config/sources.json": EXP / "resources.json",
        "config/data_manifest.json": ROOT / "data" / "e02" / "manifest.json",
        "config/vocab.json": ROOT / "data" / "e02" / "vocab.json",
        "config/data_build.json": EXP / "reports" / "data_build.json",
        "run/status.json": EXP / "run_ledger.json", "run/results.json": EXP / "results.json",
        "run/RESULTS.md": EXP / "RESULTS.md", "run/runtime.json": EXP / "reports" / "runtime.json",
        "run/wheel_check.json": EXP / "reports" / "wheel_check.json", "run/power.json": EXP / "reports" / "power.json",
        "run/audit.md": EXP / "reports" / "audit.md",
        "outcomes/per_item.json": ROOT / "data" / "e02" / "results" / "per_item.json",
        "outcomes/predictions.json": ROOT / "data" / "e02" / "results" / "predictions.json",
        "outcomes/references.jsonl": ROOT / "data" / "e02" / "references.jsonl",
    }
    for dst, src in copies.items():
        put(src, out / dst)
    files = [{"path": f"checkpoints/{r}/model.npz", "sha256": sha(out / "checkpoints" / r / "model.npz"),
              "role": "model (default)" if r == results["default_model"] else "model (research)"} for r in runs]
    files.append({"path": "resources/cmudict/cmudict.dict", "sha256": sha(out / "resources/cmudict/cmudict.dict"),
                  "role": "pronunciation provider (phone-input models)"})
    (out / "prepare_manifest.json").write_text(json.dumps({"files": files}, indent=1) + "\n", encoding="utf-8")
    sums = sorted((p.relative_to(out).as_posix(), sha(p)) for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums), encoding="utf-8")
    return {"runs": runs, "files": len(sums) + 1}


def verify(out: Path, zpath: Path) -> dict:
    """Check the ZIP: integrity, safe paths, checksums after extraction, resource manifest."""
    checks = {}
    with zipfile.ZipFile(zpath) as z:
        checks["testzip_ok"] = z.testzip() is None
        names = z.namelist()
        checks["safe_relative_paths"] = all(not n.startswith("/") and ".." not in PurePosixPath(n).parts
                                            and n.startswith(out.name + "/") for n in names)
        checks["nothing_forbidden"] = not [n for n in names if any(f in n for f in FORBIDDEN)]
        with tempfile.TemporaryDirectory() as t:
            z.extractall(t)
            ext = Path(t) / out.name
            bad = [l for l in (ext / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
                   if sha(ext / l.split("  ", 1)[1]) != l.split("  ", 1)[0]]
            checks["sha256sums_match"] = not bad
            listed = {l.split("  ", 1)[1] for l in (ext / "SHA256SUMS").read_text(encoding="utf-8").splitlines()}
            present = {p.relative_to(ext).as_posix() for p in ext.rglob("*") if p.is_file()} - {"SHA256SUMS"}
            checks["every_file_listed"] = listed == present
            p = subprocess.run([sys.executable, "-m", "khmer_translit.cli", "prepare", "--manifest",
                                str(ext / "prepare_manifest.json")], capture_output=True, text=True,
                               env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
            checks["prepare_manifest_ok"] = p.returncode == 0
    return {"members": len(names), "checks": checks, "ok": all(checks.values())}


def main() -> int:
    """Build the bundle and ZIP, then verify; writes `<bundle>.verify.json` next to the ZIP."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="replace an existing bundle folder (it is generated)")
    args = ap.parse_args()
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)[1]
    out = ROOT / "dist" / f"english-phonetic-{version}"
    if out.exists():
        if not args.force:
            print(f"{out} exists; pass --force to rebuild it", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    out.mkdir(parents=True)
    info = build(out)
    zpath = out.parent / f"{out.name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out.parent).as_posix())
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rep = {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "bundle": str(out.relative_to(ROOT)), "zip": str(zpath.relative_to(ROOT)), "zip_sha256": sha(zpath),
           "zip_bytes": zpath.stat().st_size, **info, **verify(out, zpath)}
    (out.parent / f"{out.name}.verify.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
