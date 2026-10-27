#!/usr/bin/env python3
"""Experiment E02: check the installed wheel outside the checkout (Task 5/6 product verification).

    .venv/bin/python scripts/english_phonetic_wheelcheck.py --letters models/e02/letters_s0 \
        --phones models/e02/phones_s0 [--wheel dist/x.whl] [--out reports/wheel_check.json]

Builds the wheel and sdist with `uv build` (unless `--wheel` is given), installs the wheel into a fresh venv in a
temporary folder (NumPy from the uv cache, else PyPI), and runs every check from another temporary folder with no
`PYTHONPATH`, no `KHMER_TRANSLIT_CMUDICT` and, for the API checks, sockets disabled. Each check records its command,
exit status and whether it passed. Output: `experiments/E02-english-phonetic/reports/wheel_check.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "E02-english-phonetic" / "reports" / "wheel_check.json"
CMU = ROOT / "data" / "src" / "cmudict" / "cmudict.dict"

API = r"""
import json, socket, sys
def _no_net(*a, **k):
    raise OSError("network disabled by wheel check")
socket.socket = _no_net
socket.create_connection = _no_net
import khmer_translit
from khmer_translit.english import Transliterator
letters = Transliterator.from_pretrained(sys.argv[1])
phones = Transliterator.from_pretrained(sys.argv[2], cmudict_path=sys.argv[3])
words = ["coffee", "Computer", "", "new york", "zzyzxq", "record"]
a = [r.to_dict() for r in phones.predict_many(words, top_k=3)]
b = [r.to_dict() for r in phones.predict_many(words, top_k=3)]
heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "wandb", "khmer_tts", "pocket_tts", "yaml"))
print(json.dumps({"package_file": khmer_translit.__file__, "deterministic": a == b,
                  "statuses": [r["status"] for r in a], "inputs": [r["input"] for r in a],
                  "pron_status": [(r["pronunciation"] or {}).get("status") for r in a],
                  "supplied": phones.predict("record", pronunciation="R IH0 K AO1 R D").pronunciation["status"],
                  "letters_top": [c["text"] for c in letters.predict("coffee", top_k=3).to_dict()["candidates"]],
                  "heavy_modules": heavy}, ensure_ascii=False))
"""


def run(cmd: list[str], cwd: Path, env: dict, stdin: str | None = None) -> dict:
    """Run one command and keep what a reader needs to judge it."""
    p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True)
    return {"cmd": " ".join(Path(c).name if os.sep in c else c for c in cmd), "exit": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr.strip()[-400:]}


def main() -> int:
    """Build, install and check; exit 0 only if every check passed."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--letters", required=True)
    ap.add_argument("--phones", required=True)
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    letters, phones = (str(Path(m).resolve()) for m in (args.letters, args.phones))
    tmp = Path(tempfile.mkdtemp(prefix="e02-wheel-"))
    try:
        if args.wheel:
            wheel = Path(args.wheel).resolve()
            built = [wheel.name]
        else:
            subprocess.run(["uv", "build", "-q", "-o", str(tmp / "dist")], cwd=ROOT, check=True)
            wheel = next((tmp / "dist").glob("*.whl"))
            built = sorted(p.name for p in (tmp / "dist").iterdir())
        venv = tmp / "venv"
        subprocess.run(["uv", "venv", "-q", "-p", "3.10", str(venv)], check=True)
        ienv = {**os.environ, "VIRTUAL_ENV": str(venv)}
        how = "offline (uv cache)"
        if subprocess.run(["uv", "pip", "install", "-q", "--offline", str(wheel)], env=ienv).returncode:
            subprocess.run(["uv", "pip", "install", "-q", str(wheel)], env=ienv, check=True)
            how = "numpy from PyPI"
        work = tmp / "elsewhere"
        work.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "KHMER_TRANSLIT_CMUDICT", "VIRTUAL_ENV")}
        cli, py = str(venv / "bin" / "khmer-translit"), str(venv / "bin" / "python")
        checks = {}

        r = run([cli, "predict", "--model", letters, "--word", "coffee", "--json", "--top-k", "3"], work, env)
        d = json.loads(r["stdout"]) if r["exit"] == 0 else {}
        checks["letters_word_json"] = {**r, "ok": r["exit"] == 0 and d.get("status") == "ok"
                                       and d.get("pronunciation") is None and 1 <= len(d.get("candidates", [])) <= 3}
        r = run([cli, "predict", "--model", letters, "--input", "-", "--jsonl"], work, env,
                stdin="computer\n\nhello world\nJohnson\nកាហ្វេ\n")
        rows = [json.loads(l) for l in r["stdout"].splitlines()]
        checks["letters_batch_jsonl"] = {**r, "ok": r["exit"] == 1 and [x["id"] for x in rows] == [1, 2, 3, 4, 5]
                                         and [x["status"] for x in rows] == ["ok", "empty", "unsupported_multiword",
                                                                            "ok", "unsupported_characters"]}
        r = run([cli, "predict", "--model", phones, "--word", "coffee"], work, env)
        checks["phones_without_cmudict_refused"] = {**r, "ok": r["exit"] == 2 and "KHMER_TRANSLIT_CMUDICT" in r["stderr"]}
        r = run([cli, "predict", "--model", phones, "--cmudict", str(CMU), "--word", "record", "--json"], work, env)
        d = json.loads(r["stdout"]) if r["exit"] == 0 else {}
        checks["phones_with_cmudict"] = {**r, "ok": r["exit"] == 0 and d["pronunciation"]["status"] == "dictionary"
                                         and d["pronunciation"]["ambiguous"] is True}
        r = run([cli, "predict", "--model", phones, "--cmudict", str(CMU), "--word", "record",
                 "--pronunciation", "R IH0 K AO1 R D", "--json"], work, env)
        d = json.loads(r["stdout"]) if r["exit"] == 0 else {}
        checks["supplied_pronunciation"] = {**r, "ok": r["exit"] == 0 and d["pronunciation"]["status"] == "provided"}
        r = run([cli, "predict", "--model", phones, "--cmudict", str(CMU), "--word", "record",
                 "--pronunciation", "r i k o r d"], work, env)
        checks["unparsed_pronunciation_refused"] = {**r, "ok": r["exit"] == 1 and "unknown_phone" in r["stdout"]}
        r = run([cli, "inspect", "--model", phones, "--cmudict", str(CMU)], work, env)
        d = json.loads(r["stdout"]) if r["exit"] == 0 else {}
        checks["inspect"] = {**r, "ok": r["exit"] == 0 and d.get("mode") == "phones" and "CMUdict" in d["requires"][0]}
        bad = work / "corrupt.npz"
        bad.write_bytes(b"not a model")
        r = run([cli, "predict", "--model", str(bad), "--word", "coffee"], work, env)
        checks["corrupt_checkpoint_refused"] = {**r, "ok": r["exit"] == 2}
        r = run([cli, "predict", "--model", str(work / "absent.npz"), "--word", "coffee"], work, env)
        checks["missing_checkpoint_refused"] = {**r, "ok": r["exit"] == 2}
        r1 = run([cli, "predict", "--model", letters, "--input", "-", "--jsonl", "--top-k", "4"], work, env,
                 stdin="banana\ntelevision\nSmith\n")
        r2 = run([cli, "predict", "--model", letters, "--input", "-", "--jsonl", "--top-k", "4"], work, env,
                 stdin="banana\ntelevision\nSmith\n")
        checks["repeat_run_determinism"] = {**r1, "ok": r1["exit"] == 0 and r1["stdout"] == r2["stdout"]}
        (work / "api.py").write_text(API, encoding="utf-8")
        r = run([py, "-I", str(work / "api.py"), letters, phones, str(CMU)], work, env)
        d = json.loads(r["stdout"].splitlines()[-1]) if r["exit"] == 0 else {}
        checks["api_offline_no_checkout"] = {
            **r, "ok": r["exit"] == 0 and str(venv) in d["package_file"] and d["deterministic"]
            and d["inputs"][2:4] == ["", "new york"]
            and d["statuses"] == ["ok", "ok", "empty", "unsupported_multiword", "ok", "ok"]
            and d["pron_status"][4] == "unavailable" and d["supplied"] == "provided" and not d["heavy_modules"]}
        out = {"checked": time.strftime("%Y-%m-%d %H:%M"), "wheel": wheel.name, "built": built, "install": how,
               "python": "3.10 (uv venv)", "cwd": "temporary folder outside the checkout",
               "env": "no PYTHONPATH, no KHMER_TRANSLIT_CMUDICT; API check with sockets disabled",
               "models": {"letters": letters, "phones": phones},
               "all_ok": all(c["ok"] for c in checks.values()), "checks": checks}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for k, c in checks.items():
            print(f"{'ok  ' if c['ok'] else 'FAIL'} {k} (exit {c['exit']})")
        return 0 if out["all_ok"] else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
