#!/usr/bin/env python3
"""E02 letters-only release candidate: provenance facts, wheel build, isolated install check, local bundle.

Nothing here publishes anything. The bundle stays in `dist/` (git-ignored); the committed evidence goes to
`experiments/E02-release/reports/`.

Steps (run in this order; each refuses to run on a checkpoint whose sha256 differs from `CHECKPOINT_SHA`):
    audit    re-hash the checkpoint lineage and the data chain, count training rows by source (aggregates only)
             → reports/provenance_facts.json
    build    wheel + sdist from a clean `git archive` of HEAD (not the working tree) → dist/<NAME>/wheel/
    verify   fresh venv outside the checkout (no PYTHONPATH, no .pth bridge, no system site-packages; NumPy from
             the declared requirement, uv cache first), CLI/API checks with sockets disabled, parity with the
             recorded E02 outputs, latency and memory → reports/install_verify.json
    bundle   checkpoint, wheel/sdist, documents, Hugging Face folder, SHA256SUMS, .tar.gz + .sha256 → dist/

Final E02 references and the sealed reserve are never read: parity uses the recorded *model outputs* of the
exploratory proxy track (`predictions.json`, inputs and candidates only) and the dev inputs.

Example:
    .venv/bin/python scripts/e02_release.py audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "e02" / "letters_s0"
CHECKPOINT = RUN / "model.npz"
CHECKPOINT_SHA = "8901050b1093cf90f60523c7941d6c12dbd8571d71698e37abcf874ba437904b"   # FINAL_REPORT.md
BEST_PT_SHA = "199df48c4efc4d2f79f5a4a205173ada84eb37349972934f5bcf0ab1e8dad848"     # done.json best_sha256
EXP = ROOT / "experiments" / "E02-release"
REPORTS = EXP / "reports"
DOCS = ROOT / "release" / "e02"
E02 = ROOT / "experiments" / "E02-english-phonetic"
DATA = ROOT / "data" / "e02"
NAME = "khmer-translit-e02-letters-0.2.0"          # distinct from the historical dist/english-phonetic-0.1.0
DIST = ROOT / "dist" / NAME
FORBIDDEN_SUFFIXES = (".pt", ".bin", ".safetensors", ".tsv", ".jsonl", ".parquet", ".bz2", ".env")
FORBIDDEN_NAMES = ("references", "reserve", "cmudict", "wandb", "__pycache__", "origin_key", "review")
HEAVY = ("torch", "transformers", "peft", "safetensors", "wandb", "khmer_tts", "pocket_tts")


def sha(p: Path) -> str:
    """sha256 of a file, streamed.

    Args:
        p: File path.

    Returns:
        Hex digest.

    Example:
        >>> len(sha(Path(__file__)))
        64
    """
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path: Path, obj) -> dict:
    """Write `obj` as indented UTF-8 JSON (creating folders) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return obj


def check_checkpoint() -> None:
    """Refuse to continue unless the intended E02 letters checkpoint is present and unchanged.

    Raises:
        FileNotFoundError: If `models/e02/letters_s0/model.npz` is missing.
        RuntimeError: If its sha256 differs from the one recorded in FINAL_REPORT.md.
    """
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"{CHECKPOINT} is missing; no substitute is allowed")
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError(f"{CHECKPOINT} sha256 differs from the recorded {CHECKPOINT_SHA}")


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


# ------------------------------------------------------------------ audit
def audit() -> dict:
    """Re-derive the checkpoint lineage and data-chain facts used by PROVENANCE_AUDIT.md (aggregates only)."""
    check_checkpoint()
    import numpy as np
    meta = json.loads(str(np.load(CHECKPOINT)["__meta__"]))
    done = json.loads((RUN / "done.json").read_text())
    prov = json.loads((RUN / "provenance.json").read_text())
    cfg = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checkpoint": {
        "path": str(CHECKPOINT.relative_to(ROOT)), "sha256": sha(CHECKPOINT), "matches_report": True,
        "bytes": CHECKPOINT.stat().st_size, "meta": {k: v for k, v in meta.items() if k not in ("src_tokens",
                                                                                               "tgt_vocab")},
        "best_pt_sha256": sha(RUN / "best.pt"), "best_pt_matches_done_and_meta":
            sha(RUN / "best.pt") == done["best_sha256"] == meta["checkpoint_sha256"] == BEST_PT_SHA,
        "parity": json.loads((RUN / "parity.json").read_text())}}
    out["training_run"] = {k: done[k] for k in ("arm", "seed", "mode", "init_sha256", "p2g_init_file_sha256",
                                                "best_epoch", "supervised_steps", "replay_steps", "train_rows",
                                                "dev_items", "params")}
    out["training_run"]["config_init"] = cfg.get("init")
    out["training_run"]["git_commit"] = prov["git_commit"]
    out["training_run"]["git_dirty"] = prov["git_dirty"]
    out["training_run"]["inputs_rehashed"] = {p: (sha(ROOT / p) == h if (ROOT / p).exists() else "missing")
                                              for p, h in prov["inputs"].items()}
    out["data_build"] = {"manifest_sha256": sha(DATA / "manifest.json"),
                         "manifest_matches_meta": sha(DATA / "manifest.json") == meta["data_manifest_sha256"],
                         "inputs_rehashed": {p: (sha(Path(p) if p.startswith("/") else ROOT / p) == h
                                                 if (Path(p) if p.startswith("/") else ROOT / p).exists()
                                                 else "missing")
                                             for p, h in manifest["input_sha256"].items()},
                         "outputs_rehashed": {p: sha(DATA / p) == h for p, h in manifest["output_sha256"].items()},
                         "counts": {k: manifest["counts"][k] for k in ("mined_rows_in", "supervised_dropped",
                                                                       "supervised_kept")}}
    rows = list(csv.DictReader(open(DATA / "train.tsv", encoding="utf-8"), delimiter="\t"))
    origin = {}
    for name in ("mined_kmwiki.tsv", "mined_fineweb.tsv"):
        for r in list(csv.reader(open(ROOT / "data" / name, encoding="utf-8"), delimiter="\t"))[1:]:
            origin.setdefault((r[0], unicodedata.normalize("NFC", r[1])), set()).add(name.split("_")[1][:-4])
    by = Counter("+".join(sorted(origin.get((r["word"], r["khmer"]), {"UNMATCHED"}))) for r in rows)
    out["train_tsv"] = {"rows": len(rows), "by_part": dict(Counter(r["part"] for r in rows)),
                        "by_source_column": dict(Counter(r["source"] for r in rows)),
                        "by_tier": dict(Counter(r["tier"] for r in rows)), "by_mined_file": dict(by)}
    pairs = {(r["word"], r["khmer"]) for r in rows}
    words = {r["word"] for r in rows}

    def overlap(path: Path, a: int = 0, b: int = 1) -> dict:
        s = set()
        for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t"):
            if len(r) > b:
                s.add((r[a].strip().lower(), unicodedata.normalize("NFC", r[b].strip())))
        return {"rows": len(s), "identical_pairs_in_train": len(pairs & s),
                "words_in_train": len(words & {w for w, _ in s})}
    out["exclusion_checks"] = {
        "note": "set intersections only; a training row is attributed to the mined files above, so an identical "
                "pair here is a coincidence of spelling, not a copied label",
        "llm_drafts.tsv": overlap(ROOT / "data" / "llm_drafts.tsv"),
        "llm_drafts_r2.tsv": overlap(ROOT / "data" / "llm_drafts_r2.tsv"),
        "synth/synthetic_pairs.tsv (Gemini)": overlap(ROOT / "data" / "synth" / "synthetic_pairs.tsv"),
        "synth/examples.tsv (owner picks)": overlap(ROOT / "data" / "synth" / "examples.tsv"),
        "owner_picks.tsv": overlap(ROOT / "data" / "owner_picks.tsv"),
        "pairs.tsv (Chhunneng-derived romanisation pairs)": overlap(ROOT / "data" / "pairs.tsv")}
    dev = [json.loads(x) for x in open(DATA / "dev_references.jsonl", encoding="utf-8")]
    prim = [d for d in dev if not set(d["flags"]) & {"pronunciation_ambiguous", "possible_translation"}]
    out["selection_data"] = {"file": "data/e02/dev_references.jsonl", "sha256": sha(DATA / "dev_references.jsonl"),
                             "items": len(dev), "primary_items_used_for_selection": len(prim),
                             "provenance_of_primary": dict(Counter(s for d in prim for s in d["sources"])),
                             "flags": dict(Counter(f for d in dev for f in d["flags"]))}
    return write_json(REPORTS / "provenance_facts.json", out)


# ------------------------------------------------------------------ build
def build() -> dict:
    """Build wheel + sdist from a clean export of HEAD (uncommitted changes cannot leak into the artifacts)."""
    check_checkpoint()
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    wheel_dir = DIST / "wheel"
    if wheel_dir.exists():
        shutil.rmtree(wheel_dir)
    with tempfile.TemporaryDirectory(prefix="e02-src-") as d:
        arc = Path(d) / "src.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(arc), "HEAD"], cwd=ROOT, check=True)
        src = Path(d) / "src"
        src.mkdir()
        with tarfile.open(arc) as t:
            t.extractall(src, filter="data")
        subprocess.run(["uv", "build", "-q", "-o", str(wheel_dir)], cwd=src, check=True)
    (wheel_dir / ".gitignore").unlink(missing_ok=True)
    files = sorted(p.name for p in wheel_dir.iterdir())
    out = {"commit": commit, "working_tree_dirty_ignored": dirty, "files": files,
           "sha256": {f: sha(wheel_dir / f) for f in files}}
    return write_json(REPORTS / "build.json", out)


# ------------------------------------------------------------------ verify
NO_NET = r"""
import socket
def _no_net(*a, **k):
    raise OSError("network disabled by the E02 install check")
socket.socket = _no_net
socket.create_connection = _no_net
socket.getaddrinfo = _no_net
"""

CLI_WRAP = NO_NET + r"""
import sys
from khmer_translit.cli import main
code = main(sys.argv[1:])
heavy = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
print("HEAVY_MODULES=" + ",".join(heavy), file=sys.stderr)
sys.exit(code)
"""

API = NO_NET + r"""
import json, resource, sys, time
t0 = time.perf_counter()
import khmer_translit, numpy
from khmer_translit.english import Transliterator
tl = Transliterator.from_pretrained(sys.argv[1])
load_s = time.perf_counter() - t0
words = json.loads(open(sys.argv[2], encoding="utf-8").read())
one = tl.predict("coffee", top_k=3).to_dict()
edge = ["", "   ", "new york", "c3po", "café", "ខ្មែរ", "a" * 32, "a" * 33, "o'neil", "wi-fi", "COFFEE"]
er = [tl.predict(w).to_dict() for w in edge]
a = [r.to_dict() for r in tl.predict_many(words["dev"], top_k=4)]
b = [r.to_dict() for r in tl.predict_many(words["dev"], top_k=4)]
proxy = [r.to_dict() for r in tl.predict_many(words["proxy"], top_k=4)]
heavy = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
print(json.dumps({"package_file": khmer_translit.__file__, "numpy": [numpy.__version__, numpy.__file__],
                  "info": tl.info(), "single": one, "edge": er, "dev_first": a, "dev_second": b, "proxy": proxy,
                  "peak_rss_mib_with_batches": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "heavy_modules": heavy}, ensure_ascii=False))
"""

PERF = NO_NET + r"""
import json, resource, sys, time
t0 = time.perf_counter()
from khmer_translit.english import Transliterator
tl = Transliterator.from_pretrained(sys.argv[1])
load_s = time.perf_counter() - t0
rss_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
words = json.loads(open(sys.argv[2], encoding="utf-8").read())["dev"]
for w in words:
    tl.predict(w)                                           # one warm pass
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] * 1e3
lat = []
for _ in range(3):
    for w in words:
        s = time.perf_counter(); tl.predict(w); lat.append(time.perf_counter() - s)
lat4 = []
for w in words:
    s = time.perf_counter(); tl.predict(w, top_k=4); lat4.append(time.perf_counter() - s)
s = time.perf_counter()
for i in range(0, len(words), 16):
    tl.predict_many(words[i:i + 16])
b16 = (time.perf_counter() - s) / len(words)
print(json.dumps({"load_s": load_s, "rss_after_load_mib": rss_load,
                  "single_answer_ms": {"p50": q(lat, .5), "p95": q(lat, .95), "max": max(lat) * 1e3, "n": len(lat)},
                  "top4_ms": {"p50": q(lat4, .5), "p95": q(lat4, .95), "n": len(lat4)},
                  "batch16_ms_per_word": b16 * 1e3,
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def parity_inputs() -> dict:
    """Dev inputs (the `input` field only) and the recorded proxy-track outputs of letters_s0 (no references)."""
    dev = [json.loads(x)["input"] for x in open(DATA / "dev_references.jsonl", encoding="utf-8")]
    rec = json.loads((DATA / "results" / "predictions.json").read_text(encoding="utf-8"))["letters_s0"]
    proxy = rec["final_proxy"]
    return {"dev": dev, "proxy_words": [v["input"] for v in proxy.values()],
            "proxy_recorded": [[c[0] for c in v["candidates"]] for v in proxy.values()],
            "proxy_scores": [[c[1] for c in v["candidates"]] for v in proxy.values()]}


def historical_outputs(words: list[str]) -> list[dict]:
    """The same checkpoint through the package source at the E02 training commit (the recorded implementation)."""
    commit = json.loads((RUN / "provenance.json").read_text())["git_commit"]
    with tempfile.TemporaryDirectory(prefix="e02-hist-") as d:
        arc = Path(d) / "src.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(arc), commit, "src/khmer_translit"], cwd=ROOT,
                       check=True)
        with tarfile.open(arc) as t:
            t.extractall(d, filter="data")
        code = ("import json,sys; from khmer_translit.english import Transliterator as T; "
                "tl=T.from_pretrained(sys.argv[1]); ws=json.load(open(sys.argv[2]));"
                "print(json.dumps([r.to_dict() for r in tl.predict_many(ws, top_k=4)], ensure_ascii=False))")
        (Path(d) / "w.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        p = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(Path(d) / 'src')!r}); "
                            + code, str(CHECKPOINT), str(Path(d) / "w.json")], capture_output=True, text=True,
                           check=True, cwd=d)
        return json.loads(p.stdout)


def verify() -> dict:
    """Install the built wheel in a genuinely fresh venv and check behaviour, parity, isolation and cost."""
    check_checkpoint()
    wheel = next((DIST / "wheel").glob("*.whl"))
    built = json.loads((REPORTS / "build.json").read_text())
    if sha(wheel) != built["sha256"][wheel.name]:
        raise RuntimeError("the wheel differs from the one `build` recorded")
    words = parity_inputs()
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wheel": wheel.name,
           "wheel_sha256": sha(wheel), "built_from_commit": built["commit"],
           "hardware": {"cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                     if l.startswith("model name")), platform.processor()),
                        "logical_cpus": os.cpu_count(), "os": platform.platform(),
                        "note": "laptop, CPU only, other programs running; NumPy default threading"}}
    checks: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="e02-install-") as d:
        d = Path(d)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "KHMER_TRANSLIT", "CONDA"))}
        env |= {"CUDA_VISIBLE_DEVICES": "", "HOME": os.environ.get("HOME", str(d))}
        run = lambda *c, stdin=None, cwd=d: subprocess.run(list(map(str, c)), cwd=cwd, env=env, input=stdin,  # noqa: E731
                                                           capture_output=True, text=True)
        r = run("uv", "venv", "-q", "--python", "3.10", d / "venv")
        py, site = d / "venv" / "bin" / "python", None
        cfg = (d / "venv" / "pyvenv.cfg").read_text()
        out["venv"] = {"created": r.returncode == 0, "pyvenv_cfg": cfg.strip().splitlines()}
        r = run("uv", "pip", "install", "--offline", "--python", py, wheel)
        how = "offline from the uv cache"
        if r.returncode:
            r = run("uv", "pip", "install", "--python", py, wheel)      # NumPy from PyPI (~16 MiB) if not cached
            how = "from PyPI"
        out["install"] = {"returncode": r.returncode, "how": how, "log": (r.stderr + r.stdout)[-800:]}
        if r.returncode:
            out["ok"] = False
            return write_json(REPORTS / "install_verify.json", out)
        site = next((d / "venv" / "lib").glob("python3*/site-packages"))
        pth = {p.name: p.read_text(errors="replace")[:200] for p in site.glob("*.pth")}
        frz = run("uv", "pip", "freeze", "--python", py).stdout.splitlines()
        out["environment"] = {"packages": frz, "pth_files": pth, "system_site_packages":
                              "include-system-site-packages = true" in cfg,
                              "pth_points_into_checkout": any(str(ROOT) in v for v in pth.values()),
                              "pythonpath_set": "PYTHONPATH" in env}
        checks["clean_environment"] = {"ok": not out["environment"]["system_site_packages"]
                                       and not out["environment"]["pth_points_into_checkout"]
                                       and not out["environment"]["pythonpath_set"]
                                       and sorted(x.replace(" @ ", "==").split("==")[0] for x in frz) == ["khmer-translit", "numpy"]}
        work = d / "elsewhere"
        work.mkdir()
        wrap = work / "cli_wrap.py"
        wrap.write_text(f"HEAVY = {HEAVY!r}\n" + CLI_WRAP, encoding="utf-8")
        model = d / "model" / "model.npz"
        model.parent.mkdir()
        shutil.copy(CHECKPOINT, model)
        cli = lambda *a, stdin=None: run(py, "-I", wrap, *a, stdin=stdin, cwd=work)  # noqa: E731

        r = cli("predict", "--model", model, "--word", "coffee", "--json", "--top-k", "3")
        j = json.loads(r.stdout) if r.returncode == 0 else {}
        checks["cli_single_json"] = {"ok": r.returncode == 0 and j.get("status") == "ok"
                                     and 1 <= len(j.get("candidates", [])) <= 3 and j.get("backend") == "e02"
                                     and "HEAVY_MODULES=\n" in r.stderr + "\n",
                                     "output": j, "stderr_tail": r.stderr[-200:]}
        batch = "computer\n\nhello world\nJohnson\nកាហ្វេ\n" + "b" * 33 + "\n" + "b" * 32 + "\ncafé\n"
        r = cli("predict", "--model", model, "--input", "-", "--jsonl", stdin=batch)
        rows = [json.loads(x) for x in r.stdout.splitlines()]
        want = ["ok", "empty", "unsupported_multiword", "ok", "unsupported_characters", "too_long", "ok", "ok"]
        checks["cli_batch_jsonl_alignment"] = {
            "ok": r.returncode == 1 and [x["id"] for x in rows] == list(range(1, 9))
            and [x["input"] for x in rows] == batch.split("\n")[:-1] and [x["status"] for x in rows] == want
            and rows[5]["candidates"] == [], "statuses": [x["status"] for x in rows], "exit": r.returncode}
        r1 = cli("predict", "--model", model, "--input", "-", "--jsonl", "--top-k", "4",
                 stdin="banana\ntelevision\nSmith\n")
        r2 = cli("predict", "--model", model, "--input", "-", "--jsonl", "--top-k", "4",
                 stdin="banana\ntelevision\nSmith\n")
        checks["cli_repeat_determinism"] = {"ok": r1.returncode == 0 and r1.stdout == r2.stdout}
        bad = work / "corrupt.npz"
        bad.write_bytes(b"not a model")
        r = cli("predict", "--model", bad, "--word", "coffee")
        checks["corrupt_checkpoint_refused"] = {"ok": r.returncode == 2, "stderr": r.stderr.strip()[-200:]}
        trunc = work / "truncated.npz"
        trunc.write_bytes(CHECKPOINT.read_bytes()[: CHECKPOINT.stat().st_size // 2])
        r = cli("predict", "--model", trunc, "--word", "coffee")
        checks["truncated_checkpoint_refused"] = {"ok": r.returncode == 2, "stderr": r.stderr.strip()[-200:]}
        r = cli("predict", "--model", work / "absent.npz", "--word", "coffee")
        checks["missing_checkpoint_refused"] = {"ok": r.returncode == 2, "stderr": r.stderr.strip()[-200:]}
        r = cli("predict", "--model", model, "--sha256", CHECKPOINT_SHA, "--word", "coffee")
        r_bad = cli("predict", "--model", model, "--sha256", "0" * 64, "--word", "coffee")
        checks["expected_hash_enforced"] = {"ok": r.returncode == 0 and r_bad.returncode == 2}
        r = cli("predict", "--model", model, "--word", "coffee", "--top-k", "9")
        checks["top_k_bound"] = {"ok": r.returncode == 2, "stderr": r.stderr.strip()[-160:]}
        r = cli("inspect", "--model", model)
        j = json.loads(r.stdout) if r.returncode == 0 else {}
        checks["cli_inspect"] = {"ok": r.returncode == 0 and j.get("mode") == "letters", "output": j}

        wfile = work / "words.json"
        wfile.write_text(json.dumps({"dev": words["dev"], "proxy": words["proxy_words"]}, ensure_ascii=False),
                         encoding="utf-8")
        api = work / "api.py"
        api.write_text(f"HEAVY = {HEAVY!r}\n" + API, encoding="utf-8")
        r = run(py, "-I", api, model, wfile, cwd=work)
        if r.returncode:
            checks["api"] = {"ok": False, "stderr": r.stderr[-800:]}
        else:
            a = json.loads(r.stdout.splitlines()[-1])
            edge = {e["input"]: (e["status"], [c["text"] for c in e["candidates"]]) for e in a["edge"]}
            checks["api_single"] = {"ok": a["single"]["status"] == "ok" and len(a["single"]["candidates"]) == 3,
                                    "output": [c["text"] for c in a["single"]["candidates"]]}
            checks["api_edge_inputs"] = {
                "ok": edge[""][0] == "empty" and edge["new york"][0] == "unsupported_multiword"
                and edge["c3po"][0] == "unsupported_characters" and edge["ខ្មែរ"][0] == "unsupported_characters"
                and edge["a" * 33] == ("too_long", []) and edge["a" * 32][0] == "ok"
                and edge["COFFEE"][1] == checks["api_single"]["output"][:1], "statuses": edge}
            checks["api_determinism"] = {"ok": a["dev_first"] == a["dev_second"]}
            checks["api_imports_from_fresh_venv"] = {"ok": str(site) in a["package_file"]
                                                     and str(site) in a["numpy"][1], "package_file":
                                                     a["package_file"], "numpy": a["numpy"]}
            checks["no_heavy_or_speech_modules"] = {"ok": not a["heavy_modules"], "found": a["heavy_modules"]}
            got = [[c["text"] for c in r_["candidates"]] for r_ in a["proxy"]]
            gs = [[c["log_score"] for c in r_["candidates"]] for r_ in a["proxy"]]
            gap = max(abs(x - y) for g, w in zip(gs, words["proxy_scores"]) for x, y in zip(g, w))
            checks["parity_recorded_e02_outputs"] = {
                "ok": got == words["proxy_recorded"] and gap < 1e-4, "items": len(got),
                "candidate_lists_equal": sum(g == w for g, w in zip(got, words["proxy_recorded"])),
                "max_abs_log_score_gap": gap,
                "what": "letters_s0 top-4 beam outputs recorded by english_phonetic_eval.py on the 400-item "
                        "exploratory proxy track (data/e02/results/predictions.json); inputs and outputs only"}
            hist = historical_outputs(words["dev"])
            checks["parity_e02_training_commit_code"] = {
                "ok": [h["candidates"] for h in hist] == [x["candidates"] for x in a["dev_first"]],
                "items": len(hist), "what": f"package source at {json.loads((RUN / 'provenance.json').read_text())['git_commit'][:7]} (the E02 training commit) vs the installed wheel, dev inputs, top-4 with scores"}
            perf = work / "perf.py"
            perf.write_text(PERF, encoding="utf-8")
            p = run(py, "-I", perf, model, wfile, cwd=work)
            out["performance"] = json.loads(p.stdout.splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr}
            out["performance"]["conditions"] = (
                f"own fresh process; {len(words['dev'])} dev words; the single answer (beam 4, top 1) timed over 3 "
                "passes one word at a time after a warm pass; top-4 one pass; batches of 16 one pass; "
                "sockets disabled")
            out["performance"]["peak_rss_mib_api_check_with_400_word_batches"] = a["peak_rss_mib_with_batches"]
            out["info"] = a["info"]
    out["checks"] = checks
    out["ok"] = all(c["ok"] for c in checks.values())
    out["offline_method"] = ("Python-level: socket.socket/create_connection/getaddrinfo replaced before the "
                             "package is imported (OS-level network namespaces are not permitted on this machine)")
    return write_json(REPORTS / "install_verify.json", out)


# ------------------------------------------------------------------ bundle
def bundle(replace: bool = False) -> dict:
    """Assemble `dist/<NAME>/` around the built wheel; refuse to replace a previously hashed archive silently."""
    check_checkpoint()
    arc = DIST.parent / f"{NAME}.tar.gz"
    hist = DIST.parent / f"{NAME}.replaced.jsonl"
    if arc.exists():
        if not replace:
            raise FileExistsError(f"{arc} exists (sha256 {sha(arc)}); pass --replace to rebuild it (the old hash "
                                  f"is logged in {hist.name})")
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps({"replaced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "sha256": sha(arc)}) + "\n")
    wheel_dir = DIST / "wheel"
    if not any(wheel_dir.glob("*.whl")):
        raise FileNotFoundError("run `build` first")
    for p in DIST.iterdir():
        if p.name != "wheel":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    (DIST / "checkpoint").mkdir()
    shutil.copy(CHECKPOINT, DIST / "checkpoint" / "model.npz")
    for p in sorted(DOCS.glob("*.md")):
        shutil.copy(p, DIST / p.name)
    shutil.copy(DOCS / "requirements.txt", DIST / "requirements.txt")
    shutil.copy(ROOT / "LICENSE", DIST / "LICENSE")                      # code: PolyForm Noncommercial 1.0.0
    ev = DIST / "evidence"
    ev.mkdir()
    for p in (REPORTS / "provenance_facts.json", REPORTS / "build.json", REPORTS / "install_verify.json",
              E02 / "results.json", E02 / "reports" / "runtime.json"):
        if p.exists():
            shutil.copy(p, ev / p.name)
    hf = DIST / "huggingface"
    hf.mkdir()
    shutil.copy(DOCS / "MODEL_CARD.md", hf / "README.md")
    shutil.copy(CHECKPOINT, hf / "model.npz")
    shutil.copy(DOCS / "LICENSE-weights.md", hf / "LICENSE")               # weights: CC BY-NC-SA 4.0
    for f in ("BENCHMARK_CARD.md", "PROVENANCE_AUDIT.md", "NOTICES.md"):
        shutil.copy(DOCS / f, hf / f)
    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    bad = [str(p.relative_to(DIST)) for p in files
           if p.suffix in FORBIDDEN_SUFFIXES or any(n in p.name.lower() for n in FORBIDDEN_NAMES)]
    if bad:
        raise RuntimeError(f"bundle holds files it must not: {bad}")
    (DIST / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(DIST)}\n" for p in files), encoding="utf-8")
    with tarfile.open(arc, "w:gz") as t:
        t.add(DIST, arcname=NAME)
    (DIST.parent / f"{NAME}.tar.gz.sha256").write_text(f"{sha(arc)}  {arc.name}\n", encoding="utf-8")
    out = {"bundle": str(DIST.relative_to(ROOT)), "archive": str(arc.relative_to(ROOT)), "archive_sha256": sha(arc),
           "archive_bytes": arc.stat().st_size, "checkpoint_sha256": sha(DIST / "checkpoint" / "model.npz"),
           "files": {str(p.relative_to(DIST)): sha(p) for p in files}}
    return write_json(REPORTS / "bundle.json", out)


STEPS = {"audit": audit, "build": build, "verify": verify, "bundle": bundle}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--replace", action="store_true", help="bundle: rebuild an existing archive (old hash logged)")
    a = ap.parse_args()
    res = STEPS[a.step](**({"replace": a.replace} if a.step == "bundle" else {}))
    print(json.dumps(res, ensure_ascii=False, default=str)[:3000])
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
