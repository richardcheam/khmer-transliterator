"""E10 release preparation for the ByT5 research preview: export, real-checkpoint verification, local bundle.

Nothing here publishes anything. The bundle and the exported adapter stay local (`dist/`, `models/e10/release/`).

Steps:
    export     the frozen release-candidate checkpoint → models/e10/release/adapter/ (safetensors + config)
    verify     parity with E09/E10 cached predictions, offline loading, refusal checks, latency and memory
               → experiments/E10-release-cycle/reports/release_verify.json
    bundle     wheel + sdist, adapter, documents, Hugging Face folder, SHA256SUMS, .tar.gz → dist/
    install    fresh venv outside the repository (uv, offline from the local cache), CLI checks there
               → experiments/E10-release-cycle/reports/install_check.json

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10_release.py verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import e10  # noqa: E402  (patches E07/E09 bookkeeping to E10)

ROOT = e10.ROOT
VERSION = "0.2.0"
NAME = f"khmer-translit-byt5-{VERSION}"
RELEASE_DOCS = ROOT / "release" / "byt5"          # committed documents
ADAPTER_OUT = ROOT / "models" / "e10" / "release" / "adapter"
DIST = ROOT / "dist" / NAME
REPORTS = e10.REPORTS
BASE_LOCAL = ROOT / "models" / "e06" / "byt5-small"
SMOKE = ["coffee", "london", "taxi", "doctor", "computer", "Elizabeth", "o'neil", "wi-fi"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def frozen() -> dict:
    return json.loads((REPORTS / "candidates_frozen.json").read_text())


def rc_dir(rc: str) -> Path:
    return e10.E08_DIR if rc == "e08" else e10.MODELS / rc


def export() -> dict:
    """Write the release candidate's adapter tensors as safetensors, with a config recording its provenance."""
    import torch
    from khmer_translit import byt5
    fz = frozen()
    rc = fz["release_candidate"]
    d = rc_dir(rc)
    done = json.loads((d / "done.json").read_text())
    if sha(d / "best.pt") != done["best_sha256"]:
        raise RuntimeError("best.pt changed")
    state = torch.load(d / "best.pt", weights_only=True)
    cfg = done["cfg"]
    out = byt5.export_adapter(state, ADAPTER_OUT, {
        "name": NAME, "version": VERSION, "status": "research preview; unverified phonetic quality",
        "task": "one English word or name → Khmer script approximating its English (General American) pronunciation",
        "base": {"repo": byt5.BASE_REPO, "revision": byt5.BASE_REVISION,
                 "weights_file": "pytorch_model.bin", "weights_sha256": e10.B.BASE_SHA256["pytorch_model.bin"]},
        "adapter": {"type": "LoRA (khmer_translit.byt5.LoRALinear)", "r": cfg["adapter"]["r"],
                    "alpha": cfg["adapter"]["alpha"], "targets": cfg["adapter"]["targets"],
                    "modules": {"attn": ["q", "k", "v", "o"], "ffn": ["wi_0", "wi_1", "wo"]}},
        "decode": {"single": "greedy", "candidate_beam": 4, "candidate_search": "early_stopping='never', "
                   "length_penalty 0", "max_new_tokens": byt5.MAX_NEW_TOKENS, "max_input_bytes": byt5.MAX_SRC_BYTES,
                   "batch_size": 32},
        "input": "khmer_translit.phonetic_source.normalize_word (NFKC, lower case, accents, apostrophes and hyphens "
                 "removed; [a-z]+; ≤ 32 letters), then UTF-8 bytes + 3 and EOS",
        "provenance": {"candidate": rc, "checkpoint_sha256": done["best_sha256"], "update": done["best"]["update"],
                       "training_rows": done["rows"], "data_arm": "E08 R+S (sha256 " + e10.ARM_SHA + ")",
                       "recipe": {k: cfg[k] for k in ("lr", "batch", "max_updates", "warmup", "schedule", "seed")
                                  if k in cfg} | {"sampler": cfg.get("sampler", "row_level")},
                       "experiments": ["E07", "E08", "E09", "E10"]}})
    e10.set_state("export", "done", adapter_sha256=out["adapter_sha256"], candidate=rc)
    return out


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def verify() -> dict:
    """Real-checkpoint checks of the packaged backend (one model in memory; run when no training is active)."""
    from khmer_translit import byt5
    os.environ["HF_HUB_OFFLINE"] = "1"                        # any download attempt fails loudly
    fz = frozen()
    rc = fz["release_candidate"]
    e10.register_checkpoints()
    out = {"candidate": rc, "adapter_sha256": json.loads((ADAPTER_OUT / byt5.CONFIG_FILE).read_text())["adapter_sha256"]}
    t = time.perf_counter()
    tl = byt5.ByT5Transliterator.from_pretrained(ADAPTER_OUT, base=BASE_LOCAL, offline=True, threads=6)
    out["load_seconds"] = round(time.perf_counter() - t, 2)
    out["rss_after_load_mb"] = round(peak_rss_mb())
    # parity with the cached research predictions (same order and batching: dev rows, greedy bs 32, beam bs 16)
    rows = e10.e09.dev_rows()
    s = e10.e09.checkpoint_sha(rc)
    cache = lambda d: {r["id"]: r for r in e10.e07.load_jsonl(e10.e09.cache_path(rc, d, "dev", s, rows))}  # noqa: E731
    g, b = cache("greedy"), cache("beam4_can")
    words = [w for _, w in rows]
    t = time.perf_counter()
    single = tl.predict_many(words)
    out["batch_seconds_276_words_greedy"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    four = tl.predict_many(words, top_k=4)
    out["batch_seconds_276_words_top4"] = round(time.perf_counter() - t, 2)
    mism1 = [i for (i, _), r in zip(rows, single) if r.text != (e10.e09.top1(g[i]["cands"]) or "")]
    mism4 = [i for (i, _), r in zip(rows, four) if [c.text for c in r.candidates] != e10.e09.cand_list(b[i]["cands"])[:4]]
    out["parity_dev"] = {"items": len(rows), "single_mismatches": len(mism1), "top4_mismatches": len(mism4),
                         "examples": (mism1 + mism4)[:5]}
    lat = []
    for w in SMOKE * 3:
        t = time.perf_counter()
        tl.predict(w)
        lat.append(time.perf_counter() - t)
    out["batch1_greedy_ms"] = {"median": round(median(lat) * 1000, 1), "max": round(max(lat) * 1000, 1)}
    lat4 = []
    for w in SMOKE:
        t = time.perf_counter()
        tl.predict(w, top_k=4)
        lat4.append(time.perf_counter() - t)
    out["batch1_top4_ms_median"] = round(median(lat4) * 1000, 1)
    out["refusals"] = {w: tl.predict(w).status for w in ("", "new york", "ខ្មែរ", "b" * 33, "c3po")}
    out["smoke"] = {w: tl.predict(w).text for w in SMOKE}
    out["peak_rss_mb"] = round(peak_rss_mb())
    out["threads"] = 6
    out["offline"] = "HF_HUB_OFFLINE=1 for the whole check; local base folder; no network access needed"
    out["ok"] = not mism1 and not mism4 and all(v != "ok" for v in out["refusals"].values())
    e10.set_state("release_verify", "done" if out["ok"] else "failed")
    return e10.e07.write_json(REPORTS / "release_verify.json", out) and out


def build_wheel() -> list[Path]:
    wd = DIST / "wheel"
    shutil.rmtree(wd, ignore_errors=True)
    subprocess.run(["uv", "build", "--out-dir", str(wd)], cwd=ROOT, check=True, capture_output=True)
    return sorted(wd.iterdir())


def bundle() -> dict:
    """Assemble the local release bundle (no upload): wheel/sdist, adapter, documents, HF folder, checksums."""
    from khmer_translit import byt5
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    wheels = build_wheel()
    (DIST / "wheel" / ".gitignore").unlink(missing_ok=True)          # written by `uv build`, not a release file
    shutil.copytree(ADAPTER_OUT, DIST / "adapter")
    for p in sorted(RELEASE_DOCS.glob("*.md")):
        shutil.copy(p, DIST / p.name)
    ev = DIST / "evidence"
    ev.mkdir()
    for p in ("results.json", "reports/candidates_frozen.json", "reports/gate.json", "reports/diagnosis.json",
              "reports/release_verify.json", "reports/install_check.json", "confirmation/frozen.json",
              "confirmation/input_manifest.jsonl", "review/review_manifest.json"):
        src = e10.EXP / p
        if src.exists():
            shutil.copy(src, ev / src.name)
    hf = DIST / "huggingface"
    hf.mkdir()
    shutil.copy(RELEASE_DOCS / "MODEL_CARD.md", hf / "README.md")
    for f in (byt5.ADAPTER_FILE, byt5.CONFIG_FILE):
        shutil.copy(ADAPTER_OUT / f, hf / f)
    for f in ("NOTICES.md", "DATA_PROVENANCE.md", "BENCHMARK_CARD.md"):
        shutil.copy(RELEASE_DOCS / f, hf / f)
    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    bad = [str(p) for p in files if p.suffix in (".pt", ".bin", ".npz", ".env") or "__pycache__" in p.parts
           or p.name in ("origin_key.json", "judgments_template.tsv", "review_batch.json")]
    if bad:
        raise RuntimeError(f"bundle holds files it must not: {bad[:5]}")
    sums = "".join(f"{sha(p)}  {p.relative_to(DIST)}\n" for p in files)
    (DIST / "SHA256SUMS").write_text(sums, encoding="utf-8")
    arc = DIST.parent / f"{NAME}.tar.gz"
    with tarfile.open(arc, "w:gz") as t:
        t.add(DIST, arcname=NAME)
    out = {"bundle": str(DIST.relative_to(ROOT)), "archive": str(arc.relative_to(ROOT)), "archive_sha256": sha(arc),
           "archive_bytes": arc.stat().st_size, "files": [str(p.relative_to(DIST)) for p in files],
           "wheels": [p.name for p in wheels], "excluded": "model binaries other than the adapter, base weights, "
           "restricted data (mined pairs, references, review key/batch), caches, secrets"}
    (DIST.parent / f"{NAME}.tar.gz.sha256").write_text(f"{out['archive_sha256']}  {arc.name}\n")
    e10.set_state("bundle", "done", archive_sha256=out["archive_sha256"])
    return e10.e07.write_json(REPORTS / "bundle.json", out) and out


def install() -> dict:
    """Install the built wheel with the `byt5` extra in a fresh venv outside the repository (offline, from uv's
    cache), then run the CLI there with HF_HUB_OFFLINE=1 on the exported adapter and the local base."""
    wheel = next((DIST / "wheel").glob("*.whl"))
    out = {"wheel": wheel.name, "wheel_sha256": sha(wheel)}
    with tempfile.TemporaryDirectory(prefix="kt-install-") as d:
        d = Path(d)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH"))}
        env |= {"HF_HUB_OFFLINE": "1", "HF_HOME": str(d / "hf"), "CUDA_VISIBLE_DEVICES": ""}
        run = lambda *c, **kw: subprocess.run(list(c), cwd=d, env=env, capture_output=True, text=True, **kw)  # noqa: E731
        r = run("uv", "venv", "--python", "3.10", str(d / "venv"))
        out["venv"] = r.returncode == 0
        py = d / "venv" / "bin" / "python"
        r = run("uv", "pip", "install", "--offline", "--python", str(py), f"khmer-translit[byt5] @ {wheel.as_uri()}")
        out["install"] = {"returncode": r.returncode, "tail": (r.stderr or r.stdout)[-600:]}
        out["byt5_extra_source"] = "resolved offline from uv's cache"
        if r.returncode:
            # uv's cache keeps torch's files but not the index metadata needed to resolve it offline, and fetching
            # torch (1-3 GB) is out of budget. Install the wheel and its base dependency offline, and borrow only
            # the extra's packages from the research environment through a .pth path entry. Such an entry
            # does not run the research environment's own .pth hooks, so its editable khmer_translit stays out.
            r = run("uv", "pip", "install", "--offline", "--python", str(py), wheel.as_uri())
            out["install"] = {"returncode": r.returncode, "tail": (r.stderr or r.stdout)[-600:],
                              "byt5_extra_offline": "unresolvable from uv's cache (index metadata missing)"}
            if r.returncode:
                return e10.e07.write_json(REPORTS / "install_check.json", out) and out
            research = next((ROOT / ".venv" / "lib").glob("python3*/site-packages"))
            site = next((d / "venv" / "lib").glob("python3*/site-packages"))
            (site / "zz_research_deps.pth").write_text(f"{research}\n", encoding="utf-8")
            out["byt5_extra_source"] = (f"torch/transformers/safetensors borrowed from {research.relative_to(ROOT)} "
                                        "via a .pth path entry; khmer_translit itself from the wheel")
        cli = d / "venv" / "bin" / "khmer-translit"
        (d / "words.txt").write_text("\n".join(SMOKE + ["", "new york"]) + "\n", encoding="utf-8")
        t = time.perf_counter()
        r = run(str(cli), "predict", "--backend", "byt5", "--model", str(ADAPTER_OUT), "--base", str(BASE_LOCAL),
                "--offline", "--threads", "6", "--input", "words.txt", "--jsonl")
        out["cli_seconds"] = round(time.perf_counter() - t, 1)
        rows = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
        out["cli"] = {"returncode": r.returncode, "rows": len(rows), "ids": [x["id"] for x in rows],
                      "statuses": [x["status"] for x in rows], "outputs": {x["input"]: x["candidates"][0]["text"]
                                                                           for x in rows if x["candidates"]}}
        r2 = run(str(cli), "predict", "--backend", "byt5", "--model", str(ADAPTER_OUT), "--base", str(BASE_LOCAL),
                 "--offline", "--word", "coffee", "--top-k", "4", "--json")
        out["cli_top4"] = {"returncode": r2.returncode, "ok": '"candidates"' in r2.stdout}
        r3 = run(str(cli), "predict", "--backend", "byt5", "--model", str(ADAPTER_OUT), "--offline", "--word", "coffee")
        out["offline_without_base_or_cache"] = {"returncode": r3.returncode, "stderr": r3.stderr.strip()[-300:]}
        r4 = run(str(py), "-c", "import khmer_translit, khmer_translit.translit as t; "
                 "print(all(hasattr(t, f) for f in ('guess','guess_many','caps_reads_as_word','configure','word_tag',"
                 "'pick','load')), khmer_translit.__file__)")
        out["public_api_present"] = r4.stdout.strip()
        out["imports_from_wheel"] = str(site) in r4.stdout if "site" in locals() else str(d) in r4.stdout
        out["downloads"] = {"hf_home_files": [str(p.relative_to(d)) for p in (d / "hf").rglob("*") if p.is_file()]
                            if (d / "hf").exists() else []}
        ref = json.loads((REPORTS / "release_verify.json").read_text())["smoke"]
        out["matches_repo_outputs"] = all(out["cli"]["outputs"].get(w) == ref[w] for w in SMOKE if ref.get(w))
    out["ok"] = (out["install"]["returncode"] == 0 and out["cli"]["returncode"] == 1 and out["cli"]["rows"] == 10
                 and out["cli"]["statuses"][-2:] == ["empty", "unsupported_multiword"] and out["cli_top4"]["ok"]
                 and out["offline_without_base_or_cache"]["returncode"] == 2 and out["matches_repo_outputs"]
                 and not out["downloads"]["hf_home_files"] and out["imports_from_wheel"])
    e10.set_state("install_check", "done" if out["ok"] else "failed")
    return e10.e07.write_json(REPORTS / "install_check.json", out) and out


STEPS = {"export": export, "verify": verify, "bundle": bundle, "install": install}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    a = ap.parse_args()
    print(json.dumps(STEPS[a.step](), ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
