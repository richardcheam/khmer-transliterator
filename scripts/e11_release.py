"""E11 release preparation (ByT5-small + LoRA on the real-data arm R) and a small human-judged candidate-coverage check.

Nothing here trains, publishes, tags or changes serving. The exported adapter, the bundle and every candidate or
judgment stay local (`models/e11/release/`, `dist/`, `data/e11/coverage/`); reports with hashes and aggregates go to
`experiments/E11-real-data-byt5/release/` and `.../coverage/`.

Steps:
    identity      E11's checkpoint, base pin, adapter configuration, data hashes and lineage → release/identity.json
    cov-select    freeze the coverage rule and sample before any candidate is read → coverage/frozen.json
    cov-generate  E11 greedy + canonical beam-4 (E09 settings; cached generations reused by hash) → coverage/generation.json
    cov-batches   reuse identical earlier judgments; one blinded click-review batch of the rest (≤ 120) → coverage/batches.json
    cov-import    validate the owner's answers; the coverage diagnostic → coverage/results.json
    export        best.pt LoRA tensors → models/e11/release/adapter (safetensors); tensor-by-tensor completeness
    verify        packaged backend on the real adapter: dev/eval parity, refusals, errors, reload, timing (offline)
    bundle        dist/khmer-translit-byt5-e11-0.2.0/ (wheel, sdist, adapter, documents, HF folder, checksums, .tar.gz)
    install       fresh venv outside the repository, CPU-only torch from the official index, CLI/API checks with
                  networking disabled → release/install_check.json
    status

Model-heavy steps (cov-generate, verify, install) refuse to start while another process keeps the CPU busy (for
example a TTS benchmark), so they never distort someone else's measurement; pass --force to override.

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e11_release.py identity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import e06  # noqa: E402
import e07  # noqa: E402
import e09  # noqa: E402
import e10  # noqa: E402  (pure helpers only)
import e11  # noqa: E402
import e13  # noqa: E402  (frozen-manifest and review helpers; bookkeeping overridden below)
from khmer_translit import phonetic_source as ps  # noqa: E402

ROOT = e06.ROOT
EXP = ROOT / "experiments" / "E11-real-data-byt5"
REL = EXP / "release"
COV = EXP / "coverage"
COV_DATA = ROOT / "data" / "e11" / "coverage"           # git-ignored: candidates, batch, origin key, judgments
LEDGER = REL / "ledger.jsonl"
STATE_FILE = REL / "state.json"
E11_DIR = ROOT / "models" / "e11" / "r_s0"
E11_BEST_SHA = "286292a4f21131d0dd238d48919485192b0b0eb268d005bbc804d3ff78229713"
ARM_R = ROOT / "data" / "e08" / "training" / "arm_R.tsv"
BASE_LOCAL = ROOT / "models" / "e06" / "byt5-small"
E11_REVIEW = ROOT / "data" / "e11" / "review"
E13_COLL_DATA = ROOT / "data" / "e13" / "collection"
OTHER_CHECKPOINTS = {"e08": ROOT / "models" / "e08" / "rs_s0", "e10": ROOT / "models" / "e10",
                     "e12": ROOT / "models" / "e12" / "long_s0", "e13": ROOT / "models" / "e13" / "pron_s0"}

VERSION = "0.2.0"                                       # package version (code unchanged); the bundle id names E11
BUILD = "e11-r_s0"
NAME = f"khmer-translit-byt5-e11-{VERSION}"
ADAPTER_OUT = ROOT / "models" / "e11" / "release" / "adapter"
DOCS = ROOT / "release" / "byt5-e11"                    # committed documents
DIST = ROOT / "dist" / NAME
HF_REPO = "ric912/en-km-phonetic-byt5-lora"             # proposed; not created
THREADS = 6
SMOKE = ["coffee", "london", "taxi", "doctor", "computer", "elizabeth", "o'neil", "wi-fi"]

COV_SEED = "e11-coverage-v1"
COV_EXTRA = 6                       # historical-dev irregular words without earlier judgments
MAX_NEW_JUDGMENTS = 120
BEAM_K = 4
LABELS = "ABCDE"
COV_RULE = (
    "Representative irregular development sample, fixed before any candidate is read: "
    "(A) every irregular family of E13's development collection (20; drawn by E13's declared pattern sampler from "
    "fresh pools, judged once on E11 greedy and one canonical-beam alternative); "
    "(B) every word of E11's 40-word representative dev review that meets E06's registered difficulty rule "
    "(e06.is_difficult on the first CMUdict reading); E11's 20-word disagreement sample is failure-enriched and "
    "excluded; "
    f"(C) {COV_EXTRA} further historical-dev families meeting the same rule, unambiguous in CMUdict, outside every "
    "E11 review word and every E12/E13 test and E13 collection family, in seeded order "
    f"(sha256 of '{COV_SEED}:<word>'). Nothing is selected from test sets or from model failures.")

e07.LEDGER = LEDGER
e07.STATE = ROOT / "state" / "e11_release"
e09.LEDGER = LEDGER
e09.CHECKPOINTS["e11"] = E11_DIR
e11.STATE_FILE = STATE_FILE
e11.LEDGER = LEDGER
state, set_state = e11.state, e11.set_state


def sha(p: Path) -> str:
    return e06.sha(p)


def write(p: Path, obj) -> dict:
    e07.write_json(p, obj)
    return obj


def peak_rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def busy(own: set[int] | None = None, threshold: float = 50.0) -> list[str]:
    """Other processes using more than `threshold`% CPU right now (sampled over one second), as short descriptions.

    Examples:
        >>> isinstance(busy(), list)
        True
    """
    own = own or {os.getpid(), os.getppid()}
    r = subprocess.run(["top", "-b", "-n", "2", "-d", "1", "-w", "200"], capture_output=True, text=True)
    blocks = r.stdout.split("\n\n")
    rows, out = blocks[-1].splitlines() if blocks else [], []
    for line in rows:
        f = line.split(None, 11)
        if len(f) < 12 or not f[0].isdigit() or int(f[0]) in own:
            continue
        try:
            cpu = float(f[8].replace(",", "."))
        except ValueError:
            continue
        if cpu > threshold:
            out.append(f"pid {f[0]} {cpu:.0f}% {f[11][:60]}")
    return out


def require_quiet(step: str, force: bool) -> None:
    """Refuse a model-heavy step while another process is busy (it would distort both measurements)."""
    others = busy()
    if others and not force:
        set_state(step, "deferred", busy=others)
        raise SystemExit(f"{step}: deferred, other CPU work is running ({'; '.join(others)}); rerun when it is done "
                         "or pass --force")


# ------------------------------------------------------------------ 1. identity and lineage
def arm_counts() -> dict:
    """Rows of arm R by provenance and part, and the optimisation hash E11 recorded (`e07.rows_hash` of its rows)."""
    lines = ARM_R.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, x.split("\t"))) for x in lines[1:]]
    opt, _ = e11.arm_rows()
    return {"by_provenance_part": dict(Counter(f"{r['provenance']}:{r['part']}" for r in rows)),
            "optimisation_rows": sum(r["part"] == "train" for r in rows),
            "words": len({r["word"] for r in rows if r["part"] == "train"}),
            "optimisation_hash_recomputed": e07.rows_hash(opt)}


def identity() -> dict:
    """Verify that the release candidate is exactly E11 `r_s0` update 3,000 and record everything inference needs."""
    import torch
    done = json.loads((E11_DIR / "done.json").read_text())
    summ = json.loads((E11_DIR / "summary.json").read_text())
    reg = json.loads((EXP / "registration.json").read_text())
    frozen = json.loads((EXP / "reports" / "frozen.json").read_text())
    prov = json.loads((E11_DIR / "provenance.json").read_text())
    got = sha(E11_DIR / "best.pt")
    sd = torch.load(E11_DIR / "best.pt", weights_only=True, map_location="cpu")
    n_params = sum(v.numel() for v in sd.values())
    others = {}
    for k, d in OTHER_CHECKPOINTS.items():
        for p in sorted(d.rglob("best.pt")) if d.exists() else []:
            others[f"{k}:{p.parent.name}"] = sha(p)
    arm = arm_counts()
    base_sha = sha(BASE_LOCAL / "pytorch_model.bin")
    frozen_sha = json.dumps(frozen).count(E11_BEST_SHA) > 0
    checks = {
        "checkpoint_sha256_matches_records": got == E11_BEST_SHA == done["best_sha256"] == summ["best_sha256"],
        "selected_update_3000": done["best"]["update"] == 3000 and done["updates"] == 3000,
        "frozen_report_names_checkpoint": frozen_sha,
        "optimisation_rows_4764": done["rows"] == 4764 == reg["data"]["optimisation_rows"] == arm["optimisation_rows"],
        "arm_file_sha256": sha(ARM_R) == reg["data"]["sha256"],
        "data_hash_matches_run_ident": done["ident"]["data"] == reg["data"]["optimisation_hash"]
        == arm["optimisation_hash_recomputed"],
        "no_ai_mode_rows": not any("ai_mode" in k for k in arm["by_provenance_part"]),
        "fresh_adapter_init": "fresh LoRA adapter" in reg["init"] and "not initialised from E08/E10" in reg["init"],
        "differs_from_other_checkpoints": got not in others.values(),
        "adapter_r16_a32_attn_ffn": done["cfg"]["adapter"] == {"r": 16, "alpha": 32, "dropout": 0.05,
                                                               "targets": ["attn", "ffn"]},
        "lora_tensors_only": all(k.endswith((".A", ".B")) for k in sd) and n_params == summ["params"]["trainable"],
        "base_weights_sha256": base_sha == done["ident"]["base"]["pytorch_model.bin"] == reg["base_sha256"]["pytorch_model.bin"],
    }
    from khmer_translit import byt5
    out = {
        "candidate": {"experiment": "E11", "run": "r_s0", "checkpoint": "models/e11/r_s0/best.pt",
                      "checkpoint_sha256": got, "update": done["best"]["update"], "selection": done["cfg"]["select"],
                      "lora_tensors": len(sd), "lora_parameters": n_params},
        "base": {"repo": byt5.BASE_REPO, "revision": byt5.BASE_REVISION, "local_folder": str(BASE_LOCAL.relative_to(ROOT)),
                 "weights_file": "pytorch_model.bin", "weights_sha256": base_sha,
                 "config_sha256": sha(BASE_LOCAL / "config.json"),
                 "revision_evidence": "experiments/E09-byt5-decoding/registration.json and "
                                      "experiments/E07-byt5-diagnosis/reports/environment.json"},
        "adapter_config": done["cfg"]["adapter"] | {"base_dropout": done["cfg"]["base_dropout"]},
        "recipe": {k: done["cfg"][k] for k in ("optimizer", "lr", "betas", "weight_decay", "grad_clip", "batch",
                                               "warmup", "schedule", "max_updates", "seed", "input")},
        "training_data": {"arm": str(ARM_R.relative_to(ROOT)), "arm_sha256": sha(ARM_R),
                          "optimisation_hash": done["ident"]["data"], **arm,
                          "rows_by_provenance_registered": reg["data"]["rows_by_provenance"]},
        "lineage": {"init": reg["init"], "registered": reg["registered"], "cfg_hash": reg["cfg_hash"],
                    "training_commit": prov["git_commit"], "training_tree_dirty": prov["git_dirty"],
                    "training_diff_sha256": prov["git_diff_sha256"], "source_sha256_registered": reg["source_sha256"],
                    "packages": prov["packages"], "elapsed_hours": done["elapsed_hours"]},
        "other_checkpoints_sha256": others,
        "inference_resources": ["google/byt5-small @ " + byt5.BASE_REVISION + " (config.json, pytorch_model.bin)",
                                "adapter.safetensors + adapter_config.json (exported from best.pt)",
                                "khmer-translit[byt5] (torch, transformers, safetensors); no CMUdict, no tokenizer files"],
        "evidence_gaps": ([f"training ran from a dirty tree at {prov['git_commit'][:8]}; the registered source "
                           "hashes (source_sha256) pin the scripts that ran"] if prov["git_dirty"] else []),
        "checks": checks, "ok": all(checks.values())}
    set_state("identity", "done" if out["ok"] else "failed")
    return write(REL / "identity.json", out)


# ------------------------------------------------------------------ 2. coverage check
def judged_index() -> tuple[dict, dict]:
    """Earlier explicit judgments by (word, intended reading, NFC candidate) under docs/PHONETIC_POLICY.md, from E13's
    development collection and E11's dev review, with their provenance; conflicting duplicates are dropped and listed.

    Returns:
        ({key: {"judgment", "source"}}, {"conflicts": [...], "sources": {...}})
    """
    idx, conflicts = {}, []

    def add(key, j, src):
        if key in idx and idx[key]["judgment"] != j:
            conflicts.append({"key": list(key), "judgments": [idx[key]["judgment"], j]})
            idx[key]["judgment"] = None
        elif key not in idx:
            idx[key] = {"judgment": j, "source": src}

    items = {it["item_id"]: it for it in json.loads((E13_COLL_DATA / "items.json").read_text())["items"]}
    seen, _ = e13.read_judgments(E13_COLL_DATA, list(items.values()))
    for (iid, lb), j in seen.items():
        it = items[iid]
        text = next(c["text"] for c in it["candidates"] if c["label"] == lb)
        add((it["word"], it["intended_reading"], e06.nfc(text)), j, f"e13_collection:{iid}:{lb}")
    for r in read_tsv(E11_REVIEW / "judgments.tsv"):
        if r.get("judgment") in e10.JUDGMENTS:
            add((r["word"], r["intended_reading"], e06.nfc(r["candidate"])), r["judgment"],
                f"e11_review:{r['item_id']}:{r['label']}")
    idx = {k: v for k, v in idx.items() if v["judgment"]}
    meta = {"conflicts": conflicts, "sources": {
        "e13_collection": {"files_sha256": {p.name: sha(p) for p in sorted(E13_COLL_DATA.glob("judgments_*.tsv"))},
                           "provenance": json.loads((E13_COLL_DATA / "judgments.provenance.json").read_text())},
        "e11_review": {"files_sha256": {"judgments.tsv": sha(E11_REVIEW / "judgments.tsv")},
                       "provenance": json.loads((E11_REVIEW / "judgments.tsv.provenance.json").read_text())}},
        "policy": "docs/PHONETIC_POLICY.md", "policy_sha256": sha(e13.POLICY),
        "policy_unchanged_since": "2026-10-28 (commit 4b75ab0), before both judgment sets"}
    return idx, meta


def read_tsv(p: Path) -> list[dict]:
    import csv
    import io
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8")), delimiter="\t"))


def coverage_sample() -> tuple[list[dict], dict]:
    """The frozen rule applied to development inputs only (no candidate or judgment of these words is read)."""
    cmu = ps.cmudict()[0]
    a = [{"id": f"a{n:02d}", "pool": "A_e13_dev_irregular", "word": r["word"], "reading": r["pronunciation"],
          "source_id": r["id"], "pattern": r["pattern"]}
         for n, r in enumerate((r for r in e13.frozen_rows(e13.COLL) if r["split"] == "dev" and r["category"] == "irregular"), 1)]
    key = json.loads((E11_REVIEW / "origin_key.json").read_text())
    rev = json.loads((E11_REVIEW / "review_batch.json").read_text())["items"]
    rep = [it for it in rev if key[it["item_id"]]["sample"] == "representative"]
    b = [{"id": f"b{n:02d}", "pool": "B_e11_review_representative", "word": it["word"],
          "reading": it["intended_reading"], "source_id": key[it["item_id"]]["dev_id"]}
         for n, it in enumerate((it for it in rep if it["word"] in cmu
                                 and e06.is_difficult(it["word"], tuple(cmu[it["word"]][0]))), 1)]
    blocked = set()
    for w in [it["word"] for it in rev] + [r["word"] for r in e13.frozen_rows(e13.COLL)] \
            + [r["word"] for r in e13.frozen_rows(e13.TEST)] \
            + [r["word"] for r in e11.jl(e13.E12_TEST / "input_manifest.jsonl")] + [x["word"] for x in a + b]:
        blocked |= e13.e12.family_keys(w)
    _, arm = e11.arm_rows()
    train = {r["word"] for r in arm}
    dev = e06.primary(e06.items("dev"))
    pool = sorted({i.word for i in dev if i.word in cmu and not e06.is_ambiguous(cmu[i.word])
                   and e06.is_difficult(i.word, tuple(cmu[i.word][0])) and not (e13.e12.family_keys(i.word) & blocked)
                   and i.word not in train},
                  key=lambda w: e06.order_key(COV_SEED, w))
    ids = {i.word: i.id for i in dev}
    c = [{"id": f"c{n:02d}", "pool": "C_historical_dev", "word": w, "reading": " ".join(cmu[w][0]),
          "source_id": ids[w]} for n, w in enumerate(pool[:COV_EXTRA], 1)]
    rows = a + b + c
    worst = sum(5 - (r["pool"] != "C_historical_dev") for r in rows)  # greedy already judged in pools A and B
    meta = {"pools": dict(Counter(r["pool"] for r in rows)), "eligible_extra": len(pool),
            "worst_case_new_judgments": worst}
    return rows, meta


def cov_select() -> dict:
    """Freeze the coverage rule and sample (commit before cov-generate)."""
    p = COV / "frozen.json"
    if p.exists():
        raise RuntimeError(f"{p} exists: the sample is frozen")
    rows, meta = coverage_sample()
    if not 20 <= len(rows) <= 30 or meta["worst_case_new_judgments"] > MAX_NEW_JUDGMENTS:
        raise RuntimeError(f"sample outside the registered bounds: {len(rows)} words, {meta}")
    COV.mkdir(parents=True, exist_ok=True)
    (COV / "input_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
    out = {"frozen": e06.now(), "rule": COV_RULE, "words": len(rows), **meta,
           "manifest_sha256": sha(COV / "input_manifest.jsonl"),
           "generation": {"model": "e11", "checkpoint_sha256": E11_BEST_SHA, "greedy": e09.setting("greedy"),
                          "beam": {"setting": "beam4_can", **e09.setting("beam4_can"), **e09.COMMON,
                                   "setting_hash": e09.setting_hash("beam4_can")},
                          "candidates": f"E11 greedy + up to {BEAM_K} distinct canonical-beam candidates, the same "
                                        "limit for every word; no other model"},
           "budget": {"max_new_judgments": MAX_NEW_JUDGMENTS,
                      "reduction": "if the worst case exceeded the cap, pool C would shrink from the end before "
                                   "generation (not needed)"},
           "purpose": "diagnostic: does E11 already generate acceptable alternatives for irregular dev words but rank "
                      "them poorly? Not a release gate and not evidence that a ranker would succeed.",
           "metrics": ["greedy top-1 acceptable", "canonical-beam top-1 acceptable",
                       "≥ 1 confirmed-acceptable candidate among the 4 beam candidates",
                       "≥ 1 confirmed-acceptable among greedy + 4 beam candidates", "uncertainty and coverage",
                       "words recoverable by choosing another candidate (greedy not confirmed acceptable, another "
                       "candidate confirmed acceptable)"]}
    set_state("cov_select", "done", words=len(rows))
    return write(p, out)


def cov_rows() -> list[dict]:
    fz = json.loads((COV / "frozen.json").read_text())
    p = COV / "input_manifest.jsonl"
    if sha(p) != fz["manifest_sha256"]:
        raise RuntimeError("coverage manifest changed after freezing")
    return e07.load_jsonl(p)


def cov_generate(force: bool = False) -> dict:
    """Greedy + canonical beam-4 for the frozen words (E09 settings), reusing cached generations by hash."""
    rows = cov_rows()
    sha_ = e09.checkpoint_sha("e11")
    if sha_ != E11_BEST_SHA:
        raise RuntimeError("E11 checkpoint changed")
    coll, dev = e13.coll_rows(), e09.dev_rows()
    need = [(n, split, rs) for split, rs in (("e13coll", coll), ("dev", dev)) for n in ("greedy", "beam4_can")
            if not e09.cache_path("e11", n, split, sha_, rs).exists()]
    m = None
    if need:
        require_quiet("cov_generate", force)
        m = e09.load("e11")[0]
        for n, split, rs in need:
            e09.generate(m, "e11", n, split, rs, sha_)
        e09.unload(m)
    files, recs = {}, {}
    for split, rs in (("e13coll", coll), ("dev", dev)):
        for n in ("greedy", "beam4_can"):
            p = e09.cache_path("e11", n, split, sha_, rs)
            files[p.name] = sha(p)
            recs[(split, n)] = {r["id"]: r for r in e07.load_jsonl(p)}
    out_rows = []
    for r in rows:
        split = "e13coll" if r["pool"].startswith("A") else "dev"
        g = e09.top1(recs[(split, "greedy")][r["source_id"]]["cands"])
        beam = e09.cand_list(recs[(split, "beam4_can")][r["source_id"]]["cands"])[:BEAM_K]
        out_rows.append({**r, "greedy": g, "beam": beam})
    COV_DATA.mkdir(parents=True, exist_ok=True)
    (COV_DATA / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                                               encoding="utf-8")
    out = {"generated": e06.now(), "checkpoint_sha256": sha_, "new_generation": [f"{n}:{s}" for n, s, _ in need],
           "cache_files_sha256": files, "settings": {n: {**e09.setting(n), **e09.COMMON, "setting_hash": e09.setting_hash(n)}
                                                     for n in ("greedy", "beam4_can")},
           "batching": "the cached runs' own row sets (E13 collection: 100 words; historical dev: 276 items), so the "
                       "padding and therefore the numerics match every earlier E11 output",
           "candidates_per_word": dict(Counter(len({x for x in [o["greedy"], *o["beam"]] if x}) for o in out_rows)),
           "beam_fewer_than_4": sum(len(o["beam"]) < BEAM_K for o in out_rows),
           "greedy_equals_beam_top1": sum(bool(o["beam"]) and o["greedy"] == o["beam"][0] for o in out_rows),
           "candidates_sha256": sha(COV_DATA / "candidates.jsonl"), "local": str(COV_DATA.relative_to(ROOT))}
    set_state("cov_generate", "done")
    return write(COV / "generation.json", out)


def label_texts(texts: list[str], seed: str) -> list[tuple[str, str]]:
    """Distinct texts in seeded order with labels A–E (rank and origin hidden).

    Examples:
        >>> [lb for lb, _ in label_texts(["ក", "ខ", "ក"], "s")]
        ['A', 'B']
    """
    uniq = list(dict.fromkeys(texts))
    return [(LABELS[n], t) for n, t in enumerate(sorted(uniq, key=lambda t: e06.order_key(seed, t)))]


def cov_batches() -> dict:
    """Reuse identical earlier judgments; one blinded batch of the unjudged candidates (≤ 120)."""
    cands = e07.load_jsonl(COV_DATA / "candidates.jsonl")
    idx, meta = judged_index()
    items, key, reused = [], {}, []
    for r in cands:
        origins = {}
        for o, t in [("greedy", r["greedy"])] + [(f"beam{k}", t) for k, t in enumerate(r["beam"], 1)]:
            if t:
                origins.setdefault(e06.nfc(t), []).append(o)
        new = []
        for t, o in origins.items():
            hit = idx.get((r["word"], r["reading"], t))
            if hit:
                reused.append({"id": r["id"], "text": t, "origins": o, **hit})
            else:
                new.append(t)
        if new:
            lab = label_texts(new, f"{COV_SEED}:labels:{r['id']}")
            items.append({"item_id": r["id"], "word": r["word"], "intended_reading": r["reading"],
                          "candidates": [{"label": lb, "text": t} for lb, t in lab]})
            key[r["id"]] = {lb: origins[t] for lb, t in lab}
    n_new = sum(len(i["candidates"]) for i in items)
    if n_new > MAX_NEW_JUDGMENTS:
        raise RuntimeError(f"{n_new} new judgments > {MAX_NEW_JUDGMENTS}")
    e07.write_json(COV_DATA / "origin_key.json", key)
    e07.write_json(COV_DATA / "items.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    e07.write_json(COV_DATA / "reused.json", {"reused": reused, **meta})
    files = e13.write_batches(COV_DATA, items, "E11 candidate-coverage review", MAX_NEW_JUDGMENTS) if items else []
    out = {"created": e06.now(), "words": len(cands), "words_with_new_candidates": len(items),
           "new_judgments_needed": n_new, "reused_judgments": len(reused),
           "reused_by_source": dict(Counter(x["source"].split(":")[0] for x in reused)),
           "reuse_rule": "identical word, intended reading and NFC candidate text, same policy file (sha256 "
                         f"{meta['policy_sha256'][:12]}…); provenance kept per reused judgment",
           "reuse_conflicts": len(meta["conflicts"]),
           "sha256": {f: sha(COV_DATA / f) for f in ["origin_key.json", "items.json", "reused.json", *files]},
           "blinding": "no ranks, scores or model origins on the page; labels in seeded order; the origin key is read "
                       "only by cov-import",
           "review": "python scripts/e12_review.py --reviewer <name> --data data/e11/coverage "
                     "--title 'E11 coverage review'"}
    set_state("cov_batches", "awaiting_owner_coverage_judgments" if items else "done", new=n_new)
    return write(COV / "batches.json", out)


def coverage_metrics(cands: list[dict], judg: dict[tuple[str, str], str]) -> dict:
    """The coverage diagnostic from candidates and judgments by (row id, text).

    Examples:
        >>> m = coverage_metrics([{"id": "a", "greedy": "ក", "beam": ["ខ", "ក"]}],
        ...                      {("a", "ក"): "not_acceptable", ("a", "ខ"): "acceptable"})
        >>> m["greedy_top1"].get("acceptable", 0), m["beam_top1"]["acceptable"], m["recoverable_words"]
        (0, 1, 1)
    """
    def j(r, t):
        return judg.get((r["id"], e06.nfc(t))) if t else None

    n = len(cands)
    g = Counter(j(r, r["greedy"]) or "no_output" for r in cands)
    b1 = Counter(j(r, r["beam"][0]) if r["beam"] else "no_output" for r in cands)
    any4 = sum(any(j(r, t) == "acceptable" for t in r["beam"]) for r in cands)
    any5 = sum(any(j(r, t) == "acceptable" for t in [r["greedy"], *r["beam"]] if t) for r in cands)
    rec = [r["id"] for r in cands if j(r, r["greedy"]) != "acceptable"
           and any(j(r, t) == "acceptable" for t in r["beam"] if e06.nfc(t) != e06.nfc(r["greedy"] or ""))]
    texts = {(r["id"], e06.nfc(t)) for r in cands for t in [r["greedy"], *r["beam"]] if t}
    js = Counter(judg.get(k, "missing") for k in texts)
    share = lambda k: round(k / n, 4) if n else None  # noqa: E731
    return {"words": n, "greedy_top1": dict(g), "greedy_top1_acceptable_share": share(g["acceptable"]),
            "beam_top1": dict(b1), "beam_top1_acceptable_share": share(b1["acceptable"]),
            "any_acceptable_in_beam4": any4, "any_acceptable_in_beam4_share": share(any4),
            "any_acceptable_in_greedy_or_beam4": any5, "recoverable_words": len(rec), "recoverable_ids": rec,
            "acceptable_per_word": dict(Counter(sum(judg.get((r["id"], t)) == "acceptable" for t in
                                                    {e06.nfc(x) for x in [r["greedy"], *r["beam"]] if x}) for r in cands)),
            "candidate_judgments": dict(js), "distinct_candidates": len(texts),
            "uncertain_share_of_candidates": round(js["uncertain"] / len(texts), 4) if texts else None,
            "judgment_coverage": round(1 - js["missing"] / len(texts), 4) if texts else None}


def cov_import() -> dict:
    """Validate the owner's coverage answers and report the diagnostic (aggregates only in the repository)."""
    its = json.loads((COV_DATA / "items.json").read_text())["items"]
    v = e13.validate(COV_DATA, its) if its else {"status": "complete", "seen": {}, "provenance": None,
                                                  "judgment_files_sha256": {}}
    if v["status"] != "complete":
        out = {k: x for k, x in v.items() if k != "seen"}
        set_state("cov_import", "awaiting_owner_coverage_judgments" if v["status"] == "awaiting" else v["status"])
        return {"status": "awaiting_owner_coverage_judgments" if v["status"] == "awaiting" else v["status"], **out}
    cands = e07.load_jsonl(COV_DATA / "candidates.jsonl")
    judg = {(x["id"], x["text"]): x["judgment"] for x in json.loads((COV_DATA / "reused.json").read_text())["reused"]}
    text_of = {(it["item_id"], c["label"]): c["text"] for it in its for c in it["candidates"]}
    judg |= {(iid, text_of[(iid, lb)]): jj for (iid, lb), jj in v["seen"].items()}
    pools = sorted({r["pool"] for r in cands})
    out = {"status": "scored", "scored": e06.now(), "label": "development diagnostic only (oracle coverage is not "
           "system accuracy; a ranker was not built or tested)",
           "overall": coverage_metrics(cands, judg),
           "by_pool": {p: coverage_metrics([r for r in cands if r["pool"] == p], judg) for p in pools},
           "new_judgments": {"provenance": v["provenance"], "files_sha256": v["judgment_files_sha256"],
                             "counts": dict(Counter(v["seen"].values()))},
           "judge": "project owner (one, not independent); blinded; explicit answers per spelling",
           "interpretation": "words whose greedy output is not confirmed acceptable but whose beam list holds a "
                             "confirmed-acceptable spelling are the most a perfect reranker over these 4 candidates "
                             "could recover; a real ranker would recover fewer, and nothing here measures one"}
    for d in (out["overall"], *out["by_pool"].values()):
        d.pop("recoverable_ids", None)
    set_state("cov_import", "done")
    return write(COV / "results.json", out)


# ------------------------------------------------------------------ 3. export
def adapter_config(done: dict) -> dict:
    from khmer_translit import byt5
    reg = json.loads((EXP / "registration.json").read_text())
    cfg = done["cfg"]
    return {
        "name": NAME, "version": VERSION, "build": BUILD,
        "status": "release candidate (local); research quality, one non-independent judge; not published",
        "task": "one English word or name → Khmer script approximating its English (General American) pronunciation",
        "base": {"repo": byt5.BASE_REPO, "revision": byt5.BASE_REVISION, "weights_file": "pytorch_model.bin",
                 "weights_sha256": done["ident"]["base"]["pytorch_model.bin"]},
        "adapter": {"type": "LoRA (khmer_translit.byt5.LoRALinear; not a PEFT checkpoint)", "r": cfg["adapter"]["r"],
                    "alpha": cfg["adapter"]["alpha"], "targets": cfg["adapter"]["targets"],
                    "modules": {"attn": ["q", "k", "v", "o"], "ffn": ["wi_0", "wi_1", "wo"]}},
        "decode": {"single": "greedy", "candidate_beam": 4, "candidate_search": "early_stopping='never', "
                   "length_penalty 0", "max_new_tokens": byt5.MAX_NEW_TOKENS, "max_input_bytes": byt5.MAX_SRC_BYTES,
                   "batch_size": 32},
        "input": "khmer_translit.phonetic_source.normalize_word (NFKC, lower case, accents, apostrophes and hyphens "
                 "removed; [a-z]+; ≤ 32 letters), then UTF-8 bytes + 3 and EOS",
        "provenance": {"candidate": "e11", "experiment": "E11-real-data-byt5", "run": "r_s0",
                       "checkpoint_sha256": done["best_sha256"], "update": done["best"]["update"],
                       "training_rows": done["rows"], "init": reg["init"],
                       "data_arm": f"E08 arm R (data/e08/training/arm_R.tsv, sha256 {reg['data']['sha256']}); "
                                   "no Google AI Mode rows",
                       "rows_by_provenance": reg["data"]["rows_by_provenance"],
                       "recipe": {k: cfg[k] for k in ("lr", "batch", "max_updates", "warmup", "schedule", "seed")},
                       "experiments": ["E07", "E08", "E09", "E11", "E12", "E13"]}}


def export() -> dict:
    """E11 best.pt → safetensors adapter; every tensor compared exactly with the checkpoint."""
    import torch
    from safetensors.torch import load_file
    from khmer_translit import byt5
    done = json.loads((E11_DIR / "done.json").read_text())
    if sha(E11_DIR / "best.pt") != E11_BEST_SHA:
        raise RuntimeError("E11 best.pt changed")
    sd = torch.load(E11_DIR / "best.pt", weights_only=True, map_location="cpu")
    if ADAPTER_OUT.exists():
        shutil.rmtree(ADAPTER_OUT)
    cfg = byt5.export_adapter(sd, ADAPTER_OUT, adapter_config(done))
    back = load_file(str(ADAPTER_OUT / byt5.ADAPTER_FILE))
    exact = set(back) == set(sd) and all(torch.equal(back[k], sd[k]) and back[k].dtype == sd[k].dtype for k in sd)
    out = {"adapter": str(ADAPTER_OUT.relative_to(ROOT)), "adapter_sha256": cfg["adapter_sha256"],
           "config_sha256": sha(ADAPTER_OUT / byt5.CONFIG_FILE), "tensors": len(back),
           "parameters": sum(v.numel() for v in back.values()), "dtype": sorted({str(v.dtype) for v in back.values()}),
           "bitwise_equal_to_checkpoint": exact, "source_checkpoint_sha256": E11_BEST_SHA,
           "format": "safetensors (no pickle); the training checkpoint (.pt, pickle-based) is not shipped"}
    if not exact:
        raise RuntimeError("exported tensors differ from the checkpoint")
    set_state("export", "done", adapter_sha256=cfg["adapter_sha256"])
    return write(REL / "export.json", out)


# ------------------------------------------------------------------ 4. verification
def hardware() -> dict:
    cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                if x.startswith("model name")), platform.processor())
    mem = next(int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal"))
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gib": round(mem / 2 ** 20, 1), "os": platform.platform(),
            "python": platform.python_version()}


def frozen_greedy(split: str) -> dict[str, str | None]:
    return {r["id"]: r["prediction"] for r in e07.load_jsonl(EXP / "predictions" / f"e11_{split}_greedy.jsonl")}


def error_checks(byt5, tl_cls) -> dict:
    """Missing/corrupt adapter, wrong base revision, wrong base weights and a missing base, each must fail loudly."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="kt-err-") as d:
        d = Path(d)

        def attempt(name, adapter, base, **kw):
            try:
                tl_cls.from_pretrained(adapter, base=base, offline=True, **kw)
                out[name] = "loaded (unexpected)"
            except Exception as e:  # noqa: BLE001  (the type is the result)
                out[name] = f"{type(e).__name__}: {str(e)[:120]}"

        attempt("missing_adapter", d / "nope", BASE_LOCAL)
        bad = d / "corrupt"
        shutil.copytree(ADAPTER_OUT, bad)
        raw = bytearray((bad / byt5.ADAPTER_FILE).read_bytes())
        raw[-1] ^= 0xFF
        (bad / byt5.ADAPTER_FILE).write_bytes(bytes(raw))
        attempt("corrupt_adapter", bad, BASE_LOCAL)
        rev = d / "revision"
        shutil.copytree(ADAPTER_OUT, rev)
        c = json.loads((rev / byt5.CONFIG_FILE).read_text())
        c["base"]["revision"] = "0" * 40
        (rev / byt5.CONFIG_FILE).write_text(json.dumps(c))
        attempt("wrong_base_revision", rev, BASE_LOCAL)
        fake = d / "fakebase"
        fake.mkdir()
        shutil.copy(BASE_LOCAL / "config.json", fake / "config.json")
        (fake / "pytorch_model.bin").write_bytes(b"not the base")
        attempt("wrong_base_weights", ADAPTER_OUT, fake)
        attempt("missing_base_folder", ADAPTER_OUT, d / "no-base")
        attempt("missing_base_offline_no_cache", ADAPTER_OUT, None)
        attempt("expected_sha_mismatch", ADAPTER_OUT, BASE_LOCAL, expected_sha256="0" * 64)
    want = {"missing_adapter": "FileNotFoundError", "corrupt_adapter": "ModelFormatError",
            "wrong_base_revision": "ModelFormatError", "wrong_base_weights": "ModelFormatError",
            "missing_base_folder": "FileNotFoundError", "missing_base_offline_no_cache": "FileNotFoundError",
            "expected_sha_mismatch": "ModelFormatError"}
    return {"results": out, "ok": all(out[k].startswith(v) for k, v in want.items())}


def verify(force: bool = False) -> dict:
    """Real-adapter checks of the packaged backend in this environment (offline; one model in memory)."""
    require_quiet("verify", force)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="kt-hf-")         # an empty cache: no hidden Hub copy
    from khmer_translit import byt5
    exp = json.loads((REL / "export.json").read_text())
    t = time.perf_counter()
    tl = byt5.ByT5Transliterator.from_pretrained(ADAPTER_OUT, base=BASE_LOCAL, offline=True, threads=THREADS,
                                                 expected_sha256=exp["adapter_sha256"])
    out = {"hardware": hardware(), "threads": THREADS, "adapter_sha256": exp["adapter_sha256"],
           "load_seconds": round(time.perf_counter() - t, 2), "rss_after_load_mb": peak_rss_mb()}
    sha_ = e09.checkpoint_sha("e11")
    parity = {}
    for split, rows in (("dev", e09.dev_rows()), ("eval", e09.eval_rows())):
        words = [w for _, w in rows]
        fz = frozen_greedy(split)
        t = time.perf_counter()
        single = tl.predict_many(words)
        sec = time.perf_counter() - t
        mism = [i for (i, _), r in zip(rows, single) if (r.text if r.status == "ok" else None) != fz.get(i)]
        parity[split] = {"items": len(rows), "greedy_mismatches_vs_frozen_predictions": len(mism),
                         "examples": mism[:5], "greedy_seconds": round(sec, 2),
                         "greedy_words_per_second": round(len(rows) / sec, 1)}
    rows = e09.dev_rows()
    beam = {r["id"]: r for r in e07.load_jsonl(e09.cache_path("e11", "beam4_can", "dev", sha_, rows))}
    t = time.perf_counter()
    four = tl.predict_many([w for _, w in rows], top_k=4)
    sec = time.perf_counter() - t
    mism4 = [i for (i, _), r in zip(rows, four) if [c.text for c in r.candidates] != e09.cand_list(beam[i]["cands"])[:4]]
    parity["dev_top4"] = {"items": len(rows), "mismatches_vs_research_canonical_beam": len(mism4),
                          "examples": mism4[:5], "seconds": round(sec, 2), "words_per_second": round(len(rows) / sec, 1)}
    out["parity"] = parity
    lat = []
    for w in SMOKE * 3:
        t = time.perf_counter()
        tl.predict(w)
        lat.append(time.perf_counter() - t)
    lat4 = []
    for w in SMOKE:
        t = time.perf_counter()
        tl.predict(w, top_k=4)
        lat4.append(time.perf_counter() - t)
    out["latency_ms"] = {"single_word_greedy": {"median": round(median(lat) * 1000, 1), "max": round(max(lat) * 1000, 1),
                                                "calls": len(lat)},
                         "single_word_top4": {"median": round(median(lat4) * 1000, 1), "calls": len(lat4)}}
    refuse = {"empty": "", "multiword": "new york", "khmer_script": "ខ្មែរ", "digits": "c3po", "overlength_33": "b" * 33,
              "at_limit_32": "b" * 32, "whitespace": "  "}
    res = {k: tl.predict(w) for k, w in refuse.items()}
    out["inputs"] = {k: {"status": r.status, "candidates": len(r.candidates)} for k, r in res.items()}
    out["no_silent_truncation"] = res["overlength_33"].status == "too_long" and not res["overlength_33"].candidates
    batch = ["coffee", "", "new york", "london", "b" * 33, "taxi"]
    br = tl.predict_many(batch)
    out["batch_alignment"] = {"inputs": batch, "statuses": [r.status for r in br],
                              "aligned": [r.input for r in br] == batch}
    smoke = {w: tl.predict(w).text for w in SMOKE}
    top4 = {w: [c.text for c in tl.predict(w, top_k=4).candidates] for w in SMOKE[:3]}
    out["smoke"], out["smoke_top4"] = smoke, top4
    out["peak_rss_mb"] = peak_rss_mb()
    del tl
    tl2 = byt5.ByT5Transliterator.from_pretrained(ADAPTER_OUT, base=BASE_LOCAL, offline=True, threads=THREADS)
    out["reload_deterministic"] = ({w: tl2.predict(w).text for w in SMOKE} == smoke
                                   and {w: [c.text for c in tl2.predict(w, top_k=4).candidates] for w in SMOKE[:3]} == top4)
    out["errors"] = error_checks(byt5, byt5.ByT5Transliterator)
    out["decoding"] = tl2.info()["decoding"]
    out["offline"] = "HF_HUB_OFFLINE=1 and an empty HF_HOME for the whole check; local base folder"
    out["ok"] = (all(p.get("greedy_mismatches_vs_frozen_predictions", p.get("mismatches_vs_research_canonical_beam")) == 0
                     for p in parity.values())
                 and out["no_silent_truncation"] and out["batch_alignment"]["aligned"] and out["reload_deterministic"]
                 and out["errors"]["ok"] and res["at_limit_32"].status == "ok"
                 and all(res[k].status != "ok" for k in refuse if k != "at_limit_32"))
    set_state("verify", "done" if out["ok"] else "failed")
    return write(REL / "verify.json", out)


# ------------------------------------------------------------------ 5. bundle
EVIDENCE = {"identity.json": REL / "identity.json", "export.json": REL / "export.json",
            "verify.json": REL / "verify.json", "install_check.json": REL / "install_check.json",
            "e11_results.json": EXP / "results.json", "e11_frozen.json": EXP / "reports" / "frozen.json",
            "e11_dev_review_scores.json": EXP / "reports" / "review_scores.json",
            "e12_test_scores.json": ROOT / "experiments" / "E12-longer-schedule" / "reports" / "test_scores.json",
            "e13_test_scores.json": ROOT / "experiments" / "E13-pronunciation-guided" / "reports" / "test_scores.json",
            "coverage_frozen.json": COV / "frozen.json", "coverage_results.json": COV / "results.json"}
FORBIDDEN = (".pt", ".bin", ".npz", ".env", ".tsv")
FORBIDDEN_NAMES = ("origin_key.json", "judgments", "review_batch.json", "items.json", "candidates.jsonl",
                   "reused.json", "arm_R", "review_log.jsonl")


def build_wheel() -> list[Path]:
    wd = DIST / "wheel"
    shutil.rmtree(wd, ignore_errors=True)
    subprocess.run(["uv", "build", "--out-dir", str(wd)], cwd=ROOT, check=True, capture_output=True)
    (wd / ".gitignore").unlink(missing_ok=True)
    return sorted(wd.iterdir())


def scan(files: list[Path], base: Path) -> list[str]:
    """Files the bundle must not hold (training checkpoints, base weights, data, review keys, judgments, secrets)."""
    bad = []
    for p in files:
        rel = str(p.relative_to(base))
        if p.suffix in FORBIDDEN or any(x in rel for x in FORBIDDEN_NAMES) or "__pycache__" in p.parts:
            bad.append(rel)
        elif p.suffix in (".md", ".json", ".txt") and os.environ.get("WANDB_API_KEY", "\0") in p.read_text("utf-8", "ignore"):
            bad.append(rel + " (secret)")
    return bad


def bundle() -> dict:
    """Assemble the local E11 bundle (no upload) from the E11 adapter, E11 documents and E11 evidence."""
    from khmer_translit import byt5
    exp = json.loads((REL / "export.json").read_text())
    if sha(ADAPTER_OUT / byt5.ADAPTER_FILE) != exp["adapter_sha256"]:
        raise RuntimeError("exported adapter changed")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    wheels = build_wheel()
    shutil.copytree(ADAPTER_OUT, DIST / "adapter")
    for p in sorted(DOCS.glob("*.md")):
        shutil.copy(p, DIST / p.name)
    shutil.copy(DOCS / "requirements.txt", DIST / "requirements.txt")
    shutil.copy(ROOT / "LICENSE", DIST / "LICENSE-code")
    ev = DIST / "evidence"
    ev.mkdir()
    missing = []
    for name, src in EVIDENCE.items():
        if src.exists():                                  # local absolute paths → <repo>
            (ev / name).write_text(src.read_text().replace(str(ROOT), "<repo>"))
        else:
            missing.append(name)
    hf = DIST / "huggingface"
    hf.mkdir()
    shutil.copy(DOCS / "MODEL_CARD.md", hf / "README.md")
    for f in (byt5.ADAPTER_FILE, byt5.CONFIG_FILE):
        shutil.copy(ADAPTER_OUT / f, hf / f)
    for f in ("NOTICES.md", "DATA_PROVENANCE.md", "EVALUATION.md", "LIMITATIONS.md", "REFERENCES.md"):
        shutil.copy(DOCS / f, hf / f)
    shutil.copy(ROOT / "LICENSE", hf / "LICENSE-code")
    shutil.copy(DOCS / "LICENSE-weights.md", hf / "LICENSE")             # adapter weights: CC BY-NC-SA 4.0 (D198)
    hf_files = sorted(p for p in hf.iterdir() if p.is_file())
    (hf / "MANIFEST.json").write_text(json.dumps({"files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                                            for p in hf_files}, "base": exp | {"base_repo": byt5.BASE_REPO,
                                                            "base_revision": byt5.BASE_REVISION}}, indent=1) + "\n")
    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    bad = scan(files, DIST)
    if bad:
        raise RuntimeError(f"bundle holds files it must not: {bad[:5]}")
    (DIST / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(DIST)}\n" for p in files), encoding="utf-8")
    arc = DIST.parent / f"{NAME}.tar.gz"
    with tarfile.open(arc, "w:gz") as t:
        t.add(DIST, arcname=NAME)
    out = {"bundle": str(DIST.relative_to(ROOT)), "archive": str(arc.relative_to(ROOT)), "archive_sha256": sha(arc),
           "archive_bytes": arc.stat().st_size, "files": [str(p.relative_to(DIST)) for p in files],
           "wheels": {p.name: sha(p) for p in wheels}, "adapter_sha256": exp["adapter_sha256"],
           "evidence_missing": missing, "hf_folder": str(hf.relative_to(ROOT)),
           "excluded": "training checkpoints (.pt), base weights, arm R and every mined pair, references, review "
                       "batches, origin keys, judgments, caches, W&B files, secrets"}
    (DIST.parent / f"{NAME}.tar.gz.sha256").write_text(f"{out['archive_sha256']}  {arc.name}\n")
    set_state("bundle", "done", archive_sha256=out["archive_sha256"])
    return write(REL / "bundle.json", out)


# ------------------------------------------------------------------ 6. isolated install
NO_NET = """import socket
def _no_net(*a, **k):
    raise OSError("network disabled by the E11 install check")
socket.socket.connect = socket.socket.connect_ex = _no_net
socket.create_connection = socket.getaddrinfo = _no_net
"""


def install(force: bool = False) -> dict:
    """Fresh uv venv outside the repository; the wheel with its `byt5` extra resolved from the declared requirements
    (CPU-only torch via uv's `--torch-backend cpu`, official indexes); then the CLI and API with networking disabled
    (`unshare -rn` when the kernel allows it, else a socket guard loaded before any package code), the local base
    folder and an empty Hugging Face cache."""
    require_quiet("install", force)
    from khmer_translit import byt5
    wheel = next((DIST / "wheel").glob("*.whl"))
    exp = json.loads((REL / "export.json").read_text())
    adapter = DIST / "adapter"
    out = {"wheel": wheel.name, "wheel_sha256": sha(wheel), "hardware": hardware()}
    uv_cache = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv"))
    du = lambda p: int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0] or 0)  # noqa: E731
    with tempfile.TemporaryDirectory(prefix="kt-e11-install-") as d:
        d = Path(d)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA", "UV_PROJECT"))}
        env |= {"HF_HOME": str(d / "hf"), "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
        run = lambda *c, **kw: subprocess.run(list(c), cwd=d, capture_output=True, text=True, **{"env": env, **kw})  # noqa: E731
        r = run("uv", "venv", "--python", "3.10", str(d / "venv"))
        py = d / "venv" / "bin" / "python"
        cfg_txt = (d / "venv" / "pyvenv.cfg").read_text()
        out["venv"] = {"returncode": r.returncode, "system_site_packages": "include-system-site-packages = true" in cfg_txt}
        before = du(uv_cache)
        t = time.perf_counter()
        r = run("uv", "pip", "install", "--python", str(py), "--torch-backend", "cpu", f"khmer-translit[byt5] @ {wheel.as_uri()}")
        out["install"] = {"returncode": r.returncode, "seconds": round(time.perf_counter() - t, 1),
                          "tail": (r.stderr or r.stdout)[-800:],
                          "uv_cache_growth_mb": round((du(uv_cache) - before) / 2 ** 20, 1),
                          "command": "uv pip install --torch-backend cpu 'khmer-translit[byt5] @ file://…whl'"}
        if r.returncode:
            out["ok"] = False
            set_state("install_check", "failed")
            return write(REL / "install_check.json", out)
        site = next((d / "venv" / "lib").glob("python3*/site-packages"))
        pth = {p.name: p.read_text()[:200] for p in site.glob("*.pth")}
        frz = run("uv", "pip", "freeze", "--python", str(py)).stdout.splitlines()
        out["environment"] = {"site_packages_mb": round(du(site) / 2 ** 20), "packages": frz,
                              "pth_files": pth, "pth_points_into_repo": any(str(ROOT) in v for v in pth.values()),
                              "nvidia_packages": [x for x in frz if x.lower().startswith(("nvidia", "triton"))]}
        probe_src = ("import socket\ntry:\n    s = socket.create_connection(('pypi.org', 443), timeout=3)\n"
                     "    print('CONNECTED')\nexcept OSError as e:\n    print('BLOCKED', type(e).__name__)\n")
        probe = run("unshare", "-rn", str(py), "-c", probe_src)
        if probe.returncode == 0 and "BLOCKED" in probe.stdout:
            method = "kernel: network namespace without interfaces (unshare -rn)"
            pyrun = lambda src, *a: run("unshare", "-rn", str(py), "-c", src, *a)  # noqa: E731
        else:                                   # unprivileged user namespaces refused (AppArmor): guard in Python
            method = ("Python-level: socket connect/connect_ex/create_connection/getaddrinfo raise before any package "
                      f"code is imported (unshare -rn refused: {(probe.stderr or probe.stdout).strip()[-90:]})")
            pyrun = lambda src, *a: run(str(py), "-c", NO_NET + src, *a)  # noqa: E731
            probe = pyrun(probe_src)
        out["network_disabled"] = {"method": method, "returncode": probe.returncode,
                                   "probe": (probe.stdout + probe.stderr).strip()[-160:]}
        r = pyrun("import khmer_translit, torch, transformers; print(khmer_translit.__file__); "
                  "print(torch.__version__, torch.version.cuda, transformers.__version__)")
        lines = r.stdout.splitlines()
        out["imports"] = {"khmer_translit": lines[0] if lines else r.stderr[-300:],
                          "from_venv": bool(lines) and lines[0].startswith(str(site)),
                          "torch_transformers": lines[1] if len(lines) > 1 else None}
        cli_main = "import sys\nfrom khmer_translit.cli import main\nsys.exit(main(sys.argv[1:]))\n"
        cli = lambda *a: pyrun(cli_main, *a)  # noqa: E731
        base_args = ["--backend", "byt5", "--model", str(adapter), "--base", str(BASE_LOCAL), "--offline",
                     "--threads", str(THREADS), "--sha256", exp["adapter_sha256"]]
        c = {}
        r = cli("predict", *base_args, "--word", "coffee")
        c["single_word"] = {"returncode": r.returncode, "stdout": r.stdout.strip()}
        r = run(str(d / "venv" / "bin" / "khmer-translit"), "predict", *base_args, "--word", "coffee",
                env=env | {"HF_HUB_OFFLINE": "1"})   # the installed console script itself (offline flags only)
        c["console_script"] = {"returncode": r.returncode, "stdout": r.stdout.strip(),
                               "same_as_guarded": r.stdout.strip() == c["single_word"]["stdout"]}
        r = cli("predict", *base_args, "--word", "london", "--top-k", "4", "--json")
        c["top4_json"] = {"returncode": r.returncode, "candidates": [x["text"] for x in json.loads(r.stdout)["candidates"]]
                          if r.returncode == 0 else r.stderr[-300:]}
        words = SMOKE + ["", "new york", "b" * 33, "ខ្មែរ"]
        (d / "words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
        r = cli("predict", *base_args, "--input", "words.txt", "--jsonl")
        rows = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
        c["batch_jsonl"] = {"returncode": r.returncode, "rows": len(rows), "ids": [x["id"] for x in rows],
                            "inputs_aligned": [x["input"] for x in rows] == words,
                            "statuses": [x["status"] for x in rows]}
        r = cli("inspect", *base_args)
        info = json.loads(r.stdout) if r.returncode == 0 else {}
        c["inspect"] = {"returncode": r.returncode, "candidate": info.get("provenance", {}).get("candidate"),
                        "checkpoint_sha256": info.get("provenance", {}).get("checkpoint_sha256"),
                        "base_revision": info.get("base", {}).get("revision")}
        r = cli("predict", "--backend", "byt5", "--model", str(d / "missing"), "--base", str(BASE_LOCAL),
                "--offline", "--word", "coffee")
        c["missing_adapter"] = {"returncode": r.returncode, "stderr": r.stderr.strip()[-200:]}
        bad = d / "corrupt"
        shutil.copytree(adapter, bad)
        raw = bytearray((bad / byt5.ADAPTER_FILE).read_bytes())
        raw[100] ^= 0xFF
        (bad / byt5.ADAPTER_FILE).write_bytes(bytes(raw))
        r = cli("predict", "--backend", "byt5", "--model", str(bad), "--base", str(BASE_LOCAL), "--offline",
                "--word", "coffee")
        c["corrupt_adapter"] = {"returncode": r.returncode, "stderr": r.stderr.strip()[-200:]}
        r = cli("predict", "--backend", "byt5", "--model", str(adapter), "--offline", "--word", "coffee")
        c["missing_base_offline"] = {"returncode": r.returncode, "stderr": r.stderr.strip()[-200:]}
        r = cli("predict", "--model", str(adapter), "--word", "coffee")
        c["default_backend_is_not_byt5"] = {"returncode": r.returncode, "stderr": r.stderr.strip()[-160:]}
        out["cli"] = c
        # Python API in the isolated environment: dev parity, determinism on reload, timing
        rows = e09.dev_rows()
        fz = frozen_greedy("dev")
        (d / "dev_words.json").write_text(json.dumps([w for _, w in rows]))
        script = f"""
import json, time, resource
from khmer_translit.byt5 import ByT5Transliterator
words = json.load(open("dev_words.json"))
t = time.perf_counter()
tl = ByT5Transliterator.from_pretrained({str(adapter)!r}, base={str(BASE_LOCAL)!r}, offline=True, threads={THREADS})
load = time.perf_counter() - t
t = time.perf_counter(); out = tl.predict_many(words); sec = time.perf_counter() - t
lat = []
for w in {SMOKE!r} * 3:
    t = time.perf_counter(); tl.predict(w); lat.append(time.perf_counter() - t)
tl2 = ByT5Transliterator.from_pretrained({str(adapter)!r}, base={str(BASE_LOCAL)!r}, offline=True, threads={THREADS})
same = [r.text for r in tl2.predict_many(words[:64])] == [r.text for r in out[:64]]
top4 = {{w: [c.text for c in tl.predict(w, top_k=4).candidates] for w in {SMOKE[:3]!r}}}
print(json.dumps({{"preds": [r.text if r.status == "ok" else None for r in out], "load_s": round(load, 2),
                  "greedy_s": round(sec, 2), "lat_ms": sorted(round(x * 1000, 1) for x in lat), "reload_same": same,
                  "top4": top4, "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}},
                 ensure_ascii=False))
"""
        r = pyrun(script)
        if r.returncode == 0:
            res = json.loads(r.stdout.splitlines()[-1])
            mism = [i for (i, _), p in zip(rows, res["preds"]) if p != fz.get(i)]
            out["api"] = {"dev_items": len(rows), "greedy_mismatches_vs_frozen_predictions": len(mism),
                          "examples": mism[:5], "load_seconds": res["load_s"], "greedy_dev_seconds": res["greedy_s"],
                          "greedy_words_per_second": round(len(rows) / res["greedy_s"], 1),
                          "single_word_ms_median": median(res["lat_ms"]), "reload_deterministic": res["reload_same"],
                          "top4": res["top4"], "peak_rss_mb": res["peak_rss_mb"], "threads": THREADS}
        else:
            out["api"] = {"returncode": r.returncode, "stderr": r.stderr[-600:]}
        out["hf_home_files"] = [str(p.relative_to(d)) for p in (d / "hf").rglob("*") if p.is_file()] \
            if (d / "hf").exists() else []
    ver = json.loads((REL / "verify.json").read_text()) if (REL / "verify.json").exists() else {}
    top4_ref = ver.get("smoke_top4", {})
    a = out.get("api", {})
    out["checks"] = {
        "install": out["install"]["returncode"] == 0, "no_system_site_packages": not out["venv"]["system_site_packages"],
        "no_repo_pth_bridge": not out["environment"]["pth_points_into_repo"],
        "cpu_only_torch": not out["environment"]["nvidia_packages"] and "None" in (out["imports"]["torch_transformers"] or ""),
        "imports_from_venv": out["imports"]["from_venv"],
        "network_disabled": "BLOCKED" in out["network_disabled"]["probe"],
        "console_script": c["console_script"]["returncode"] == 0 and c["console_script"]["same_as_guarded"],
        "single_word": c["single_word"]["returncode"] == 0, "top4": c["top4_json"]["returncode"] == 0,
        "batch_aligned": c["batch_jsonl"]["inputs_aligned"] and c["batch_jsonl"]["ids"] == list(range(1, len(words) + 1))
        and c["batch_jsonl"]["statuses"][-4:] == ["empty", "unsupported_multiword", "too_long", "unsupported_characters"],
        "inspect_names_e11": c["inspect"]["candidate"] == "e11" and c["inspect"]["checkpoint_sha256"] == E11_BEST_SHA,
        "missing_adapter_exit_2": c["missing_adapter"]["returncode"] == 2,
        "corrupt_adapter_exit_2": c["corrupt_adapter"]["returncode"] == 2,
        "missing_base_exit_2": c["missing_base_offline"]["returncode"] == 2,
        "default_backend_refuses_adapter": c["default_backend_is_not_byt5"]["returncode"] == 2,
        "dev_parity": a.get("greedy_mismatches_vs_frozen_predictions") == 0,
        "reload_deterministic": bool(a.get("reload_deterministic")),
        "top4_matches_repo": not top4_ref or a.get("top4") == top4_ref,
        "no_hub_downloads": not out["hf_home_files"]}
    out["ok"] = all(out["checks"].values())
    set_state("install_check", "done" if out["ok"] else "failed")
    return write(REL / "install_check.json", out)


def status() -> dict:
    return state()


STEPS = {"identity": identity, "cov-select": cov_select, "cov-generate": cov_generate, "cov-batches": cov_batches,
         "cov-import": cov_import, "export": export, "verify": verify, "bundle": bundle, "install": install,
         "status": status}
HEAVY = {"cov-generate", "verify", "install"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--force", action="store_true", help="run a model-heavy step even if other CPU work is running")
    a = ap.parse_args()
    fn = STEPS[a.step]
    res = fn(force=a.force) if a.step in HEAVY else fn()
    print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
