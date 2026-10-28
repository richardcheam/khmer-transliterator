"""E09 ByT5 decoding (inference only): why does greedy beat beam-4 on E07/E08, and can decoding help?

Reuses E07's model building, generation (`e07.gen`), teacher-forced losses and scoring; no training, no downloads.
Plan: `experiments/E09-byt5-decoding/PLAN.md`.

Steps (cache in `state/e09/cache/`, outputs in `experiments/E09-byt5-decoding/`):
    register    versions, hashes of sources, checkpoints, references and saved predictions → registration.json
    reproduce   registered greedy + beam-4 on all dev items vs the saved E07/E08 files → reports/reproduction.json
    audit       generation and scoring checks, teacher-forced score reconstruction, H1 → reports/generation_audit.json
    dev         every registered setting on dev, both models → reports/dev_comparison.json, dev_per_item.jsonl
    diagnose    the balanced disagreement sample with sequence scores → reports/disagreements.json
    select      dev selection per model (frozen; commit before eval) → reports/selected_settings.json
    eval        new eval predictions for selected settings only, frozen → predictions/, reports/eval_frozen.json
    analyse     matched historical-eval comparisons → results.json, results_per_item.jsonl

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e09.py reproduce
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E09-byt5-decoding"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
LEDGER = EXP / "runs" / "ledger.jsonl"
CACHE = ROOT / "state" / "e09" / "cache"
GiB = 1024 ** 3
LIMITS = {"threads": 6, "rss_max_bytes": 5 * GiB, "min_available_bytes": int(1.5 * GiB), "pressure_checks": 3,
          "active_hours": 4.0}

CHECKPOINTS = {"e07": ROOT / "models" / "e07" / "full", "e08": ROOT / "models" / "e08" / "rs_s0"}
SAVED = {"e07": lambda s: e07.PRED / f"byt5_e07_{s}.jsonl",
         "e08": lambda s: ROOT / "experiments" / "E08-data-expansion" / "predictions" / f"byt5_rs_s0_{s}.jsonl"}

COMMON = {"max_new_tokens": e07.MAX_NEW, "do_sample": False, "logits_processors": None}


def _beam(k: int, early_stopping, length_penalty: float = 0.0) -> dict:
    return {"num_beams": k, "ret": k, "length_penalty": length_penalty, "early_stopping": early_stopping, "bs": 16}


SETTINGS = {"greedy": {"num_beams": 1, "ret": 1, "bs": 32},
            **{f"beam{k}_reg": _beam(k, True) for k in (2, 4, 8)},
            **{f"beam{k}_can": _beam(k, "never") for k in (2, 4, 8)}}
FOLLOW_UP = {"beam4_can_lp1": _beam(4, "never", 1.0)}          # PLAN.md §3; only under its trigger
PRIORITY = ["greedy", "beam4_reg", "beam4_can", "beam2_reg", "beam2_can", "beam8_reg", "beam8_can"]
REGISTERED = {"greedy", "beam4_reg"}                            # identical to the saved E07/E08 predictions
BOOT = {"n": e05.N_BOOT, "seed": e05.BOOT_SEED}


def setting(name: str) -> dict:
    return SETTINGS.get(name) or FOLLOW_UP[name]


def setting_hash(name: str) -> str:
    """Hash of everything that defines a generation setting (the batch size included: padding changes numerics).

    Examples:
        >>> setting_hash("beam4_reg") != setting_hash("beam4_can"), len(setting_hash("greedy"))
        (True, 64)
    """
    return e07.cfg_hash({"setting": setting(name), **COMMON})


def data_hash(rows: list[tuple[str, str]]) -> str:
    return e07.rows_hash(rows)


def cache_path(model: str, name: str, split: str, ckpt_sha: str, rows: list[tuple[str, str]]) -> Path:
    """One file per (checkpoint, setting, data): a changed input of any kind gets a new file, never a stale hit."""
    h = e07.sha_text(f"{ckpt_sha}|{setting_hash(name)}|{data_hash(rows)}")[:16]
    return CACHE / f"{model}_{name}_{split}_{h}.jsonl"


# ------------------------------------------------------------------ bookkeeping
def ledger() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def ledger_add(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": e06.now(), **rec}, ensure_ascii=False) + "\n")


def active_hours(entries: list[dict] | None = None) -> float:
    """Active experiment time so far: the `seconds` of every recorded step (generation, scoring, analysis).

    Examples:
        >>> active_hours([{"event": "step", "seconds": 1800}, {"event": "generate", "seconds": 1800}])
        1.0
    """
    entries = ledger() if entries is None else entries
    return sum(e.get("seconds", 0.0) for e in entries) / 3600


def check_budget() -> None:
    if active_hours() >= LIMITS["active_hours"]:
        raise RuntimeError(f"active-time budget of {LIMITS['active_hours']} h used up")


def timed(step: str):
    """Decorator: record a step's seconds in the ledger (the active-time budget) and refuse it once spent."""
    def wrap(fn):
        def run(*a, **kw):
            check_budget()
            t = time.perf_counter()
            out = fn(*a, **kw)
            ledger_add({"event": "step", "step": step, "seconds": round(time.perf_counter() - t, 1)})
            return out
        run.__doc__ = fn.__doc__
        return run
    return wrap


# ------------------------------------------------------------------ models and generation
def checkpoint_sha(model: str) -> str:
    done = json.loads((CHECKPOINTS[model] / "done.json").read_text())
    sha = e06.sha(CHECKPOINTS[model] / "best.pt")
    if sha != done["best_sha256"]:
        raise RuntimeError(f"{model} best.pt changed")
    return sha


def load(model: str):
    """One checkpoint on the pinned base (strict adapter load), in eval mode; returns (model, done.json)."""
    cfg = e07.full_cfg()
    checkpoint_sha(model)
    m = e07.build(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    e07.load_adapter(m, CHECKPOINTS[model] / "best.pt")
    m.eval()
    return m, json.loads((CHECKPOINTS[model] / "done.json").read_text())


def unload(m) -> None:
    del m
    gc.collect()


def gen_kwargs(name: str) -> dict:
    s = setting(name)
    kw = {"beam": s["num_beams"], "ret": s["ret"], "bs": s["bs"]}
    if s["num_beams"] > 1 and (s["early_stopping"] is not True or s["length_penalty"] != 0.0):
        kw |= {"early_stopping": s["early_stopping"], "length_penalty": s["length_penalty"]}
    return kw


def generate(m, model: str, name: str, split: str, rows: list[tuple[str, str]], ckpt_sha: str,
             guard: e07.Guard | None = None) -> list[dict]:
    """Cached generation: `[{id, word, cands: [{text, status, score, ids}]}]` in `rows` order.

    A finished file is reused (resume); a missing or partial one is regenerated in full and written atomically.
    Memory is checked between chunks; sustained pressure stops the step before anything is written.
    """
    p = cache_path(model, name, split, ckpt_sha, rows)
    if p.exists():
        recs = e07.load_jsonl(p)
        if [(r["id"], r["word"]) for r in recs] == rows:
            return recs
    check_budget()
    guard = guard or e07.Guard(limits=LIMITS)
    kw = gen_kwargs(name)
    t = time.perf_counter()
    out, step = [], 64
    for i in range(0, len(rows), step):
        if guard.check() == "memory_pressure":
            ledger_add({"event": "stopped", "reason": "memory_pressure", "model": model, "setting": name,
                        "split": split})
            raise RuntimeError("sustained memory pressure: stopped before writing")
        chunk = rows[i:i + step]
        res = e07.gen(m, [B.encode_src(w) for _, w in chunk], **kw)
        out += [{"id": rid, "word": w, "cands": [{"text": tx, "status": st, "score": sc, "ids": ids}
                                                 for tx, st, sc, ids in r]} for (rid, w), r in zip(chunk, res)]
    sec = round(time.perf_counter() - t, 1)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    os.replace(tmp, p)
    ledger_add({"event": "generate", "model": model, "setting": name, "split": split, "items": len(rows),
                "seconds": sec, "setting_hash": setting_hash(name), "checkpoint_sha256": ckpt_sha,
                "data_hash": data_hash(rows), "file": p.name, "peak_rss_mb": round(guard.peak_rss / 2 ** 20)})
    return out


def top1(cands: list[dict]) -> str | None:
    """E07/E08 top-1: the first non-empty, well-decoded candidate (NFC), else None.

    Examples:
        >>> top1([{"text": "", "status": "empty"}, {"text": "ក", "status": "ok"}])
        'ក'
    """
    ok = e05.dedup([c["text"] for c in cands if c["status"] == "ok" and c["text"]])
    return ok[0] if ok else None


def cand_list(cands: list[dict]) -> list[str]:
    return e05.dedup([c["text"] for c in cands if c["status"] == "ok" and c["text"]])


def tf_logprobs(m, pairs: list[tuple[str, list[int]]], bs: int = 32) -> list[float]:
    """Teacher-forced sum log-probability of each `(word, target ids)` pair, EOS included when present."""
    if not pairs:
        return []
    tl = e07.token_losses(m, [B.encode_src(w) for w, _ in pairs], [ids for _, ids in pairs], bs)
    return [round(-sum(v for _, v in ex), 6) for ex in tl]


REPEAT = re.compile(r"(.{1,4})\1{2,}")


def repeated(t: str | None) -> bool:
    """A 1–4 code-point unit repeated ≥ 3 times in a row.

    Examples:
        >>> repeated("ហាហាហា"), repeated("ហាហា"), repeated(None)
        (True, False, False)
    """
    return bool(t and REPEAT.search(t))


# ------------------------------------------------------------------ data
def dev_rows() -> list[tuple[str, str]]:
    return [(i.id, i.word) for i in e06.items("dev")]


def eval_rows() -> list[tuple[str, str]]:
    return [(i.id, i.word) for i in e06.items("eval")]


def saved(model: str, split: str) -> dict[str, dict]:
    return {r["id"]: r for r in e07.load_jsonl(SAVED[model](split))}


def check_alignment(items, preds: dict) -> None:
    """Every item has exactly one prediction and nothing else (raises otherwise)."""
    e05.check_ids({i.id for i in items}, preds)


# ------------------------------------------------------------------ register
@timed("register")
def register() -> dict:
    import torch
    import transformers
    files = ["scripts/e09.py", "scripts/e07.py", "scripts/e06_byt5.py", "scripts/e06.py", "scripts/e05.py",
             "src/khmer_translit/evaluation.py", "research/requirements-byt5.txt",
             "research/requirements-byt5.lock.txt", "experiments/E09-byt5-decoding/PLAN.md",
             "experiments/E09-byt5-decoding/config.yaml", "experiments/E07-byt5-diagnosis/full_recipe.json"]
    preds = {f"{m}_{s}": e06.sha(SAVED[m](s)) for m in SAVED for s in ("dev", "eval")}
    out = {"registered": e06.now(), "git": e07.git_rev(), "python": sys.version.split()[0],
           "torch": torch.__version__, "transformers": transformers.__version__, "threads": LIMITS["threads"],
           "limits": LIMITS, "base": {"revision": B.REVISION, "sha256": B.check_base(),
                                      "generation_config": json.loads((B.BASE / "generation_config.json").read_text())},
           "tokenizer": {"scheme": "UTF-8 byte + 3", "pad": B.PAD, "eos": B.EOS, "unk": 2, "offset": B.OFFSET,
                         "decoder_start": 0, "input": "lower-cased word + EOS (e06_byt5.encode_src)"},
           "checkpoints": {m: {"path": str((CHECKPOINTS[m] / "best.pt").relative_to(ROOT)), "sha256": checkpoint_sha(m),
                               "update": json.loads((CHECKPOINTS[m] / "done.json").read_text())["best"]["update"]}
                           for m in CHECKPOINTS},
           "recipe_sha256": e06.sha(e07.FULL_PATH),
           "settings": {n: {**setting(n), **COMMON, "hash": setting_hash(n)} for n in [*SETTINGS, *FOLLOW_UP]},
           "follow_up_note": "beam4_can_lp1 only under the PLAN.md §3 trigger",
           "references": {"dev_references.jsonl": e06.sha(e06.E03 / "dev_references.jsonl"),
                          "eval_references.jsonl": e06.sha(e06.E03 / "eval_references.jsonl"),
                          "dev_items": len(e06.items("dev")), "dev_primary": len(e06.primary(e06.items("dev"))),
                          "eval_items": len(e06.items("eval")), "eval_primary": len(e06.primary(e06.items("eval")))},
           "saved_predictions_sha256": preds, "source_sha256": {f: e06.sha(ROOT / f) for f in files},
           "bootstrap": BOOT, "selection": "dev top-1 macro, then lower dev mean CER, then lower generation seconds"}
    return e07.write_json(EXP / "registration.json", out) and out


# ------------------------------------------------------------------ reproduce
def compare_saved(recs: list[dict], sv: dict[str, dict], name: str, tol: float = 1e-4) -> dict:
    """Item-by-item agreement of regenerated records with a saved E07/E08 file (texts, statuses, scores).

    Examples:
        >>> r = [{"id": "a", "cands": [{"text": "ក", "status": "ok", "score": -1.0, "ids": []}]}]
        >>> compare_saved(r, {"a": {"greedy": "ក", "greedy_status": "ok"}}, "greedy")["mismatches"]
        0
    """
    bad, max_d = [], 0.0
    for r in recs:
        s = sv[r["id"]]
        if name == "greedy":
            c = r["cands"][0]
            same = (c["text"] if c["status"] == "ok" else None) == s["greedy"] and c["status"] == s["greedy_status"]
        else:
            raw = s["beam4_raw"]
            same = len(raw) == len(r["cands"]) and all(
                a["text"] == b["text"] and a["status"] == b["status"] for a, b in zip(r["cands"], raw))
            if same:
                max_d = max([max_d, *(abs(a["score"] - b["score"]) for a, b in zip(r["cands"], raw))])
                same = max_d <= tol
            same = same and cand_list(r["cands"]) == s["beam4"]
        if not same:
            bad.append(r["id"])
    return {"items": len(recs), "mismatches": len(bad), "examples": bad[:5], "max_abs_score_diff": round(max_d, 7)}


@timed("reproduce")
def reproduce() -> dict:
    rows, out = dev_rows(), {}
    for model in CHECKPOINTS:
        m, _ = load(model)
        sha = checkpoint_sha(model)
        sv = saved(model, "dev")
        out[model] = {n: compare_saved(generate(m, model, n, "dev", rows, sha), sv, n) for n in ("greedy", "beam4_reg")}
        unload(m)
    out["exact"] = all(v["mismatches"] == 0 for k, d in out.items() if k != "exact" for v in d.values())
    return e07.write_json(REPORTS / "reproduction.json", out) and out


# ------------------------------------------------------------------ audit
def h1_evidence(greedy: list[dict], beam: list[dict], g_lp: list[float], b_lp: list[float]) -> dict:
    """Search-error evidence for one beam setting: where the beam top-1 differs from greedy, how often the model
    gives the greedy output a higher teacher-forced log-prob (canonical beam search would rarely allow this)."""
    diff = [(g, b, gl, bl) for g, b, gl, bl in zip(greedy, beam, g_lp, b_lp)
            if b is not None and top1(b["cands"]) != top1(g["cands"])]
    worse = [x for x in diff if x[3] < x[2] - 1e-4]
    shorter = [x for x in diff if len(top1(x[1]["cands"]) or "") < len(top1(x[0]["cands"]) or "")]
    junk = Counter()
    for b in beam:
        for c in b["cands"]:
            if c["status"] != "ok" or len(c["text"]) <= 1:
                junk[c["status"] if c["status"] != "ok" else "one_char"] += 1
    return {"items": len(greedy), "beam_top1_differs": len(diff), "beam_top1_less_probable_than_greedy": len(worse),
            "share_less_probable": round(len(worse) / max(1, len(diff)), 3),
            "beam_top1_shorter_than_greedy": len(shorter),
            "empty_or_one_char_or_bad_candidates": dict(junk),
            "rank1_empty": sum(b["cands"][0]["status"] == "empty" for b in beam)}


def score_reconstruction(m, recs: list[dict]) -> dict:
    """Generation scores vs teacher-forced sum log-probs of the same ids (lp 0 → identical up to numerics)."""
    pairs = [(r["word"], c["ids"]) for r in recs for c in r["cands"] if c["ids"]]
    gen = [c["score"] for r in recs for c in r["cands"] if c["ids"]]
    tf = tf_logprobs(m, pairs)
    d = [abs(a - b) for a, b in zip(gen, tf)]
    return {"sequences": len(d), "max_abs_diff": round(max(d), 6), "mean_abs_diff": round(mean(d), 7)}


@timed("audit")
def audit() -> dict:
    rows = dev_rows()
    dev_all = e06.items("dev")
    ev_all = e06.items("eval")
    checks = {}
    max_ref = max(len(B.encode_tgt(a)) for i in dev_all + ev_all for a in i.accepted)
    checks["byte_budget"] = {"max_reference_ids_with_eos": max_ref, "max_new_tokens": e07.MAX_NEW,
                             "ok": max_ref <= e07.MAX_NEW}
    checks["nfc"] = {"references_nfc": all(a == e06.nfc(a) for i in dev_all + ev_all for a in i.accepted),
                     "outputs": "e06_byt5.decode_ids NFC-normalises; evaluation.item_outcomes NFC-normalises both sides"}
    for model in CHECKPOINTS:
        m, done = load(model)
        sha = checkpoint_sha(model)
        sample = [B.encode_src(w) for _, w in rows[:16]]
        a = e07.gen(m, sample)
        b = e07.gen(m, sample)
        gc_ = m.generation_config.to_dict()
        keys = ("do_sample", "num_beams", "temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram_size",
                "bad_words_ids", "forced_bos_token_id", "forced_eos_token_id", "min_length", "min_new_tokens",
                "length_penalty", "early_stopping", "decoder_start_token_id", "eos_token_id", "pad_token_id",
                "max_length", "renormalize_logits", "suppress_tokens", "begin_suppress_tokens")
        g = generate(m, model, "greedy", "dev", rows, sha)
        b4 = generate(m, model, "beam4_reg", "dev", rows, sha)
        g_lp = tf_logprobs(m, [(r["word"], r["cands"][0]["ids"]) for r in g])
        bt = [next((c for c in r["cands"] if c["status"] == "ok" and c["text"]), None) for r in b4]
        b_lp = tf_logprobs(m, [(r["word"], c["ids"]) for r, c in zip(b4, bt) if c])
        it = iter(b_lp)
        b_lp = [next(it) if c else float("-inf") for c in bt]
        checks[model] = {
            "checkpoint": {"sha256": sha, "update": done["best"]["update"], "adapter_load": "strict (e07.load_adapter)"},
            "eval_mode": {"training_flag": m.training,
                          "dropout_off_same_output_twice": [x[0][:3] for x in a] == [x[0][:3] for x in b]},
            "generation_config": {k: gc_.get(k) for k in keys},
            "inputs": {"encoding": "both paths call e06_byt5.encode_src and the same padding/mask code in e07.gen",
                       "example_ids": sample[0]},
            "no_eos": {"greedy": sum(r["cands"][0]["status"] == "no_eos" for r in g),
                       "beam4_any": sum(c["status"] == "no_eos" for r in b4 for c in r["cands"])},
            "malformed_utf8": {"greedy": sum(r["cands"][0]["status"] == "malformed_utf8" for r in g),
                               "beam4_any": sum(c["status"] == "malformed_utf8" for r in b4 for c in r["cands"])},
            "score_reconstruction": {"greedy": score_reconstruction(m, g), "beam4_reg": score_reconstruction(m, b4)},
            "h1_beam4_reg": h1_evidence(g, b4, g_lp, b_lp),
            "alignment": {"dev_ids_match_saved": set(saved(model, "dev")) == {i for i, _ in rows},
                          "eval_ids_match_saved": set(saved(model, "eval")) == {i.id for i in ev_all}}}
        unload(m)
    return e07.write_json(REPORTS / "generation_audit.json", checks) and checks


# ------------------------------------------------------------------ dev comparison
def setting_metrics(items, recs: list[dict], seconds: float | None) -> tuple[dict, dict]:
    """Dev/eval metrics for one (model, setting): top-1 macro, CER, validity, failure counts, length, cost."""
    by = {r["id"]: r for r in recs}
    preds = {i.id: cand_list(by[i.id]["cands"]) for i in items}
    check_alignment(items, preds)
    tab, per = e07.system_table(items, preds)
    tops = [preds[i.id][0] if preds[i.id] else None for i in items]
    raw1 = Counter(by[i.id]["cands"][0]["status"] if by[i.id]["cands"] else "none" for i in items)
    n_c = [len(preds[i.id]) for i in items]
    tab |= {"raw_rank1_status": dict(raw1), "top1_empty_or_none": sum(t is None for t in tops),
            "truncated_no_eos_any": sum(c["status"] == "no_eos" for i in items for c in by[i.id]["cands"]),
            "repeated_top1": sum(repeated(t) for t in tops),
            "mean_top1_codepoints": round(mean(len(t or "") for t in tops), 3),
            "mean_top1_bytes": round(mean(len((t or "").encode()) for t in tops), 3),
            "distinct_candidates_per_item": round(mean(n_c), 3), "generation_seconds": seconds}
    return tab, per


def gen_seconds(model: str, name: str, split: str) -> float | None:
    s = [e["seconds"] for e in ledger() if e.get("event") == "generate" and e["model"] == model
         and e["setting"] == name and e["split"] == split]
    return s[-1] if s else None


def contrasts(items, per: dict[str, dict], pairs: dict[str, tuple[str, str]]) -> dict:
    """Paired family-bootstrap contrasts (top-1 points and CER reduction points), one shared resampling matrix."""
    idx = e05.boot_indices(len({i.family for i in items}))
    fm = {s: e05.family_means(items, {i: v["top1"] for i, v in p.items()}) for s, p in per.items()}
    fc = {s: e05.family_means(items, {i: -v["cer"] for i, v in p.items()}) for s, p in per.items()}
    out = {}
    for k, (a, b) in pairs.items():
        flips = Counter((per[a][i.id]["top1"], per[b][i.id]["top1"]) for i in items)
        out[k] = {"top1": e05.contrast(fm, [a], [b], idx), "cer_reduction": e05.contrast(fc, [a], [b], idx),
                  "items_plus_only": flips[(1, 0)], "items_minus_only": flips[(0, 1)]}
    return out


def run_settings(names: list[str], split: str, rows) -> None:
    for model in CHECKPOINTS:
        m, _ = load(model)
        sha = checkpoint_sha(model)
        for n in names:
            generate(m, model, n, split, rows, sha)
        unload(m)


def dev_table(names: list[str]) -> tuple[dict, dict]:
    rows, dev = dev_rows(), e06.primary(e06.items("dev"))
    tab, per = {}, {}
    for model in CHECKPOINTS:
        sha = checkpoint_sha(model)
        for n in names:
            p = cache_path(model, n, "dev", sha, rows)
            if p.exists():
                tab[f"{model}:{n}"], per[f"{model}:{n}"] = setting_metrics(dev, e07.load_jsonl(p),
                                                                           gen_seconds(model, n, "dev"))
    return tab, per


def follow_up_trigger(tab: dict) -> dict:
    """PLAN.md §3: run beam4_can_lp1 only if beam4_can top-1 is shorter than greedy on ≥ 60% of differing items and
    beam4_can's dev top-1 is below greedy's (per model; the follow-up runs for both models if either triggers)."""
    rows, out = dev_rows(), {}
    dev_ids = {i.id for i in e06.primary(e06.items("dev"))}
    for model in CHECKPOINTS:
        sha = checkpoint_sha(model)
        g = {r["id"]: top1(r["cands"]) for r in e07.load_jsonl(cache_path(model, "greedy", "dev", sha, rows))}
        b = {r["id"]: top1(r["cands"]) for r in e07.load_jsonl(cache_path(model, "beam4_can", "dev", sha, rows))}
        diff = [i for i in dev_ids if g[i] != b[i]]
        short = sum(len(b[i] or "") < len(g[i] or "") for i in diff)
        lower = tab[f"{model}:beam4_can"]["top1_macro"] < tab[f"{model}:greedy"]["top1_macro"]
        out[model] = {"differing": len(diff), "beam_shorter": short, "share_shorter": round(short / max(1, len(diff)), 3),
                      "beam4_can_below_greedy": lower, "trigger": lower and short >= 0.6 * max(1, len(diff))}
    out["run_follow_up"] = any(v["trigger"] for v in out.values() if isinstance(v, dict))
    return out


@timed("dev")
def dev() -> dict:
    rows = dev_rows()
    run_settings(PRIORITY, "dev", rows)
    tab, _ = dev_table(PRIORITY)
    trig = follow_up_trigger(tab)
    names = list(PRIORITY)
    if trig["run_follow_up"]:
        ledger_add({"event": "follow_up_registered", "setting": "beam4_can_lp1", "evidence": trig,
                    "value": "length_penalty 1.0 (score per generated byte)",
                    "expected": "longer beam-4 top-1, closer to greedy length; dev top-1 at least beam4_can's",
                    "why_smallest": "one value, one width, the canonical stopping rule already verified"})
        run_settings(["beam4_can_lp1"], "dev", rows)
        names.append("beam4_can_lp1")
    tab, per = dev_table(names)
    dev_items = e06.primary(e06.items("dev"))
    pairs = {}
    for model in CHECKPOINTS:
        for n in names:
            if n != "greedy":
                pairs[f"{model}:{n}_minus_greedy"] = (f"{model}:{n}", f"{model}:greedy")
        pairs[f"{model}:beam4_can_minus_beam4_reg"] = (f"{model}:beam4_can", f"{model}:beam4_reg")
    pairs["e08_minus_e07:greedy"] = ("e08:greedy", "e07:greedy")
    out = {"items": len(dev_items), "families": len({i.family for i in dev_items}), "settings": names,
           "table": tab, "contrasts": contrasts(dev_items, per, pairs), "follow_up_trigger": trig,
           "note": "dev is the checkpoint-selection set for both models; oracle coverage (top4_macro) is a diagnostic,"
                   " not top-1 quality"}
    with open(EXP / "dev_per_item.jsonl", "w", encoding="utf-8") as f:
        for i in dev_items:
            f.write(json.dumps({"id": i.id, "word": i.word, **{s: {"top1": per[s][i.id]["prediction"],
                                                                   "correct": per[s][i.id]["top1"]} for s in per}},
                               ensure_ascii=False) + "\n")
    return e07.write_json(REPORTS / "dev_comparison.json", out) and out


# ------------------------------------------------------------------ disagreements
GROUPS = ("greedy_right_beam_wrong", "beam_right_greedy_wrong", "both_wrong_different", "agree")


def group_of(g_ok: int, b_ok: int, same: bool) -> str | None:
    """Disagreement group of one item (None: both wrong with the same output, or both right and different).

    Examples:
        >>> group_of(1, 0, False), group_of(0, 1, False), group_of(0, 0, False), group_of(1, 1, True)
        ('greedy_right_beam_wrong', 'beam_right_greedy_wrong', 'both_wrong_different', 'agree')
    """
    if same:
        return "agree"
    if g_ok and not b_ok:
        return GROUPS[0]
    if b_ok and not g_ok:
        return GROUPS[1]
    return GROUPS[2] if not g_ok and not b_ok else None


def balanced_sample(groups: dict[str, list[str]], model: str, k: int = 5) -> dict[str, list[str]]:
    """Seeded sample of up to k ids per group (sorted ids first, so the draw does not depend on file order)."""
    return {g: sorted(random.Random(f"e09:sample:{model}:{g}").sample(sorted(ids), min(k, len(ids))))
            for g, ids in groups.items()}


@timed("diagnose")
def diagnose() -> dict:
    rows = dev_rows()
    dev = {i.id: i for i in e06.primary(e06.items("dev"))}
    out = {"label": "development items only; references are mined and unreviewed, shown as stored; agent "
                    "inspection (if any) is non-independent", "items": []}
    for model in CHECKPOINTS:
        sha = checkpoint_sha(model)
        load_c = lambda n: {r["id"]: r for r in e07.load_jsonl(cache_path(model, n, "dev", sha, rows))}  # noqa: E731
        g, br, bc = load_c("greedy"), load_c("beam4_reg"), load_c("beam4_can")
        hit = lambda t, i: int(bool(t) and e06.nfc(t) in {e06.nfc(a) for a in i.accepted})  # noqa: E731
        grp = {x: [] for x in GROUPS}
        for i in dev.values():
            gt, bt = top1(g[i.id]["cands"]), top1(br[i.id]["cands"])
            k = group_of(hit(gt, i), hit(bt, i), gt == bt)
            if k:
                grp[k].append(i.id)
        pick = balanced_sample(grp, model)
        m, _ = load(model)
        for k, ids in pick.items():
            for iid in ids:
                i = dev[iid]
                refs = list(dict.fromkeys([i.preferred, *i.accepted]))
                seqs = ([("greedy", 0, g[iid]["cands"][0])] + [("beam4_reg", r, c) for r, c in enumerate(br[iid]["cands"])]
                        + [("beam4_can", r, c) for r, c in enumerate(bc[iid]["cands"])])
                lps = tf_logprobs(m, [(i.word, c["ids"]) for _, _, c in seqs if c["ids"]]
                                  + [(i.word, B.encode_tgt(a)) for a in refs])
                it = iter(lps)
                cand = [{"source": s, "rank": r + 1, "text": c["text"], "status": c["status"], "gen_score": c["score"],
                         "tf_logprob": next(it) if c["ids"] else None, "bytes": len(c["ids"]),
                         "eos": B.EOS in c["ids"], "correct": hit(c["text"], i)} for s, r, c in seqs]
                out["items"].append({"model": model, "group": k, "id": iid, "english": i.word,
                                     "reference_preferred": i.preferred, "reference_accepted": i.accepted,
                                     "reference_tf_logprob": {a: next(it) for a in refs}, "candidates": cand})
        out[f"{model}_group_sizes"] = {x: len(v) for x, v in grp.items()}
        unload(m)
    return e07.write_json(REPORTS / "disagreements.json", out) and out


# ------------------------------------------------------------------ selection
def select_setting(tab: dict[str, dict]) -> str:
    """Highest top-1 macro, then lower mean CER, then lower generation seconds; `tab` maps setting → metrics.

    Examples:
        >>> select_setting({"greedy": {"top1_macro": .5, "mean_cer": .2, "generation_seconds": 10},
        ...                 "beam4": {"top1_macro": .5, "mean_cer": .2, "generation_seconds": 40},
        ...                 "beam8": {"top1_macro": .49, "mean_cer": .1, "generation_seconds": 5}})
        'greedy'
    """
    return min(tab, key=lambda n: (-tab[n]["top1_macro"], tab[n]["mean_cer"], tab[n]["generation_seconds"] or 0))


@timed("select")
def select() -> dict:
    d = json.loads((REPORTS / "dev_comparison.json").read_text())
    out = {"frozen": e06.now(), "rule": "dev top-1 macro, then lower dev mean CER, then lower generation seconds",
           "models": {}}
    for model in CHECKPOINTS:
        tab = {n: d["table"][f"{model}:{n}"] for n in d["settings"]}
        n = select_setting(tab)
        out["models"][model] = {"setting": n, "setting_hash": setting_hash(n), "definition": {**setting(n), **COMMON},
                                "checkpoint_sha256": checkpoint_sha(model),
                                "dev": {k: tab[n][k] for k in ("top1_macro", "mean_cer", "generation_seconds")}}
    return e07.write_json(REPORTS / "selected_settings.json", out) and out


def selection_committed() -> bool:
    import subprocess
    p = str((REPORTS / "selected_settings.json").relative_to(ROOT))
    logged = subprocess.run(["git", "log", "--oneline", "--", p], cwd=ROOT, capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain", "--", p], cwd=ROOT, capture_output=True, text=True).stdout
    return bool(logged.strip()) and not dirty.strip()


# ------------------------------------------------------------------ eval
def eval_settings() -> dict[str, list[str]]:
    sel = json.loads((REPORTS / "selected_settings.json").read_text())["models"]
    return {m: list(dict.fromkeys(["beam4_reg", "greedy", sel[m]["setting"]])) for m in CHECKPOINTS}


@timed("eval")
def evaluate() -> dict:
    """New eval predictions only for selected settings that are not registered ones (those reuse the saved files)."""
    if not selection_committed():
        raise RuntimeError("commit reports/selected_settings.json before historical eval")
    rows, frozen = eval_rows(), {}
    PRED.mkdir(parents=True, exist_ok=True)
    for model, names in eval_settings().items():
        new = [n for n in names if n not in REGISTERED]
        if not new:
            continue
        m, _ = load(model)
        sha = checkpoint_sha(model)
        for n in new:
            recs = generate(m, model, n, "eval", rows, sha)
            p = PRED / f"{model}_{n}_eval.jsonl"
            p.write_text("".join(json.dumps({"id": r["id"], "word": r["word"], "setting": n,
                                             "candidates": cand_list(r["cands"]),
                                             "raw": [{k: c[k] for k in ("text", "status", "score")} for c in r["cands"]]},
                                            ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
            frozen[p.name] = e06.sha(p)
        unload(m)
    out = {"frozen": e06.now(), "sha256": frozen,
           "reused_saved": {m: {n: str(SAVED[m]("eval").relative_to(ROOT)) for n in REGISTERED} for m in CHECKPOINTS}}
    return e07.write_json(REPORTS / "eval_frozen.json", out) and out


def eval_preds(model: str, name: str) -> dict[str, list[str]]:
    if name == "greedy":
        return {i: [r["greedy"]] if r["greedy"] else [] for i, r in saved(model, "eval").items()}
    if name == "beam4_reg":
        return {i: r["beam4"] for i, r in saved(model, "eval").items()}
    return {r["id"]: r["candidates"] for r in e07.load_jsonl(PRED / f"{model}_{name}_eval.jsonl")}


@timed("analyse")
def analyse() -> dict:
    frozen = json.loads((REPORTS / "eval_frozen.json").read_text())
    for name, h in frozen["sha256"].items():
        if e06.sha(PRED / name) != h:
            raise RuntimeError(f"{name} changed after freezing")
    ev_all, prim = e06.items("eval"), e06.primary(e06.items("eval"))
    sel = {m: v["setting"] for m, v in json.loads((REPORTS / "selected_settings.json").read_text())["models"].items()}
    tab, per = {}, {}
    for model, names in eval_settings().items():
        for n in names:
            p = eval_preds(model, n)
            check_alignment(ev_all, p)
            recs = [{"id": i, "cands": [{"text": t, "status": "ok"} for t in c]} for i, c in p.items()]
            tab[f"{model}:{n}"], per[f"{model}:{n}"] = setting_metrics(prim, recs, None)
    pairs = {}
    for model in CHECKPOINTS:
        pairs[f"{model}:greedy_minus_beam4_reg"] = (f"{model}:greedy", f"{model}:beam4_reg")
        if sel[model] not in REGISTERED:
            pairs[f"{model}:selected_minus_beam4_reg"] = (f"{model}:{sel[model]}", f"{model}:beam4_reg")
            pairs[f"{model}:selected_minus_greedy"] = (f"{model}:{sel[model]}", f"{model}:greedy")
    pairs["e08_minus_e07:greedy"] = ("e08:greedy", "e07:greedy")
    pairs["e08_minus_e07:selected"] = (f"e08:{sel['e08']}", f"e07:{sel['e07']}")
    pairs["e08_minus_e07:beam4_reg"] = ("e08:beam4_reg", "e07:beam4_reg")
    out = {"label": "exploratory: the historical E03 eval set (228 primary items, 222 families) has been examined "
                    "repeatedly (E03–E09); references are mined and unreviewed. Intervals are conditional on these two "
                    "checkpoints (no seed variance) and exclude the uncertainty of dev selection.",
           "selected": sel, "table": tab, "contrasts": contrasts(prim, {k: per[k] for k in tab}, pairs),
           "dev": json.loads((REPORTS / "dev_comparison.json").read_text())["table"]}
    with open(EXP / "results_per_item.jsonl", "w", encoding="utf-8") as f:
        for i in prim:
            f.write(json.dumps({"id": i.id, "word": i.word, "family": i.family,
                                **{s: {"top1": per[s][i.id]["prediction"], "correct": per[s][i.id]["top1"],
                                       "cer": round(per[s][i.id]["cer"], 4)} for s in per}}, ensure_ascii=False) + "\n")
    return e07.write_json(EXP / "results.json", out) and out


STEPS = {"register": register, "reproduce": reproduce, "audit": audit, "dev": dev, "diagnose": diagnose,
         "select": select, "eval": evaluate, "analyse": analyse}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(LIMITS["threads"])
    res = STEPS[a.step]()
    print(json.dumps(res, ensure_ascii=False, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
