"""E11: ByT5-small + the E07 recipe on E08's frozen real-data arm R (no Google AI Mode labels), one seed-0 run.

Reuses `scripts/e07.py` (model, resumable training loop, memory guard, budget), E09's cached generation and E10's
scoring and review helpers. Only the data, the run folder, the ledger, the limits and the experiment name differ.

Steps (in order):
    verify     re-derive arm R's counts, provenance, dedup, family and quarantine checks → reports/data_checks.json
               (+ local data/e11/pair_sources.json: every source of each training pair)
    register   freeze recipe, data, code hashes, exposure, evaluation plan and decision rule → registration.json
    train      the one authorised run (seed 0, ≤ 3,000 updates / 6 h; resumable) → models/e11/r_s0/
    freeze     greedy dev/eval predictions of E07, E11 and E08, written and hashed before scoring → reports/frozen.json
    evaluate   historical scoring (exploratory), contrasts, validity, lengths, error kinds → results.json
    review     blinded ≤ 60-word dev comparison of the three greedy outputs → data/e11/review/ (local) + manifest
    import     validate and score independent judgments, if a file exists (none is expected)
    status     phases and budgets

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e11.py verify
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e08  # noqa: E402
import e09  # noqa: E402
import e10  # noqa: E402  (pure helpers only; its bookkeeping is overridden below)
from khmer_translit import reference_data as rd  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E11-real-data-byt5"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
STATE_FILE = EXP / "state.json"
LEDGER = EXP / "runs" / "ledger.jsonl"
DATA = ROOT / "data" / "e11"                     # git-ignored: pair-level metadata, the review batch and its key
MODELS = ROOT / "models" / "e11"
RUN = "r_s0"
OUT = MODELS / RUN
GiB = 1024 ** 3
# The guard currently in use (E10 decision 8: owner-relaxed thresholds) with this task's limits: 6 threads, 1 run
LIMITS = {"threads": 6, "rss_max_bytes": int(6.5 * GiB), "min_available_bytes": int(0.4 * GiB), "pressure_checks": 3,
          "full_runs": 1, "full_max_updates": 3000, "full_max_hours": 6.0, "training_hours_total": 6.0,
          "wall_hours_total": 1e9, "inference_hours_total": 4.0}
ARM = e08.DATA / "training" / "arm_R.tsv"
ARM_SHA = "d00706cf1e16ccb81f9d367301d9a3fa637a46f79e2e8c8f0b2e501b57f8ec88"   # E08 training_registration.json
ARM_RS = e08.DATA / "training" / "arm_RS.tsv"
ARM_RS_SHA = "03a4988f84cdc97aff6b8356e019d1076c8d09a1ecec4004a2a909ed3e229b0f"
COMPARED = ("e07", "e11", "e08")      # e07: primary baseline; e08: historical reference only (AI Mode-influenced)
LENGTH_BINS = (("le4", 0, 4), ("5to8", 5, 8), ("ge9", 9, 99))                   # English letters, registered

e07.LEDGER = LEDGER
e07.EXPERIMENT = EXPERIMENT
e07.STATE = ROOT / "state" / "e11"
e07.LIMITS = LIMITS
e09.LEDGER = LEDGER
e09.LIMITS = {**e09.LIMITS, "active_hours": LIMITS["inference_hours_total"]}
e09.CHECKPOINTS["e11"] = OUT


# ------------------------------------------------------------------ state
def state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"phases": {}, "errors": []}


def set_state(phase: str, status: str, **info) -> dict:
    s = state()
    s["phases"][phase] = {"status": status, "updated": e06.now(), **info}
    s["updated"] = e06.now()
    return e07.write_json(STATE_FILE, s) and s


def inference_hours() -> float:
    """Active inference/evaluation hours (generation events and timed steps; training is counted separately)."""
    return sum(e.get("seconds", 0.0) for e in e07.ledger() if e.get("event") in ("generate", "step")) / 3600


def timed(step: str):
    def wrap(fn):
        def run(*a, **kw):
            if inference_hours() >= LIMITS["inference_hours_total"]:
                raise RuntimeError("inference/evaluation budget used up")
            t = time.perf_counter()
            try:
                out = fn(*a, **kw)
            except Exception as e:
                s = state()
                s["errors"].append({"time": e06.now(), "step": step, "error": repr(e)[:500]})
                e07.write_json(STATE_FILE, s)
                raise
            e07.ledger_add({"event": "step", "step": step, "seconds": round(time.perf_counter() - t, 1)})
            return out
        run.__doc__ = fn.__doc__
        return run
    return wrap


# ------------------------------------------------------------------ data
def read_arm(path: Path, sha: str) -> list[dict]:
    """Rows of a frozen arm file; refuses a changed file."""
    if e06.sha(path) != sha:
        raise RuntimeError(f"{path.name} differs from its frozen hash")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]]


def arm_rows() -> tuple[list[tuple[str, str]], list[dict]]:
    """(optimisation pairs in file order, all rows) of the frozen arm R."""
    rows = read_arm(ARM, ARM_SHA)
    return [(r["word"], r["khmer"]) for r in rows if r["part"] == "train"], rows


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@timed("verify")
def verify() -> dict:
    """Arm R's construction checks, re-derived from the files themselves (not from the E08 summaries)."""
    opt, rows = arm_rows()
    pairs = [(r["word"], r["khmer"]) for r in rows]
    pset = set(pairs)
    prov = Counter(r["provenance"] for r in rows)
    # every source of each pair: the arm keeps only the first provenance (control, then recovered, then ALT)
    src: dict[tuple[str, str], set] = defaultdict(set)
    for r in jl(e08.DATA / "recovery" / "control.jsonl"):
        src[(r["word"], r["khmer"])].add("control")
    rec = jl(e08.DATA / "recovery" / "recovered.jsonl")
    alt = jl(e08.DATA / "attested" / "alt_eligible.jsonl")
    for r in rec:
        src[(r["word"], r["target"])].add("recovered")
    for r in alt:
        src[(r["word"], r["target"])].add("attested_alt")
    # AI Mode spellings (accepted, additional, quarantined) and what arm R+S adds over R
    syn = set()
    for f in ("accepted.jsonl", "additional.jsonl", "quarantine.jsonl"):
        for r in jl(e08.DATA / "synthetic" / f):
            for k in [r.get("preferred"), *(r.get("alternatives") or [])]:
                if k:
                    syn.add((r["english"], nfc(k)))
    rs = read_arm(ARM_RS, ARM_RS_SHA)
    rs_pairs = {(r["word"], r["khmer"]) for r in rs}
    syn_identical = [p for p in pairs if (p[0], nfc(p[1])) in syn]
    # quarantined or rejected E08 material
    quarantine = set()
    for f in ("recovery/error_or_translation.jsonl", "recovery/uncertain.jsonl",
              "attested/alt_error_or_translation.jsonl", "attested/alt_uncertain.jsonl"):
        quarantine |= {(r["word"], r["target"]) for r in jl(e08.DATA / f)}
    q_overlap = [p for p in pairs if p in quarantine]
    # dropped on the way into R (dedup or train-val family exclusion), from the source files
    tv = {r["word"] for r in rows if r["part"] == "train_val"}
    tv_stems = {s for w in tv for s in rd.stems(w)}
    dropped = Counter()
    for name, recs in (("recovered", rec), ("attested_alt", alt)):
        for r in recs:
            p = (r["word"], r["target"])
            if r["word"] in tv or rd.related(r["word"], tv_stems):
                dropped[f"{name}:train_val_family"] += 1
            elif p in pset and name not in {x["provenance"] for x in rows if (x["word"], x["khmer"]) == p}:
                dropped[f"{name}:duplicate_of_earlier_source"] += 1
    e07.check_disjoint(opt)
    prot = e08.protected()
    added_prot = sorted({r["word"] for r in rows if r["provenance"] != "control" and e08.is_protected(r["word"], prot)})
    e07_opt, _, _ = e07.load_train()
    control_opt = [(r["word"], r["khmer"]) for r in rows if r["provenance"] == "control" and r["part"] == "train"]
    out = {"arm": str(ARM.relative_to(ROOT)), "sha256": e06.sha(ARM), "matches_registered_hash": True,
           "rows": len(rows), "optimisation_rows": len(opt), "train_val_rows": len(rows) - len(opt),
           "optimisation_words": len({w for w, _ in opt}), "rows_by_provenance": dict(prov),
           "optimisation_rows_by_provenance": dict(Counter(r["provenance"] for r in rows if r["part"] == "train")),
           "source_file_rows": {"recovered.jsonl": len(rec), "alt_eligible.jsonl": len(alt)},
           "dropped_from_source_files": dict(dropped),
           "control_equals_e07_training_rows": sorted(control_opt) == sorted(e07_opt),
           "dedup": {"exact_duplicate_pairs": len(pairs) - len(pset),
                     "nfc_duplicate_pairs": len(pairs) - len({(w, nfc(k)) for w, k in pairs}),
                     "targets_not_nfc": sum(k != nfc(k) for _, k in pairs),
                     "pairs_with_several_sources": dict(Counter("+".join(sorted(src[p])) for p in pairs
                                                                if len(src[p]) > 1)),
                     "pairs_without_a_source_record": sum(1 for p in pairs if not src[p])},
           "synthetic": {"ai_mode_rows_in_R": prov.get("google_ai_mode_synthetic", 0),
                         "R_is_subset_of_RS": pset <= rs_pairs,
                         "RS_minus_R_by_provenance": dict(Counter(r["provenance"] for r in rs
                                                                  if (r["word"], r["khmer"]) not in pset)),
                         "R_pairs_identical_to_an_ai_mode_spelling": len(syn_identical),
                         "those_pairs_by_source": dict(Counter("+".join(sorted(src[p])) for p in syn_identical))},
           "quarantine": {"known_quarantined_pairs": len(quarantine), "R_pairs_also_in_a_quarantine_file": len(q_overlap),
                          "those_pairs_by_source": dict(Counter("+".join(sorted(src[p])) for p in q_overlap))},
           "families": {"dev_eval_word_overlap": 0, "added_rows_in_e08_protected_families": len(added_prot),
                        "added_rows_in_train_val_families": sum(1 for r in rows if r["provenance"] != "control"
                                                                and (r["word"] in tv or rd.related(r["word"], tv_stems)))},
           "byte_limit_rejections": sum(1 for w, k in opt if B.fits(w, k)),
           "alternatives": {"words_with_several_spellings": sum(n > 1 for n in Counter(w for w, _ in opt).values()),
                            "max_spellings_per_word": max(Counter(w for w, _ in opt).values())}}
    DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(DATA / "pair_sources.json", [{"word": w, "khmer": k, "sources": sorted(src[(w, k)])}
                                                for w, k in pairs])
    ok = (not out["dedup"]["exact_duplicate_pairs"] and not out["dedup"]["nfc_duplicate_pairs"]
          and not out["synthetic"]["ai_mode_rows_in_R"] and out["synthetic"]["R_is_subset_of_RS"]
          and set(out["synthetic"]["RS_minus_R_by_provenance"]) == {"google_ai_mode_synthetic"}
          and not out["dedup"]["pairs_without_a_source_record"] and not added_prot
          and not out["families"]["added_rows_in_train_val_families"] and not out["byte_limit_rejections"]
          and out["control_equals_e07_training_rows"])
    out["ok"] = ok
    set_state("verify", "done" if ok else "failed")
    return e07.write_json(REPORTS / "data_checks.json", out) and out


# ------------------------------------------------------------------ registration
def recipe() -> dict:
    """The E07 recipe, unchanged (hash-checked against E07's registration)."""
    return e07.full_cfg()


def draw_counts(n_rows: int, cfg: dict) -> Counter:
    """Row draws of `e07.fit`'s default order (row-level; each epoch a seeded permutation) over the whole run."""
    order = lambda ep: random.Random(f"{cfg['seed']}:order:{ep}").sample(range(n_rows), n_rows)  # noqa: E731
    c, ep, pos, perm = Counter(), 0, 0, order(0)
    for _ in range(cfg["max_updates"] * cfg["grad_accumulation"]):
        if pos + cfg["batch"] > n_rows:
            ep, pos, perm = ep + 1, 0, order(ep + 1)
        c.update(perm[pos:pos + cfg["batch"]])
        pos += cfg["batch"]
    return c


def exposure(cfg: dict, opt, rows) -> dict:
    prov = [r["provenance"] for r in rows if r["part"] == "train"]
    draws = draw_counts(len(opt), cfg)
    by_prov, by_word = Counter(), Counter()
    for i, n in draws.items():
        by_prov[prov[i]] += n
        by_word[opt[i][0]] += n
    tot = sum(draws.values())
    pw = sorted(by_word.values())
    return {"sampling": "row-level: each epoch a seeded permutation of all optimisation rows (E07's e07.fit default); "
                        "a word with k spellings is drawn k times per epoch; no source weighting",
            "effective_batch": cfg["batch"] * cfg["grad_accumulation"], "updates": cfg["max_updates"],
            "draws_total": tot, "epochs": round(tot / len(opt), 2),
            "draws_by_provenance": {p: {"draws": n, "share": round(n / tot, 4)} for p, n in sorted(by_prov.items())},
            "draws_per_word": {"min": pw[0], "median": pw[len(pw) // 2], "max": pw[-1]},
            "e07_reference": {"rows": 2255, "epochs": round(cfg["max_updates"] * cfg["batch"] / 2255, 2)}}


DECISION_RULE = {
    "primary": "E11 greedy − E07 greedy on the historical eval split (paired family bootstrap, 10,000, seed 20261027)",
    "useful_real_data_only_candidate": "eval exact-agreement difference CI lower bound > 0 and eval CER not higher "
                                       "(CER-reduction point estimate ≥ 0); dev top-1 not below E07's",
    "regression_relative_to_e07": "eval difference CI upper bound < 0, or point ≤ −3.0 points with eval CER higher",
    "uncertain_alternative": "anything else",
    "note": "classification of historical reference agreement only; phonetic quality and publication permissions "
            "are reported separately; no tuning after evaluation",
}


@timed("register")
def register() -> dict:
    """Freeze recipe, data and code hashes, exposure, the evaluation plan and the decision rule (before training)."""
    checks = json.loads((REPORTS / "data_checks.json").read_text())
    if not checks.get("ok"):
        raise RuntimeError("data checks did not pass")
    cfg = recipe()
    opt, rows = arm_rows()
    files = ["scripts/e11.py", "scripts/e07.py", "scripts/e06_byt5.py", "scripts/e06.py", "scripts/e09.py",
             "scripts/e10.py", "experiments/E07-byt5-diagnosis/full_recipe.json", "experiments/E11-real-data-byt5/PLAN.md",
             "research/requirements-byt5.txt", "research/requirements-byt5.lock.txt"]
    reg = {"registered": e06.now(), "experiment": EXPERIMENT, "run": RUN, "recipe": cfg, "cfg_hash": e07.cfg_hash(cfg),
           "init": "pinned google/byt5-small base + fresh LoRA adapter (seed 0); not initialised from E08/E10",
           "data": {"arm": str(ARM.relative_to(ROOT)), "sha256": ARM_SHA, "optimisation_hash": e07.rows_hash(opt),
                    "optimisation_rows": len(opt), "train_val_rows": len(rows) - len(opt),
                    "rows_by_provenance": checks["optimisation_rows_by_provenance"]},
           "exposure": exposure(cfg, opt, rows), "limits": LIMITS,
           "guard": "stop (resumable) after 3 consecutive readings with process RSS > 6.5 GiB or MemAvailable < 0.4 GiB",
           "selection": "dev greedy top-1 macro, then lower dev CER, then earlier update; checkpoints 600/1200/1800/"
                        "2400/3000",
           "evaluation": {"decoding": "greedy (headline and selection)", "compared": list(COMPARED),
                          "splits": ["dev (selection set)", "eval (historical, examined since E03, unreviewed refs)"],
                          "bootstrap": "paired by family, 10,000 resamples, seed 20261027",
                          "length_bins_english_letters": [list(b) for b in LENGTH_BINS],
                          "error_kinds": "e10.error_kind against the closest reference (vowels_only, marks_only, "
                                         "consonants_changed, consonant_added, consonant_dropped, both)",
                          "missing_ending": "the prediction's consonant skeleton is a proper prefix of the closest "
                                            "reference's, or a candidate ended without EOS",
                          "validity": "statuses, orthography, empty, repeated outputs"},
           "decision_rule": DECISION_RULE,
           "source_sha256": {f: e06.sha(ROOT / f) for f in files}, "base_sha256": B.BASE_SHA256}
    set_state("register", "done", cfg_hash=reg["cfg_hash"])
    return e07.write_json(EXP / "registration.json", reg) and reg


# ------------------------------------------------------------------ training
def train() -> dict:
    """The one authorised run; resumable (config, data and base hashes are checked by `e07.fit`)."""
    reg = json.loads((EXP / "registration.json").read_text())
    cfg = recipe()
    opt, _ = arm_rows()
    if e07.cfg_hash(cfg) != reg["cfg_hash"] or e07.rows_hash(opt) != reg["data"]["optimisation_hash"]:
        raise RuntimeError("recipe or data differ from the registration")
    e07.check_disjoint(opt)
    dev = e06.primary(e06.items("dev"))
    set_state("train", "running")
    done = e07.fit(RUN, "full", cfg, opt, lambda m: e07.dev_selection(m, dev), set(cfg["eval_at"]), OUT)
    set_state("train", "done" if "best" in done else "stopped", stop=done.get("stop"), best=done.get("best"))
    return done


# ------------------------------------------------------------------ freeze and evaluate
def recs(model: str, split: str) -> list[dict]:
    rows = e09.dev_rows() if split == "dev" else e09.eval_rows()
    return e07.load_jsonl(e09.cache_path(model, "greedy", split, e09.checkpoint_sha(model), rows))


@timed("freeze")
def freeze() -> dict:
    """Greedy dev/eval predictions of every compared model, written and hashed before any historical scoring."""
    done = json.loads((OUT / "done.json").read_text())
    for c in COMPARED:
        m = None
        sha = e09.checkpoint_sha(c)
        for split, rows in (("dev", e09.dev_rows()), ("eval", e09.eval_rows())):
            if not e09.cache_path(c, "greedy", split, sha, rows).exists():
                m = m or e09.load(c)[0]
                e09.generate(m, c, "greedy", split, rows, sha)
        if m is not None:
            e09.unload(m)
    PRED.mkdir(parents=True, exist_ok=True)
    files = {}
    for c in COMPARED:
        for split in ("dev", "eval"):
            p = PRED / f"{c}_{split}_greedy.jsonl"
            p.write_text("".join(json.dumps({"id": r["id"], "word": r["word"], "prediction": e09.top1(r["cands"]),
                                             "status": r["cands"][0]["status"] if r["cands"] else "none"},
                                            ensure_ascii=False) + "\n" for r in recs(c, split)), encoding="utf-8")
            files[p.name] = e06.sha(p)
    out = {"frozen": e06.now(), "selected": {"run": RUN, "update": done["best"]["update"],
                                             "dev_top1": done["best"]["dev_top1"], "dev_cer": done["best"]["dev_cer"],
                                             "checkpoint_sha256": e09.checkpoint_sha("e11")},
           "baselines": {c: {"checkpoint": str((e09.CHECKPOINTS[c] / "best.pt").relative_to(ROOT)),
                             "sha256": e09.checkpoint_sha(c)} for c in ("e07", "e08")},
           "decoding": {"setting": "greedy", **e09.setting("greedy"), **e09.COMMON, "hash": e09.setting_hash("greedy")},
           "prediction_files_sha256": files}
    set_state("freeze", "done")
    return e07.write_json(REPORTS / "frozen.json", out) and out


def missing_ending(pred: str | None, refs: list[str]) -> bool:
    """True if the prediction's consonant skeleton is a proper prefix of the closest reference's skeleton.

    Examples:
        >>> missing_ending("កុំពូ", ["កុំពូទ័រ"])
        True
        >>> missing_ending("កុំពូទ័រ", ["កុំពូទ័រ"])
        False
    """
    if not pred:
        return False
    best = min(refs, key=lambda a: e07.cer_of(pred, [a]))
    s, r = e10.skeleton(pred), e10.skeleton(best)
    return len(s) < len(r) and r.startswith(s)


def length_bin(word: str) -> str:
    n = len(word)
    return next(name for name, lo, hi in LENGTH_BINS if lo <= n <= hi)


@timed("evaluate")
def evaluate() -> dict:
    """Historical reference agreement (exploratory) for the frozen greedy predictions, with paired contrasts."""
    fz = json.loads((REPORTS / "frozen.json").read_text())
    for c in COMPARED:
        for split in ("dev", "eval"):
            if e06.sha(PRED / f"{c}_{split}_greedy.jsonl") != fz["prediction_files_sha256"][f"{c}_{split}_greedy.jsonl"]:
                raise RuntimeError("frozen predictions changed")
    out = {"label": "Historical mined-reference agreement only. Eval has been examined repeatedly since E03 and its "
                    "references are unreviewed web-mined spellings; no number here measures phonetic accuracy.",
           "frozen": fz["frozen"], "selected": fz["selected"]}
    for split in ("dev", "eval"):
        items = e06.primary(e06.items(split))
        tab, per = {}, {}
        for c in COMPARED:
            tab[c], per[c] = e09.setting_metrics(items, recs(c, split), e09.gen_seconds(c, "greedy", split))
        con = {"e11_minus_e07": e10.seed_contrast(items, per, ["e11"], ["e07"]),
               "e08_minus_e07 (historical reference)": e10.seed_contrast(items, per, ["e08"], ["e07"]),
               "e11_minus_e08 (historical reference)": e10.seed_contrast(items, per, ["e11"], ["e08"])}
        kinds = {c: dict(Counter(e10.error_kind(per[c][i.id]["prediction"], i.accepted) for i in items
                                 if not per[c][i.id]["top1"])) for c in COMPARED}
        ending = {c: sum(missing_ending(per[c][i.id]["prediction"], i.accepted) for i in items) for c in COMPARED}
        bins = {}
        for name, _, _ in LENGTH_BINS:
            sub = [i for i in items if length_bin(i.word) == name]
            bins[name] = {"items": len(sub), **{c: round(e06.macro(sub, {i.id: per[c][i.id]["top1"] for i in sub}), 3)
                                                for c in COMPARED}}
        opsl = {c: e10.operational(recs(c, split)) for c in COMPARED}
        flips = Counter((per["e07"][i.id]["top1"], per["e11"][i.id]["top1"]) for i in items)
        out[split] = {"items": len(items), "families": len({i.family for i in items}), "table": tab, "contrasts": con,
                      "error_kinds": kinds, "missing_ending": ending, "length_bins_top1": bins, "operational": opsl,
                      "e07_vs_e11_item_outcomes": {"both_right": flips[(1, 1)], "only_e07": flips[(1, 0)],
                                                   "only_e11": flips[(0, 1)], "both_wrong": flips[(0, 0)]}}
        with open(EXP / f"results_per_item_{split}.jsonl", "w", encoding="utf-8") as f:
            for i in items:
                f.write(json.dumps({"id": i.id, "word": i.word, "family": i.family, "length_bin": length_bin(i.word),
                                    **{c: {"prediction": per[c][i.id]["prediction"], "correct": per[c][i.id]["top1"],
                                           "cer": round(per[c][i.id]["cer"], 4)} for c in COMPARED}},
                                   ensure_ascii=False) + "\n")
    done = json.loads((OUT / "done.json").read_text())
    out["training"] = {"history": [{k: h[k] for k in ("update", "dev_top1", "dev_cer")} for h in done["history"]],
                       "selected_update": done["best"]["update"], "updates": done["updates"],
                       "active_training_hours": done["elapsed_hours"],
                       "segments": [e for e in e07.ledger() if e.get("event") == "segment"],
                       "peak_rss_mb": done.get("peak_rss_mb"), "min_available_mb": done.get("min_available_mb")}
    out["decision"] = classify(out)
    set_state("evaluate", "done", decision=out["decision"]["class"])
    return e07.write_json(EXP / "results.json", out) and out


def classify(res: dict) -> dict:
    """Apply the registered decision rule to the historical contrasts."""
    c = res["eval"]["contrasts"]["e11_minus_e07"]
    d, lo, hi = c["top1"]["diff_pp"], *c["top1"]["ci95_pp"]
    cer_red = c["cer_reduction"]["diff_pp"]
    dev_ok = res["dev"]["table"]["e11"]["top1_macro"] >= res["dev"]["table"]["e07"]["top1_macro"]
    if lo > 0 and cer_red >= 0 and dev_ok:
        cls = "useful_real_data_only_candidate"
    elif hi < 0 or (d <= -3.0 and cer_red < 0):
        cls = "regression_relative_to_e07"
    else:
        cls = "uncertain_alternative"
    return {"class": cls, "eval_top1_diff_pp": d, "ci95_pp": [lo, hi], "eval_cer_reduction_pp": cer_red,
            "dev_not_below_e07": dev_ok, "rule": DECISION_RULE}


# ------------------------------------------------------------------ review (dev only, blinded)
REVIEW_DIR = EXP / "review"
REVIEW_DATA = DATA / "review"
REVIEW_SEED = "e11-dev-review"
N_REPRESENTATIVE, N_DISAGREEMENT = 40, 20


def difference(texts: list[str]) -> str:
    """How the distinct candidates of one word differ: `single` (all identical), `vowels_or_marks` (same consonant
    skeleton) or `consonants_or_syllables`."""
    if len(texts) < 2:
        return "single"
    return "vowels_or_marks" if len({e10.skeleton(t) for t in texts}) == 1 else "consonants_or_syllables"


@timed("review")
def review() -> dict:
    """≤ 60 dev words: 40 representative (one per sampled family) and 20 where E07, E11 and E08 disagree; the
    distinct greedy outputs are shown blinded (≤ 3 per word), with the intended English reading."""
    dev = e06.primary(e06.items("dev"))
    g = {c: {r["id"]: e09.top1(r["cands"]) for r in recs(c, "dev")} for c in COMPARED}
    fams = sorted({i.family for i in dev}, key=lambda f: e06.order_key(f"{REVIEW_SEED}:rep", f))
    by_fam = defaultdict(list)
    for i in dev:
        by_fam[i.family].append(i)
    rep = [sorted(by_fam[f], key=lambda i: i.id)[0] for f in fams[:N_REPRESENTATIVE]]
    used = {i.family for i in rep}
    pool = [i for i in dev if i.family not in used and len({e06.nfc(g[c][i.id] or "") for c in COMPARED}) > 1]
    dis = sorted(pool, key=lambda i: e06.order_key(f"{REVIEW_SEED}:dis", i.id))[:N_DISAGREEMENT]
    items, key = [], {}
    order = sorted([("representative", i) for i in rep] + [("disagreement", i) for i in dis],
                   key=lambda x: e06.order_key(f"{REVIEW_SEED}:order", x[1].id))
    for n, (sample, i) in enumerate(order, 1):
        rid = f"r{n:02d}"
        cands = e10.review_candidates([(c, g[c][i.id]) for c in COMPARED], f"{REVIEW_SEED}:{rid}")
        items.append({"item_id": rid, "word": i.word, "intended_reading": i.intended_phones, "accent": i.accent,
                      "candidates_differ_in": difference([t for _, t, _ in cands]),
                      "candidates": [{"label": lb, "text": t} for lb, t, _ in cands]})
        key[rid] = {"dev_id": i.id, "sample": sample, "origins": {lb: o for lb, _, o in cands}}
    REVIEW_DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(REVIEW_DATA / "review_batch.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    e07.write_json(REVIEW_DATA / "origin_key.json", key)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["item_id", "word", "intended_reading", "label", "candidate", "judgment"])
    for it in items:
        for c in it["candidates"]:
            w.writerow([it["item_id"], it["word"], it["intended_reading"], c["label"], c["text"], ""])
        w.writerow([it["item_id"], it["word"], it["intended_reading"], "NONE", "(no candidate acceptable)", ""])
    (REVIEW_DATA / "judgments_template.tsv").write_text(buf.getvalue(), encoding="utf-8")
    man = {"created": e06.now(), "items": len(items), "representative": len(rep), "disagreement": len(dis),
           "candidates": sum(len(it["candidates"]) for it in items),
           "candidates_per_item": dict(Counter(len(it["candidates"]) for it in items)),
           "differ_in": {s: dict(Counter(it["candidates_differ_in"] for it in items
                                         if key[it["item_id"]]["sample"] == s)) for s in ("representative", "disagreement")},
           "judgments": list(e10.JUDGMENTS) + ["none_acceptable (on the NONE row)"],
           "sha256": {p.name: e06.sha(p) for p in sorted(REVIEW_DATA.glob("*")) if p.is_file()},
           "local_files": str(REVIEW_DATA.relative_to(ROOT)),
           "note": "dev items (the selection set): a development comparison, not a fresh test. Only model outputs are "
                   "shown (no mined reference); origins hidden; the key stays local."}
    set_state("review", "prepared", items=len(items))
    return e07.write_json(REVIEW_DIR / "review_manifest.json", man) and man


@timed("import")
def import_judgments(judgments: str | None = None) -> dict:
    """Validate and score an independent judgment file (TSV like the template, plus `<file>.provenance.json`)."""
    path = Path(judgments) if judgments else REVIEW_DATA / "judgments.tsv"
    if not path.exists():
        set_state("review_scored", "skipped", reason="no judgment file (independent review not collected)")
        return {"status": "no_judgments", "path": str(path)}
    prov = path.with_suffix(path.suffix + ".provenance.json")
    if not prov.exists():
        raise FileNotFoundError(f"{prov} is required (who judged, when, under which policy, independence)")
    batch = json.loads((REVIEW_DATA / "review_batch.json").read_text())
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t"))
    # a partial review is scored on the items it judges (all of their candidates must be judged); coverage is reported
    judged = {r["item_id"] for r in rows if r.get("judgment")}
    total = len(batch["items"])
    batch = {**batch, "items": [it for it in batch["items"] if it["item_id"] in judged]}
    rows = [r for r in rows if r["item_id"] in judged]
    v = e10.validate_judgments(rows, batch)
    if not v["ok"]:
        return {"status": "invalid", **v}
    key = json.loads((REVIEW_DATA / "origin_key.json").read_text())
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["label"] != "NONE" and r["judgment"]:
            k = key[r["item_id"]]
            for o in k["origins"][r["label"]]:
                acc[k["sample"]][o].append(r["judgment"] == "acceptable")
    items = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # sample → origin → [items with an acceptable output, items]
    for it in batch["items"]:
        k = key[it["item_id"]]
        ok = {r["label"] for r in rows if r["item_id"] == it["item_id"] and r["judgment"] == "acceptable"}
        for lb, origins in k["origins"].items():
            for o in origins:
                items[k["sample"]][o][0] += lb in ok
                items[k["sample"]][o][1] += 1
    out = {"status": "scored", "provenance": json.loads(prov.read_text()),
           "coverage": {"judged_items": len(batch["items"]), "batch_items": total,
                        "by_sample": dict(Counter(key[it["item_id"]]["sample"] for it in batch["items"]))},
           "acceptable_items": {s: {o: {"acceptable": a, "items": n, "share": round(a / n, 3)}
                                    for o, (a, n) in sorted(d.items())} for s, d in items.items()},
           "acceptable_share": {s: {o: round(mean(v), 3) for o, v in d.items()} for s, d in acc.items()}}
    set_state("review_scored", "done")
    return e07.write_json(REPORTS / "review_scores.json", out) and out


def status() -> dict:
    s = state()
    s["inference_hours"] = round(inference_hours(), 3)
    s["training"] = e07.budget_left("full")
    return s


STEPS = {"verify": verify, "register": register, "train": train, "freeze": freeze, "evaluate": evaluate,
         "review": review, "import": import_judgments, "status": status}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--judgments")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(LIMITS["threads"])
    res = STEPS[a.step](**({"judgments": a.judgments} if a.step == "import" and a.judgments else {}))
    print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
