"""E10 release cycle for the ByT5 English → Khmer transliterator: diagnosis, review preparation, a bounded training
cycle, candidate freezing and evaluation. Packaging lives in `khmer_translit.byt5` and `scripts/e10_release.py`.

Reuses E07 (model building, resumable training loop, guard, scoring), E08 (frozen data arm, protected families) and
E09 (corrected generation, cache by checkpoint/setting/data hash). Plan: `experiments/E10-release-cycle/PLAN.md`.

Steps (state `experiments/E10-release-cycle/state.json`, ledger `runs/ledger.jsonl`):
    diagnose     E08 training behaviour and dev errors → reports/diagnosis.json
    review       blinded dev-review batch (≤ 60 words, ≤ 4 candidates) → review/ + data/e10/review/ (key, local)
    confirm      fresh confirmation-input manifest (≈ 120 families, before new outputs) → confirmation/
    import       validate and score review judgments (`--judgments FILE`) → reports/review_scores.json
    register     freeze the candidate recipes and selection rule → recipes/*.json, registration.json
    train        one registered run (`--run A_s0` …; resumable) → models/e10/<run>/
    gate         the replication gate on dev → reports/gate.json
    freeze       candidate identities and decoding before historical eval → reports/candidates_frozen.json
    evaluate     dev + historical eval + confirmation diagnostics → results.json, results_per_item.jsonl
    status       print state.json

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e10.py diagnose
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e08  # noqa: E402
import e09  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E10-release-cycle"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
STATE_FILE = EXP / "state.json"
LEDGER = EXP / "runs" / "ledger.jsonl"
DATA = ROOT / "data" / "e10"
MODELS = ROOT / "models" / "e10"
GiB = 1024 ** 3
# Memory guard relaxed by the owner on 2026-10-28 during A_s0 ("i am willing to use all resource"): was RSS 5 GiB and
# 1.5 GiB available; now only a last-resort stop before the kernel's OOM killer (resumable from last.pt)
LIMITS = {"threads": 6, "rss_max_bytes": int(6.5 * GiB), "min_available_bytes": int(0.4 * GiB), "pressure_checks": 3,
          "full_runs": 4, "full_max_updates": 3000, "full_max_hours": 6.0, "training_hours_total": 24.0,
          "wall_hours_total": 1e9, "inference_hours_total": 4.0}
E08_DIR = ROOT / "models" / "e08" / "rs_s0"
ARM = e08.DATA / "training" / "arm_RS.tsv"
ARM_SHA = "03a4988f84cdc97aff6b8356e019d1076c8d09a1ecec4004a2a909ed3e229b0f"

# E10 bookkeeping for the shared E07 training loop: own ledger, limits, experiment name and state folder
e07.LEDGER = LEDGER
e07.EXPERIMENT = EXPERIMENT
e07.STATE = ROOT / "state" / "e10"
e07.LIMITS = LIMITS
# E09's generation cache, keyed by checkpoint hash: shared, so identical (checkpoint, setting, data) runs once
e09.LEDGER = LEDGER
e09.LIMITS = {**e09.LIMITS, "active_hours": LIMITS["inference_hours_total"]}


# ------------------------------------------------------------------ state and ledger
def state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"phases": {}, "errors": []}


def set_state(phase: str, status: str, **info) -> dict:
    """Record a phase's status (and details) in the machine-readable state file; survives interruptions."""
    s = state()
    s["phases"][phase] = {"status": status, "updated": e06.now(), **info}
    s["updated"] = e06.now()
    e07.write_json(STATE_FILE, s)
    return s


def inference_hours() -> float:
    """Active inference/evaluation hours: E09-style `generate` events plus timed E10 steps (not training)."""
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
def arm_rows() -> tuple[list[tuple[str, str]], list[dict]]:
    """(optimisation pairs in file order, all rows) of the frozen E08 arm R+S; refuses a changed file."""
    if e06.sha(ARM) != ARM_SHA:
        raise RuntimeError("arm_RS.tsv differs from the frozen hash")
    lines = ARM.read_text(encoding="utf-8").splitlines()
    rows = [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]]
    return [(r["word"], r["khmer"]) for r in rows if r["part"] == "train"], rows


KH_MARKS = {"៉", "៊"}                    # muusikatoan ៉, triisap ៊
KH_VOWEL = set(chr(c) for c in range(0x17B6, 0x17C6)) | {"ំ", "ះ", "ៈ"}
KH_CONS = set(chr(c) for c in range(0x1780, 0x17A3))


def skeleton(t: str) -> str:
    """Consonant skeleton (consonant letters only), to tell a missing syllable from a vowel change.

    Examples:
        >>> skeleton("ហ្គាវីន"), skeleton("ម៉ានី")
        ('ហគវន', 'មន')
    """
    return "".join(c for c in t if c in KH_CONS)


def error_kind(pred: str | None, refs: list[str]) -> str:
    """Category of a wrong top-1 against the closest accepted reference (by edit distance).

    `empty`; `marks_only` (differs only in ៉/៊); `vowels_only`; `consonant_dropped` (the reference's consonant
    skeleton loses letters, e.g. a missing syllable or ending); `consonant_added`; `consonants_changed`; `both`.

    Examples:
        >>> error_kind("មានី", ["ម៉ានី"]), error_kind("កាត", ["កេត"]), error_kind("រ៉ាល", ["រ៉ាល់ហ្វ"])
        ('marks_only', 'vowels_only', 'consonant_dropped')
    """
    from khmer_translit.evaluation import edit_distance
    if not pred:
        return "empty"
    ref = min(refs, key=lambda a: edit_distance(pred, a))
    strip = lambda s, drop: "".join(c for c in s if c not in drop)  # noqa: E731
    if strip(pred, KH_MARKS) == strip(ref, KH_MARKS):
        return "marks_only"
    sp, sr = skeleton(pred), skeleton(ref)
    if sp == sr:
        return "vowels_only"
    if len(sp) < len(sr):
        return "consonant_dropped"
    if len(sp) > len(sr):
        return "consonant_added"
    vow_same = [c for c in pred if c in KH_VOWEL] == [c for c in ref if c in KH_VOWEL]
    return "consonants_changed" if vow_same else "both"


# ------------------------------------------------------------------ diagnosis
def draw_counts(n_rows: int, cfg: dict) -> Counter:
    """Row draws of `e07.fit`'s deterministic row-level order over the run (same as E08's exposure report)."""
    order = lambda ep: random.Random(f"{cfg['seed']}:order:{ep}").sample(range(n_rows), n_rows)  # noqa: E731
    c, ep, pos, perm = Counter(), 0, 0, order(0)
    for _ in range(cfg["max_updates"] * cfg["grad_accumulation"]):
        if pos + cfg["batch"] > n_rows:
            ep, pos, perm = ep + 1, 0, order(ep + 1)
        c.update(perm[pos:pos + cfg["batch"]])
        pos += cfg["batch"]
    return c


@timed("diagnose")
def diagnose() -> dict:
    """E08 training behaviour (loss, checkpoints, exposure, fit by provenance) and dev error categories."""
    cfg = e07.full_cfg()
    opt, rows = arm_rows()
    prov = [r["provenance"] for r in rows if r["part"] == "train"]
    done = json.loads((E08_DIR / "done.json").read_text())
    met = [json.loads(x) for x in (E08_DIR / "metrics.jsonl").read_text().splitlines()]
    tr = [x for x in met if "loss" in x]
    windows = {f"{a + 1}-{a + 300}": round(mean(x["loss"] for x in tr if a < x["update"] <= a + 300), 4)
               for a in range(0, cfg["max_updates"], 300)}
    e07_tr = [json.loads(x) for x in (ROOT / "models/e07/full/metrics.jsonl").read_text().splitlines()]
    e07_tr = [x for x in e07_tr if "loss" in x]
    # exposure by word
    draws = draw_counts(len(opt), cfg)
    by_word, k_of = Counter(), Counter(w for w, _ in opt)
    for i, n in draws.items():
        by_word[opt[i][0]] += n
    per_word = sorted(by_word.values())
    multi = {w for w, k in k_of.items() if k > 1}
    share_multi = sum(by_word[w] for w in multi) / sum(by_word.values())
    by_k = {str(k): {"words": sum(1 for w in k_of if k_of[w] == k),
                     "mean_draws_per_word": round(mean(by_word[w] for w in k_of if k_of[w] == k), 1)}
            for k in sorted(set(k_of.values()))}
    prov_of_word = defaultdict(set)
    for (w, _), p in zip(opt, prov):
        prov_of_word[w].add(p)
    # fit by provenance: teacher-forced loss and greedy exact reproduction on the training rows
    m, _ = e09.load("e08")
    tl = e07.token_losses(m, [B.encode_src(w) for w, _ in opt], [B.encode_tgt(k) for _, k in opt], bs=64)
    per_byte = [mean(v for _, v in ex) for ex in tl]
    g = e07.gen(m, [B.encode_src(w) for w in dict.fromkeys(w for w, _ in opt)], bs=64)
    gtop = {w: r[0][0] for w, r in zip(dict.fromkeys(w for w, _ in opt), g)}
    accepted = defaultdict(set)
    for w, k in opt:
        accepted[w].add(e06.nfc(k))
    fit = {}
    for p in sorted(set(prov)):
        idx = [i for i, q in enumerate(prov) if q == p]
        words = sorted({opt[i][0] for i in idx})
        fit[p] = {"rows": len(idx), "words": len(words),
                  "mean_loss_per_byte": round(mean(per_byte[i] for i in idx), 4),
                  "median_loss_per_byte": round(median(per_byte[i] for i in idx), 4),
                  "rows_loss_gt_1": sum(per_byte[i] > 1.0 for i in idx),
                  "greedy_reproduces_this_row": round(mean(gtop[opt[i][0]] == e06.nfc(opt[i][1]) for i in idx), 3),
                  "greedy_reproduces_any_spelling_of_word": round(mean(gtop[w] in accepted[w] for w in words), 3)}
    e09.unload(m)
    # dev errors (E08 greedy, cached by E09)
    rows_dev = e09.dev_rows()
    sha = e09.checkpoint_sha("e08")
    gdev = {r["id"]: e09.top1(r["cands"]) for r in e07.load_jsonl(e09.cache_path("e08", "greedy", "dev", sha, rows_dev))}
    dev = e06.primary(e06.items("dev"))
    wrong = [i for i in dev if not (gdev[i.id] and e06.nfc(gdev[i.id]) in {e06.nfc(a) for a in i.accepted})]
    kinds = Counter(error_kind(gdev[i.id], i.accepted) for i in wrong)
    lens = {"pred_minus_ref_codepoints_mean": round(mean(len(gdev[i.id] or "") - min(len(a) for a in i.accepted)
                                                         for i in wrong), 3)}
    out = {"model": {"checkpoint": str((E08_DIR / "best.pt").relative_to(ROOT)), "update": done["best"]["update"]},
           "checkpoint_progression_dev_greedy": [{k: h[k] for k in ("update", "dev_top1", "dev_cer")}
                                                 for h in done["history"]],
           "train_loss_by_window": windows,
           "e07_train_loss_last_300": round(mean(x["loss"] for x in e07_tr[-300:]), 4),
           "lr_schedule": "linear warm-up 100, linear decay to 0 at 3,000 (E07 recipe)",
           "exposure": {"rows": len(opt), "words": len(k_of), "epochs": round(sum(draws.values()) / len(opt), 2),
                        "draws_per_word": {"min": per_word[0], "median": median(per_word), "max": per_word[-1]},
                        "words_with_several_spellings": len(multi), "share_of_draws_on_those_words": round(share_multi, 3),
                        "by_spellings_per_word": by_k,
                        "words_in_several_provenances": sum(len(v) > 1 for v in prov_of_word.values())},
           "fit_by_provenance": fit,
           "dev_errors": {"items": len(dev), "wrong": len(wrong), "kinds": dict(kinds), **lens,
                          "note": "categories are automatic, against the closest mined reference (unreviewed)"}}
    set_state("diagnose", "done")
    return e07.write_json(REPORTS / "diagnosis.json", out) and out


# ------------------------------------------------------------------ review preparation (dev only)
REVIEW_DIR = EXP / "review"                    # committed: instructions, manifest with hashes
REVIEW_DATA = DATA / "review"                  # local: the batch (contains mined references) and the origin key
REVIEW_SEED = "e10-dev-review"
N_REPRESENTATIVE, N_DISAGREEMENT, MAX_CANDS = 40, 20, 4
JUDGMENTS = ("acceptable", "not_acceptable", "uncertain")
LABELS = "ABCD"


def review_candidates(sources: list[tuple[str, str | None]], seed: str) -> list[tuple[str, str, list[str]]]:
    """Distinct candidates (≤ 4) from ordered `(origin, text)` pairs, shuffled deterministically; one text may
    have several origins. Returns `[(label, text, origins)]`.

    Examples:
        >>> [(l, t) for l, t, _ in review_candidates([("x", "ក"), ("y", "ក"), ("z", "ខ")], "s")]
        [('A', 'ក'), ('B', 'ខ')]
    """
    by: dict[str, list[str]] = {}
    for origin, t in sources:
        if t:
            t = e06.nfc(t)
            if t in by or len(by) < MAX_CANDS:
                by.setdefault(t, []).append(origin)
    texts = sorted(by, key=lambda t: e06.order_key(seed, t))
    return [(LABELS[n], t, by[t]) for n, t in enumerate(texts)]


@timed("review")
def review() -> dict:
    """A compact blinded dev-review batch from cached predictions (no new generation): 40 representative dev words
    and 20 E07/E08 greedy disagreements, ≤ 4 candidates each (E08 greedy, E07 greedy, the mined reference, E08
    canonical beam-4 runner-up), origins hidden, deterministic order."""
    rows = e09.dev_rows()
    dev = e06.primary(e06.items("dev"))
    load_c = lambda m, n: {r["id"]: r for r in e07.load_jsonl(e09.cache_path(m, n, "dev", e09.checkpoint_sha(m), rows))}  # noqa: E731
    g8, g7, b8 = load_c("e08", "greedy"), load_c("e07", "greedy"), load_c("e08", "beam4_can")
    inspected = {x["id"] for x in json.loads((ROOT / "experiments/E09-byt5-decoding/reports/disagreements.json")
                                            .read_text())["items"]}
    fams = sorted({i.family for i in dev}, key=lambda f: e06.order_key(f"{REVIEW_SEED}:rep", f))
    by_fam = defaultdict(list)
    for i in dev:
        by_fam[i.family].append(i)
    rep = [sorted(by_fam[f], key=lambda i: i.id)[0] for f in fams[:N_REPRESENTATIVE]]
    used = {i.family for i in rep}
    dis_pool = [i for i in dev if i.family not in used and e09.top1(g8[i.id]["cands"]) != e09.top1(g7[i.id]["cands"])]
    dis = sorted(dis_pool, key=lambda i: e06.order_key(f"{REVIEW_SEED}:dis", i.id))[:N_DISAGREEMENT]
    items, key = [], {}
    order = sorted([("representative", i) for i in rep] + [("disagreement", i) for i in dis],
                   key=lambda x: e06.order_key(f"{REVIEW_SEED}:order", x[1].id))
    for n, (sample, i) in enumerate(order, 1):
        rid = f"rv{n:02d}"
        runner = (e09.cand_list(b8[i.id]["cands"])[1:2] or [None])[0]
        cands = review_candidates([("e08_greedy", e09.top1(g8[i.id]["cands"])), ("e07_greedy", e09.top1(g7[i.id]["cands"])),
                                   ("mined_reference", i.preferred), ("e08_beam4_canonical_rank2", runner)],
                                  f"{REVIEW_SEED}:{rid}")
        items.append({"item_id": rid, "word": i.word, "pronunciation": i.intended_phones, "accent": i.accent,
                      "candidates": [{"label": lb, "text": t} for lb, t, _ in cands]})
        key[rid] = {"dev_id": i.id, "sample": sample, "inspected_in_e09": i.id in inspected,
                    "origins": {lb: o for lb, _, o in cands}}
    REVIEW_DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(REVIEW_DATA / "review_batch.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    e07.write_json(REVIEW_DATA / "origin_key.json", key)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["item_id", "word", "pronunciation", "label", "candidate", "judgment"])
    for it in items:
        for c in it["candidates"]:
            w.writerow([it["item_id"], it["word"], it["pronunciation"], c["label"], c["text"], ""])
        w.writerow([it["item_id"], it["word"], it["pronunciation"], "NONE", "(no candidate acceptable)", ""])
    (REVIEW_DATA / "judgments_template.tsv").write_text(buf.getvalue(), encoding="utf-8")
    man = {"created": e06.now(), "items": len(items), "representative": len(rep), "disagreement": len(dis),
           "candidates": sum(len(it["candidates"]) for it in items),
           "inspected_in_e09": sum(v["inspected_in_e09"] for v in key.values()),
           "sha256": {p.name: e06.sha(p) for p in sorted(REVIEW_DATA.glob("*")) if p.is_file()},
           "local_files": str(REVIEW_DATA.relative_to(ROOT)),
           "note": "dev items (the checkpoint-selection set, historically examined): a development review, not a fresh "
                   "test. Origins hidden; the key stays local."}
    e07.write_json(REVIEW_DIR / "review_manifest.json", man)
    set_state("review_prepared", "done", items=len(items))
    return man


def read_judgments(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rd_ = csv.DictReader(lines, delimiter="\t")
    return [{k: (v or "").strip() for k, v in r.items()} for r in rd_]


def validate_judgments(rows: list[dict], batch: dict) -> dict:
    """Problems in a judgment file: unknown ids or labels, edited candidates, invalid values, duplicates, missing
    judgments and `NONE` contradicting an `acceptable` candidate. Blank judgments count as missing.

    Examples:
        >>> b = {"items": [{"item_id": "rv01", "candidates": [{"label": "A", "text": "ក"}]}]}
        >>> v = validate_judgments([{"item_id": "rv01", "label": "A", "candidate": "ក", "judgment": "acceptable"},
        ...                         {"item_id": "rv01", "label": "NONE", "candidate": "", "judgment": "none_acceptable"}], b)
        >>> v["ok"], v["inconsistent"]
        (False, ['rv01'])
    """
    want = {(it["item_id"], c["label"]): c["text"] for it in batch["items"] for c in it["candidates"]}
    items = {it["item_id"] for it in batch["items"]}
    seen, prob = {}, defaultdict(list)
    for r in rows:
        k = (r.get("item_id", ""), r.get("label", ""))
        j = r.get("judgment", "")
        if k[0] not in items:
            prob["unknown_item"].append(k[0])
            continue
        if k[1] == "NONE":
            if j not in ("", "none_acceptable"):
                prob["invalid_value"].append(f"{k[0]}:{k[1]}:{j}")
            elif j:
                if k in seen and seen[k] != j:
                    prob["duplicate_conflict"].append(f"{k[0]}:{k[1]}")
                seen[k] = j
            continue
        if k not in want:
            prob["unknown_label"].append(f"{k[0]}:{k[1]}")
            continue
        if r.get("candidate") and e06.nfc(r["candidate"]) != want[k]:
            prob["edited_candidate"].append(f"{k[0]}:{k[1]}")
        if j and j not in JUDGMENTS:
            prob["invalid_value"].append(f"{k[0]}:{k[1]}:{j}")
            continue
        if not j:
            continue
        if k in seen:
            prob["duplicate_conflict" if seen[k] != j else "duplicate"].append(f"{k[0]}:{k[1]}")
        seen[k] = j
    missing = sorted(f"{a}:{b}" for a, b in want if (a, b) not in seen)
    incons = sorted({a for (a, b), j in seen.items() if b == "NONE"
                     and any(seen.get((a, lb)) == "acceptable" for (x, lb) in want if x == a)})
    out = {**{k: sorted(set(v)) for k, v in prob.items()}, "missing": missing, "inconsistent": incons,
           "judged": len([k for k in seen if k[1] != "NONE"])}
    out["ok"] = not any(out[k] for k in out if k not in ("judged",))
    return out


@timed("import")
def import_judgments(judgments: str | None = None) -> dict:
    """Validate a judgment file and, if valid and its provenance is declared, score each hidden origin."""
    batch = json.loads((REVIEW_DATA / "review_batch.json").read_text())
    if not judgments:
        p = REVIEW_DATA / "judgments.tsv"
        if not p.exists():
            set_state("review_scored", "skipped", reason="no judgment file (independent review not collected)")
            return {"status": "no_judgments", "expected": str(p.relative_to(ROOT))}
    p = Path(judgments) if judgments else REVIEW_DATA / "judgments.tsv"
    prov = p.with_suffix(".provenance.json")
    rows = read_judgments(p)
    v = validate_judgments(rows, batch)
    out = {"file": str(p), "sha256": e06.sha(p), "validation": v,
           "provenance": json.loads(prov.read_text()) if prov.exists() else None}
    if not v["ok"] or out["provenance"] is None:
        out["status"] = "invalid" if not v["ok"] else "provenance_missing"
        return e07.write_json(REPORTS / "review_scores.json", out) and out
    key = json.loads((REVIEW_DATA / "origin_key.json").read_text())
    j = {(r["item_id"], r["label"]): r["judgment"] for r in rows if r.get("judgment")}
    scores = {}
    for sample in ("representative", "disagreement"):
        ids = [i for i, k in key.items() if k["sample"] == sample]
        per = defaultdict(Counter)
        for i in ids:
            for lb, origins in key[i]["origins"].items():
                for o in origins:
                    per[o][j[(i, lb)]] += 1
            per["any_candidate"]["acceptable" if any(j[(i, lb)] == "acceptable" for lb in key[i]["origins"])
                                 else "none"] += 1
        scores[sample] = {o: dict(c) for o, c in per.items()}
    out |= {"status": "scored", "track": out["provenance"].get("track"), "scores": scores}
    set_state("review_scored", "done", track=out["track"])
    return e07.write_json(REPORTS / "review_scores.json", out) and out


# ------------------------------------------------------------------ confirmation manifest (fresh inputs)
CONFIRM_DIR = EXP / "confirmation"
CONFIRM_SEED = "e10-confirmation"
CONFIRM_CATEGORIES = {"ordinary": 40, "irregular": 40, "names": 40}


@timed("confirm")
def confirm() -> dict:
    """≈ 120 fresh English word families, chosen before any new candidate output: ordinary words (FineWeb-2 Khmer
    Latin-token count ≥ 2, random order), irregular spellings (E06's patterns) and names (SCOWL capitalised), each
    with one CMUdict reading. Excludes protected families, historical Latin tokens, every E08 training word and
    synthetic input, romanised-Khmer forms, abbreviations and ambiguous readings."""
    from khmer_translit import phonetic_source as ps
    from khmer_translit import reference_data as rd
    man = CONFIRM_DIR / "input_manifest.jsonl"
    if man.exists():
        f = json.loads((CONFIRM_DIR / "frozen.json").read_text())
        if e06.sha(man) != f["sha256"]:
            raise RuntimeError("confirmation manifest differs from its frozen hash")
        return f
    cmu, _ = ps.cmudict()
    prot = e08.protected()
    hist, _ = e06.historical_universe()
    hist_stems = e06.stem_set(hist)
    opt, _ = arm_rows()
    train_words = {w for w, _ in opt} | {json.loads(x)["english"] for x in
                                         e08.MANIFEST.read_text(encoding="utf-8").splitlines()}
    train_stems = e06.stem_set(train_words)
    romanised = set((ROOT / "models/model/romanised_forms.txt").read_text(encoding="utf-8").split())
    lower, names = e06.candidate_words()
    excluded, pool = Counter(), []
    for w, kind in sorted([(w, "lower") for w in lower] + [(w, "name") for w in names]):
        if w not in cmu or not 3 <= len(w) <= 14:
            excluded["no_cmudict_or_length"] += 1
        elif e08.is_protected(w, prot):
            excluded["protected_family"] += 1
        elif not e06.is_fresh(w, hist_stems) or w in train_words or rd.related(w, train_stems):
            excluded["historical_or_training_family"] += 1
        elif w in romanised:
            excluded["romanised_khmer_form"] += 1
        elif e08.is_abbreviation(w, cmu[w][0]):
            excluded["abbreviation_like"] += 1
        elif e06.is_ambiguous(cmu[w]):
            excluded["ambiguous_reading"] += 1
        else:
            pool.append((w, kind))
    freq = e08.web_counts({w for w, _ in pool})
    used, picked = set(), {c: [] for c in CONFIRM_CATEGORIES}

    def take(cat, cands):
        for w in sorted(cands, key=lambda w: e06.order_key(f"{CONFIRM_SEED}:{cat}", w)):
            if len(picked[cat]) >= CONFIRM_CATEGORIES[cat]:
                return
            st = rd.stems(w) or {w}
            if used & st:
                continue
            used.update(st)
            picked[cat].append(w)
    lw = [w for w, k in pool if k == "lower"]
    take("names", [w for w, k in pool if k == "name"])
    take("irregular", [w for w in lw if e06.is_difficult(w, cmu[w][0])])
    take("ordinary", [w for w in lw if freq[w] >= 2 and not e06.is_difficult(w, cmu[w][0])])
    rows = [{"id": f"cf{n:03d}", "word": w, "category": c, "pronunciation": " ".join(cmu[w][0]),
             "accent": "en-US-cmudict"} for n, (c, w) in enumerate(((c, w) for c in CONFIRM_CATEGORIES
                                                                    for w in picked[c]), 1)]
    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    man.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    f = {"frozen": e06.now(), "sha256": e06.sha(man), "n": len(rows),
         "per_category": {c: len(v) for c, v in picked.items()}, "pool": len(pool), "excluded": dict(excluded),
         "before_outputs": "frozen before any E10 candidate was trained or run on these words",
         "limitations": ["candidate sources are E06's word lists (SCOWL + CMUdict), not usage-weighted English",
                         "ordinary words need ≥ 2 Latin-token hits in one FineWeb-2 Khmer shard: a weak frequency proxy",
                         "names are those SCOWL lists capitalised and CMUdict reads; brand and place names included",
                         "no references exist: without independent judgments only operational diagnostics are reported"]}
    e07.write_json(CONFIRM_DIR / "frozen.json", f)
    set_state("confirmation_manifest", "done", n=len(rows))
    return f


# ------------------------------------------------------------------ recipes, registration, training
RECIPES = EXP / "recipes"
RUNS = {"A_s0": ("A", 0), "A_s1": ("A", 1), "E08_s1": ("E08", 1)}       # at most these; s1 runs need the gate
GATE = {"min_dev_top1_gain_pp": 3.0, "dev_cer_not_higher": True,
        "baseline": "E08 rs_s0 selected checkpoint: dev greedy top-1 0.542, CER 0.1513 (done.json)"}


def recipe_a() -> dict:
    """Candidate A: E07/E08 recipe with word-level sampling and peak LR 2e-3 (everything else unchanged)."""
    cfg = json.loads(e07.FULL_PATH.read_text())
    return {**cfg, "lr": 0.002, "sampler": "word_uniform_spelling",
            "changes_from_e08": ["word-level sampler: each epoch visits every English word once (seeded order) and "
                                 "draws one of its spellings uniformly (seeded), instead of visiting every row",
                                 "peak learning rate 2e-3 instead of 1e-3 (same warm-up and linear decay)"],
            "rationale": "reports/diagnosis.json: (1) the 1,031 words with several spellings take 46.6% of the draws "
                         "(one word up to 147 draws vs a median of 19), so alternatives multiply exposure; (2) E08 was "
                         "still improving at its last checkpoint with the LR decayed to 0, its final train loss (0.147) "
                         "is twice E07's and it reproduces only 57% of its control rows: under-fitted at 3,000 updates. "
                         "The two changes are combined for performance discovery; their separate contributions stay "
                         "unresolved (no ablation)."}


def recipe(name: str, seed: int) -> dict:
    cfg = recipe_a() if name == "A" else json.loads(e07.FULL_PATH.read_text())
    return {**cfg, "seed": seed}


def word_order(rows: list[tuple[str, str]], seed: int):
    """Word-level epoch order: every word once per epoch (seeded), one of its rows drawn uniformly (seeded).

    Examples:
        >>> o = word_order([("a", "ក"), ("a", "ខ"), ("b", "គ")], 0)
        >>> sorted(rows[0] for rows in [[("a", "ក"), ("a", "ខ"), ("b", "គ")][i] for i in o(0)])
        ['a', 'b']
        >>> o(3) == word_order([("a", "ក"), ("a", "ខ"), ("b", "គ")], 0)(3)
        True
    """
    of: dict[str, list[int]] = {}
    for i, (w, _) in enumerate(rows):
        of.setdefault(w, []).append(i)
    words = list(of)

    def order(ep: int) -> list[int]:
        ws = random.Random(f"{seed}:words:{ep}").sample(words, len(words))
        return [random.Random(f"{seed}:pick:{ep}:{w}").choice(of[w]) for w in ws]
    return order


def order_for(cfg: dict, rows):
    return word_order(rows, cfg["seed"]) if cfg.get("sampler") == "word_uniform_spelling" else None


def exposure(cfg: dict, rows, prov: list[str]) -> dict:
    """Draws per provenance and per word over the run, for any sampler."""
    order = order_for(cfg, rows) or (lambda ep: random.Random(f"{cfg['seed']}:order:{ep}").sample(range(len(rows)),
                                                                                                    len(rows)))
    c, ep, pos, perm = Counter(), 0, 0, order(0)
    for _ in range(cfg["max_updates"]):
        if pos + cfg["batch"] > len(perm):
            ep, pos, perm = ep + 1, 0, order(ep + 1)
        c.update(perm[pos:pos + cfg["batch"]])
        pos += cfg["batch"]
    bp, bw = Counter(), Counter()
    for i, n in c.items():
        bp[prov[i]] += n
        bw[rows[i][0]] += n
    tot = sum(c.values())
    v = sorted(bw.values())
    multi = {w for w, k in Counter(w for w, _ in rows).items() if k > 1}
    return {"draws": tot, "epochs_started": ep + 1,
            "share_by_provenance": {p: round(n / tot, 4) for p, n in sorted(bp.items())},
            "draws_per_word": {"min": v[0], "median": median(v), "max": v[-1]},
            "share_on_multi_spelling_words": round(sum(bw[w] for w in multi) / tot, 4)}


@timed("register")
def register() -> dict:
    """Freeze recipe A and the E08 baseline recipe (seed 1 replication), the selection rule and the gate."""
    if (EXP / "registration.json").exists():
        return json.loads((EXP / "registration.json").read_text())
    opt, rows = arm_rows()
    prov = [r["provenance"] for r in rows if r["part"] == "train"]
    RECIPES.mkdir(parents=True, exist_ok=True)
    e07.write_json(RECIPES / "A.json", recipe_a())
    reg = {"registered": e06.now(), "git": e07.git_rev(), "data": {"arm": str(ARM.relative_to(ROOT)), "sha256": ARM_SHA,
                                                                   "rows_hash": e07.rows_hash(opt), "rows": len(opt)},
           "base": {"revision": B.REVISION, "sha256": B.BASE_SHA256},
           "recipes": {"A": {"file": "recipes/A.json", "sha256": e06.sha(RECIPES / "A.json")},
                       "E08": {"file": str(e07.FULL_PATH.relative_to(ROOT)), "sha256": e06.sha(e07.FULL_PATH)}},
           "runs": {r: {"recipe": n, "seed": sd, "cfg_hash": e07.cfg_hash(recipe(n, sd))} for r, (n, sd) in RUNS.items()},
           "plan": "A_s0 first; A_s1 and E08_s1 only if the gate passes; no other run",
           "selection": "dev greedy top-1 macro (E03 dev primary, NFC exact against any accepted mined reference; a "
                        "proxy, no reviewed dev judgments exist), then lower dev CER, then earlier checkpoint, at "
                        "updates 600/1,200/1,800/2,400/3,000",
           "gate": GATE, "limits": LIMITS,
           "exposure": {"E08_rs_s0": exposure(recipe("E08", 0), opt, prov), "A_s0": exposure(recipe("A", 0), opt, prov)}}
    e07.write_json(EXP / "registration.json", reg)
    set_state("register", "done")
    return reg


def gate_result() -> dict | None:
    p = REPORTS / "gate.json"
    return json.loads(p.read_text()) if p.exists() else None


def train(run: str | None = None) -> dict:
    """One registered run (resumable). Seed-1 runs refuse unless the gate passed."""
    if run not in RUNS:
        raise ValueError(f"--run must be one of {sorted(RUNS)}")
    reg = json.loads((EXP / "registration.json").read_text())
    name, seed = RUNS[run]
    cfg = recipe(name, seed)
    if e07.cfg_hash(cfg) != reg["runs"][run]["cfg_hash"]:
        raise RuntimeError(f"{run}: recipe differs from the registration")
    if seed == 1 and not (gate_result() or {}).get("passed"):
        raise RuntimeError(f"{run}: the replication gate has not passed")
    opt, _ = arm_rows()
    e07.check_disjoint(opt)
    dev = e06.primary(e06.items("dev"))
    set_state(f"train_{run}", "running")
    done = e07.fit(run, "full", cfg, opt, lambda m: e07.dev_selection(m, dev), set(cfg["eval_at"]), MODELS / run,
                   order=order_for(cfg, opt))
    set_state(f"train_{run}", "done" if "best" in done else "stopped", stop=done.get("stop"),
              best=done.get("best"))
    return done


@timed("gate")
def gate() -> dict:
    """Replication gate on dev (a resource-allocation rule, not proof of phonetic improvement)."""
    base = json.loads((E08_DIR / "done.json").read_text())["best"]
    a = json.loads((MODELS / "A_s0" / "done.json").read_text())["best"]
    gain = round((a["dev_top1"] - base["dev_top1"]) * 100, 2)
    passed = gain >= GATE["min_dev_top1_gain_pp"] and a["dev_cer"] <= base["dev_cer"]
    out = {"A_s0": {k: a[k] for k in ("update", "dev_top1", "dev_cer")},
           "E08_rs_s0": {k: base[k] for k in ("update", "dev_top1", "dev_cer")},
           "gain_pp": gain, "passed": passed, "rule": GATE}
    set_state("gate", "passed" if passed else "failed", gain_pp=gain)
    return e07.write_json(REPORTS / "gate.json", out) and out


# ------------------------------------------------------------------ freeze and evaluate
DECODINGS = {"single": "greedy", "candidates": "beam4_can"}      # E09-corrected; equal candidate budget (4)
RC_RULE = ("Keep E08 rs_s0 unless the gate passed AND the seed-1 replication supports the direction: A_s1 dev "
           "greedy top-1 > E08_s1 dev greedy top-1 with dev CER not higher. Then A_s0 (or A_s1, whichever has the "
           "higher dev top-1, then lower CER) becomes the research release candidate. Historical eval cannot change "
           "this choice.")


def trained_runs() -> list[str]:
    return [r for r in RUNS if (MODELS / r / "done.json").exists()]


def register_checkpoints() -> None:
    """Make E10 runs loadable through E09's cached generation (same adapter architecture as E07/E08)."""
    for r in trained_runs():
        e09.CHECKPOINTS[r] = MODELS / r


@timed("freeze")
def freeze() -> dict:
    """Candidate identities, decoding and the release-candidate rule, frozen before historical eval scoring."""
    register_checkpoints()
    cands = ["e08", *trained_runs()]
    rc = "e08"
    g = gate_result() or {}
    if g.get("passed") and {"A_s1", "E08_s1"} <= set(trained_runs()):
        a1, b1 = (json.loads((MODELS / r / "done.json").read_text())["best"] for r in ("A_s1", "E08_s1"))
        if a1["dev_top1"] > b1["dev_top1"] and a1["dev_cer"] <= b1["dev_cer"]:
            a0 = json.loads((MODELS / "A_s0" / "done.json").read_text())["best"]
            rc = max((("A_s0", a0), ("A_s1", a1)), key=lambda x: (x[1]["dev_top1"], -x[1]["dev_cer"]))[0]
    out = {"frozen": e06.now(), "candidates": {c: {"checkpoint": str((e09.CHECKPOINTS[c] / "best.pt").relative_to(ROOT)),
                                                   "sha256": e09.checkpoint_sha(c),
                                                   "update": json.loads((e09.CHECKPOINTS[c] / "done.json").read_text())["best"]["update"],
                                                   "dev_greedy": {k: json.loads((e09.CHECKPOINTS[c] / "done.json").read_text())["best"][k]
                                                                  for k in ("dev_top1", "dev_cer")}}
                                               for c in cands},
           "decoding": {k: {"setting": v, **e09.setting(v), **e09.COMMON, "hash": e09.setting_hash(v)}
                        for k, v in DECODINGS.items()},
           "gate": g, "release_candidate_rule": RC_RULE, "release_candidate": rc,
           "base": {"repo": "google/byt5-small", "revision": B.REVISION, "sha256": B.BASE_SHA256}}
    set_state("freeze", "done", release_candidate=rc)
    return e07.write_json(REPORTS / "candidates_frozen.json", out) and out


def confirm_rows() -> list[tuple[str, str]]:
    return [(r["id"], r["word"]) for r in e07.load_jsonl(CONFIRM_DIR / "input_manifest.jsonl")]


def operational(recs: list[dict]) -> dict:
    """Diagnostics without references: statuses, orthography, empty/truncated/repeated outputs, length, diversity."""
    tops = [e09.top1(r["cands"]) for r in recs]
    return {"items": len(recs), "rank1_status": dict(Counter(r["cands"][0]["status"] for r in recs)),
            "no_output": sum(t is None for t in tops),
            "orthography_not_ok": sum(e06.orthography(t) != "ok" for t in tops if t),
            "repeated": sum(e09.repeated(t) for t in tops), "truncated_no_eos": sum(
                c["status"] == "no_eos" for r in recs for c in r["cands"]),
            "mean_codepoints": round(mean(len(t or "") for t in tops), 2), "distinct_top1": len({t for t in tops if t}),
            "max_share_one_output": round(Counter(t for t in tops if t).most_common(1)[0][1] / len(recs), 4)}


def seed_contrast(items, per: dict, plus: list[str], minus: list[str]) -> dict:
    idx = e05.boot_indices(len({i.family for i in items}))
    fm = {s: e05.family_means(items, {i: v["top1"] for i, v in per[s].items()}) for s in plus + minus}
    fc = {s: e05.family_means(items, {i: -v["cer"] for i, v in per[s].items()}) for s in plus + minus}
    return {"top1": e05.contrast(fm, plus, minus, idx), "cer_reduction": e05.contrast(fc, plus, minus, idx)}


@timed("evaluate")
def evaluate() -> dict:
    """Dev, historical eval (exploratory) and confirmation diagnostics for every frozen candidate."""
    fz = json.loads((REPORTS / "candidates_frozen.json").read_text())
    register_checkpoints()
    sets = {"dev": e09.dev_rows(), "eval": e09.eval_rows(), "confirmation": confirm_rows()}
    for c in fz["candidates"]:
        m, _ = e09.load(c)
        sha = e09.checkpoint_sha(c)
        for split, rows in sets.items():
            for d in DECODINGS.values():
                e09.generate(m, c, d, split, rows, sha)
        e09.unload(m)
    load_c = lambda c, d, split: e07.load_jsonl(e09.cache_path(c, d, split, e09.checkpoint_sha(c), sets[split]))  # noqa: E731
    out = {"label": "Track 1 (historical mined-reference agreement) for dev and eval; eval examined repeatedly "
                    "(E03–E10), exploratory. Confirmation: operational diagnostics only (no references, no "
                    "independent judgments).", "frozen": fz["frozen"], "release_candidate": fz["release_candidate"]}
    per_split = {}
    for split in ("dev", "eval"):
        items = e06.primary(e06.items(split))
        tab, per = {}, {}
        for c in fz["candidates"]:
            for k, d in DECODINGS.items():
                recs = load_c(c, d, split)
                tab[f"{c}:{k}"], per[f"{c}:{k}"] = e09.setting_metrics(items, recs, e09.gen_seconds(c, d, split))
        con = {}
        for c in fz["candidates"]:
            if c != "e08":
                con[f"{c}_minus_e08:single"] = seed_contrast(items, per, [f"{c}:single"], ["e08:single"])
        if {"A_s0", "A_s1", "E08_s1"} <= set(fz["candidates"]):
            con["A_s1_minus_E08_s1:single"] = seed_contrast(items, per, ["A_s1:single"], ["E08_s1:single"])
            con["A_seed_mean_minus_E08_seed_mean:single"] = seed_contrast(
                items, per, ["A_s0:single", "A_s1:single"], ["e08:single", "E08_s1:single"])
        wrong = {s: Counter(error_kind(per[s][i.id]["prediction"], i.accepted) for i in items if not per[s][i.id]["top1"])
                 for s in per if s.endswith(":single")}
        out[split] = {"items": len(items), "families": len({i.family for i in items}), "table": tab,
                      "contrasts": con, "error_kinds_single": {k: dict(v) for k, v in wrong.items()}}
        per_split[split] = per
    out["confirmation"] = {f"{c}:{k}": operational(load_c(c, d, "confirmation"))
                           for c in fz["candidates"] for k, d in DECODINGS.items()}
    tops = {c: {r["id"]: e09.top1(r["cands"]) for r in load_c(c, "greedy", "confirmation")} for c in fz["candidates"]}
    if len(tops) > 1:
        out["confirmation_agreement_with_e08_single"] = {
            c: round(mean(tops[c][i] == tops["e08"][i] for i in tops["e08"]), 3) for c in tops if c != "e08"}
    PRED.mkdir(parents=True, exist_ok=True)
    for c in fz["candidates"]:
        for split in ("eval", "confirmation"):
            recs = load_c(c, "greedy", split)
            beams = {r["id"]: e09.cand_list(r["cands"]) for r in load_c(c, "beam4_can", split)}
            (PRED / f"{c}_{split}.jsonl").write_text("".join(json.dumps(
                {"id": r["id"], "word": r["word"], "single": e09.top1(r["cands"]), "candidates4": beams[r["id"]]},
                ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    with open(EXP / "results_per_item.jsonl", "w", encoding="utf-8") as f:
        for split in ("dev", "eval"):
            for i in e06.primary(e06.items(split)):
                f.write(json.dumps({"split": split, "id": i.id, "word": i.word, "family": i.family,
                                    **{s: {"top1": p[i.id]["prediction"], "correct": p[i.id]["top1"],
                                           "cer": round(p[i.id]["cer"], 4)} for s, p in per_split[split].items()}},
                                   ensure_ascii=False) + "\n")
    set_state("evaluate", "done")
    return e07.write_json(EXP / "results.json", out) and out


def status() -> dict:
    s = state()
    s["inference_hours"] = round(inference_hours(), 3)
    s["training"] = e07.budget_left("full")
    return s


STEPS = {"diagnose": diagnose, "review": review, "import": import_judgments, "confirm": confirm,
         "register": register, "train": train, "gate": gate, "freeze": freeze, "evaluate": evaluate,
         "status": status}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--run")
    ap.add_argument("--judgments")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(LIMITS["threads"])
    kw = {k: v for k, v in (("run", a.run), ("judgments", a.judgments)) if v is not None}
    res = STEPS[a.step](**kw)
    print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
