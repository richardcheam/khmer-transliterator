"""E12: E11's ByT5-small recipe with a longer schedule (6,000 updates, linear decay over 6,000), one seed-0 run.

Same base, LoRA, dropout, optimiser, row-level sampling and frozen real-data arm R as E11; only the schedule
horizon, the checkpoint grid and the (automatic, CER-first) selection rule change. A fresh 120-family test-input
set is frozen before training; its outputs are generated only after selection and judged blind by the owner.

Reuses `scripts/e07.py` (model, resumable loop, memory guard, budget), E09's cached generation, E10's review
helpers and E11's data and scoring helpers. E08 appears in historical tables only.

Steps (in order):
    verify     starting state: E11 checkpoint, base, arm R, recipe, owner dev judgments, E09 decoding
               → reports/start_checks.json
    testset    120 fresh word families (inputs only), frozen and hashed → test/input_manifest.jsonl, test/frozen.json
    register   recipe, discrepancies vs E11, exposure, LR plan, limits, selection and test analysis → registration.json
    train      the one authorised run (≤ 6,000 updates / 12 h; resumable) → models/e12/long_s0/
    freeze     selected checkpoint + greedy dev/eval predictions (E12, E11, E07, E08) → reports/frozen.json
    evaluate   historical diagnostics, curves, dev-judgment coverage → results.json
    testgen    greedy E12 and E11 outputs on the test inputs, hashed → data/e12/test/ (local), test/predictions.json
    batches    blinded numbered batches + separate origin key → data/e12/test/ (local), test/batches.json
    import     validate owner judgments; when complete, the predeclared analysis → reports/test_scores.json
    status     phases and budgets

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e12.py verify
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e08  # noqa: E402
import e09  # noqa: E402
import e10  # noqa: E402  (pure helpers only)
import e11  # noqa: E402  (data and scoring helpers; its bookkeeping is overridden below)
from khmer_translit import reference_data as rd  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E12-longer-schedule"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
TEST = EXP / "test"
STATE_FILE = EXP / "state.json"
LEDGER = EXP / "runs" / "ledger.jsonl"
DATA = ROOT / "data" / "e12"                     # git-ignored: test outputs, blinded batches, origin key, judgments
TEST_DATA = DATA / "test"
MODELS = ROOT / "models" / "e12"
RUN = "long_s0"
OUT = MODELS / RUN
GiB = 1024 ** 3
# The guard in use since E10 (owner-relaxed thresholds), with this task's limits: 6 threads, 1 run, 6,000 / 12 h
LIMITS = {"threads": 6, "rss_max_bytes": int(6.5 * GiB), "min_available_bytes": int(0.4 * GiB), "pressure_checks": 3,
          "full_runs": 1, "full_max_updates": 6000, "full_max_hours": 12.0, "training_hours_total": 12.0,
          "wall_hours_total": 1e9, "inference_hours_total": 4.0}
EVAL_AT = (1200, 2400, 3000, 3600, 4800, 6000)
E11_BEST_SHA = "286292a4f21131d0dd238d48919485192b0b0eb268d005bbc804d3ff78229713"
E11_CFG_HASH = "f5d7cf9230f15987c2252487cda366e80207a4a93f81c5da3795a7a300a3c3f6"
E11_DATA_HASH = "5dd38022c60b0b24eb98cd66bccf1edf23dcc619f23b2c8208a1c2d2a07c1d61"
E11_REVIEW = ROOT / "data" / "e11" / "review"
COMPARED = ("e12", "e11", "e07", "e08")   # e11: baseline; e07 and e08: historical columns only
TESTED = ("e12", "e11")

e07.LEDGER = LEDGER
e07.EXPERIMENT = EXPERIMENT
e07.STATE = ROOT / "state" / "e12"
e07.LIMITS = LIMITS
e09.LEDGER = LEDGER
e09.LIMITS = {**e09.LIMITS, "active_hours": LIMITS["inference_hours_total"]}
e09.CHECKPOINTS["e11"] = ROOT / "models" / "e11" / "r_s0"
e09.CHECKPOINTS["e12"] = OUT
e11.STATE_FILE = STATE_FILE          # e11's timed() and set_state() write to this experiment
e11.LEDGER = LEDGER
state, set_state = e11.state, e11.set_state


def timed(step: str):
    return e11.timed(step)


# ------------------------------------------------------------------ 1. starting state
@timed("verify")
def verify() -> dict:
    """Starting-state checks (E11 checkpoint and recipe, base, arm R, owner dev judgments, E09 decoding)."""
    from khmer_translit import evaluation  # noqa: F401  (environment check: package importable)
    chk = {}
    done = json.loads((e09.CHECKPOINTS["e11"] / "done.json").read_text())
    fz = json.loads((e11.REPORTS / "frozen.json").read_text())
    sha = e09.checkpoint_sha("e11")
    chk["e11_checkpoint"] = {"update": done["best"]["update"], "sha256": sha,
                             "pass": sha == E11_BEST_SHA == fz["selected"]["checkpoint_sha256"]
                             and done["best"]["update"] == 3000}
    chk["base"] = {"revision": B.REVISION, "sha256": B.BASE_SHA256,
                   "pass": B.REVISION == "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
                   and e06.sha(B.BASE / "pytorch_model.bin") == B.BASE_SHA256["pytorch_model.bin"]}
    opt, rows = e11.arm_rows()
    chk["arm_R"] = {"sha256": e11.ARM_SHA, "optimisation_rows": len(opt), "train_val_rows": len(rows) - len(opt),
                    "optimisation_hash": e07.rows_hash(opt),
                    "pass": len(opt) == 4764 and e07.rows_hash(opt) == E11_DATA_HASH
                    and json.loads((e11.REPORTS / "data_checks.json").read_text())["ok"]}
    e07.check_disjoint(opt)
    cfg = e11.recipe()
    chk["e11_recipe"] = {"cfg_hash": e07.cfg_hash(cfg), "values": recipe_values(cfg),
                         "pass": e07.cfg_hash(cfg) == E11_CFG_HASH == json.loads(
                             (e11.EXP / "registration.json").read_text())["cfg_hash"]}
    jt = E11_REVIEW / "judgments.tsv"
    jrows = list(csv.DictReader(io.StringIO(jt.read_text(encoding="utf-8")), delimiter="\t"))
    cand = [r for r in jrows if r["label"] != "NONE"]
    key = json.loads((E11_REVIEW / "origin_key.json").read_text())
    dev = {i.id: i for i in e06.items("dev")}
    fams = {dev[k["dev_id"]].family for k in key.values()}
    train_stems = e06.stem_set({w for w, _ in e11.arm_rows()[0]} | {r["word"] for r in rows})
    chk["owner_dev_judgments"] = {
        "items": len({r["item_id"] for r in jrows}), "candidates": len(cand),
        "blank_candidate_judgments": sum(not r["judgment"] for r in cand),
        "values": dict(Counter(r["judgment"] for r in cand)), "sha256": e06.sha(jt),
        "provenance": json.loads((E11_REVIEW / "judgments.tsv.provenance.json").read_text())["judge"],
        "families": len(fams), "families_in_training": sorted(f for f in fams if rd.related(f, train_stems)
                                                              or f in {r["word"] for r in rows}),
        "role": "development evidence (dev words, the selection set), not test evidence"}
    chk["owner_dev_judgments"]["pass"] = (chk["owner_dev_judgments"]["items"] == 60
                                          and chk["owner_dev_judgments"]["blank_candidate_judgments"] == 0
                                          and not chk["owner_dev_judgments"]["families_in_training"])
    chk["decoding"] = {"greedy": e09.setting("greedy"), "common": e09.COMMON,
                       "canonical_beam_early_stopping": e09.setting("beam4_can").get("early_stopping"),
                       "pass": e09.setting("greedy")["num_beams"] == 1 and e09.COMMON["max_new_tokens"] == 121
                       and e09.setting("beam4_can").get("early_stopping") == "never"}
    out = {"checked": e06.now(), "checks": chk, "ok": all(c["pass"] for c in chk.values())}
    set_state("verify", "done" if out["ok"] else "failed")
    return e07.write_json(REPORTS / "start_checks.json", out) and out


def recipe_values(cfg: dict) -> dict:
    """The recipe fields this experiment must keep from E11 (adapter, dropout, optimiser, batch, warm-up)."""
    return {k: cfg[k] for k in ("adapter", "base_dropout", "optimizer", "lr", "betas", "weight_decay", "grad_clip",
                                "batch", "grad_accumulation", "warmup", "schedule", "seed")}


# ------------------------------------------------------------------ 2–3. fresh test inputs (before training)
TEST_SEED = "e12-test"
N_TEST = 120
CATEGORIES = ("ordinary", "irregular", "names")              # 40 each
LENGTH_QUOTA = {"le4": 13, "5to8": 14, "ge9": 13}            # per category (English letters; E11's bins)
VOWEL_CAP, CLUSTER_CAP = 5, 24                               # per category, first pass only (see `pick_balanced`)
NAME_DERIVED = ("ian", "ism", "ist", "ese", "ic", "ish", "ite", "ology", "ize", "ise", "ized", "izes", "izing")
OFFENSIVE = {"dink"}                                         # removed on reading the inputs (an ethnic slur sense)
AMENDMENTS = ["before freezing and before any output: name filter extended to -ize/-ise forms and to plurals of a "
              "listed word (anglicize, americanize, cajuns, eurodollars appeared as 'names'); 'dink' excluded as a "
              "slur; the set was redrawn with the same seed"]
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def phone_features(phones: tuple[str, ...]) -> dict:
    """Stressed vowel, consonant-cluster presence and syllable count of one CMUdict reading.

    Examples:
        >>> phone_features(("S", "T", "R", "IY1", "T"))
        {'stressed_vowel': 'IY', 'cluster': True, 'syllables': 1}
        >>> phone_features(("K", "AE1", "T"))["cluster"]
        False
    """
    base = [re.sub(r"\d", "", p) for p in phones]
    stressed = [b for p, b in zip(phones, base) if p.endswith("1")]
    cl = any(base[i] not in VOWELS and base[i + 1] not in VOWELS for i in range(len(base) - 1))
    return {"stressed_vowel": stressed[0] if stressed else "none", "cluster": cl,
            "syllables": sum(b in VOWELS for b in base)}


def exclusion(word: str, kind: str, ctx: dict) -> str | None:
    """Why a candidate word cannot enter the test set (None if it can). Order matters only for the counts.

    `ctx` holds the CMUdict, protected families, historical stems, training words and stems, romanised forms.

    Examples:
        >>> ctx = {"cmu": {"table": [("T", "EY1", "B", "AH0", "L")]}, "prot": {"words": set(), "stems": set()},
        ...        "hist_stems": set(), "train_words": set(), "train_stems": set(), "romanised": set(), "extra": set(),
        ...        "extra_stems": set()}
        >>> exclusion("table", "lower", ctx), exclusion("chair", "lower", ctx)
        (None, 'no_cmudict_or_length')
        >>> exclusion("table", "lower", {**ctx, "prot": {"words": {"table"}, "stems": set()}})
        'protected_family'
    """
    cmu = ctx["cmu"]
    if word not in cmu or not 3 <= len(word) <= 14:
        return "no_cmudict_or_length"
    if e08.is_protected(word, ctx["prot"]):
        return "protected_family"
    if word in ctx["extra"] or rd.related(word, ctx["extra_stems"]):
        return "earlier_review_or_test_set"
    if not e06.is_fresh(word, ctx["hist_stems"]):
        return "historical_family"
    if word in ctx["train_words"] or rd.related(word, ctx["train_stems"]):
        return "training_family"
    if word in ctx["romanised"]:
        return "romanised_khmer_form"
    if e08.is_abbreviation(word, cmu[word][0]):
        return "abbreviation_like"
    if e06.is_ambiguous(cmu[word]):
        return "ambiguous_reading"
    if kind == "name" and (word.endswith(NAME_DERIVED) or (word.endswith("s") and word[:-1] in ctx.get("listed", ()))):
        return "derived_name_form"
    if word in OFFENSIVE:
        return "offensive"
    return None


SUFFIXES = ("ness", "ing", "est", "ers", "er", "en", "ed", "es", "ly", "s")


def family_keys(word: str) -> set[str]:
    """Stricter family keys for one-item-per-family inside the test set: the project's stems plus the word with
    common inflectional/derivational suffixes stripped (remainder ≥ 3 letters).

    Examples:
        >>> sorted(family_keys("lighten") & family_keys("lightest"))
        ['light']
    """
    keys = set(rd.stems(word)) | {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            keys.add(word[: -len(suf)])
    return keys


def pick_balanced(cands: list[str], cmu: dict, seed: str, used: set[str]) -> tuple[list[str], dict]:
    """40 words of one category: length quotas per bin, in seeded order; a first pass caps any stressed vowel at
    `VOWEL_CAP` and clusters at `CLUSTER_CAP`, a second pass fills what is left without caps. One word per family
    (stems in `used`, updated in place). Shortfalls in a length bin are filled from the other bins and reported."""
    order = sorted(cands, key=lambda w: e06.order_key(seed, w))
    picked, vow, clu, short = [], Counter(), 0, {}
    for capped in (True, False):
        for b, q in LENGTH_QUOTA.items():
            for w in order:
                if sum(e11.length_bin(x) == b for x in picked) >= q:
                    break
                if w in picked or e11.length_bin(w) != b:
                    continue
                f = phone_features(cmu[w][0])
                if capped and (vow[f["stressed_vowel"]] >= VOWEL_CAP or (f["cluster"] and clu >= CLUSTER_CAP)):
                    continue
                st = family_keys(w)
                if used & st:
                    continue
                used.update(st)
                picked.append(w)
                vow[f["stressed_vowel"]] += 1
                clu += f["cluster"]
    for b, q in LENGTH_QUOTA.items():
        short[b] = q - sum(e11.length_bin(x) == b for x in picked)
    for w in order:                                   # fill a length shortfall from any bin
        if len(picked) >= sum(LENGTH_QUOTA.values()):
            break
        st = family_keys(w)
        if w not in picked and not used & st:
            used.update(st)
            picked.append(w)
    return picked, {"length_shortfall_filled_from_other_bins": {b: n for b, n in short.items() if n > 0}}


def manifest_rows() -> list[dict]:
    """The frozen test inputs (hash-checked)."""
    p = TEST / "input_manifest.jsonl"
    f = json.loads((TEST / "frozen.json").read_text())
    if e06.sha(p) != f["sha256"]:
        raise RuntimeError("test input manifest differs from its frozen hash")
    return e11.jl(p)


@timed("testset")
def testset() -> dict:
    """120 fresh English word families (inputs and readings only), sampled and frozen before training."""
    from khmer_translit import phonetic_source as ps
    if (TEST / "frozen.json").exists():
        manifest_rows()
        return json.loads((TEST / "frozen.json").read_text())
    cmu, cmu_sha = ps.cmudict()
    prot = e08.protected()
    hist, _ = e06.historical_universe()
    opt, rows = e11.arm_rows()
    train_words = {r["word"] for r in rows} | {json.loads(x)["english"] for x in
                                               e08.MANIFEST.read_text(encoding="utf-8").splitlines()}
    extra = {json.loads(x)["word"] for x in
             (ROOT / "experiments/E10-release-cycle/confirmation/input_manifest.jsonl").read_text().splitlines()}
    for p in (ROOT / "data/e10/review/review_batch.json", E11_REVIEW / "review_batch.json"):
        extra |= {it["word"] for it in json.loads(p.read_text())["items"]}
    ctx = {"cmu": cmu, "prot": prot, "hist_stems": e06.stem_set(hist), "train_words": train_words,
           "train_stems": e06.stem_set(train_words), "extra": extra, "extra_stems": e06.stem_set(extra),
           "romanised": set((ROOT / "models/model/romanised_forms.txt").read_text(encoding="utf-8").split())}
    lower, names = e06.candidate_words()
    ctx["listed"] = lower | names
    excluded, pool = Counter(), []
    for w, kind in sorted([(w, "lower") for w in lower] + [(w, "name") for w in names]):
        why = exclusion(w, kind, ctx)
        if why:
            excluded[why] += 1
        else:
            pool.append((w, kind))
    freq = e08.web_counts({w for w, k in pool if k == "lower"})
    lw = [w for w, k in pool if k == "lower"]
    pools = {"names": [w for w, k in pool if k == "name"],
             "irregular": [w for w in lw if e06.is_difficult(w, cmu[w][0])],
             "ordinary": [w for w in lw if freq[w] >= 2 and not e06.is_difficult(w, cmu[w][0])]}
    used, picked, notes = set(), {}, {}
    for c in ("names", "irregular", "ordinary"):          # rarer pools first, so shared stems go to them
        picked[c], notes[c] = pick_balanced(pools[c], cmu, f"{TEST_SEED}:{c}", used)
    allw = [(c, w) for c in CATEGORIES for w in picked[c]]
    allw.sort(key=lambda x: e06.order_key(f"{TEST_SEED}:order", x[1]))  # mixed order → mixed batches
    out_rows = []
    for n, (c, w) in enumerate(allw, 1):
        r = cmu[w][0]
        out_rows.append({"id": f"t{n:03d}", "word": w, "category": c, "length_bin": e11.length_bin(w),
                         "pronunciation": " ".join(r), "accent": "en-US-cmudict", **phone_features(r)})
    TEST.mkdir(parents=True, exist_ok=True)
    man = TEST / "input_manifest.jsonl"
    man.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    dist = lambda k: {c: dict(sorted(Counter(r[k] for r in out_rows if r["category"] == c).items()))  # noqa: E731
                      for c in CATEGORIES}
    f = {"frozen": e06.now(), "sha256": e06.sha(man), "n": len(out_rows), "seed": TEST_SEED,
         "before_outputs": "frozen before E12 training; no model has been run on these words",
         "per_category": {c: len(picked[c]) for c in CATEGORIES},
         "distribution": {"length_bin": dist("length_bin"), "stressed_vowel": dist("stressed_vowel"),
                          "cluster": dist("cluster"), "syllables": dist("syllables")},
         "pool": {c: len(v) for c, v in pools.items()}, "excluded": dict(excluded), "notes": notes,
         "amendments": AMENDMENTS, "cmudict_sha256": cmu_sha, "protected_counts": prot["counts"],
         "sampling": {"sources": "SCOWL lower-case words (ordinary, irregular) and SCOWL capitalised entries not also "
                                 "lower-case (names), each with exactly one CMUdict reading (after stress removal)",
                      "ordinary": "≥ 2 Latin-token hits in the local FineWeb-2 Khmer shard and not irregular",
                      "irregular": "E06's registered spelling-to-sound difficulty rule (e06.is_difficult)",
                      "names": f"capitalised SCOWL entries, minus derived forms ending in {list(NAME_DERIVED)}",
                      "quotas": {"per_category": 40, "length_bins": LENGTH_QUOTA, "vowel_cap": VOWEL_CAP,
                                 "cluster_cap": CLUSTER_CAP, "order": "sha256(seed:category:word)"},
                      "one_per_family": "family_keys (reference_data.stems plus stripped suffixes), shared across all categories"},
         "exclusions": ["E03 dev/eval, E02 held-out sets, benchmark panels and the reserve (Latin inputs only, via "
                        "e08.protected; no reserve target or E02 final reference is opened)",
                        "every historical Latin token in the repository (e06.historical_universe)",
                        "every arm R word (optimisation and train-val) and every E08 synthetic input",
                        "E10 confirmation words and the E10 and E11 review words (and their families)",
                        "romanised-Khmer forms, abbreviations, ambiguous readings"],
         "e10_confirmation_audit": {"reused": False,
                                    "usage": "E10 used its outputs only for operational diagnostics and an agreement "
                                             "rate; no recipe, data or decoding choice depended on them",
                                    "why_not_reused": "unbalanced lengths (6 of 120 words ≤ 4 letters, 61 ≥ 9) and "
                                                      "derived forms among the names (ethiopian, sikhism, brazilian)"},
         "scope": "a single-judge test for this declared sample and docs/PHONETIC_POLICY.md, not a universal "
                  "English→Khmer benchmark"}
    e07.write_json(TEST / "frozen.json", f)
    set_state("testset", "done", n=len(out_rows))
    return f


# ------------------------------------------------------------------ 4. recipe and registration
SELECT_RULE = ("lowest dev greedy mean CER (historical dev set, against the mined references), then highest dev "
               "greedy top-1 exact agreement (family macro), then the earlier update; eligible: the registered "
               "checkpoints and, if the run stops early, the final state")


def recipe() -> dict:
    """E11's recipe with the 6,000-update horizon, the E12 checkpoint grid and the CER-first selection."""
    base = e11.recipe()
    return {**base, "max_updates": 6000, "max_hours": 12.0, "eval_at": list(EVAL_AT), "select": SELECT_RULE,
            "decode": {"selection": "greedy", "headline": "greedy", "max_new_tokens": e09.COMMON["max_new_tokens"]}}


def discrepancies() -> dict:
    """Recipe fields that differ from E11's implementation (everything else is asserted equal)."""
    a, b = e11.recipe(), recipe()
    if recipe_values(a) != recipe_values(b):
        raise RuntimeError("adapter / optimiser / sampling fields differ from E11")
    return {k: {"e11": a.get(k), "e12": b[k]} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def lr_plan(cfg: dict) -> dict:
    """Peak-relative learning rate at named updates (1-based) under the registered schedule.

    Examples:
        >>> p = lr_plan({"lr": 1e-3, "warmup": 100, "schedule": "linear", "max_updates": 6000})
        >>> p[1], p[100], p[3000], p[6000]
        (1e-05, 0.001, 0.000508644068, 1.69492e-07)
    """
    return {u: round(cfg["lr"] * e07.lr_factor(u - 1, cfg), 12) for u in (1, 100, 101, 1200, 3000, 3600, 4800, 6000)}


TEST_ANALYSIS = {
    "primary": "paired difference E12 − E11 in owner-confirmed top-1 acceptability on the 120 test families: "
               "share of items whose greedy output is judged `acceptable` (denominator: all 120 items; `uncertain` "
               "and `not_acceptable` count as not confirmed)",
    "per_item_judgment": "each model takes the judgment of the candidate equal to its greedy output; identical outputs "
                         "are shown once and share one judgment; an empty output counts as not_acceptable "
                         "(policy rule 9)",
    "report": ["acceptable / not_acceptable / uncertain counts per model", "confirmed-acceptable share (all items)",
               "uncertainty rate per model and share of items with any uncertain judgment",
               "paired table: both acceptable, E12 only, E11 only, neither (primary coding)",
               "family-level paired bootstrap 95% interval of the primary difference (10,000, seed 20261027; one "
               "item per family)",
               "two-sided exact binomial (McNemar) test on discordant clear pairs: one model acceptable and the other "
               "not_acceptable; pairs with an uncertain judgment are not in this test"],
    "sensitivity": ["uncertain counted as acceptable (denominator 120)",
                    "items with any uncertain judgment excluded (denominator: items with two clear judgments)"],
    "completeness": "scored only when every candidate of every item has an explicit judgment; blanks are missing, "
                    "never acceptance",
    "interpretation": "a non-significant difference is not equivalence; one non-independent judge; results hold for "
                      "this sample and policy",
    "isolation": "no test judgment enters training or checkpoint choice in this cycle",
}


@timed("register")
def register() -> dict:
    """Freeze recipe, discrepancies, data and code hashes, exposure, LR plan, selection and test analysis."""
    sc = json.loads((REPORTS / "start_checks.json").read_text())
    if not sc["ok"]:
        raise RuntimeError("starting-state checks did not pass")
    tf = testset()
    cfg = recipe()
    opt, rows = e11.arm_rows()
    files = ["scripts/e12.py", "scripts/e11.py", "scripts/e07.py", "scripts/e06_byt5.py", "scripts/e06.py",
             "scripts/e09.py", "scripts/e10.py", "experiments/E07-byt5-diagnosis/full_recipe.json",
             "experiments/E12-longer-schedule/PLAN.md", "research/requirements-byt5.txt",
             "research/requirements-byt5.lock.txt", "docs/PHONETIC_POLICY.md"]
    reg = {"registered": e06.now(), "experiment": EXPERIMENT, "run": RUN, "recipe": cfg, "cfg_hash": e07.cfg_hash(cfg),
           "discrepancies_vs_e11": discrepancies(),
           "init": "pinned google/byt5-small base + fresh LoRA adapter (seed 0); a fresh 6,000-update schedule, "
                   "not a resume of E11's exhausted scheduler; nothing from E08/E10/E11 adapters",
           "data": {"arm": str(e11.ARM.relative_to(ROOT)), "sha256": e11.ARM_SHA, "optimisation_hash": e07.rows_hash(opt),
                    "optimisation_rows": len(opt), "train_val_rows": len(rows) - len(opt)},
           "exposure": e11.exposure(cfg, opt, rows), "lr_plan": lr_plan(cfg), "limits": LIMITS,
           "guard": "stop (resumable) after 3 consecutive readings with process RSS > 6.5 GiB or MemAvailable < 0.4 GiB",
           "checkpoints": list(EVAL_AT), "selection": SELECT_RULE,
           "selection_note": "an explicitly imperfect automatic surrogate (mined references), chosen so the run can "
                             "finish unattended; it does not select for phonetic quality; owner dev judgments are "
                             "diagnostic only and never rank checkpoints",
           "evaluation": {"decoding": "greedy for E12 and E11 (identical encoding, length limit, normalisation)",
                          "historical": "dev and eval exact agreement and CER, paired family bootstrap, validity, "
                                        "missing endings, length bins, error kinds (exploratory; no tuning or further "
                                        "run may follow from them)",
                          "dev_judgment_coverage": "E12 dev outputs equal to an owner-judged E11-review candidate "
                                                   "take that judgment; others are reported as unknown"},
           "test": {"manifest_sha256": tf["sha256"], "n": tf["n"], "analysis": TEST_ANALYSIS},
           "source_sha256": {f: e06.sha(ROOT / f) for f in files}, "base_sha256": B.BASE_SHA256,
           "base_revision": B.REVISION}
    set_state("register", "done", cfg_hash=reg["cfg_hash"])
    return e07.write_json(EXP / "registration.json", reg) and reg


# ------------------------------------------------------------------ 5–6. training
def selection_key(r: dict) -> tuple:
    """Registered automatic selection: lower CER first, then higher exact agreement (`e07.fit` keeps the earlier
    update on ties, since it replaces the best only on a strictly greater key).

    Examples:
        >>> selection_key({"key": (0.53, -0.157)}) > selection_key({"key": (0.55, -0.160)})
        True
    """
    top1, neg_cer = r["key"]
    return (neg_cer, top1)


def check_resume(out: Path, ident: dict, started: list[str], run: str = RUN) -> str:
    """`fresh`, `resume` or `done`; refuses a resume state from another config/data/base, and a restart of a run
    that was started but lost its resumable state (it would be an uncounted new attempt)."""
    if (out / "done.json").exists():
        return "done"
    if (out / "last.pt").exists():
        import torch
        ck = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        if ck["ident"] != ident:
            raise RuntimeError(f"last.pt does not match the registration: {ck['ident']} vs {ident}")
        return "resume"
    if run in started:
        raise RuntimeError(f"{run} was started but has no resumable state: refusing an uncounted restart")
    return "fresh"


def train() -> dict:
    """The one authorised run; resumable (hashes checked before the model is built and again by `e07.fit`)."""
    reg = json.loads((EXP / "registration.json").read_text())
    cfg = recipe()
    opt, _ = e11.arm_rows()
    if e07.cfg_hash(cfg) != reg["cfg_hash"] or e07.rows_hash(opt) != reg["data"]["optimisation_hash"]:
        raise RuntimeError("recipe or data differ from the registration")
    manifest_rows()
    e07.check_disjoint(opt)
    ident = {"cfg": e07.cfg_hash(cfg), "data": e07.rows_hash(opt), "base": B.BASE_SHA256}
    mode = check_resume(OUT, ident, e07.budget_left("full")["started"])
    dev = e06.primary(e06.items("dev"))

    def evaluate(m):
        r = e07.dev_selection(m, dev)
        return {**r, "key": selection_key(r)}
    set_state("train", "running", mode=mode)
    done = e07.fit(RUN, "full", cfg, opt, evaluate, set(EVAL_AT), OUT)
    set_state("train", "done" if "best" in done else "stopped", stop=done.get("stop"), best=done.get("best"))
    return done


# ------------------------------------------------------------------ 7. freeze and historical diagnostics
def recs(model: str, split: str) -> list[dict]:
    rows = e09.dev_rows() if split == "dev" else e09.eval_rows()
    return e07.load_jsonl(e09.cache_path(model, "greedy", split, e09.checkpoint_sha(model), rows))


def training_curve() -> dict:
    """Dev curve, actual learning rates (every 100 updates and at checkpoints), segments and exposure."""
    hist = json.loads((OUT / "done.json").read_text())["history"]
    lrs = {}
    for ln in (OUT / "metrics.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "lr" in r and (r["update"] % 100 == 0 or r["update"] in (1, 101) or r["update"] in EVAL_AT):
            lrs[r["update"]] = r["lr"]
    segs = [e for e in e07.ledger() if e.get("event") == "segment" and e.get("run") == RUN]
    return {"dev": [{k: h[k] for k in ("update", "dev_top1", "dev_cer", "elapsed_s")} for h in hist],
            "lr_applied_at_update": lrs, "segments": segs,
            "active_training_hours": round(sum(s["hours"] for s in segs), 4)}


@timed("freeze")
def freeze() -> dict:
    """Selected checkpoint and greedy dev/eval predictions of every compared model, hashed before scoring."""
    done = json.loads((OUT / "done.json").read_text())
    for c in COMPARED:
        m, sha = None, e09.checkpoint_sha(c)
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
    eligible = set(EVAL_AT) | ({done["updates"]} if done["stop"] != "max_updates" else set())
    if done["best"]["update"] not in eligible:
        raise RuntimeError("selected update is not an eligible checkpoint")
    out = {"frozen": e06.now(), "rule": SELECT_RULE,
           "selected": {"run": RUN, "update": done["best"]["update"], "dev_top1": done["best"]["dev_top1"],
                        "dev_cer": done["best"]["dev_cer"], "checkpoint_sha256": e09.checkpoint_sha("e12")},
           "candidates": [{k: h[k] for k in ("update", "dev_top1", "dev_cer")} for h in done["history"]],
           "stop": done["stop"], "updates": done["updates"],
           "baselines": {c: {"checkpoint": str((e09.CHECKPOINTS[c] / "best.pt").relative_to(ROOT)),
                             "sha256": e09.checkpoint_sha(c)} for c in ("e11", "e07", "e08")},
           "decoding": {"setting": "greedy", **e09.setting("greedy"), **e09.COMMON, "hash": e09.setting_hash("greedy")},
           "prediction_files_sha256": files}
    set_state("freeze", "done", update=out["selected"]["update"])
    return e07.write_json(REPORTS / "frozen.json", out) and out


def judged_dev() -> dict[str, dict[str, str]]:
    """Owner judgments of the E11 dev review, by dev id and NFC candidate text."""
    key = json.loads((E11_REVIEW / "origin_key.json").read_text())
    out = defaultdict(dict)
    for r in csv.DictReader(io.StringIO((E11_REVIEW / "judgments.tsv").read_text(encoding="utf-8")), delimiter="\t"):
        if r["label"] != "NONE" and r["judgment"]:
            out[key[r["item_id"]]["dev_id"]][e06.nfc(r["candidate"])] = r["judgment"]
    return out


def judgment_coverage(preds: dict[str, str | None], judged: dict[str, dict[str, str]]) -> dict:
    """Known owner judgments of a model's dev outputs; a spelling nobody judged is `unknown`, never wrong.

    Examples:
        >>> judgment_coverage({"a": "ក", "b": "ខ"}, {"a": {"ក": "acceptable"}, "b": {"គ": "acceptable"}})
        {'judged_items': 2, 'acceptable': 1, 'unknown': 1}
    """
    c = Counter()
    for i, txt in judged.items():
        c[txt.get(e06.nfc(preds.get(i) or ""), "unknown")] += 1
    return {"judged_items": len(judged), **dict(sorted(c.items()))}


@timed("evaluate")
def evaluate() -> dict:
    """Historical diagnostics (exploratory) for the frozen greedy predictions, curves and dev-judgment coverage."""
    fz = json.loads((REPORTS / "frozen.json").read_text())
    for name, sha in fz["prediction_files_sha256"].items():
        if e06.sha(PRED / name) != sha:
            raise RuntimeError("frozen predictions changed")
    out = {"label": "Historical mined-reference agreement only (exploratory). Eval has been examined repeatedly and "
                    "its references are unreviewed; no number here measures phonetic accuracy.",
           "frozen": fz["frozen"], "selected": fz["selected"], "candidates": fz["candidates"]}
    for split in ("dev", "eval"):
        items = e06.primary(e06.items(split))
        tab, per = {}, {}
        for c in COMPARED:
            tab[c], per[c] = e09.setting_metrics(items, recs(c, split), e09.gen_seconds(c, "greedy", split))
        con = {"e12_minus_e11": e10.seed_contrast(items, per, ["e12"], ["e11"]),
               "e12_minus_e07 (historical)": e10.seed_contrast(items, per, ["e12"], ["e07"]),
               "e12_minus_e08 (historical)": e10.seed_contrast(items, per, ["e12"], ["e08"])}
        g = {c: {r["id"]: e09.top1(r["cands"]) for r in recs(c, split)} for c in COMPARED}
        kinds, ends, bins = {}, {}, {}
        for c in COMPARED:
            kinds[c] = dict(Counter(e10.error_kind(g[c][i.id], i.accepted) for i in items if not per[c][i.id]["top1"]))
            ends[c] = sum(e11.missing_ending(g[c][i.id], i.accepted) for i in items if not per[c][i.id]["top1"])
        for b, _, _ in e11.LENGTH_BINS:
            its = [i for i in items if e11.length_bin(i.word) == b]
            bins[b] = {"items": len(its), **{c: round(e06.macro(its, {i.id: per[c][i.id]["top1"] for i in its}), 3)
                                             for c in COMPARED}}
        flips = Counter(("both_right" if per["e12"][i.id]["top1"] and per["e11"][i.id]["top1"] else
                         "only_e12" if per["e12"][i.id]["top1"] else "only_e11" if per["e11"][i.id]["top1"]
                         else "both_wrong") for i in items)
        rows = e09.dev_rows() if split == "dev" else e09.eval_rows()
        out[split] = {"items": len(items), "families": len({i.family for i in items}), "table": tab,
                      "contrasts": con, "error_kinds": kinds, "missing_ending": ends, "length_bins_top1": bins,
                      "operational": {c: e10.operational(e07.load_jsonl(e09.cache_path(
                          c, "greedy", split, e09.checkpoint_sha(c), rows))) for c in COMPARED},
                      "e11_vs_e12_item_outcomes": dict(flips)}
    jd = judged_dev()
    dev_g = {c: {r["id"]: e09.top1(r["cands"]) for r in recs(c, "dev")} for c in COMPARED}
    out["dev_judgment_coverage"] = {c: judgment_coverage(dev_g[c], jd) for c in COMPARED}
    done = json.loads((OUT / "done.json").read_text())
    out["training"] = {**training_curve(), "selected_update": done["best"]["update"], "updates": done["updates"],
                       "stop": done["stop"], "peak_rss_mb": done["peak_rss_mb"],
                       "min_available_mb": done["min_available_mb"],
                       "best_at_max_update": done["best"]["update"] == done["updates"] == 6000}
    set_state("evaluate", "done")
    return e07.write_json(EXP / "results.json", out) and out


# ------------------------------------------------------------------ 8. prospective blinded test
BATCH = 30


def test_rows() -> list[tuple[str, str]]:
    return [(r["id"], r["word"]) for r in manifest_rows()]


def aligned(rows: list[tuple[str, str]], recs_: list[dict]) -> list[dict]:
    """Predictions in manifest order, refusing any id/word mismatch.

    Examples:
        >>> aligned([("t1", "cat")], [{"id": "t1", "word": "cat", "cands": []}])[0]["id"]
        't1'
    """
    if [(r["id"], r["word"]) for r in recs_] != rows:
        raise RuntimeError("test predictions are not aligned with the frozen manifest")
    return recs_


@timed("testgen")
def testgen() -> dict:
    """Greedy outputs of the selected E12 checkpoint and E11 on the frozen test inputs (after selection)."""
    fz = json.loads((REPORTS / "frozen.json").read_text())
    rows = test_rows()
    files = {}
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    for c in TESTED:
        sha = e09.checkpoint_sha(c)
        if c == "e12" and sha != fz["selected"]["checkpoint_sha256"]:
            raise RuntimeError("E12 checkpoint differs from the frozen selection")
        if not e09.cache_path(c, "greedy", "e12test", sha, rows).exists():
            m = e09.load(c)[0]
            e09.generate(m, c, "greedy", "e12test", rows, sha)
            e09.unload(m)
        rs = aligned(rows, e07.load_jsonl(e09.cache_path(c, "greedy", "e12test", sha, rows)))
        p = TEST_DATA / f"predictions_{c}.jsonl"
        p.write_text("".join(json.dumps({"id": r["id"], "word": r["word"], "prediction": e09.top1(r["cands"]),
                                         "status": r["cands"][0]["status"] if r["cands"] else "none"},
                                        ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        files[p.name] = e06.sha(p)
    out = {"generated": e06.now(), "manifest_sha256": json.loads((TEST / "frozen.json").read_text())["sha256"],
           "checkpoints": {c: e09.checkpoint_sha(c) for c in TESTED}, "decoding": fz["decoding"],
           "prediction_files_sha256": files, "local": str(TEST_DATA.relative_to(ROOT)),
           "operational": {c: e10.operational(e07.load_jsonl(e09.cache_path(c, "greedy", "e12test",
                                                                             e09.checkpoint_sha(c), rows)))
                           for c in TESTED}}
    set_state("testgen", "done")
    return e07.write_json(TEST / "predictions.json", out) and out


def test_preds() -> dict[str, dict[str, str | None]]:
    """Frozen local test predictions (hash-checked)."""
    f = json.loads((TEST / "predictions.json").read_text())
    out = {}
    for c in TESTED:
        p = TEST_DATA / f"predictions_{c}.jsonl"
        if e06.sha(p) != f["prediction_files_sha256"][p.name]:
            raise RuntimeError(f"{p.name} differs from its frozen hash")
        out[c] = {r["id"]: r["prediction"] for r in e11.jl(p)}
    return out


def build_items(manifest: list[dict], preds: dict[str, dict[str, str | None]]) -> tuple[list[dict], dict]:
    """Blinded items (deduplicated outputs, seeded labels) and the separate origin key.

    Examples:
        >>> its, key = build_items([{"id": "t1", "word": "cat", "pronunciation": "K AE1 T"}],
        ...                        {"e12": {"t1": "ខាត"}, "e11": {"t1": "ខាត"}})
        >>> [c["text"] for c in its[0]["candidates"]], key["t1"]["origins"]
        (['ខាត'], {'A': ['e12', 'e11']})
    """
    items, key = [], {}
    for r in manifest:
        cands = e10.review_candidates([(c, preds[c][r["id"]]) for c in TESTED], f"{TEST_SEED}:labels:{r['id']}")
        items.append({"item_id": r["id"], "word": r["word"], "intended_reading": r["pronunciation"],
                      "candidates": [{"label": lb, "text": t} for lb, t, _ in cands]})
        key[r["id"]] = {"origins": {lb: o for lb, _, o in cands},
                        "empty": [c for c in TESTED if not preds[c][r["id"]]]}
    return items, key


@timed("batches")
def batches() -> dict:
    """Numbered blinded batches (TSV templates + readable Markdown) and the separate origin key, all local."""
    man = manifest_rows()
    items, key = build_items(man, test_preds())
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(TEST_DATA / "origin_key.json", key)
    e07.write_json(TEST_DATA / "test_items.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    names = []
    for k in range(0, len(items), BATCH):
        n = k // BATCH + 1
        part = items[k:k + BATCH]
        buf = io.StringIO()
        w = csv.writer(buf, delimiter="\t", lineterminator="\n")
        w.writerow(["item_id", "word", "intended_reading", "label", "candidate", "judgment"])
        md = [f"# E12 test batch {n} of {-(-len(items) // BATCH)}", "",
              "For each spelling: acceptable / not_acceptable / uncertain (docs/PHONETIC_POLICY.md). "
              "Several or none may be acceptable. Do not type any Khmer.", "",
              "| # | Word | Reading (US) | Candidates |", "|---|---|---|---|"]
        for it in part:
            for c in it["candidates"]:
                w.writerow([it["item_id"], it["word"], it["intended_reading"], c["label"], c["text"], ""])
            md.append(f"| {it['item_id']} | {it['word']} | {it['intended_reading']} | "
                      + " · ".join(f"{c['label']} {c['text']}" for c in it["candidates"]) + " |")
        (TEST_DATA / f"batch_{n}.tsv").write_text(buf.getvalue(), encoding="utf-8")
        (TEST_DATA / f"batch_{n}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        names.append(n)
    man_out = {"created": e06.now(), "items": len(items), "batches": len(names), "per_batch": BATCH,
               "candidates": sum(len(it["candidates"]) for it in items),
               "single_candidate_items": sum(len(it["candidates"]) == 1 for it in items),
               "judgments": list(e10.JUDGMENTS), "local_files": str(TEST_DATA.relative_to(ROOT)),
               "fill_in": "copy batch_<n>.tsv to judgments_<n>.tsv and fill the judgment column, with "
                          "judgments.provenance.json beside them",
               "sha256": {p.name: e06.sha(p) for p in sorted(TEST_DATA.glob("*")) if p.is_file()
                          and not p.name.startswith("judgments")},
               "blinding": "no model names, scores or reference spellings in the batches; labels seeded per item; "
                           "the origin key is a separate local file read only by `import`"}
    set_state("batches", "awaiting_owner_test_judgments", items=len(items))
    return e07.write_json(TEST / "batches.json", man_out) and man_out


def read_test_judgments(items: list[dict]) -> tuple[dict, dict]:
    """(judgment per (item, label), validation problems) from every `judgments_<n>.tsv`; blanks are missing."""
    want = {(it["item_id"], c["label"]): c["text"] for it in items for c in it["candidates"]}
    seen, prob = {}, defaultdict(list)
    for p in sorted(TEST_DATA.glob("judgments_*.tsv")):
        for r in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8")), delimiter="\t"):
            k, j = (r.get("item_id", ""), r.get("label", "")), (r.get("judgment") or "").strip()
            if k not in want:
                prob["unknown_id_or_label"].append(f"{k[0]}:{k[1]}")
                continue
            if e06.nfc(r.get("candidate", "")) != want[k]:
                prob["edited_candidate"].append(f"{k[0]}:{k[1]}")
            if not j:
                continue
            if j not in e10.JUDGMENTS:
                prob["invalid_value"].append(f"{k[0]}:{k[1]}:{j}")
                continue
            if k in seen and seen[k] != j:
                prob["duplicate_conflict"].append(f"{k[0]}:{k[1]}")
            seen[k] = j
    prob["missing"] = sorted(f"{a}:{b}" for a, b in want if (a, b) not in seen)
    return seen, {k: sorted(set(v)) for k, v in prob.items()}


def per_model(items: list[dict], key: dict, seen: dict) -> dict[str, dict[str, str]]:
    """Judgment of each tested model's own output per item (shared when outputs are identical; empty = not ok)."""
    out = {c: {} for c in TESTED}
    for it in items:
        k = key[it["item_id"]]
        for lb, origins in k["origins"].items():
            for o in origins:
                out[o][it["item_id"]] = seen[(it["item_id"], lb)]
        for o in k["empty"]:
            out[o][it["item_id"]] = "not_acceptable"
    return out


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact binomial p for discordant pairs.

    Examples:
        >>> exact_mcnemar(14, 4), exact_mcnemar(0, 0)
        (0.0309, 1.0)
    """
    n = wins + losses
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n), 4)


def analyse_test(j: dict[str, dict[str, str]], a: str = "e12", b: str = "e11") -> dict:
    """The predeclared test analysis (`TEST_ANALYSIS`) for complete per-model judgments `j[model][item]`.

    Examples:
        >>> j = {"e12": {"1": "acceptable", "2": "acceptable", "3": "uncertain"},
        ...      "e11": {"1": "acceptable", "2": "not_acceptable", "3": "acceptable"}}
        >>> r = analyse_test(j)
        >>> r["primary"]["diff_pp"], r["paired"], r["exact_test"]["discordant_clear"]
        (0.0, {'both': 1, 'a_only': 1, 'b_only': 1, 'neither': 0}, [1, 0])
    """
    ids = sorted(j[a])
    if set(j[b]) != set(ids):
        raise ValueError("models judged on different items")
    n = len(ids)
    ok = {m: e05.np.array([j[m][i] == "acceptable" for i in ids], float) for m in (a, b)}
    unc = {m: e05.np.array([j[m][i] == "uncertain" for i in ids], float) for m in (a, b)}
    idx = e05.boot_indices(n)
    d = ok[a] - ok[b]
    bs = e05.np.sort(d[idx].mean(1)) * 100
    paired = Counter("both" if ok[a][k] and ok[b][k] else "a_only" if ok[a][k] else "b_only" if ok[b][k]
                     else "neither" for k in range(n))
    clear = [k for k in range(n) if not unc[a][k] and not unc[b][k]]
    wins = sum(1 for k in clear if ok[a][k] and not ok[b][k])
    losses = sum(1 for k in clear if ok[b][k] and not ok[a][k])
    lenient = {m: ok[m] + unc[m] for m in (a, b)}
    share = lambda v, den: round(float(v.sum()) / den, 4) if den else None  # noqa: E731
    return {
        "n": n, "models": {"a": a, "b": b},
        "counts": {m: dict(Counter(j[m][i] for i in ids)) for m in (a, b)},
        "confirmed_acceptable_share": {m: share(ok[m], n) for m in (a, b)},
        "uncertainty_rate": {m: share(unc[m], n) for m in (a, b)} | {"any": share(e05.np.maximum(unc[a], unc[b]), n)},
        "primary": {"diff_pp": round(float(d.mean()) * 100, 2),
                    "ci95_pp": [round(float(bs[int(0.025 * len(bs))]), 2), round(float(bs[int(0.975 * len(bs)) - 1]), 2)],
                    "bootstrap": {"n": e05.N_BOOT, "seed": e05.BOOT_SEED, "unit": "family (one item each)"}},
        "paired": {k: paired.get(k, 0) for k in ("both", "a_only", "b_only", "neither")},
        "exact_test": {"discordant_clear": [wins, losses], "p_two_sided": exact_mcnemar(wins, losses)},
        "sensitivity": {
            "uncertain_as_acceptable": {"diff_pp": round(float((lenient[a] - lenient[b]).mean()) * 100, 2),
                                        "denominator": n},
            "uncertain_excluded": {"diff_pp": round(float((ok[a][clear] - ok[b][clear]).mean()) * 100, 2)
                                   if clear else None, "denominator": len(clear)}},
        "interpretation": TEST_ANALYSIS["interpretation"]}


@timed("import")
def import_test() -> dict:
    """Validate the owner's test judgments; run the predeclared analysis only when complete and valid."""
    data = json.loads((TEST_DATA / "test_items.json").read_text())
    items, key = data["items"], json.loads((TEST_DATA / "origin_key.json").read_text())
    if not list(TEST_DATA.glob("judgments_*.tsv")):
        set_state("import", "awaiting_owner_test_judgments")
        return {"status": "awaiting_owner_test_judgments", "expected": str((TEST_DATA / "judgments_<n>.tsv"))}
    seen, prob = read_test_judgments(items)
    bad = {k: v for k, v in prob.items() if v and k != "missing"}
    if bad:
        return {"status": "invalid", **bad}
    judged = {i for i, _ in seen}
    progress = {"items_complete": len({it["item_id"] for it in items
                                       if all((it["item_id"], c["label"]) in seen for c in it["candidates"])}),
                "items": len(items), "candidates_missing": len(prob["missing"])}
    if prob["missing"]:
        set_state("import", "awaiting_owner_test_judgments", **progress)
        return {"status": "incomplete", **progress, "first_missing": prob["missing"][:10], "touched": len(judged)}
    prov = TEST_DATA / "judgments.provenance.json"
    if not prov.exists():
        return {"status": "provenance_missing", "expected": str(prov)}
    out = {"status": "scored", "provenance": json.loads(prov.read_text()), **progress,
           "judgment_files_sha256": {p.name: e06.sha(p) for p in sorted(TEST_DATA.glob("judgments_*.tsv"))},
           "track": "single owner judge, blinded, not independent of the project",
           "analysis": analyse_test(per_model(items, key, seen))}
    set_state("import", "scored")
    return e07.write_json(REPORTS / "test_scores.json", out) and out


def status() -> dict:
    s = state()
    s["inference_hours"] = round(e11.inference_hours(), 3)
    s["training"] = e07.budget_left("full")
    return s


STEPS = {"verify": verify, "testset": testset, "register": register, "train": train, "freeze": freeze,
         "evaluate": evaluate, "testgen": testgen, "batches": batches, "import": import_test, "status": status}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=list(STEPS))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(LIMITS["threads"])
    out = STEPS[a.step]()
    print(json.dumps(out, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
