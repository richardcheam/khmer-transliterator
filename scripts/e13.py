"""E13: pronunciation-guided adaptation of ByT5-small: owner-accepted difficult examples + the English reading as
input + replay of the real-data pool R, starting from E11's adapter. One combined recipe, one seed-0 run.

Every source is `<word>|<CMUdict ARPAbet with stress | <no_pron>>` (lower-cased bytes by `e06_byt5.encode_src`;
amended before training from the labelled `word: …\\npronunciation: …` form, see `AMENDMENTS`). The reading comes from the pinned local CMUdict (`khmer_translit.phonetic_source`, first
reading = the established default); a word without one gets `<no_pron>`. E11 stays the word-only baseline.

Reuses `scripts/e07.py` (model, resumable loop, guard, budget), E09's cached generation, E10's review helpers,
E11's data helpers, E12's freshness / balancing / analysis code and the local review page `scripts/e12_review.py`.

Steps (in order):
    verify      starting state → reports/start_checks.json
    inputs      100 collection families (60 train / 40 dev) and 120 test families, inputs only, frozen before any
                candidate is generated → collection/, test/ (manifests, frozen.json, overlap audit)
    collect     E11 greedy + distinct canonical-beam alternative on the 100 collection words; blinded review batch
                → data/e13/collection/ (local); state awaiting_owner_collection_judgments
    import-collection   validate the owner's collection judgments; accepted train/dev data, quarantine, unlabelled,
                feasibility check → reports/collection.json, data/e13/datasets/ (local)
    register    recipe, sampling, masking, limits, data hashes, selection, test analysis → registration.json
    pipeline    input/target/masking/limit checks, adapter loading, timing and a separate smoke update
                → reports/pipeline_checks.json, models/e13/smoke/ (local)
    train       the one authorised run (≤ 2,000 updates / 6 h; resumable) → models/e13/pron_s0/
    freeze      selected checkpoint + greedy predictions (collection dev, historical dev/eval) → reports/frozen.json
    evaluate    collection-dev and historical diagnostics, coverage, curves → results.json
    mechanism   correct / missing / mismatched pronunciation on dev items → reports/mechanism.json
    testgen     greedy E13 and E11 on the frozen test inputs → data/e13/test/ (local), test/predictions.json
    batches     blinded test batches (≤ 150 spellings each) + origin key → data/e13/test/, test/batches.json
    import      validate test judgments; when complete, the predeclared analysis → reports/test_scores.json
    status      phases and budgets

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e13.py verify
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
import e06_byt5 as B  # noqa: E402
import e07  # noqa: E402
import e08  # noqa: E402
import e09  # noqa: E402
import e10  # noqa: E402  (pure helpers only)
import e11  # noqa: E402  (data helpers; bookkeeping overridden below)
import e12  # noqa: E402  (freshness, balancing, analysis helpers; bookkeeping overridden below)
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E13-pronunciation-guided"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
PRED = EXP / "predictions"
COLL = EXP / "collection"
TEST = EXP / "test"
STATE_FILE = EXP / "state.json"
LEDGER = EXP / "runs" / "ledger.jsonl"
DATA = ROOT / "data" / "e13"                     # git-ignored: candidates, batches, origin keys, judgments, datasets
COLL_DATA = DATA / "collection"
TEST_DATA = DATA / "test"
SETS = DATA / "datasets"
MODELS = ROOT / "models" / "e13"
RUN = "pron_s0"
OUT = MODELS / RUN
SMOKE = MODELS / "smoke"
GiB = 1024 ** 3
LIMITS = {"threads": 6, "rss_max_bytes": int(6.5 * GiB), "min_available_bytes": int(0.4 * GiB), "pressure_checks": 3,
          "tiny_attempts": 0, "full_runs": 1, "full_max_updates": 2000, "full_max_hours": 6.0,
          "training_hours_total": 6.0, "wall_hours_total": 1e9, "inference_hours_total": 4.0}
EVAL_AT = (400, 800, 1200, 1600, 2000)
E11_DIR = ROOT / "models" / "e11" / "r_s0"
E11_BEST_SHA = e12.E11_BEST_SHA
E11_REVIEW = e12.E11_REVIEW
E12_TEST = ROOT / "experiments" / "E12-longer-schedule" / "test"
POLICY = ROOT / "docs" / "PHONETIC_POLICY.md"
NO_PRON = "<no_pron>"
SRC_MAX_BYTES = 192          # registered: longer sources are refused, never truncated
TGT_MAX_BYTES = B.RECIPE["max_tgt_bytes"]       # 120, unchanged
COLL_SEED, TEST_SEED, LABEL_SEED, MASK_SEED = "e13-collection", "e13-test", "e13-labels", "e13-mismatch"
COLL_PLAN = {"train": {"irregular": 40, "ordinary": 10, "names": 10},
             "dev": {"irregular": 20, "ordinary": 10, "names": 10}}
CATEGORIES = ("ordinary", "irregular", "names")
FEASIBLE = {"train_families_accepted": 30, "train_irregular_accepted": 20, "dev_families_accepted": 15}
MIX = {"block_updates": 100, "batch": 32, "reviewed_share": 0.10, "mask_share": 0.10}     # batch = effective batch
AMENDMENTS = [  # before training, after the registered pipeline smoke failed (no training had started)
    {"date": "2026-10-30", "what": "source format `word: <w>\\npronunciation: <p>` → `<w>|<p>` (same word and ARPAbet "
     "reading with stress; only the 22-byte field labels removed)",
     "why": "measured: labelled sources averaged 39 bytes (max 72); a 32-row batch peaked at 6.1 GB RSS and took "
            "about 37 s per update (≈ 20 h for 2,000 updates, over the 6 h cap), and the smoke run was killed by the "
            "system for lack of memory; the compact form averages 18 bytes (max 51)"},
    {"date": "2026-10-30", "what": "micro-batches 2 × 16 with gradient accumulation (effective batch 32 unchanged)",
     "why": "a 32-row compact batch still peaked at about 6.3 GB, at the 6.5 GB guard; e07.fit averages the two "
            "equal-size micro-batch losses, so the update is the mean of the two halves' token-mean losses"}]
TEST_BATCH_MAX_CANDIDATES = 150
COMPARED = ("e13", "e11")
SMOKE_UPDATES = 3

e07.LEDGER = LEDGER
e07.EXPERIMENT = EXPERIMENT
e07.STATE = ROOT / "state" / "e13"
e07.LIMITS = LIMITS
e09.LEDGER = LEDGER
e09.LIMITS = {**e09.LIMITS, "active_hours": LIMITS["inference_hours_total"]}
e09.CHECKPOINTS["e11"] = E11_DIR
e09.CHECKPOINTS["e13"] = OUT
e11.STATE_FILE = STATE_FILE
e11.LEDGER = LEDGER
state, set_state = e11.state, e11.set_state


def timed(step: str):
    return e11.timed(step)


# ------------------------------------------------------------------ 1. input representation
def pron(word: str) -> dict:
    """The reading used for one word: the first CMUdict reading (the established default), else none.

    Returns `{"phones": "Y AA1 T" | None, "status": dictionary|unavailable|refused, "ambiguous": bool,
    "pron_id": "cmudict:yacht" | None}`. A refused or missing word never gets invented phones.

    Examples:
        >>> pron("yacht")["phones"], pron("zzqxv")["status"]
        ('Y AA1 T', 'unavailable')
    """
    try:
        r = ps.get_pronunciation(word)
    except ps.InputError:
        return {"phones": None, "status": "refused", "ambiguous": False, "pron_id": None}
    if not r.phones:
        return {"phones": None, "status": "unavailable", "ambiguous": False, "pron_id": None}
    return {"phones": " ".join(r.phones), "status": "dictionary", "ambiguous": bool(r.ambiguous),
            "pron_id": f"cmudict:{r.word}"}


def source(word: str, phones: str | None) -> str:
    """The E13 source text (before `encode_src` lower-cases it); `phones=None` → `<no_pron>`.

    Raises:
        ValueError: if the encoded source exceeds `SRC_MAX_BYTES` (refused, never truncated).

    Examples:
        >>> source("yacht", "Y AA1 T")
        'yacht|Y AA1 T'
        >>> source("yacht", None)
        'yacht|<no_pron>'
    """
    s = f"{word}|{phones if phones else NO_PRON}"
    n = len(B.encode_src(s)) - 1
    if n > SRC_MAX_BYTES:
        raise ValueError(f"source of {word!r} has {n} bytes > {SRC_MAX_BYTES}")
    return s


def check_target(text: str) -> None:
    """Refuse an empty or overlong target (never truncated)."""
    n = len(B.encode_tgt(text)) - 1
    if not text or n > TGT_MAX_BYTES:
        raise ValueError(f"target {text!r} has {n} bytes (limit {TGT_MAX_BYTES})")


# ------------------------------------------------------------------ 2. starting state
@timed("verify")
def verify() -> dict:
    """E11 checkpoint, base, arm R, CMUdict, policy, decoding, protected manifests and review-set membership."""
    chk = {}
    done = json.loads((E11_DIR / "done.json").read_text())
    sha = e06.sha(E11_DIR / "best.pt")
    chk["e11_checkpoint"] = {"pass": sha == E11_BEST_SHA == done["best_sha256"] and done["best"]["update"] == 3000,
                             "sha256": sha, "update": done["best"]["update"]}
    chk["e11_recipe"] = {"pass": done["ident"]["cfg"] == e12.E11_CFG_HASH and done["ident"]["data"] == e12.E11_DATA_HASH,
                         "cfg_hash": done["ident"]["cfg"]}
    B.check_base()
    chk["base"] = {"pass": done["ident"]["base"] == B.BASE_SHA256, "revision": B.REVISION, "sha256": B.BASE_SHA256}
    opt, rows = e11.arm_rows()
    chk["arm_R"] = {"pass": len(opt) == 4764, "optimisation_rows": len(opt), "train_val_rows": len(rows) - len(opt),
                    "sha256": e11.ARM_SHA}
    cmu, cmu_sha = ps.cmudict()
    chk["cmudict"] = {"pass": len(cmu) > 100000, "path": str(ps.CMUDICT.relative_to(ROOT)), "sha256": cmu_sha,
                      "entries": len(cmu), "yacht": pron("yacht")}
    chk["policy"] = {"pass": POLICY.exists(), "sha256": e06.sha(POLICY)}
    chk["decoding"] = {"pass": e09.setting("greedy")["num_beams"] == 1
                       and e09.setting("beam4_can")["early_stopping"] == "never"
                       and e09.COMMON["max_new_tokens"] == 121,
                       "greedy": e09.setting("greedy"), "canonical_beam": e09.setting("beam4_can"), **e09.COMMON}
    prot = e08.protected()
    chk["protected"] = {"pass": bool(prot["counts"]), "counts": prot["counts"],
                        "access": "Latin inputs only (e08.protected); no reserve target or E02 final reference opened"}
    e11_words = [it["word"] for it in json.loads((E11_REVIEW / "review_batch.json").read_text())["items"]]
    e12_words = [r["word"] for r in e11.jl(E12_TEST / "input_manifest.jsonl")]
    train_words = {r["word"] for r in rows}
    chk["review_sets"] = {"pass": len(set(e11_words)) == 60 and len(e12_words) == 120
                          and not train_words & (set(e11_words) | set(e12_words)),
                          "e11_review_families": len(set(e11_words)), "e12_test_families": len(e12_words),
                          "in_R": len(train_words & (set(e11_words) | set(e12_words)))}
    out = {"checked": e06.now(), "checks": chk, "ok": all(c["pass"] for c in chk.values())}
    set_state("verify", "done" if out["ok"] else "failed")
    return e07.write_json(REPORTS / "start_checks.json", out) and out


# ------------------------------------------------------------------ 3. frozen inputs (collection + test)
PATTERNS = (     # declared spelling-to-sound patterns, first match wins (the irregular collection is stratified)
    ("silent_letter", lambda w, p: bool(re.match(r"(kn|wr|gn|ps|pn|rh)", w) or re.search(r"(mb|mn|gn|gm|bt)$", w)
                                        or re.search(r"(ght|alk|alm|olm|stle|tch)", w))),
    ("unusual_vowel", lambda w, p: bool(re.search(r"(ough|augh|eigh|aigh|ea|ei|ie|ui|eo|oe|ae|au|ou|oo|ue)", w))),
    ("consonant_digraph", lambda w, p: bool(re.search(r"(ph|gh|ch|th|sh|ck|qu|wh)", w))),
    ("spelling_dependent_consonant", lambda w, p: bool(("c" in w and "S" in p.split()) or ("g" in w and "JH" in p.split())
                                                       or "x" in w or ("s" in w and "Z" in p.split()))),
    ("cluster_or_ending", lambda w, p: e12.phone_features(tuple(p.split()))["cluster"]
                                       or bool(re.search(r"(tion|sion|ture|ous|que|gue|le)$", w))),
    ("other", lambda w, p: True),
)


def pattern_of(word: str, phones: str) -> str:
    """The declared pattern of a word (first matching rule).

    Examples:
        >>> pattern_of("knight", "N AY1 T"), pattern_of("bread", "B R EH1 D"), pattern_of("phone", "F OW1 N")
        ('silent_letter', 'unusual_vowel', 'consonant_digraph')
    """
    return next(n for n, f in PATTERNS if f(word, phones))


def pick_patterned(cands: list[str], cmu: dict, seed: str, n: int, used: set[str]) -> list[str]:
    """`n` words, round-robin over the declared patterns, seeded order inside each pattern, one per family."""
    by = defaultdict(list)
    for w in sorted(cands, key=lambda w: e06.order_key(seed, w)):
        by[pattern_of(w, " ".join(cmu[w][0]))].append(w)
    pos, picked = Counter(), []
    while len(picked) < n and any(pos[p] < len(by[p]) for p, _ in PATTERNS):
        for p, _ in PATTERNS:
            while pos[p] < len(by[p]) and len(picked) < n:
                w = by[p][pos[p]]
                pos[p] += 1
                if not used & e12.family_keys(w):
                    used.update(e12.family_keys(w))
                    picked.append(w)
                    break
    return picked


def pick_plain(cands: list[str], cmu: dict, seed: str, n: int, used: set[str], vowel_cap: int = 3) -> list[str]:
    """`n` words in seeded order, at most `vowel_cap` per stressed vowel in a first pass, one per family."""
    order = sorted(cands, key=lambda w: e06.order_key(seed, w))
    picked, vow = [], Counter()
    for capped in (True, False):
        for w in order:
            if len(picked) >= n:
                return picked
            v = e12.phone_features(cmu[w][0])["stressed_vowel"]
            if w in picked or (capped and vow[v] >= vowel_cap) or used & e12.family_keys(w):
                continue
            used.update(e12.family_keys(w))
            picked.append(w)
            vow[v] += 1
    return picked


def pools() -> tuple[dict, dict, Counter, dict]:
    """Fresh candidate pools (ordinary / irregular / names) after every exclusion, as in E12 plus E12's test words."""
    cmu, cmu_sha = ps.cmudict()
    prot = e08.protected()
    hist, _ = e06.historical_universe()
    _, rows = e11.arm_rows()
    train_words = {r["word"] for r in rows} | {json.loads(x)["english"] for x in
                                               e08.MANIFEST.read_text(encoding="utf-8").splitlines()}
    extra = {json.loads(x)["word"] for x in
             (ROOT / "experiments/E10-release-cycle/confirmation/input_manifest.jsonl").read_text().splitlines()}
    for p in (ROOT / "data/e10/review/review_batch.json", E11_REVIEW / "review_batch.json"):
        extra |= {it["word"] for it in json.loads(p.read_text())["items"]}
    extra |= {r["word"] for r in e11.jl(E12_TEST / "input_manifest.jsonl")}
    ctx = {"cmu": cmu, "prot": prot, "hist_stems": e06.stem_set(hist), "train_words": train_words,
           "train_stems": e06.stem_set(train_words), "extra": extra, "extra_stems": e06.stem_set(extra),
           "romanised": set((ROOT / "models/model/romanised_forms.txt").read_text(encoding="utf-8").split())}
    lower, names = e06.candidate_words()
    ctx["listed"] = lower | names
    excluded, pool = Counter(), []
    for w, kind in sorted([(w, "lower") for w in lower] + [(w, "name") for w in names]):
        why = e12.exclusion(w, kind, ctx)
        if why:
            excluded[why] += 1
        else:
            pool.append((w, kind))
    freq = e08.web_counts({w for w, k in pool if k == "lower"})
    lw = [w for w, k in pool if k == "lower"]
    out = {"names": [w for w, k in pool if k == "name"],
           "irregular": [w for w in lw if e06.is_difficult(w, cmu[w][0])],
           "ordinary": [w for w in lw if freq[w] >= 2 and not e06.is_difficult(w, cmu[w][0])]}
    return out, cmu, excluded, {"cmu_sha": cmu_sha, "prot": prot, "ctx": ctx, "train_words": train_words}


def manifest_row(id_: str, w: str, cat: str, cmu: dict, **extra) -> dict:
    r = cmu[w][0]
    p = " ".join(r)
    return {"id": id_, "word": w, "category": cat, **extra, "length_bin": e11.length_bin(w), "pronunciation": p,
            "pron_id": f"cmudict:{w}", "pron_source": "first CMUdict reading (only reading after stress removal)",
            "accent": "en-US-cmudict", "pattern": pattern_of(w, p), **e12.phone_features(r)}


def frozen_rows(folder: Path) -> list[dict]:
    """A frozen manifest (hash-checked)."""
    p = folder / "input_manifest.jsonl"
    if e06.sha(p) != json.loads((folder / "frozen.json").read_text())["sha256"]:
        raise RuntimeError(f"{p} differs from its frozen hash")
    return e11.jl(p)


def overlap_audit(rows: list[dict], info: dict) -> dict:
    """Counts that must all be 0 for a frozen manifest."""
    prot, tw = info["prot"], info["train_words"]
    tstems = e06.stem_set(tw)
    dev_eval = {i.word for s in ("dev", "eval") for i in e06.items(s)}
    ctx = info["ctx"]
    return {"protected_family": sum(e08.is_protected(r["word"], prot) for r in rows),
            "R_or_E08_word": sum(r["word"] in tw for r in rows),
            "R_or_E08_family": sum(bool(rd.related(r["word"], tstems)) for r in rows),
            "historical_dev_eval_word": sum(r["word"] in dev_eval for r in rows),
            "historical_family": sum(not e06.is_fresh(r["word"], ctx["hist_stems"]) for r in rows),
            "earlier_review_or_test_family": sum(r["word"] in ctx["extra"] or bool(rd.related(r["word"], ctx["extra_stems"]))
                                                 for r in rows),
            "duplicate_family_keys": len([k for r in rows for k in e12.family_keys(r["word"])])
                                     - len({k for r in rows for k in e12.family_keys(r["word"])})}


def write_manifest(folder: Path, rows: list[dict], meta: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    man = folder / "input_manifest.jsonl"
    man.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    f = {"frozen": e06.now(), "sha256": e06.sha(man), "n": len(rows), **meta}
    return e07.write_json(folder / "frozen.json", f) and f


@timed("inputs")
def inputs() -> dict:
    """Collection (60 train / 40 dev) and test (120) families, inputs and readings only, frozen together before any
    candidate or output is generated. One family per item across both sets."""
    if (COLL / "frozen.json").exists() and (TEST / "frozen.json").exists():
        frozen_rows(COLL), frozen_rows(TEST)
        return {"collection": json.loads((COLL / "frozen.json").read_text()),
                "test": json.loads((TEST / "frozen.json").read_text())}
    pl, cmu, excluded, info = pools()
    used: set[str] = set()
    coll = {}
    for c in ("names", "irregular", "ordinary"):                 # rarer pools first
        n = COLL_PLAN["train"][c] + COLL_PLAN["dev"][c]
        pick = (pick_patterned if c == "irregular" else pick_plain)(pl[c], cmu, f"{COLL_SEED}:{c}", n, used)
        if len(pick) < n:
            raise RuntimeError(f"collection pool {c} too small: {len(pick)} < {n}")
        coll[c] = pick
    rows, k = [], 0
    for c in CATEGORIES:
        ratio = COLL_PLAN["dev"][c] + COLL_PLAN["train"][c]
        seq = coll[c]
        if c == "irregular":                 # group by pattern (pick order inside), so each pattern is split 2:1
            pat = {w: pattern_of(w, " ".join(cmu[w][0])) for w in seq}
            rank = {p: n for n, (p, _) in enumerate(PATTERNS)}
            seq = sorted(seq, key=lambda w: (rank[pat[w]], coll[c].index(w)))
        for j, w in enumerate(seq):          # interleaved split: both splits cover every pattern
            split = "dev" if (j * COLL_PLAN["dev"][c]) % ratio >= ratio - COLL_PLAN["dev"][c] else "train"
            k += 1
            rows.append(manifest_row(f"c{k:03d}", w, c, cmu, split=split))
    rows.sort(key=lambda r: e06.order_key(f"{COLL_SEED}:order", r["word"]))
    for n, r in enumerate(rows, 1):
        r["id"] = f"c{n:03d}"
    got = Counter((r["split"], r["category"]) for r in rows)
    if any(got[(s, c)] != COLL_PLAN[s][c] for s in COLL_PLAN for c in CATEGORIES):
        raise RuntimeError(f"split counts differ from the plan: {dict(got)}")
    test, notes = {}, {}
    for c in ("names", "irregular", "ordinary"):
        test[c], notes[c] = e12.pick_balanced(pl[c], cmu, f"{TEST_SEED}:{c}", used)
    trows = [(c, w) for c in CATEGORIES for w in test[c]]
    trows.sort(key=lambda x: e06.order_key(f"{TEST_SEED}:order", x[1]))
    trows = [manifest_row(f"u{n:03d}", w, c, cmu) for n, (c, w) in enumerate(trows, 1)]
    cross = {k for r in rows for k in e12.family_keys(r["word"])} & {k for r in trows for k in e12.family_keys(r["word"])}
    common = {"cmudict_sha256": info["cmu_sha"], "protected_counts": info["prot"]["counts"], "pool": {c: len(v) for c, v in pl.items()},
              "excluded": dict(excluded),
              "before_outputs": "frozen before any E13 candidate, training or test output; no model has been run on "
                                "these words",
              "readings": "the first CMUdict reading; words whose readings differ beyond stress (heteronyms) are "
                          "excluded; names keep their English (CMUdict) reading",
              "exclusions": ["protected families (E03 dev/eval, E02 held-out, benchmark panels, reserve; Latin inputs "
                             "only)", "every historical Latin token (e06.historical_universe)", "R and E08 inputs",
                             "E10 confirmation, E10/E11 review and E12 test words and families",
                             "romanised forms, abbreviations, ambiguous readings, derived name forms, offensive"],
              "cross_set_family_overlap": len(cross)}
    cf = write_manifest(COLL, rows, {
        **common, "seed": COLL_SEED, "plan": COLL_PLAN,
        "patterns": [p for p, _ in PATTERNS],
        "pattern_counts": {s: dict(Counter(r["pattern"] for r in rows if r["split"] == s and r["category"] == "irregular"))
                           for s in COLL_PLAN},
        "selection": "irregular: E06's difficulty rule, round-robin over the declared patterns in seeded order; "
                     "ordinary and names: seeded order with ≤ 3 per stressed vowel; split interleaved in pick order",
        "overlap_audit": overlap_audit(rows, info),
        "train_dev_family_overlap": len({k for r in rows if r["split"] == "train" for k in e12.family_keys(r["word"])}
                                        & {k for r in rows if r["split"] == "dev" for k in e12.family_keys(r["word"])})})
    tf = write_manifest(TEST, trows, {
        **common, "seed": TEST_SEED, "per_category": {c: len(test[c]) for c in CATEGORIES}, "notes": notes,
        "selection": "E12's balanced sampler (length quotas 13/14/13, vowel and cluster caps) per category",
        "distribution": "declared balanced evaluation distribution (40 ordinary, 40 irregular, 40 names); not a "
                        "sample of real usage",
        "overlap_audit": overlap_audit(trows, info)})
    if cross or any(cf["overlap_audit"].values()) or any(tf["overlap_audit"].values()) or cf["train_dev_family_overlap"]:
        raise RuntimeError("overlap audit failed")
    set_state("inputs", "done", collection=len(rows), test=len(trows))
    return {"collection": cf, "test": tf}


# ------------------------------------------------------------------ 4. collection candidates and review
def coll_rows() -> list[tuple[str, str]]:
    return [(r["id"], r["word"]) for r in frozen_rows(COLL)]


def alternative(greedy: str | None, beam: list[dict]) -> str | None:
    """The first well-formed canonical-beam candidate different from the greedy output (NFC), else None.

    Examples:
        >>> alternative("ក", [{"text": "ក", "status": "ok"}, {"text": "ខ", "status": "ok"}])
        'ខ'
        >>> alternative("ក", [{"text": "ក", "status": "ok"}]) is None
        True
    """
    g = e06.nfc(greedy or "")
    for c in beam:
        if c["status"] == "ok" and c["text"] and e06.nfc(c["text"]) != g:
            return e06.nfc(c["text"])
    return None


def build_items(manifest: list[dict], sources: dict[str, dict[str, str | None]], seed: str) -> tuple[list, dict]:
    """Blinded items (deduplicated texts, seeded labels) and the separate origin key, for any named sources.

    Examples:
        >>> its, key = build_items([{"id": "c1", "word": "cat", "pronunciation": "K AE1 T"}],
        ...                        {"greedy": {"c1": "ខាត"}, "beam_alt": {"c1": None}}, "s")
        >>> [c["text"] for c in its[0]["candidates"]], key["c1"]["origins"]
        (['ខាត'], {'A': ['greedy']})
    """
    items, key = [], {}
    for r in manifest:
        cands = e10.review_candidates([(s, sources[s][r["id"]]) for s in sources], f"{seed}:{r['id']}")
        items.append({"item_id": r["id"], "word": r["word"], "intended_reading": r["pronunciation"],
                      "candidates": [{"label": lb, "text": t} for lb, t, _ in cands]})
        key[r["id"]] = {"origins": {lb: o for lb, _, o in cands}, "empty": [s for s in sources if not sources[s][r["id"]]]}
    return items, key


def write_batches(folder: Path, items: list[dict], title: str, max_cands: int) -> list[str]:
    """Fill-in TSV templates and readable Markdown, each with ≤ `max_cands` spellings (whole items)."""
    parts, cur = [], []
    for it in items:
        if cur and sum(len(x["candidates"]) for x in cur) + len(it["candidates"]) > max_cands:
            parts.append(cur)
            cur = []
        cur.append(it)
    parts += [cur] if cur else []
    names = []
    for n, part in enumerate(parts, 1):
        buf = io.StringIO()
        w = csv.writer(buf, delimiter="\t", lineterminator="\n")
        w.writerow(["item_id", "word", "intended_reading", "label", "candidate", "judgment"])
        md = [f"# {title}: batch {n} of {len(parts)}", "",
              "For each spelling: acceptable / not_acceptable / uncertain (docs/PHONETIC_POLICY.md). "
              "Several or none may be acceptable. Do not type any Khmer.", "",
              "| # | Word | Reading (US) | Candidates |", "|---|---|---|---|"]
        for it in part:
            for c in it["candidates"]:
                w.writerow([it["item_id"], it["word"], it["intended_reading"], c["label"], c["text"], ""])
            md.append(f"| {it['item_id']} | {it['word']} | {it['intended_reading']} | "
                      + " · ".join(f"{c['label']} {c['text']}" for c in it["candidates"]) + " |")
        (folder / f"batch_{n}.tsv").write_text(buf.getvalue(), encoding="utf-8")
        (folder / f"batch_{n}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        names += [f"batch_{n}.tsv", f"batch_{n}.md"]
    return names


@timed("collect")
def collect() -> dict:
    """E11 greedy and a distinct canonical-beam alternative (word-only E11 interface) for the 100 collection words;
    one blinded review batch (≤ 200 spellings) and the separate origin key, all local."""
    man = frozen_rows(COLL)
    rows = coll_rows()
    sha = e09.checkpoint_sha("e11")
    if sha != E11_BEST_SHA:
        raise RuntimeError("E11 checkpoint changed")
    m = None
    for name in ("greedy", "beam4_can"):
        if not e09.cache_path("e11", name, "e13coll", sha, rows).exists():
            m = m or e09.load("e11")[0]
            e09.generate(m, "e11", name, "e13coll", rows, sha)
    if m is not None:
        e09.unload(m)
    g = {r["id"]: e09.top1(r["cands"]) for r in e07.load_jsonl(e09.cache_path("e11", "greedy", "e13coll", sha, rows))}
    b = {r["id"]: alternative(g[r["id"]], r["cands"])
         for r in e07.load_jsonl(e09.cache_path("e11", "beam4_can", "e13coll", sha, rows))}
    items, key = build_items(man, {"greedy": g, "beam_alt": b}, f"{COLL_SEED}:labels")
    COLL_DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(COLL_DATA / "origin_key.json", key)
    e07.write_json(COLL_DATA / "items.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    (COLL_DATA / "candidates.jsonl").write_text("".join(json.dumps({"id": i, "greedy": g[i], "beam_alt": b[i]},
                                                                   ensure_ascii=False) + "\n" for i in g),
                                                encoding="utf-8")
    files = write_batches(COLL_DATA, items, "E13 collection review", 200)
    out = {"created": e06.now(), "items": len(items), "candidates": sum(len(i["candidates"]) for i in items),
           "single_candidate_items": sum(len(i["candidates"]) == 1 for i in items),
           "no_distinct_alternative": sum(v is None for v in b.values()),
           "generator": {"model": "e11", "checkpoint_sha256": sha, "greedy": e09.setting("greedy"),
                         "alternative": {"setting": "beam4_can", **e09.setting("beam4_can"),
                                         "rule": "first well-formed beam candidate different from greedy"},
                         "input": "word only (E11's interface)"},
           "sha256": {f: e06.sha(COLL_DATA / f) for f in ["origin_key.json", "items.json", "candidates.jsonl", *files]},
           "local_files": str(COLL_DATA.relative_to(ROOT)),
           "blinding": "no model names, scores, ranks or references on the page; labels seeded per item; the origin "
                       "key is read only by import-collection"}
    set_state("collect", "awaiting_owner_collection_judgments", items=len(items))
    return e07.write_json(COLL / "batches.json", out) and out


def read_judgments(folder: Path, items: list[dict]) -> tuple[dict, dict]:
    """(judgment per (item, label), validation problems) from every `judgments_<n>.tsv`; blanks are missing.

    Examples:
        >>> import tempfile
        >>> d = Path(tempfile.mkdtemp())
        >>> _ = (d / "judgments_1.tsv").write_text("item_id\\tlabel\\tcandidate\\tjudgment\\nc1\\tA\\tក\\t\\n", encoding="utf-8")
        >>> read_judgments(d, [{"item_id": "c1", "candidates": [{"label": "A", "text": "ក"}]}])[1]["missing"]
        ['c1:A']
    """
    want = {(it["item_id"], c["label"]): c["text"] for it in items for c in it["candidates"]}
    seen, prob = {}, defaultdict(list)
    for p in sorted(folder.glob("judgments_*.tsv")):
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


def validate(folder: Path, items: list[dict]) -> dict:
    """`awaiting` / `invalid` / `incomplete` / `provenance_missing` / `complete` with the judgments."""
    if not list(folder.glob("judgments_*.tsv")):
        return {"status": "awaiting"}
    seen, prob = read_judgments(folder, items)
    bad = {k: v for k, v in prob.items() if v and k != "missing"}
    if bad:
        return {"status": "invalid", **bad}
    if prob["missing"]:
        return {"status": "incomplete", "candidates_missing": len(prob["missing"]), "first_missing": prob["missing"][:10]}
    prov = folder / "judgments.provenance.json"
    if not prov.exists():
        return {"status": "provenance_missing"}
    return {"status": "complete", "seen": seen, "provenance": json.loads(prov.read_text()),
            "judgment_files_sha256": {p.name: e06.sha(p) for p in sorted(folder.glob("judgments_*.tsv"))}}


def datasets_from(man: list[dict], items: list[dict], seen: dict) -> dict[str, list[dict]]:
    """Per collection family: accepted alternatives (targets), quarantined (uncertain), rejected (kept as evidence
    only) and a label status. No target is invented; rejected or uncertain spellings never become targets.

    Examples:
        >>> man = [{"id": "c1", "word": "w", "category": "irregular", "split": "train", "pronunciation": "W",
        ...         "pron_id": "cmudict:w", "pattern": "other"}]
        >>> its = [{"item_id": "c1", "candidates": [{"label": "A", "text": "ក"}, {"label": "B", "text": "ខ"}]}]
        >>> r = datasets_from(man, its, {("c1", "A"): "acceptable", ("c1", "B"): "uncertain"})["train"][0]
        >>> r["accepted"], r["quarantined"], r["label_status"]
        (['ក'], ['ខ'], 'accepted')
    """
    by = {it["item_id"]: it for it in items}
    out = {"train": [], "dev": []}
    for r in man:
        cs = by[r["id"]]["candidates"]
        j = {c["text"]: seen[(r["id"], c["label"])] for c in cs}
        acc = [t for t, v in j.items() if v == "acceptable"]
        for t in acc:
            check_target(t)
        out[r["split"]].append({**{k: r[k] for k in ("id", "word", "category", "split", "pronunciation", "pron_id",
                                                     "pattern")},
                                "accepted": acc, "quarantined": [t for t, v in j.items() if v == "uncertain"],
                                "rejected": [t for t, v in j.items() if v == "not_acceptable"],
                                "label_status": "accepted" if acc else
                                ("uncertain_only" if any(v == "uncertain" for v in j.values()) else
                                 "none_acceptable" if cs else "no_candidate")})
    return out


def feasibility(ds: dict[str, list[dict]]) -> dict:
    """The registered minimum counts (feasibility, not power).

    Examples:
        >>> f = feasibility({"train": [{"accepted": ["x"], "category": "irregular"}] * 30,
        ...                  "dev": [{"accepted": ["x"], "category": "names"}] * 15})
        >>> f["pass"]
        True
    """
    got = {"train_families_accepted": sum(bool(r["accepted"]) for r in ds["train"]),
           "train_irregular_accepted": sum(bool(r["accepted"]) and r["category"] == "irregular" for r in ds["train"]),
           "dev_families_accepted": sum(bool(r["accepted"]) for r in ds["dev"])}
    return {"required": FEASIBLE, "got": got, "pass": all(got[k] >= v for k, v in FEASIBLE.items())}


def dataset(split: str) -> list[dict]:
    """The accepted dataset of one split (hash-checked against reports/collection.json)."""
    p = SETS / f"{split}.jsonl"
    if e06.sha(p) != json.loads((REPORTS / "collection.json").read_text())["datasets_sha256"][p.name]:
        raise RuntimeError(f"{p.name} differs from its recorded hash")
    return e11.jl(p)


@timed("import_collection")
def import_collection() -> dict:
    """Validate the owner's collection judgments and build the reviewed datasets (local) with the feasibility check."""
    man = frozen_rows(COLL)
    items = json.loads((COLL_DATA / "items.json").read_text())["items"]
    v = validate(COLL_DATA, items)
    if v["status"] != "complete":
        set_state("import_collection", "awaiting_owner_collection_judgments", detail=v["status"])
        return {**v, "status": "awaiting_owner_collection_judgments" if v["status"] in ("awaiting", "incomplete")
                else v["status"], "expected": str(COLL_DATA / "judgments_<n>.tsv")}
    seen = v["seen"]
    key = json.loads((COLL_DATA / "origin_key.json").read_text())
    ds = datasets_from(man, items, seen)
    SETS.mkdir(parents=True, exist_ok=True)
    for split, rs in ds.items():
        (SETS / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs),
                                             encoding="utf-8")
    by_origin = defaultdict(Counter)
    for (i, lb), j in seen.items():
        for o in key[i]["origins"][lb]:
            by_origin[o][j] += 1
    summ = {s: {"families": len(rs), "label_status": dict(Counter(r["label_status"] for r in rs)),
                "accepted_spellings": sum(len(r["accepted"]) for r in rs),
                "families_with_two_accepted": sum(len(r["accepted"]) == 2 for r in rs),
                "by_category": {c: {"families": sum(r["category"] == c for r in rs),
                                    "accepted": sum(bool(r["accepted"]) and r["category"] == c for r in rs)}
                                for c in CATEGORIES},
                "irregular_by_pattern": {p: {"families": sum(r["pattern"] == p and r["category"] == "irregular" for r in rs),
                                             "accepted": sum(bool(r["accepted"]) and r["pattern"] == p
                                                             and r["category"] == "irregular" for r in rs)}
                                         for p, _ in PATTERNS}}
            for s, rs in ds.items()}
    out = {"imported": e06.now(), "provenance": v["provenance"], "judgment_files_sha256": v["judgment_files_sha256"],
           "judge": "the owner: a single, project-affiliated judge (not independent)",
           "judgments": dict(Counter(seen.values())), "by_origin": {o: dict(c) for o, c in by_origin.items()},
           "datasets": summ, "datasets_sha256": {p.name: e06.sha(p) for p in sorted(SETS.glob("*.jsonl"))},
           "rules": ["accepted spellings are targets (all alternatives kept)", "rejected spellings never become "
                     "targets", "uncertain spellings are quarantined", "a family with no accepted spelling stays "
                     "unlabelled (no invented target)", "no mined spelling is deleted because a candidate was rejected",
                     "train and dev families stay separate"],
           "feasibility": feasibility(ds)}
    set_state("import_collection", "done" if out["feasibility"]["pass"] else "below_feasibility",
              **out["feasibility"]["got"])
    return e07.write_json(REPORTS / "collection.json", out) and out


# ------------------------------------------------------------------ 5. recipe, sampling and registration
SELECT_RULE = ("lowest mean CER of the greedy output against the accepted spellings of the reviewed development "
               "collection (families with ≥ 1 accepted spelling), then highest exact agreement with any accepted "
               "spelling, then the earlier update; eligible: the registered checkpoints and, if the run stops early, "
               "the final state")


def recipe() -> dict:
    """E11's adapter architecture and dropout with the E13 optimiser, schedule, sampling and input format."""
    base = e11.recipe()
    return {"adapter": base["adapter"], "base_dropout": base["base_dropout"], "optimizer": "AdamW", "lr": 3e-4,
            "betas": base["betas"], "weight_decay": base["weight_decay"], "grad_clip": base["grad_clip"],
            "batch": 16, "grad_accumulation": 2, "warmup": 100, "schedule": "linear", "max_updates": 2000,
            "max_hours": 6.0, "eval_at": list(EVAL_AT), "seed": 0,
            "init": {"adapter": "models/e11/r_s0/best.pt", "sha256": E11_BEST_SHA,
                     "note": "task adaptation from E11's verified adapter; fresh optimiser and scheduler"},
            "input": "<word>|<first CMUdict reading | <no_pron>>, lower-cased UTF-8 bytes + EOS",
            "amendments": AMENDMENTS,
            "limits": {"src_max_bytes": SRC_MAX_BYTES, "tgt_max_bytes": TGT_MAX_BYTES, "over_limit": "refused"},
            "sampling": {**MIX, "R": "R's convention: successive seeded permutations of all optimisation rows "
                                     "(seed:order:k), consumed in order",
                         "reviewed": "family uniform, then an accepted alternative uniform",
                         "block": "each block of 100 updates has exactly 2,880 R and 320 reviewed draws in seeded "
                                  "positions, and exactly 320 masked draws in independent seeded positions",
                         "mask": "pronunciation replaced by <no_pron>; the target is unchanged"},
            "select": SELECT_RULE, "decode": {"selection": "greedy", "max_new_tokens": e09.COMMON["max_new_tokens"]}}


def row_sources(r_rows: list[tuple[str, str]], reviewed: list[dict]) -> tuple[list[tuple[str, str]], dict]:
    """Training rows as (source, target): R full | R masked | reviewed full | reviewed masked, plus the layout."""
    rp = {w: pron(w)["phones"] for w in {w for w, _ in r_rows}}
    flat = [(f["word"], f["pronunciation"], t, n) for n, f in enumerate(reviewed) for t in f["accepted"]]
    for _, _, t, _ in flat:
        check_target(t)
    for _, t in r_rows:
        check_target(t)
    rows = ([(source(w, rp[w]), t) for w, t in r_rows] + [(source(w, None), t) for w, t in r_rows]
            + [(source(w, p), t) for w, p, t, _ in flat] + [(source(w, None), t) for w, _, t, _ in flat])
    alts = defaultdict(list)
    for k, (_, _, _, n) in enumerate(flat):
        alts[n].append(k)
    return rows, {"nR": len(r_rows), "nA": len(flat), "alts": [alts[n] for n in range(len(reviewed))],
                  "r_no_pron": sum(rp[w] is None for w, _ in r_rows)}


def make_order(lay: dict, seed: int = 0):
    """`order(block) -> [row index]` for `e07.fit`: one block = 100 updates of 32 draws (see `recipe()`)."""
    n_draws = MIX["block_updates"] * MIX["batch"]
    n_rev, n_mask = round(n_draws * MIX["reviewed_share"]), round(n_draws * MIX["mask_share"])
    n_r = n_draws - n_rev
    nR, nA, alts = lay["nR"], lay["nA"], lay["alts"]

    @lru_cache(maxsize=4)
    def perm(k: int) -> list[int]:
        return random.Random(f"{seed}:order:{k}").sample(range(nR), nR)

    def order(ep: int) -> list[int]:
        rng = random.Random(f"e13:{seed}:block:{ep}")
        slots = ["R"] * n_r + ["V"] * n_rev
        rng.shuffle(slots)
        masked = set(rng.sample(range(n_draws), n_mask))
        out, c = [], ep * n_r
        for pos, s in enumerate(slots):
            m = pos in masked
            if s == "R":
                i = perm(c // nR)[c % nR]
                c += 1
                out.append(i + (nR if m else 0))
            else:
                a = alts[rng.randrange(len(alts))]
                out.append(2 * nR + a[rng.randrange(len(a))] + (nA if m else 0))
        return out
    return order


def exposure(cfg: dict, lay: dict, reviewed: list[dict]) -> dict:
    """Actual draw counts of the registered order over the whole run (by source, family, mask)."""
    order = make_order(lay, cfg["seed"])
    per = MIX["block_updates"] * MIX["batch"]
    eff = cfg["batch"] * cfg["grad_accumulation"]
    if eff != MIX["batch"]:
        raise ValueError(f"effective batch {eff} differs from the sampling block's {MIX['batch']}")
    blocks = -(-cfg["max_updates"] * eff // per)
    draws = [i for b in range(blocks) for i in order(b)][: cfg["max_updates"] * eff]
    nR, nA = lay["nR"], lay["nA"]
    src = Counter("R" if i < 2 * nR else "reviewed" for i in draws)
    masked = sum((nR <= i < 2 * nR) or i >= 2 * nR + nA for i in draws)
    fam_of = {k: n for n, ks in enumerate(lay["alts"]) for k in ks}
    fam = Counter(fam_of[(i - 2 * nR) % nA] for i in draws if i >= 2 * nR)
    rr = Counter((i % nR) for i in draws if i < 2 * nR)
    fc = [fam.get(n, 0) for n in range(len(reviewed))]
    return {"draws_total": len(draws), "by_source": dict(src), "share_reviewed": round(src["reviewed"] / len(draws), 4),
            "masked": masked, "share_masked": round(masked / len(draws), 4),
            "R_epochs": round(src["R"] / nR, 2), "R_rows_seen": len(rr),
            "reviewed_families": len(reviewed), "reviewed_alternatives": nA,
            "draws_per_reviewed_family": {"min": min(fc), "median": sorted(fc)[len(fc) // 2], "max": max(fc)},
            "R_rows_without_cmudict": lay["r_no_pron"]}


def lr_plan(cfg: dict) -> dict:
    """Learning rate applied at named updates (1-based).

    Examples:
        >>> p = lr_plan({"lr": 3e-4, "warmup": 100, "schedule": "linear", "max_updates": 2000})
        >>> p[1], p[100], p[2000]
        (3e-06, 0.0003, 1.57895e-07)
    """
    return {u: round(cfg["lr"] * e07.lr_factor(u - 1, cfg), 12) for u in (1, 100, 101, 400, 800, 1200, 1600, 2000)}


TEST_ANALYSIS = {
    "primary": "paired difference E13 − E11 in owner-confirmed top-1 acceptability on the 120 test families: share "
               "of items whose greedy output is judged `acceptable` (denominator all 120; `uncertain` and "
               "`not_acceptable` count as not confirmed)",
    "per_item_judgment": "each model takes the judgment of the candidate equal to its greedy output; identical outputs "
                         "are shown once and share one judgment; an empty output counts as not_acceptable",
    "report": ["acceptable / not_acceptable / uncertain counts per model", "confirmed-acceptable share (all items)",
               "uncertainty rates", "paired table both / E13 only / E11 only / neither",
               "family-level paired bootstrap 95% interval (10,000, seed 20261027; one item per family)",
               "two-sided exact McNemar on clear discordant pairs (pairs with an uncertain are excluded)"],
    "sensitivity": ["uncertain counted as acceptable (n = 120)", "items with any uncertain excluded"],
    "secondary": "the same analysis per declared group: ordinary, irregular spelling, names (40 each); reported with "
                 "their intervals; a gain in one group with losses elsewhere is shown, never hidden",
    "decision": {"promising_improvement": "primary bootstrap 95% interval lower bound > 0",
                 "regression": "primary bootstrap 95% interval upper bound < 0",
                 "uncertain_change": "anything else",
                 "flags": "any group whose point difference is ≤ −10 points is flagged as a trade-off; missing-"
                          "pronunciation behaviour (mechanism report) is reported beside the decision"},
    "completeness": "scored only when every candidate has an explicit judgment; blanks are missing, never acceptance",
    "interpretation": "a non-significant difference is not equivalence; subgroup differences are not confirmed "
                      "findings; one non-independent judge; results hold for this declared sample and policy; the "
                      "combined result does not identify which component caused it",
    "isolation": "no test judgment enters training, recipe or checkpoint choice",
}


@timed("register")
def register() -> dict:
    """Freeze recipe, data and code hashes, sampling, exposure, LR plan, limits, selection and the test analysis."""
    if not json.loads((REPORTS / "start_checks.json").read_text())["ok"]:
        raise RuntimeError("starting-state checks did not pass")
    col = json.loads((REPORTS / "collection.json").read_text())
    if not col["feasibility"]["pass"]:
        raise RuntimeError(f"collection below the feasibility minimum: {col['feasibility']}")
    cfg = recipe()
    r_rows, _ = e11.arm_rows()
    train_ds = [r for r in dataset("train") if r["accepted"]]
    dev_ds = [r for r in dataset("dev") if r["accepted"]]
    rows, lay = row_sources(r_rows, train_ds)
    files = ["scripts/e13.py", "scripts/e12.py", "scripts/e11.py", "scripts/e07.py", "scripts/e06_byt5.py",
             "scripts/e06.py", "scripts/e09.py", "scripts/e10.py", "scripts/e12_review.py",
             "src/khmer_translit/phonetic_source.py", "experiments/E13-pronunciation-guided/PLAN.md",
             "research/requirements-byt5.txt", "research/requirements-byt5.lock.txt", "docs/PHONETIC_POLICY.md"]
    reg = {"registered": e06.now(), "experiment": EXPERIMENT, "run": RUN, "recipe": cfg, "cfg_hash": e07.cfg_hash(cfg),
           "data": {"arm_R": {"sha256": e11.ARM_SHA, "optimisation_rows": len(r_rows),
                              "optimisation_hash": e07.rows_hash(r_rows)},
                    "reviewed_train": {"families": len(train_ds), "alternatives": lay["nA"],
                                       "sha256": col["datasets_sha256"]["train.jsonl"]},
                    "reviewed_dev": {"families_with_accepted": len(dev_ds), "sha256": col["datasets_sha256"]["dev.jsonl"]},
                    "training_rows_hash": e07.rows_hash(rows), "training_rows": len(rows),
                    "cmudict_sha256": ps.cmudict()[1]},
           "exposure": exposure(cfg, lay, train_ds), "lr_plan": lr_plan(cfg), "limits": LIMITS,
           "guard": "resumable stop after 3 consecutive readings with RSS > 6.5 GiB or MemAvailable < 0.4 GiB",
           "checkpoints": list(EVAL_AT), "selection": SELECT_RULE,
           "selection_note": "the accepted references come from reviewed E11 candidates and are incomplete; CER and "
                             "exact agreement are surrogates; a novel spelling is unjudged, not wrong",
           "monitor": "historical dev (mined references, pronunciation input) at each checkpoint; never selects",
           "test": {"manifest_sha256": json.loads((TEST / "frozen.json").read_text())["sha256"],
                    "analysis": TEST_ANALYSIS},
           "amendments": AMENDMENTS,
           "hypothesis": "a bounded combined hypothesis; the learning rate, mix and masking values are not claimed "
                         "optimal; no ablation in this experiment",
           "source_sha256": {f: e06.sha(ROOT / f) for f in files}, "base_sha256": B.BASE_SHA256,
           "base_revision": B.REVISION}
    set_state("register", "done", cfg_hash=reg["cfg_hash"])
    return e07.write_json(EXP / "registration.json", reg) and reg


def registered_rows() -> tuple[dict, list, dict, list, list]:
    """(cfg, rows, layout, train families, dev families) checked against the registration."""
    reg = json.loads((EXP / "registration.json").read_text())
    cfg = recipe()
    r_rows, _ = e11.arm_rows()
    train_ds = [r for r in dataset("train") if r["accepted"]]
    dev_ds = [r for r in dataset("dev") if r["accepted"]]
    rows, lay = row_sources(r_rows, train_ds)
    if e07.cfg_hash(cfg) != reg["cfg_hash"] or e07.rows_hash(rows) != reg["data"]["training_rows_hash"]:
        raise RuntimeError("recipe or training rows differ from the registration")
    return cfg, rows, lay, train_ds, dev_ds


# ------------------------------------------------------------------ 6. model, evaluation, pipeline checks
def e11_init(adapter: dict, base_dropout: float, seed: int):
    """The pinned base with E11's verified adapter loaded strictly (the registered initial state)."""
    if e06.sha(E11_DIR / "best.pt") != E11_BEST_SHA:
        raise RuntimeError("E11 best.pt changed")
    m = e07.build(adapter, base_dropout, seed)
    e07.load_adapter(m, E11_DIR / "best.pt")
    return m


def gen_sources(model, srcs: list[str], bs: int = 32) -> list[tuple[str | None, str]]:
    """Greedy (text, status) for already formatted sources."""
    return [(r[0][0], r[0][1]) for r in e07.gen(model, [B.encode_src(s) for s in srcs], bs=bs)]


def ref_scores(outs: list[str | None], accepted: list[list[str]]) -> dict:
    """Mean CER against the accepted spellings and exact agreement with any of them.

    Examples:
        >>> ref_scores(["ក", "ខគ"], [["ក"], ["ខ"]])
        {'cer': 0.5, 'exact': 0.5, 'n': 2}
    """
    cer = [e07.cer_of(o, a) for o, a in zip(outs, accepted)]
    ex = [int(any(e07.nfc_eq(o, x) for x in a)) for o, a in zip(outs, accepted)]
    return {"cer": round(mean(cer), 4), "exact": round(mean(ex), 4), "n": len(outs)}


def selection_key(r: dict) -> tuple:
    """Lower CER first, then higher exact (`e07.fit` keeps the earlier update on ties).

    Examples:
        >>> selection_key({"cer": 0.2, "exact": 0.4}) > selection_key({"cer": 0.25, "exact": 0.6})
        True
    """
    return (-r["cer"], r["exact"])


def hist_dev_sources() -> tuple[list, list[str]]:
    items = e06.primary(e06.items("dev"))
    return items, [source(i.word, pron(i.word)["phones"]) for i in items]


def make_evaluate(dev_ds: list[dict]):
    """Checkpoint evaluation: the registered key on the reviewed dev collection + historical dev monitoring."""
    srcs = [source(r["word"], r["pronunciation"]) for r in dev_ds]
    acc = [r["accepted"] for r in dev_ds]
    items, hsrc = hist_dev_sources()

    def evaluate(m):
        r = ref_scores([t for t, _ in gen_sources(m, srcs)], acc)
        h = [t for t, _ in gen_sources(m, hsrc, bs=64)]
        hit = {i.id: int(e07.nfc_eq(o, i.preferred) or any(e07.nfc_eq(o, a) for a in i.accepted))
               for i, o in zip(items, h)}
        return {"key": selection_key(r), "coll_dev_cer": r["cer"], "coll_dev_exact": r["exact"],
                "hist_dev_top1": round(e06.macro(items, hit), 4),
                "hist_dev_cer": round(mean(e07.cer_of(o, i.accepted) for i, o in zip(items, h)), 4)}
    return evaluate


@timed("pipeline")
def pipeline() -> dict:
    """Input / target / masking / limit checks, strict adapter loading, timing and a smoke update (separate)."""
    import torch
    cfg, rows, lay, train_ds, dev_ds = registered_rows()
    chk = {}
    ex = rows[0][0]
    chk["format"] = {"pass": ex.count("|") == 1 and ex.split("|")[1] in (NO_PRON, pron(ex.split("|")[0])["phones"]),
                     "example": ex,
                     "yacht": source("yacht", pron("yacht")["phones"]), "missing": source("zzqxv", pron("zzqxv")["phones"])}
    chk["lookup"] = {"pass": pron("yacht")["phones"] == "Y AA1 T" and pron("zzqxv")["phones"] is None,
                     "R_without_cmudict": lay["r_no_pron"], "reviewed_reading_matches_lookup":
                         all(pron(r["word"])["phones"] == r["pronunciation"] for r in train_ds + dev_ds)}
    chk["lookup"]["pass"] &= chk["lookup"]["reviewed_reading_matches_lookup"]
    nR, nA = lay["nR"], lay["nA"]
    chk["masking"] = {"pass": all(rows[i][1] == rows[i + nR][1] and rows[i + nR][0].endswith(NO_PRON) for i in range(nR))
                      and all(rows[2 * nR + k][1] == rows[2 * nR + nA + k][1] for k in range(nA)),
                      "rule": "masked rows differ only in the pronunciation field; targets identical"}
    sl = [len(B.encode_src(s)) - 1 for s, _ in rows]
    tl = [len(B.encode_tgt(t)) - 1 for _, t in rows]
    chk["limits"] = {"pass": max(sl) <= SRC_MAX_BYTES and max(tl) <= TGT_MAX_BYTES, "src_max_seen": max(sl),
                     "src_mean": round(mean(sl), 1), "tgt_max_seen": max(tl), "src_limit": SRC_MAX_BYTES,
                     "tgt_limit": TGT_MAX_BYTES, "truncation": "none: batch_tensors pads only; over-limit sources raise"}
    try:
        source("a" * 200, "AH0")
        chk["limits"]["pass"] = False
    except ValueError:
        chk["limits"]["overlength_refused"] = True
    ids, mask, labels = B.batch_tensors(rows[:4])
    chk["target_eos"] = {"pass": all(int(labels[k][int((labels[k] != -100).sum()) - 1]) == B.EOS for k in range(4))
                         and all(int(ids[k][int(mask[k].sum()) - 1]) == B.EOS for k in range(4))}
    m = e11_init(cfg["adapter"], cfg["base_dropout"], cfg["seed"])
    words = e09.dev_rows()[:16]
    cached = {r["id"]: e09.top1(r["cands"]) for r in e07.load_jsonl(
        e09.cache_path("e11", "greedy", "dev", E11_BEST_SHA, e09.dev_rows()))}
    got = gen_sources(m, [w for _, w in words])
    chk["adapter_loading"] = {"pass": all(e06.nfc(t or "") == e06.nfc(cached[i] or "") for (i, _), (t, _) in zip(words, got)),
                              "check": "E11 adapter on the base reproduces E11's cached greedy dev outputs (word input)"}
    order = make_order(lay, cfg["seed"])
    blk = order(0)
    nrev = sum(i >= 2 * nR for i in blk)
    nmask = sum((nR <= i < 2 * nR) or i >= 2 * nR + nA for i in blk)
    chk["sampling"] = {"pass": len(blk) == 3200 and nrev == 320 and nmask == 320 and order(0) == blk,
                       "block_draws": len(blk), "reviewed": nrev, "masked": nmask, "deterministic": order(0) == blk}
    # smoke: a few real updates on training rows only, on a separate copy; the registered run starts again from E11
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    before = [p.detach().clone() for p in params[:4]]
    m.train()
    times, grads = [], []
    guard = e07.Guard(limits=LIMITS)
    longest = sorted(range(len(rows)), key=lambda i: -len(rows[i][0]))[:32]
    mb, acc = cfg["batch"], cfg["grad_accumulation"]
    for k in range(SMOKE_UPDATES):
        b = [rows[i] for i in (longest if k == 0 else blk[32 * k:32 * (k + 1)])]
        t = time.perf_counter()
        for j in range(acc):                         # as e07.fit: equal micro-batches, mean of their losses
            ids, mask, labels = B.batch_tensors(b[j * mb:(j + 1) * mb])
            loss = m(input_ids=ids, attention_mask=mask, labels=labels).loss / acc
            loss.backward()
        grads.append(round(float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])), 4))
        opt.step()
        opt.zero_grad()
        times.append(round(time.perf_counter() - t, 2))
        guard.check()
    changed = sum(not torch.equal(a, p.detach()) for a, p in zip(before, params[:4]))
    smoke_out = gen_sources(m, [s for s, _ in rows[2 * nR: 2 * nR + 8]])
    SMOKE.mkdir(parents=True, exist_ok=True)
    torch.save(B.lora_state(m), SMOKE / "smoke_adapter.pt")
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024     # true process peak (Linux: KiB)
    chk["smoke"] = {"pass": all(g > 0 for g in grads) and changed == 4 and all(s == "ok" for _, s in smoke_out)
                    and peak < LIMITS["rss_max_bytes"], "process_peak_rss_mb": round(peak / 2 ** 20),
                    "updates": SMOKE_UPDATES, "grad_norms": grads, "params_changed_of_4": changed,
                    "seconds_per_update": times, "longest_batch_first": True,
                    "peak_rss_mb": round(guard.peak_rss / 2 ** 20), "generation_status": [s for _, s in smoke_out],
                    "saved": str((SMOKE / "smoke_adapter.pt").relative_to(ROOT)),
                    "e11_unchanged": e06.sha(E11_DIR / "best.pt") == E11_BEST_SHA,
                    "estimated_training_hours": round(mean(times[1:]) * cfg["max_updates"] / 3600, 2),
                    "micro_batches": f"{acc} × {mb}"}
    chk["smoke"]["pass"] &= chk["smoke"]["e11_unchanged"]
    out = {"checked": e06.now(), "checks": chk, "ok": all(c["pass"] for c in chk.values()),
           "note": "a falling loss is not taken as evidence that the pronunciation is used (see mechanism)"}
    set_state("pipeline", "done" if out["ok"] else "failed")
    return e07.write_json(REPORTS / "pipeline_checks.json", out) and out


# ------------------------------------------------------------------ 7. training
def train() -> dict:
    """The one authorised run; resumable (`e12.check_resume` before the model is built; hashes again in `e07.fit`)."""
    if not json.loads((REPORTS / "pipeline_checks.json").read_text())["ok"]:
        raise RuntimeError("pipeline checks did not pass")
    cfg, rows, lay, _, dev_ds = registered_rows()
    frozen_rows(TEST)
    ident = {"cfg": e07.cfg_hash(cfg), "data": e07.rows_hash(rows), "base": B.BASE_SHA256}
    mode = e12.check_resume(OUT, ident, e07.budget_left("full")["started"], RUN)
    set_state("train", "running", mode=mode)
    done = e07.fit(RUN, "full", cfg, rows, make_evaluate(dev_ds), set(EVAL_AT), OUT,
                   guard=e07.Guard(limits=LIMITS), make=e11_init, order=make_order(lay, cfg["seed"]))
    set_state("train", "done" if "best" in done else "stopped", stop=done.get("stop"), best=done.get("best"))
    return done


# ------------------------------------------------------------------ 8. freeze, evaluation, mechanism
def src_rows(rows: list[tuple[str, str]], phones=lambda w: pron(w)["phones"]) -> list[tuple[str, str]]:
    return [(i, source(w, phones(w))) for i, w in rows]


def gen_cached(model: str, split: str, rows: list[tuple[str, str]], m=None):
    """Cached greedy generation for (id, input) rows; returns (records, model or None)."""
    sha = e09.checkpoint_sha(model)
    if not e09.cache_path(model, "greedy", split, sha, rows).exists():
        m = m or e09.load(model)[0]
        e09.generate(m, model, "greedy", split, rows, sha)
    return e07.load_jsonl(e09.cache_path(model, "greedy", split, sha, rows)), m


def dev_coll_rows() -> list[tuple[str, str]]:
    return [(r["id"], r["word"]) for r in dataset("dev")]


def pred_inputs(model: str) -> dict[str, list[tuple[str, str]]]:
    """Inputs per split for a model: word only for E11, word + reading for E13."""
    words = {"colldev": dev_coll_rows(), "dev": e09.dev_rows(), "eval": e09.eval_rows()}
    if model == "e11":
        return words
    phones = {r["id"]: r["pronunciation"] for r in dataset("dev")}
    return {"colldev": [(i, source(w, phones[i])) for i, w in words["colldev"]],
            "dev": src_rows(words["dev"]), "eval": src_rows(words["eval"])}


def preds(model: str, split: str) -> list[dict]:
    rs, m = gen_cached(model, f"e13{split}" if split == "colldev" else split, pred_inputs(model)[split])
    if m is not None:
        e09.unload(m)
    return rs


@timed("freeze")
def freeze() -> dict:
    """Selected checkpoint and greedy predictions (reviewed dev collection, historical dev/eval), hashed."""
    done = json.loads((OUT / "done.json").read_text())
    eligible = set(EVAL_AT) | ({done["updates"]} if done["stop"] != "max_updates" else set())
    if done["best"]["update"] not in eligible:
        raise RuntimeError("selected update is not an eligible checkpoint")
    PRED.mkdir(parents=True, exist_ok=True)
    files = {}
    for c in COMPARED:
        for split in ("colldev", "dev", "eval"):
            rs = preds(c, split)
            words = dict(dev_coll_rows() if split == "colldev" else e09.dev_rows() if split == "dev" else e09.eval_rows())
            p = PRED / f"{c}_{split}_greedy.jsonl"
            p.write_text("".join(json.dumps({"id": r["id"], "word": words[r["id"]], "input": r["word"],
                                             "prediction": e09.top1(r["cands"]),
                                             "status": r["cands"][0]["status"] if r["cands"] else "none"},
                                            ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
            files[p.name] = e06.sha(p)
    out = {"frozen": e06.now(), "rule": SELECT_RULE,
           "selected": {"run": RUN, "update": done["best"]["update"], "coll_dev_cer": done["best"]["coll_dev_cer"],
                        "coll_dev_exact": done["best"]["coll_dev_exact"], "checkpoint_sha256": e09.checkpoint_sha("e13")},
           "candidates": [{k: h.get(k) for k in ("update", "coll_dev_cer", "coll_dev_exact", "hist_dev_top1",
                                                  "hist_dev_cer")} for h in done["history"]],
           "stop": done["stop"], "updates": done["updates"],
           "baseline": {"e11": {"checkpoint": "models/e11/r_s0/best.pt", "sha256": e09.checkpoint_sha("e11")}},
           "decoding": {"setting": "greedy", **e09.setting("greedy"), **e09.COMMON, "hash": e09.setting_hash("greedy")},
           "inputs": {"e13": "word + first CMUdict reading (<no_pron> if none)", "e11": "word only"},
           "prediction_files_sha256": files}
    set_state("freeze", "done", update=out["selected"]["update"])
    return e07.write_json(REPORTS / "frozen.json", out) and out


def frozen_preds(model: str, split: str) -> dict[str, str | None]:
    fz = json.loads((REPORTS / "frozen.json").read_text())
    p = PRED / f"{model}_{split}_greedy.jsonl"
    if e06.sha(p) != fz["prediction_files_sha256"][p.name]:
        raise RuntimeError(f"{p.name} changed")
    return {r["id"]: r["prediction"] for r in e11.jl(p)}


def coll_judged() -> dict[str, dict[str, str]]:
    """Owner judgments of the collection, by item id and NFC text (all candidates, any judgment)."""
    items = json.loads((COLL_DATA / "items.json").read_text())["items"]
    seen, _ = read_judgments(COLL_DATA, items)
    out = defaultdict(dict)
    for it in items:
        for c in it["candidates"]:
            if (it["item_id"], c["label"]) in seen:
                out[it["item_id"]][e06.nfc(c["text"])] = seen[(it["item_id"], c["label"])]
    return out


def training_curve() -> dict:
    hist = json.loads((OUT / "done.json").read_text())["history"]
    lrs, loss = {}, defaultdict(list)
    for ln in (OUT / "metrics.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "lr" in r:
            loss[(r["update"] - 1) // 100].append(r["loss"])
            if r["update"] % 100 == 0 or r["update"] in (1, 101):
                lrs[r["update"]] = r["lr"]
    segs = [e for e in e07.ledger() if e.get("event") == "segment" and e.get("run") == RUN]
    return {"checkpoints": hist, "lr_applied_at_update": lrs,
            "loss_mean_per_100": {(k + 1) * 100: round(mean(v), 4) for k, v in sorted(loss.items())},
            "segments": segs, "active_training_hours": round(sum(s["hours"] for s in segs), 4)}


@timed("evaluate")
def evaluate() -> dict:
    """Reviewed dev-collection scores and coverage, historical diagnostics (exploratory), curves."""
    fz = json.loads((REPORTS / "frozen.json").read_text())
    out = {"label": "Exploratory. The reviewed dev references are incomplete (accepted E11 candidates); historical "
                    "references are unreviewed mined spellings. No number here measures phonetic accuracy.",
           "selected": fz["selected"], "candidates": fz["candidates"]}
    dev_ds = dataset("dev")
    labelled = [r for r in dev_ds if r["accepted"]]
    jd = coll_judged()
    g = {c: frozen_preds(c, "colldev") for c in COMPARED}
    out["collection_dev"] = {
        "families": len(dev_ds), "families_with_accepted": len(labelled),
        "scores": {c: ref_scores([g[c][r["id"]] for r in labelled], [r["accepted"] for r in labelled]) for c in COMPARED},
        "note": "E11's greedy output was itself a reviewed candidate, so E11's exact score here counts only how often "
                "its own greedy was accepted; E13 outputs can be novel (unjudged)",
        "judgment_coverage": {c: e12.judgment_coverage(g[c], jd) for c in COMPARED},
        "by_category": {cat: {c: ref_scores([g[c][r["id"]] for r in labelled if r["category"] == cat],
                                            [r["accepted"] for r in labelled if r["category"] == cat])
                              for c in COMPARED} for cat in CATEGORIES
                        if any(r["category"] == cat for r in labelled)},
        "changed_from_e11": sum(e06.nfc(g["e13"][r["id"]] or "") != e06.nfc(g["e11"][r["id"]] or "") for r in dev_ds)}
    for split in ("dev", "eval"):
        items = e06.primary(e06.items(split))
        tab, per = {}, {}
        for c in COMPARED:
            rs = preds(c, split)
            tab[c], per[c] = e09.setting_metrics(items, rs, None)
        flips = Counter(("both_right" if per["e13"][i.id]["top1"] and per["e11"][i.id]["top1"] else
                         "only_e13" if per["e13"][i.id]["top1"] else "only_e11" if per["e11"][i.id]["top1"]
                         else "both_wrong") for i in items)
        gs = {c: {r["id"]: e09.top1(r["cands"]) for r in preds(c, split)} for c in COMPARED}
        out[split] = {"items": len(items), "table": tab,
                      "e13_minus_e11": e10.seed_contrast(items, per, ["e13"], ["e11"]),
                      "error_kinds": {c: dict(Counter(e10.error_kind(gs[c][i.id], i.accepted) for i in items
                                                      if not per[c][i.id]["top1"])) for c in COMPARED},
                      "missing_ending": {c: sum(e11.missing_ending(gs[c][i.id], i.accepted) for i in items
                                                if not per[c][i.id]["top1"]) for c in COMPARED},
                      "operational": {c: e10.operational(preds(c, split)) for c in COMPARED},
                      "item_outcomes": dict(flips),
                      "cmudict_coverage": round(mean(pron(i.word)["phones"] is not None for i in items), 4)}
    done = json.loads((OUT / "done.json").read_text())
    out["training"] = {**training_curve(), "selected_update": done["best"]["update"], "updates": done["updates"],
                       "stop": done["stop"], "peak_rss_mb": done["peak_rss_mb"],
                       "min_available_mb": done["min_available_mb"]}
    set_state("evaluate", "done")
    return e07.write_json(EXP / "results.json", out) and out


def follows(out: str | None, own: str | None, donor: str | None) -> bool:
    """Whether an output under a mismatched reading is closer (CER) to the donor word's own output than to this
    word's correct-reading output.

    Examples:
        >>> follows("ខា", "ក", "ខា"), follows("ក", "ក", "ខា")
        (True, False)
    """
    if not out or not own or not donor:
        return False
    return e07.cer_of(out, [donor]) < e07.cer_of(out, [own])


@timed("mechanism")
def mechanism() -> dict:
    """Dev items only, letters fixed: correct vs missing vs a frozen mismatched (deranged) pronunciation."""
    m = e09.load("e13")[0]
    out = {"label": "diagnostic only: sensitivity is not correctness; mismatched readings are not scored as tasks"}
    sets = {"collection_dev": [(r["id"], r["word"], r["pronunciation"], r["accepted"]) for r in dataset("dev")],
            "historical_dev": [(i.id, i.word, pron(i.word)["phones"], i.accepted)
                               for i in e06.primary(e06.items("dev"))]}
    for name, its in sets.items():
        with_p = [x for x in its if x[2]]
        der = e07.derangement(len(with_p), f"{MASK_SEED}:{name}")
        conds = {"correct": [source(w, p) for _, w, p, _ in with_p],
                 "missing": [source(w, None) for _, w, _, _ in with_p],
                 "mismatched": [source(w, with_p[der[k]][2]) for k, (_, w, _, _) in enumerate(with_p)]}
        o = {c: [t for t, _ in gen_sources(m, s, bs=64)] for c, s in conds.items()}
        acc = [a for *_, a in with_p]
        lab = [k for k, a in enumerate(acc) if a]
        changed = {c: round(mean(e06.nfc(a or "") != e06.nfc(b or "") for a, b in zip(o[c], o["correct"])), 4)
                   for c in ("missing", "mismatched")}
        fol = [follows(o["mismatched"][k], o["correct"][k], o["correct"][der[k]]) for k in range(len(with_p))]
        ex = [{"word": with_p[k][1], "reading": with_p[k][2], "correct": o["correct"][k],
               "missing": o["missing"][k], "mismatched_reading": with_p[der[k]][2], "mismatched": o["mismatched"][k],
               "donor_word": with_p[der[k]][1], "donor_output": o["correct"][der[k]]}
              for k in sorted(range(len(with_p)), key=lambda k: e06.order_key(f"{MASK_SEED}:ex", with_p[k][1]))[:8]]
        out[name] = {"items": len(its), "with_cmudict": len(with_p),
                     "cmudict_coverage": round(len(with_p) / len(its), 4),
                     "output_changed_vs_correct": changed,
                     "mismatched_closer_to_donor_output": round(mean(fol), 4),
                     "reference_scores": {c: ref_scores([o[c][k] for k in lab], [acc[k] for k in lab])
                                          for c in ("correct", "missing")},
                     "examples": ex}
    e09.unload(m)
    _, rows = e11.arm_rows()
    tests = frozen_rows(TEST)
    out["dictionary_coverage"] = {"R_words": round(mean(pron(w)["phones"] is not None for w in {r["word"] for r in rows}), 4),
                                  "historical_dev": out["historical_dev"]["cmudict_coverage"],
                                  "test": round(mean(pron(r["word"])["phones"] == r["pronunciation"] for r in tests), 4)}
    set_state("mechanism", "done")
    return e07.write_json(REPORTS / "mechanism.json", out) and out


# ------------------------------------------------------------------ 9. prospective blinded test
def test_inputs(model: str) -> list[tuple[str, str]]:
    """Test inputs through each model's normal path: E11 word only; E13 word + its CMUdict lookup (asserted to be
    the frozen intended reading)."""
    man = frozen_rows(TEST)
    if model == "e11":
        return [(r["id"], r["word"]) for r in man]
    if any(pron(r["word"])["phones"] != r["pronunciation"] for r in man):
        raise RuntimeError("lookup differs from the frozen intended reading")
    return [(r["id"], source(r["word"], pron(r["word"])["phones"])) for r in man]


@timed("testgen")
def testgen() -> dict:
    """Greedy outputs of the frozen E13 selection and E11 on the test inputs (after selection is frozen)."""
    fz = json.loads((REPORTS / "frozen.json").read_text())
    if e09.checkpoint_sha("e13") != fz["selected"]["checkpoint_sha256"]:
        raise RuntimeError("E13 checkpoint differs from the frozen selection")
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    files, ops = {}, {}
    for c in COMPARED:
        rows = test_inputs(c)
        rs, m = gen_cached(c, "e13test", rows)
        if m is not None:
            e09.unload(m)
        e12.aligned(rows, rs)
        p = TEST_DATA / f"predictions_{c}.jsonl"
        p.write_text("".join(json.dumps({"id": r["id"], "input": r["word"], "prediction": e09.top1(r["cands"]),
                                         "status": r["cands"][0]["status"] if r["cands"] else "none"},
                                        ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        files[p.name] = e06.sha(p)
        ops[c] = e10.operational(rs)
    out = {"generated": e06.now(), "manifest_sha256": json.loads((TEST / "frozen.json").read_text())["sha256"],
           "checkpoints": {c: e09.checkpoint_sha(c) for c in COMPARED}, "decoding": fz["decoding"],
           "prediction_files_sha256": files, "operational": ops, "local": str(TEST_DATA.relative_to(ROOT))}
    set_state("testgen", "done")
    return e07.write_json(TEST / "predictions.json", out) and out


def test_preds() -> dict[str, dict[str, str | None]]:
    f = json.loads((TEST / "predictions.json").read_text())
    out = {}
    for c in COMPARED:
        p = TEST_DATA / f"predictions_{c}.jsonl"
        if e06.sha(p) != f["prediction_files_sha256"][p.name]:
            raise RuntimeError(f"{p.name} differs from its frozen hash")
        out[c] = {r["id"]: r["prediction"] for r in e11.jl(p)}
    return out


@timed("batches")
def batches() -> dict:
    """Blinded test batches (≤ 150 spellings each) and the separate origin key, all local."""
    man = frozen_rows(TEST)
    items, key = build_items(man, test_preds(), f"{TEST_SEED}:labels")
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    e07.write_json(TEST_DATA / "origin_key.json", key)
    e07.write_json(TEST_DATA / "items.json", {"policy": "docs/PHONETIC_POLICY.md", "items": items})
    files = write_batches(TEST_DATA, items, "E13 test", TEST_BATCH_MAX_CANDIDATES)
    out = {"created": e06.now(), "items": len(items), "candidates": sum(len(i["candidates"]) for i in items),
           "single_candidate_items": sum(len(i["candidates"]) == 1 for i in items),
           "batches": len(files) // 2, "max_candidates_per_batch": TEST_BATCH_MAX_CANDIDATES,
           "sha256": {f: e06.sha(TEST_DATA / f) for f in ["origin_key.json", "items.json", *files]},
           "blinding": "no model names, scores or references; labels seeded per item; separate origin key"}
    set_state("batches", "awaiting_owner_test_judgments", items=len(items))
    return e07.write_json(TEST / "batches.json", out) and out


def per_model(items: list[dict], key: dict, seen: dict) -> dict[str, dict[str, str]]:
    out = {c: {} for c in COMPARED}
    for it in items:
        k = key[it["item_id"]]
        for lb, origins in k["origins"].items():
            for o in origins:
                out[o][it["item_id"]] = seen[(it["item_id"], lb)]
        for o in k["empty"]:
            out[o][it["item_id"]] = "not_acceptable"
    return out


def decide(primary: dict, groups: dict) -> dict:
    """The registered classification.

    Examples:
        >>> decide({"ci95_pp": [1.0, 9.0]}, {"names": {"primary": {"diff_pp": -12.5}}})
        {'class': 'promising_improvement', 'trade_off_groups': ['names']}
        >>> decide({"ci95_pp": [-5.0, 9.0]}, {})["class"]
        'uncertain_change'
    """
    lo, hi = primary["ci95_pp"]
    cls = "promising_improvement" if lo > 0 else "regression" if hi < 0 else "uncertain_change"
    return {"class": cls, "trade_off_groups": [g for g, r in groups.items() if r["primary"]["diff_pp"] <= -10]}


def analyse(j: dict[str, dict[str, str]], man: list[dict]) -> dict:
    """Primary + secondary (per group) analysis and the registered decision."""
    a = e12.analyse_test(j, "e13", "e11")
    a["interpretation"] = TEST_ANALYSIS["interpretation"]
    cat = {r["id"]: r["category"] for r in man}
    groups = {}
    for c in CATEGORIES:
        sub = {m: {i: v for i, v in j[m].items() if cat[i] == c} for m in j}
        g = e12.analyse_test(sub, "e13", "e11")
        groups[c] = {k: g[k] for k in ("n", "counts", "confirmed_acceptable_share", "primary", "paired", "exact_test")}
    return {**a, "secondary_groups": groups, "decision": decide(a["primary"], groups)}


@timed("import")
def import_test() -> dict:
    """Validate the owner's test judgments; the predeclared analysis only when complete and valid."""
    data = json.loads((TEST_DATA / "items.json").read_text())
    items, key = data["items"], json.loads((TEST_DATA / "origin_key.json").read_text())
    v = validate(TEST_DATA, items)
    if v["status"] != "complete":
        set_state("import", "awaiting_owner_test_judgments", detail=v["status"])
        return {**v, "status": "awaiting_owner_test_judgments" if v["status"] in ("awaiting", "incomplete")
                else v["status"], "expected": str(TEST_DATA / "judgments_<n>.tsv")}
    man = frozen_rows(TEST)
    out = {"status": "scored", "provenance": v["provenance"], "judgment_files_sha256": v["judgment_files_sha256"],
           "track": "single owner judge, blinded, not independent of the project",
           "analysis": analyse(per_model(items, key, v["seen"]), man)}
    mech = REPORTS / "mechanism.json"
    if mech.exists():
        mj = json.loads(mech.read_text())
        out["missing_pronunciation_behaviour"] = {k: mj[k]["reference_scores"] for k in ("collection_dev", "historical_dev")}
    set_state("import", "scored", decision=out["analysis"]["decision"]["class"])
    return e07.write_json(REPORTS / "test_scores.json", out) and out


def status() -> dict:
    s = state()
    s["inference_hours"] = round(e11.inference_hours(), 3)
    s["training"] = e07.budget_left("full")
    return s


STEPS = {"verify": verify, "inputs": inputs, "collect": collect, "import-collection": import_collection,
         "register": register, "pipeline": pipeline, "train": train, "freeze": freeze, "evaluate": evaluate,
         "mechanism": mechanism, "testgen": testgen, "batches": batches, "import": import_test, "status": status}


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
