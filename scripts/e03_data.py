#!/usr/bin/env python3
"""Experiment E03: the repaired selection rule and the dictionary-assessable, consonant-screened (DACS) data.

    .venv/bin/python scripts/e03_data.py exclusions   # isolated: held-out sets from frozen files -> data/e03/exclusions.json
    .venv/bin/python scripts/e03_data.py build        # candidates, screen, quarantine, splits -> data/e03/
    .venv/bin/python scripts/e03_data.py check        # re-verify every hash in data/e03/manifest.json

**The repair.** The miner (`scripts/translit_mine.py`) labelled a pair `khmer_like` when *either* its Latin side is
a romanised-Khmer form (`models/model/romanised_forms.txt`) *or* its Khmer side is a known word (Google lexicon or
`known_targets.txt`), and E02 dropped every `khmer_like` row. `classify` separates the two conditions:
- `latin_romanised`: still excluded (declared policy). The detector is a heuristic list; Latin forms that are also
  common English words (SCOWL) are recorded as `romanised_english_collision`, not declared settled;
- `khmer_known_*`: **not** a reason to reject. Those pairs are the recovered candidates.
E02's build and files are not touched.

**After recovery** the E02 checks apply unchanged and in the same order (plain a–z, well-formed Khmer target, the
D135 noise filter over all mined forms of the word, a CMUdict reading, held-out family, held-out target). Then:
- assessable track: the target has an exact Google-lexicon reading (else `missing_reading`, kept outside);
- screen: the frozen E02-diagnostics consonant comparison (`e02_diagnostics.compat_row`, rules frozen at e4c0857),
  a screening proxy, not a certification;
- quarantine: `compound_extra_component` (the target is an aligned loan plus extra lexicon word(s), e.g. +គណនី
  "account") is an established misalignment; any other `suspected_conflict` is unresolved; both are kept on file;
- retained: `compatible_evidence` pairs; several retained targets of one word are alternatives, kept together.
Vowels are reported separately (syllable counts, vowel pairs) and never used to accept or reject.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "e03"
EXP = ROOT / "experiments" / "E03-assessable-phones"
REPORTS = EXP / "reports"
SALT = "e03-assessable-phones"
SHARES = {"train": 80, "dev": 10}            # % of families (split_bucket is 0-99); the rest: eval
TRAIN_VAL_SHARE = 4                          # % of training families: loss tie-break only
MINED = {"kmwiki": DATA / "mined_kmwiki.tsv", "fineweb": DATA / "mined_fineweb.tsv"}
DETECTORS = {"romanised_forms": ROOT / "models" / "model" / "romanised_forms.txt",
             "known_targets": ROOT / "models" / "model" / "known_targets.txt"}
EXCLUSION_SOURCES = {"e02_references": DATA / "e02" / "references.jsonl",
                     "v1_pick_test": DATA / "benchmark" / "v1" / "pick_test.tsv",
                     "v1_dev41": DATA / "benchmark" / "v1" / "dev41.tsv",
                     "v1_list_test": DATA / "benchmark" / "v1" / "list_test.tsv",
                     "reserve_latin": DATA / "e2" / "reserve.json", "e01_list_val": DATA / "list_split.json"}
TRANSLATION_SIM = 0.15                       # E02's possible-translation screen (rule similarity)
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def sha(path: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """A headed TSV."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


# ------------------------------------------------------------------ the repaired rule
def classify(latin: str, khmer: str, romanised: frozenset, lexicon_keys: frozenset, known_targets: frozenset,
             english_words: frozenset) -> dict:
    """The two conditions the old `khmer_like` class merged, kept apart.

    Returns:
        `{"latin_romanised", "khmer_in_lexicon", "khmer_in_known_targets", "old_class", "romanised_english_collision"}`.
        `old_class` reproduces the miner's rule exactly (`khmer_like` iff either condition).

    Examples:
        >>> c = classify("paris", "ប៉ារីស", frozenset(), frozenset({"ប៉ារីស"}), frozenset(), frozenset({"paris"}))
        >>> c["old_class"], c["latin_romanised"], c["khmer_in_lexicon"]
        ('khmer_like', False, True)
    """
    lat_r = latin in romanised
    in_lex, in_known = khmer in lexicon_keys, khmer in known_targets
    return {"latin_romanised": lat_r, "khmer_in_lexicon": in_lex, "khmer_in_known_targets": in_known,
            "old_class": "khmer_like" if lat_r or in_lex or in_known else "foreign",
            "romanised_english_collision": lat_r and latin in english_words}


def recovery_reason(c: dict) -> str | None:
    """Why a mined `khmer_like` pair is (not) a recovered candidate; None for `foreign` rows (already E02 data)."""
    if c["old_class"] != "khmer_like":
        return None
    if c["latin_romanised"]:
        return "excluded:latin_romanised_collision" if c["romanised_english_collision"] else "excluded:latin_romanised"
    return "recovered:khmer_target_known_only"


def filter_reason(lat: str, w: str, status: str, km: str, kept: list[str], cmu: dict, held_words: set,
                  held_stems: set, held_targets: set, tgt_chars: set) -> str | None:
    """E02's checks, in E02's order, for one recovered pair; None if it passes.

    Examples:
        >>> filter_reason("connected", "connected", "ok", "ខុនណិច", ["ខុនណិច"], {"connected": ()}, {"connect"},
        ...               {"connect"}, set(), set("ខុនណិច"))
        'heldout_related'
    """
    if status != "ok" or w != lat:
        return "not_plain_a_z"
    if not ps.is_khmer_word(km) or len(km) > ps.MAX_TGT:
        return "bad_khmer_or_length"
    if km not in kept:
        return "noise_filter"
    if w not in cmu:
        return "no_english_reading"
    if w in held_words or rd.related(w, held_stems):
        return "heldout_related"
    if km in held_targets:
        return "heldout_target"
    if set(km) - tgt_chars:
        return "target_chars_outside_vocabulary"
    return None


# ------------------------------------------------------------------ screens
def compound_extra_component(word: str, target: str, km_lex: dict, supplied: tuple[str, ...]) -> str | None:
    """The extra lexicon word(s) if `target` = a loan that fits the English consonants better + lexicon word(s).

    A split is tried at every consonant-letter boundary; the tail must segment completely into lexicon words of
    ≥ 2 characters, and the head must align with strictly fewer unexplained consonant operations than the whole.

    Examples:
        >>> lex = {"គណនី": ["k ɔ . n ɔ . n ii"], "ប្លុក": ["ɓ l o k"]}
        >>> compound_extra_component("block", "ប្លុកគណនី", lex, ("B", "L", "AA1", "K"))
        'គណនី'
    """
    import e02_diagnostics as DG
    from khmer_translit import lexicon
    en = DG.en_skeleton(supplied)
    whole_sk = DG.km_skeleton(target)
    if whole_sk is None:
        return None
    whole = DG.align(en, whole_sk)[0]
    if whole == 0:
        return None
    for i in range(2, len(target) - 1):
        head, tail = target[:i], target[i:]
        if tail[0] not in DG.KM_LETTER or head[-1] == DG.COENG:
            continue
        parts = lexicon.segment(tail, km_lex)
        if not all(ok and len(p) >= 2 for p, ok in parts):
            continue
        hsk = DG.km_skeleton(head)
        if hsk is not None and DG.align(en, hsk)[0] < whole:
            return tail
    return None


def syllables(google: str) -> int:
    """Syllables in a Google transcription (dot-separated groups)."""
    return len([s for s in google.split(".") if s.strip()])


def vowel_report(supplied: tuple[str, ...], readings: list[str]) -> dict:
    """English vowel count vs Khmer syllable count (closest reading); descriptive only."""
    n_en = sum(p.rstrip("012") in VOWELS for p in supplied)
    diffs = [syllables(r) - n_en for r in readings]
    best = min(diffs, key=abs)
    return {"english_vowels": n_en, "khmer_syllables_closest": n_en + best, "syllable_diff": best}


# ------------------------------------------------------------------ isolated exclusion step
def exclusions() -> dict:
    """Held-out words, stems and targets from the frozen files E02 held out (mechanical read; only sets/counts out).

    The same sets E02's build used: every E02 benchmark item (dev, final, proxy) word and accepted spelling, the v1
    English panels' words, the reserve's Latin words, E01's list-validation words. No reference is inspected, scored
    or used for selection; the output goes to `data/e03/exclusions.json` (local), counts to the report.
    """
    import english_phonetic_data as ED
    items = rd.load_reference_items(EXCLUSION_SOURCES["e02_references"])
    words = {it.word for it in items}
    targets = {ps.nfc(a) for it in items for a in it.accepted}
    for name in ("pick_test", "dev41", "list_test"):
        words |= {ps.normalize_word(r[0])[0] for r in ED._tsv_noheader(EXCLUSION_SOURCES[f"v1_{name}"])}
    words |= {ps.normalize_word(w)[0] for w in json.loads(EXCLUSION_SOURCES["reserve_latin"].read_text(encoding="utf-8"))}
    words |= set(json.loads(EXCLUSION_SOURCES["e01_list_val"].read_text(encoding="utf-8"))["val"])
    words.discard("")
    stems = {s for w in words for s in rd.stems(w)}
    out = {"note": "held-out sets for E03 (mechanical read of the files E02 held out; reserve: Latin words only)",
           "sources_sha256": {k: sha(p) for k, p in EXCLUSION_SOURCES.items()},
           "counts": {"words": len(words), "stems": len(stems), "targets": len(targets)},
           "words": sorted(words), "stems": sorted(stems), "targets": sorted(targets)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exclusions.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    return {k: out[k] for k in ("note", "sources_sha256", "counts")}


# ------------------------------------------------------------------ build
def build() -> dict:
    """Build every E03 data file and the audit (see the module docstring)."""
    import e02_diagnostics as DG
    import english_phonetic_data as ED
    from khmer_translit import lexicon
    excl = json.loads((OUT / "exclusions.json").read_text(encoding="utf-8"))
    held_words, held_stems, held_targets = set(excl["words"]), set(excl["stems"]), set(excl["targets"])
    romanised = frozenset(DETECTORS["romanised_forms"].read_text(encoding="utf-8").split())
    known_targets = frozenset(ps.nfc(k) for k in DETECTORS["known_targets"].read_text(encoding="utf-8").split())
    lex_keys = frozenset(ps.nfc(k) for k in lexicon.keys())
    english = frozenset((DATA / "en_common_words.txt").read_text(encoding="utf-8").split())
    km_lex, lat_lex = DG.load_lexicon()
    cmu, cmu_sha = ps.cmudict()
    vocab = json.loads((DATA / "e02" / "vocab.json").read_text(encoding="utf-8"))
    tgt_chars = set(vocab["tgt_chars"])

    # all mined rows, merged per (latin, khmer) as E02 did; class conditions per row
    forms: dict[str, dict[str, list]] = defaultdict(dict)
    stored_class: dict[tuple[str, str], set] = defaultdict(set)
    for src, path in MINED.items():
        for r in read_tsv(path):
            km = ps.nfc(r["khmer"])
            o = forms[r["latin"]].setdefault(km, [0, 0.0, set(), set()])
            o[0] += int(r["count"])
            o[1] = max(o[1], float(r["similarity"]))
            o[2].add(r["class"])
            o[3].add(src)
            stored_class[(r["latin"], km)].add(r["class"])
    counts: dict[str, Counter] = {"recovery": Counter(), "filters": Counter(), "track": Counter(),
                                  "screen": Counter(), "decision": Counter()}
    reproduce = Counter()
    cands = []
    for lat in sorted(forms):
        w, st = ps.normalize_word(lat)
        kept = ED.keep_forms({k: (v[0], v[1]) for k, v in forms[lat].items()})
        for km in sorted(forms[lat]):
            c = classify(lat, km, romanised, lex_keys, known_targets, english)
            reproduce["match" if {c["old_class"]} == stored_class[(lat, km)] else "differ"] += 1
            why = recovery_reason(c)
            if why is None:
                continue
            counts["recovery"][why] += 1
            if w in cmu and km in km_lex:                 # reconciles with the diagnostics' 8,725 (both conditions)
                counts["recovery"]["any_khmer_like_with_cmudict_word_and_exact_reading_before_filters"] += 1
            if not why.startswith("recovered"):
                continue
            n, sim, _cls, srcs = forms[lat][km]
            if w in cmu and km in km_lex:
                counts["recovery"]["recovered_with_cmudict_word_and_exact_reading_before_filters"] += 1
            rec = {"latin": lat, "word": w, "target": km, "sources": sorted(srcs), "mined_count": n,
                   "mined_max_similarity": sim, "original_exclusion": "khmer_like (E02: dropped)",
                   "conditions": c, "eligibility": [why]}
            rec["filter"] = filter_reason(lat, w, st, km, kept, cmu, held_words, held_stems, held_targets, tgt_chars)
            counts["filters"][rec["filter"] or "passed"] += 1
            cands.append(rec)

    # assessable track, screen, quarantine
    for rec in cands:
        if rec["filter"]:
            rec["decision"] = "filtered:" + rec["filter"]
            continue
        w, km = rec["word"], rec["target"]
        readings = cmu[w]
        supplied = tuple(readings[0])
        strip = lambda ph: tuple(p.rstrip("012") for p in ph)
        rec["pronunciation"] = {"supplied": " ".join(supplied), "alternatives": [" ".join(x) for x in readings[1:]],
                                "ambiguous": len({strip(x) for x in readings}) > 1}
        exact = km_lex.get(km)
        rec["khmer_readings"] = exact or []
        rec["rule_similarity"] = round(ED.rule_similarity(w, km), 3)
        if not exact:
            rec["track"] = "missing_reading"
            rec["decision"] = "outside_track:missing_reading"
            counts["track"]["missing_reading"] += 1
            continue
        rec["track"] = "assessable"
        counts["track"]["assessable"] += 1
        scr = DG.compat_row(w, km, supplied, readings, km_lex, lat_lex)
        rec["screen"] = {k: scr.get(k) for k in ("category", "level", "cost_supplied", "cost_best_alternative",
                                                 "reasons", "ops", "evidence_track")}
        counts["screen"][scr["category"]] += 1
        rec["vowels"] = vowel_report(supplied, exact)
        extra = compound_extra_component(w, km, km_lex, supplied)
        rec["compound_extra_component"] = extra
        rec["possible_translation"] = rec["rule_similarity"] < TRANSLATION_SIM
        if extra:
            rec["decision"] = "quarantine:established_misalignment"
        elif scr["category"] == "unknown":
            rec["decision"] = "quarantine:unknown_screen"
        elif scr["category"] != "compatible_evidence":
            rec["decision"] = "quarantine:unresolved_conflict"
        else:
            rec["decision"] = "retained"
        counts["decision"][rec["decision"]] += 1

    # families and splits over the retained words
    retained = [r for r in cands if r.get("decision") == "retained"]
    words = sorted({r["word"] for r in retained})
    fam = rd.group_families(words)
    split_of = {}
    for w in words:
        b = rd.split_bucket(fam[w], SALT)
        split_of[w] = "train" if b < SHARES["train"] else "dev" if b < SHARES["train"] + SHARES["dev"] else "eval"
    held_eval_targets = {r["target"] for r in retained if split_of[r["word"]] != "train"}
    for r in retained:
        r["family"], r["split"] = fam[r["word"]], split_of[r["word"]]
        if r["split"] == "train" and r["target"] in held_eval_targets:
            r["split"], r["decision"] = None, "filtered:target_held_in_dev_or_eval"
            counts["decision"]["retained"] -= 1
            counts["decision"]["filtered:target_held_in_dev_or_eval"] += 1
    retained = [r for r in retained if r["decision"] == "retained"]
    for r in retained:
        if r["split"] == "train":
            r["part"] = "train_val" if rd.split_bucket(r["family"], SALT + ":tv") < TRAIN_VAL_SHARE else "train"
    write_outputs(cands, retained, counts, reproduce, excl, cmu_sha)
    return audit(cands, retained, counts, reproduce, excl)


def reference_items(retained: list[dict], split: str) -> list[rd.ReferenceItem]:
    """Dev or eval references: one item per word, all retained targets accepted (alternatives together)."""
    by_word = defaultdict(list)
    for r in retained:
        if r["split"] == split:
            by_word[r["word"]].append(r)
    items = []
    for w, rs in sorted(by_word.items()):
        pron = ps.get_pronunciation(w)
        flags = (["pronunciation_ambiguous"] if pron.ambiguous else []) + \
                (["possible_translation"] if all(x["possible_translation"] for x in rs) else [])
        pref = max(rs, key=lambda x: (x["mined_count"], x["target"]))["target"]
        items.append(rd.ReferenceItem(
            f"{w}#default", w, w, rs[0]["family"], "default", " ".join(pron.phones), "en-US-cmudict",
            sorted({x["target"] for x in rs}), pref, sorted({f"mined_{s}" for x in rs for s in x["sources"]}),
            "proxy", "dev" if split == "dev" else "final", flags,
            {"reviewer": "none", "policy": "E03 dictionary-assessable, consonant-screened mined pairs (screening proxy)",
             "partition": "E03 " + split}))
    return items


def write_outputs(cands, retained, counts, reproduce, excl, cmu_sha) -> None:
    """data/e03/: candidates.jsonl (full audit table), train.tsv, dev/eval references, vocab, manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in cands:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with open(OUT / "train.tsv", "w", encoding="utf-8") as f:
        f.write("word\tkhmer\tsource\ttier\tphones\tpart\n")
        for r in sorted((r for r in retained if r["split"] == "train"), key=lambda r: (r["word"], r["target"])):
            f.write(f"{r['word']}\t{r['target']}\tmined_dacs\tproxy\t{r['pronunciation']['supplied']}\t{r['part']}\n")
    dev, ev = reference_items(retained, "dev"), reference_items(retained, "eval")
    rd.validate_items(dev + ev)
    rd.save_reference_items(dev, OUT / "dev_references.jsonl")
    rd.save_reference_items(ev, OUT / "eval_references.jsonl")
    (OUT / "vocab.json").write_bytes((DATA / "e02" / "vocab.json").read_bytes())
    files = ["candidates.jsonl", "train.tsv", "dev_references.jsonl", "eval_references.jsonl", "vocab.json",
             "exclusions.json"]
    inputs = {**{f"mined_{k}": p for k, p in MINED.items()}, **DETECTORS,
              "lexicon": DATA / "lexicon" / "km_lexicon.tsv", "cmudict": ps.CMUDICT,
              "en_words": DATA / "en_common_words.txt", "e02_vocab": DATA / "e02" / "vocab.json"}
    manifest = {"experiment": "E03-assessable-phones", "cmudict_sha256": cmu_sha,
                "files": {f: sha(OUT / f) for f in files},
                "inputs": {k: {"path": str(Path(p).relative_to(ROOT)), "sha256": sha(p)} for k, p in inputs.items()},
                "counts": {"supervised_kept": {"mined_dacs": sum(r["split"] == "train" for r in retained)},
                           **{k: dict(v) for k, v in counts.items()}}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def audit(cands, retained, counts, reproduce, excl) -> dict:
    """The data audit (counts only; no reference content), written to `reports/data_audit.json`."""
    rows = [r for r in cands if r.get("track") == "assessable"]
    by_split = defaultdict(list)
    for r in retained:
        by_split[r["split"]].append(r)

    def words(rs):
        return len({r["word"] for r in rs})

    alt_groups = Counter()
    per_word = defaultdict(list)
    for r in retained:
        per_word[r["word"]].append(r)
    for w, rs in per_word.items():
        alt_groups[str(len(rs))] += 1
    mixed = sum(1 for w in {r["word"] for r in rows}
                if {x["decision"] for x in rows if x["word"] == w} >= {"retained"}
                and any(x["decision"].startswith("quarantine") for x in rows if x["word"] == w))
    vow = Counter(str(r["vowels"]["syllable_diff"]) for r in rows)
    vow_ret = Counter(str(r["vowels"]["syllable_diff"]) for r in retained)
    src = Counter("+".join(r["sources"]) for r in retained)
    a = {
        "selection_rule_reproduction": dict(reproduce),
        "exclusion_counts": excl["counts"],
        "recovery": dict(counts["recovery"]),
        "filters_in_order": dict(counts["filters"]),
        "assessable_track": dict(counts["track"]),
        "screen_on_assessable": dict(counts["screen"]),
        "decisions": dict(counts["decision"]),
        "retained": {"pairs": len(retained), "words": words(retained),
                     "by_split": {s: {"pairs": len(v), "words": words(v), "families": len({r['family'] for r in v})}
                                  for s, v in sorted(by_split.items())},
                     "train_val_pairs": sum(r.get("part") == "train_val" for r in retained),
                     "targets_per_word": dict(sorted(alt_groups.items())),
                     "sources": dict(src),
                     "cmudict_ambiguous_pairs": sum(r["pronunciation"]["ambiguous"] for r in retained),
                     "possible_translation_flag": sum(r["possible_translation"] for r in retained)},
        "words_with_retained_and_quarantined_targets": mixed,
        "vowels_descriptive": {"syllable_minus_english_vowels_all_assessable": dict(sorted(vow.items(), key=lambda x: int(x[0]))),
                               "retained": dict(sorted(vow_ret.items(), key=lambda x: int(x[0])))},
        "pronunciation_coverage": {"cmudict_default": len([r for r in cands if r.get("pronunciation")]),
                                   "google_exact_reading": len(rows)},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_audit.json").write_text(json.dumps(a, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return a


def overlap() -> dict:
    """Counts only: E03 words/targets per split also present in historical data or earlier models' training."""
    rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    ret = [r for r in rows if r.get("decision") == "retained"]
    e02 = read_tsv(DATA / "e02" / "train.tsv")
    p2g = {r["khmer"] for r in read_tsv(DATA / "e02" / "p2g.tsv")}
    e01 = [l.split("\t") for l in (DATA / "extra_train.tsv").read_text(encoding="utf-8").splitlines()[1:] if l]
    known = set(DETECTORS["known_targets"].read_text(encoding="utf-8").split())
    lst = ROOT.parent / "pocket-tts-khmer" / "configs" / "latin_khmer.tsv"
    list_words = {ps.normalize_word(l.split("\t")[0])[0] for l in lst.read_text(encoding="utf-8").splitlines()[1:]
                  if l} if lst.exists() else set()
    excl = json.loads((OUT / "exclusions.json").read_text(encoding="utf-8"))
    refs = {"e02_train_words": {r["word"] for r in e02}, "e02_train_targets": {r["khmer"] for r in e02},
            "e02_p2g_targets (phones_p2g model only)": p2g,
            "e01_mined_extra_train_words": {ps.normalize_word(r[0])[0] for r in e01 if r},
            "e01_mined_extra_train_targets": {ps.nfc(r[1]) for r in e01 if len(r) > 1},
            "known_targets (models A/B/E training targets)": {ps.nfc(k) for k in known},
            "loanword_list_words (speech repo)": list_words,
            "e03_heldout_words": set(excl["words"]), "e03_heldout_targets": set(excl["targets"])}
    out = {}
    for split in ("train", "dev", "eval"):
        rs = [r for r in ret if r["split"] == split]
        words, targets = {r["word"] for r in rs}, {r["target"] for r in rs}
        out[split] = {"words": len(words), "targets": len(targets),
                      **{k: len((words if "words" in k else targets) & v) for k, v in refs.items()}}
    out["note"] = ("counts of shared words/targets; E03 targets are Google-lexicon words by construction, so the "
                   "E02 P2G pretraining saw them as spellings (E03 trains no P2G)")
    (REPORTS / "overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def check() -> int:
    """Re-verify data/e03 hashes against the manifest."""
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    bad = [f for f, h in m["files"].items() if sha(OUT / f) != h]
    bad += [k for k, v in m["inputs"].items() if sha(ROOT / v["path"]) != v["sha256"]]
    print(json.dumps({"ok": not bad, "changed": bad}))
    return 1 if bad else 0


def main() -> int:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["exclusions", "build", "overlap", "check"])
    a = ap.parse_args()
    if a.step == "check":
        return check()
    print(json.dumps({"exclusions": exclusions, "build": build, "overlap": overlap}[a.step](), ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
