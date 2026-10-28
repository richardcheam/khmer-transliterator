"""E08 data expansion for the E07 ByT5-small model: recovery of excluded mined pairs, source pilots, a frozen
200-word synthetic input manifest, Google AI Mode prompts and a strict response importer. No training.

Four provenances stay in separate manifests (`data/e08/`, local; summaries in `experiments/E08-data-expansion/`):
    control     the E03 optimisation split, unchanged (`data/e03/train.tsv`, hash-checked)
    recovered   E03-excluded mined pairs re-admitted by a documented screen-weakness rule
    attested    pairs from newly audited public sources (only where permission is clear)
    synthetic   `google_ai_mode_synthetic` responses, imported from files the owner saves

Steps:
    prepare             recover → sources → attested → inputs → prompts → arms C, R (resumable; frozen files verified)
    import --responses  parse, validate and import saved AI Mode responses; write regeneration prompts
    freeze              after the import: build arm RS and freeze all arm hashes (no training)
    status              current state (e.g. `awaiting_ai_mode_responses`)

Example:
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/e08.py prepare
    .venv/bin/python scripts/e08.py import --responses data/e08/ai_mode_responses
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05  # noqa: E402
import e06  # noqa: E402
from khmer_translit import phonetic_source as ps  # noqa: E402
from khmer_translit import reference_data as rd  # noqa: E402

ROOT = e06.ROOT
EXPERIMENT = "E08-data-expansion"
EXP = ROOT / "experiments" / EXPERIMENT
REPORTS = EXP / "reports"
SYN = EXP / "synthetic"
PROMPTS = SYN / "prompts"
MANIFEST = SYN / "input_manifest.jsonl"
DATA = ROOT / "data" / "e08"                      # git-ignored (data policy): pair manifests, sources, responses
RESPONSES = DATA / "ai_mode_responses"
IMPORTED = DATA / "synthetic"
E03 = ROOT / "data" / "e03"
SOURCE = "google_ai_mode_synthetic"
SEED = "e08-synthetic-inputs"
BATCHES, PER_BATCH = 10, 20
CATEGORIES = {"common": 40, "irregular": 40, "vowels": 40, "clusters": 40, "names": 40}
VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
MAX_CHARS = 40                                   # 120 UTF-8 bytes: the E07 target limit
ID_RE = re.compile(r"^e08s\d{3}$")


def write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def write_jsonl(p: Path, rows) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def order(seed: str, key: str) -> str:
    return e06.order_key(seed, key)


# ------------------------------------------------------------------ protected English families (input-only)
def protected() -> dict:
    """Protected English words and stems, from Latin inputs only (no protected Khmer reference is opened).

    Sources:
    - `data/e03/exclusions.json` (E03's mechanical read of E02's held-out sets: E02 benchmark words, the v1 panels,
      the reserve's Latin words, E01's list validation)
    - the input words of E03 dev and eval
    - the E06 fresh pool and the E06 discovery words
    - the E05 and E06 review queues (English column)
    - the benchmark panels and the reserve, through E06's input-only reader
    """
    excl = json.loads(e06.guard(E03 / "exclusions.json").read_text(encoding="utf-8"))
    parts = {"e03_exclusions": set(excl["words"])}
    for split in ("dev", "eval"):
        parts[f"e03_{split}"] = {json.loads(x)["word"] for x in
                                 e06.guard(E03 / f"{split}_references.jsonl").read_text(encoding="utf-8").splitlines()}
    parts["e06_pool"] = {r["word"] for r in e06.load_pool()}
    parts["e06_discovery"] = set(json.loads((e06.DISCOVERY / "words.json").read_text()))
    for q in ("e05", "e06"):
        lines = e06.guard(ROOT / "data" / q / "review" / "review_queue.tsv").read_text(encoding="utf-8").splitlines()[1:]
        parts[f"{q}_review_queue"] = {ln.split("\t")[1].lower() for ln in lines}
    parts["input_only_panels_and_reserve"] = set().union(*(e06.input_only_words(k) for k in e06.INPUT_ONLY))
    words = {e06.norm_word(w) for s in parts.values() for w in s} - {""}
    stems = set(excl["stems"]) | {s for w in words for s in rd.stems(w)}
    return {"words": words, "stems": stems, "held_targets": set(excl["targets"]),
            "counts": {k: len(v) for k, v in parts.items()} | {"words": len(words), "stems": len(stems)}}


def is_protected(word: str, prot: dict) -> bool:
    """Same family rule as E02/E03 (`reference_data.related`)."""
    return word in prot["words"] or rd.related(word, prot["stems"])


# ------------------------------------------------------------------ recovery rules (documented screen weaknesses)
LETTER_PHONES = {"b": {"p", "ɓ", "b"}, "d": {"d", "ɗ", "t"}, "l": {"l"}, "n": {"n"}, "m": {"m"}, "s": {"s", "h"},
                 "t": {"t", "th"}, "r": {"r"}, "p": {"p", "ph"}, "k": {"k", "kh"}, "c": {"k", "kh"},
                 "g": {"k", "g"}, "f": {"f", "ph"}, "z": {"s", "h"}}


def explain(op: list, word: str, target: str) -> str | None:
    """The documented rule that explains one unexplained screen operation, or None.

    Rules (E02-diagnostics §2 type 5, E03 REPORT "uncertainty that remains"):
    - `coda_h`: English s/z/sh read as Khmer coda h, written ស (Khmer final ស is pronounced h)
    - `doubled_letter`: an extra Khmer consonant where the English spelling doubles that letter (អានណា *anna*)
    - `y_glide`: an inserted y glide where the spelling has i/y before a vowel (អាណាតូលីយ៉ា)
    - `y_unwritten`: English y not written
    - `final_ic`: final k written ច after ិ (Google reads អាឡិច as "ʔ aa . l ə c")
    - `hk_is_g`: ហ្ក read as g (a gap in the frozen reader)

    Examples:
        >>> explain(["sub", "S", "h"], "ace", "អេស"), explain(["del_b", "", "n"], "anna", "អានណា")
        ('coda_h', 'doubled_letter')
        >>> explain(["del_b", "", "p"], "adel", "អាប់ឌែល") is None
        True
    """
    kind, en, km = op[0], op[1], op[2]
    if kind == "sub" and en in ("S", "Z", "SH") and km == "h" and "ស" in target:
        return "coda_h"
    if kind == "del_b" and km == "j" and re.search(r"[iy][aeiou]", word):
        return "y_glide"
    if kind == "del_b" and any(km in LETTER_PHONES.get(a, ()) for a, b in zip(word, word[1:]) if a == b):
        return "doubled_letter"
    if kind == "del_a" and en == "Y":
        return "y_unwritten"
    if kind == "sub" and en == "K" and km == "c" and "ិច" in target:
        return "final_ic"
    if kind == "sub" and en == "G" and km == "k" and "ហ្ក" in target:
        return "hk_is_g"
    return None


UNEXPLAINED = ("sub", "del_a", "del_b")
READING_SWITCH = "fits the Google lexicon's Khmer reading of the word, which departs from the English one"


def recovery_decision(rec: dict) -> tuple[str, list[str]]:
    """`(category, rules)`: category `recovered`, `error_or_translation` or `uncertain` for one screened row.

    Examples:
        >>> r = {"screen": {"category": "suspected_conflict", "level": "1", "ops": [["sub", "S", "h"]],
        ...                 "reasons": ["vowels not assessed"]}, "word": "ace", "target": "អេស",
        ...      "possible_translation": False, "compound_extra_component": None}
        >>> recovery_decision(r)
        ('recovered', ['coda_h'])
        >>> recovery_decision({**r, "possible_translation": True})[0]
        'error_or_translation'
    """
    s = rec.get("screen") or {}
    if rec.get("possible_translation"):
        return "error_or_translation", []
    if s.get("category") in (None, "unknown"):
        return "uncertain", []
    un = [o for o in s.get("ops") or [] if o[0] in UNEXPLAINED]
    rules = [explain(o, rec["word"], rec["target"]) for o in un]
    if READING_SWITCH in (s.get("reasons") or []):
        return "uncertain", []                   # the target may follow a non-English reading: not switched silently
    if rec.get("compound_extra_component"):
        compound_ok = s.get("category") == "compatible_evidence" and (
            re.search(r"([a-z])\1", rec["word"]) or re.search(r"[iy][aeiou]", rec["word"]))
        if not compound_ok:
            return ("error_or_translation" if s.get("level") == "3+" else "uncertain"), []
        return "recovered", sorted(set(r for r in rules if r)) + ["compound_flag_doubled_or_glide"]
    if s.get("category") == "compatible_evidence":
        return "recovered", ["screen_compatible_alternative"]
    if all(rules) and rules:
        return "recovered", sorted(set(rules))
    if s.get("level") == "3+" and not any(rules):
        return "error_or_translation", []
    return "uncertain", []


def recover() -> dict:
    """Re-screen E03's excluded pairs; write control / recovered / error / uncertain manifests and audit samples."""
    import e02_diagnostics as DG
    import e03_data
    import english_phonetic_data as ED
    km_lex, lat_lex = DG.load_lexicon()
    cmu, _ = ps.cmudict()
    prot = protected()
    cands = read_jsonl(e06.guard(E03 / "candidates.jsonl"))
    train_lines = e06.guard(E03 / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]
    control = [dict(zip(("word", "khmer", "source", "tier", "phones", "part"), ln.split("\t"))) for ln in train_lines]
    control_pairs = {(r["word"], ps.nfc(r["khmer"])) for r in control}
    control_words = {r["word"] for r in control}
    held = {json.loads(x)["word"] for s in ("dev", "eval")
            for x in e06.guard(E03 / f"{s}_references.jsonl").read_text(encoding="utf-8").splitlines()}
    held_stems = {s for w in held for s in rd.stems(w)}
    held_targets = {ps.nfc(a) for s in ("dev", "eval") for x in
                    e06.guard(E03 / f"{s}_references.jsonl").read_text(encoding="utf-8").splitlines()
                    for a in json.loads(x)["accepted"]}
    out = {"recovered": [], "error_or_translation": [], "uncertain": []}
    counts = Counter()
    reconsider = ("quarantine:unresolved_conflict", "quarantine:established_misalignment", "quarantine:unknown_screen",
                  "outside_track:missing_reading", "filtered:noise_filter")
    seen = set()
    for c in cands:
        d = c["decision"]
        if d not in reconsider:
            counts[f"not_reconsidered:{d}"] += 1
            continue
        w, km = c["word"], ps.nfc(c["target"])
        if (w, km) in seen:
            counts["duplicate_pair"] += 1
            continue
        seen.add((w, km))
        rec = {k: c.get(k) for k in ("word", "latin", "target", "sources", "mined_count", "mined_max_similarity",
                                     "original_exclusion")} | {"e03_decision": d}
        if d == "filtered:noise_filter":         # E03 never screened these: run the same frozen screen now
            if w not in cmu or not ps.is_khmer_word(km) or len(km) > ps.MAX_TGT or c["latin"] != w:
                counts["noise_filter:not_screenable"] += 1
                continue
            readings = cmu[w]
            scr = DG.compat_row(w, km, tuple(readings[0]), readings, km_lex, lat_lex)
            rec["screen"] = {k: scr.get(k) for k in ("category", "level", "ops", "reasons", "evidence_track")}
            rec["pronunciation"] = {"supplied": " ".join(readings[0]), "alternatives": [" ".join(x) for x in readings[1:]]}
            rec["compound_extra_component"] = e03_data.compound_extra_component(w, km, km_lex, tuple(readings[0]))
            rec["possible_translation"] = ED.rule_similarity(w, km) < e03_data.TRANSLATION_SIM
        else:
            rec |= {k: c.get(k) for k in ("screen", "pronunciation", "compound_extra_component", "possible_translation")}
        cat, rules = recovery_decision(rec)
        if cat == "recovered":
            why = ("protected_family" if is_protected(w, prot) else
                   "dev_eval_family" if w in held or rd.related(w, held_stems) else
                   "target_in_dev_eval_or_heldout" if km in held_targets or km in prot["held_targets"] else
                   "exact_pair_in_control" if (w, km) in control_pairs else None)
            if why:
                counts[f"recovered_but_excluded:{why}"] += 1
                continue
            rec["recovery_rule"] = rules
            rec["alternative_of_control_word"] = w in control_words
            rec["evidence"] = {"screen_level": (rec.get("screen") or {}).get("level"),
                               "screen_ops": (rec.get("screen") or {}).get("ops"),
                               "khmer_readings": km_lex.get(km, [])}
        rec["category"] = cat
        out[cat].append(rec)
        counts[f"{d}→{cat}"] += 1
    for cat, rows in out.items():
        write_jsonl(DATA / "recovery" / f"{cat}.jsonl", rows)
    write_jsonl(DATA / "recovery" / "control.jsonl",
                [{"word": r["word"], "khmer": ps.nfc(r["khmer"]), "part": r["part"], "phones": r["phones"],
                  "source": r["source"]} for r in control])
    rule_counts = Counter(x for r in out["recovered"] for x in r["recovery_rule"])
    fam = rd.group_families(sorted({r["word"] for r in out["recovered"]} | control_words))
    rec_words = {r["word"] for r in out["recovered"]}
    samples = {cat: [{"word": r["word"], "target": r["target"], "e03_decision": r["e03_decision"],
                      "rules": r.get("recovery_rule"), "ops": (r.get("screen") or {}).get("ops"),
                      "phones": (r.get("pronunciation") or {}).get("supplied")}
                     for r in sorted(rows, key=lambda r: order("e08-audit-" + cat, r["word"] + r["target"]))[:20]]
               for cat, rows in out.items()}
    summary = {"control": {"rows": len(control), "train": sum(r["part"] == "train" for r in control),
                           "train_val": sum(r["part"] == "train_val" for r in control),
                           "sha256": e06.sha(E03 / "train.tsv")},
               "reconsidered_counts": dict(sorted(counts.items())),
               "manifests": {k: len(v) for k, v in out.items()},
               "recovered": {"pairs": len(out["recovered"]), "words": len(rec_words),
                             "families": len({fam[w] for w in rec_words}),
                             "new_words_not_in_control": len(rec_words - control_words),
                             "alternatives_of_control_words": sum(r["alternative_of_control_word"] for r in out["recovered"]),
                             "families_shared_with_control": len({fam[w] for w in rec_words} & {fam[w] for w in control_words}),
                             "rules": dict(rule_counts),
                             "sources": dict(Counter(s for r in out["recovered"] for s in r["sources"]))},
               "protection": prot["counts"], "audit_samples": samples}
    write_json(REPORTS / "recovery_summary.json", summary)
    return {k: v for k, v in summary.items() if k != "audit_samples"}


# ------------------------------------------------------------------ source pilots (read-only, bounded)
def alt_pilot(n: int = 100) -> dict:
    """ALT Parallel Corpus (CC BY 4.0): English name tokens with an exact Khmer match among letters-model guesses.

    The letters model is only a search constraint: an attested pair is a translator's spelling found verbatim, then
    checked by the frozen consonant screen. A pilot of `n` sentence pairs in sha256 order; nothing is accepted here.
    """
    import e02_diagnostics as DG
    km_lex, lat_lex = DG.load_lexicon()
    cmu, _ = ps.cmudict()
    src = DATA / "sources" / "alt"
    en = dict(ln.split("\t", 1) for ln in (src / "data_en.txt").read_text(encoding="utf-8").splitlines() if "\t" in ln)
    km = dict(ln.split("\t", 1) for ln in (src / "data_khm.txt").read_text(encoding="utf-8").splitlines() if "\t" in ln)
    ids = sorted(set(en) & set(km), key=lambda i: order("e08-alt", i))[:n]
    toks = []
    for i in ids:
        words = re.findall(r"[A-Za-z]+", en[i])
        for j, t in enumerate(words):
            if j > 0 and t[0].isupper() and t[1:].islower() and t.lower() in cmu:
                toks.append((i, t.lower()))
    guesses = e06.g_predict("G_s0", [t for _, t in toks])
    hits = []
    for (i, t), g in zip(toks, guesses):
        found = [c for c in g if len(c) >= 2 and c in km[i]]
        if found:
            scr = DG.compat_row(t, found[0], tuple(cmu[t][0]), cmu[t], km_lex, lat_lex)
            hits.append({"sentence": i, "english": t, "khmer": found[0], "screen": scr.get("category"),
                         "level": scr.get("level")})
    return {"sentences": len(ids), "capitalised_cmudict_tokens": len(toks), "exact_hits": len(hits),
            "screen_compatible": sum(h["screen"] == "compatible_evidence" for h in hits), "hits": hits}


def google_pilot(n: int = 100) -> dict:
    """Google Khmer lexicon (CC BY 4.0, local): Latin-script entries whose reading equals a Khmer-script entry's."""
    import e02_diagnostics as DG
    km_lex, lat_lex = DG.load_lexicon()
    cmu, _ = ps.cmudict()
    by_read = defaultdict(set)
    for k, rs in km_lex.items():
        for r in rs:
            by_read[r].add(k)
    lat = sorted(lat_lex, key=lambda w: order("e08-google", w))[:n]
    rows = []
    for w in lat:
        m = sorted({k for r in lat_lex[w] for k in by_read.get(r, ())})
        rows.append({"latin": w, "readings": lat_lex[w], "khmer_same_reading": m[:5], "n_matches": len(m),
                     "in_cmudict": w in cmu})
    return {"latin_entries_total": len(lat_lex), "khmer_entries_total": len(km_lex), "sampled": len(rows),
            "with_exact_reading_match": sum(r["n_matches"] > 0 for r in rows),
            "with_unique_match": sum(r["n_matches"] == 1 for r in rows),
            "in_cmudict": sum(r["in_cmudict"] for r in rows), "rows": rows}


def khmerologie_pilot() -> dict:
    """The Khmerologie loanword PDF: text extraction quality and entry shape (no reuse terms → lead only)."""
    txt = (DATA / "sources" / "khmerologie" / "v3.txt").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in txt.splitlines() if re.search("[ក-៿]", ln)]
    with_paren_en = [ln for ln in lines if re.search(r"\([a-z][a-z ]+\)", ln)]
    write_json(DATA / "sources" / "khmerologie" / "inspected_lines.json", lines[:100])   # local only: no reuse terms
    return {"pdf_sha256": e06.sha(DATA / "sources" / "khmerologie" / "mots-anglais-en-khmer-v3.pdf"),
            "khmer_lines": len(lines), "lines_with_parenthesised_english": len(with_paren_en),
            "inspected_lines": "100 lines inspected; kept locally in data/e08/sources/khmerologie/inspected_lines.json "
                               "(no reuse terms)"}


STOP = set("""the a an and or of in on at to for by with from as is was are were be been his her its it this that
these those he she they we you i our their new old one two first last mr mrs ms dr st no not all any but so if""".split())
DEPENDENT = re.compile(r"^[ា-៓]")         # a match must not be followed by a vowel sign, sign or coeng


def alt_attested() -> dict:
    """Attested ALT pairs over the whole corpus (CC BY 4.0), with the pilot's spurious-match fixes registered here:
    a function-word stoplist, Khmer matches of ≥ 3 characters, and a match not followed by a dependent sign.

    Acceptance is the frozen screen and the recovery rules (`recovery_decision`), never the letters model, which
    only proposes the strings that are searched for. Pairs are separate from `recovered` (provenance `alt_cc_by`).
    """
    import e02_diagnostics as DG
    import e03_data
    import english_phonetic_data as ED
    km_lex, lat_lex = DG.load_lexicon()
    cmu, _ = ps.cmudict()
    prot = protected()
    src = DATA / "sources" / "alt"
    en = dict(ln.split("\t", 1) for ln in (src / "data_en.txt").read_text(encoding="utf-8").splitlines() if "\t" in ln)
    km = dict(ln.split("\t", 1) for ln in (src / "data_khm.txt").read_text(encoding="utf-8").splitlines() if "\t" in ln)
    occ = defaultdict(list)
    for i in sorted(set(en) & set(km)):
        for j, t in enumerate(re.findall(r"[A-Za-z]+", en[i])):
            w = t.lower()
            if j > 0 and t[0].isupper() and t[1:].islower() and w in cmu and len(w) >= 3 and w not in STOP:
                occ[w].append(i)
    words = sorted(occ)
    guesses = {}
    for k in range(0, len(words), 200):                  # chunks: the NumPy beam holds a whole batch in memory
        guesses |= dict(zip(words[k:k + 200], e06.g_predict("G_s0", words[k:k + 200])))
    control = {r["word"] for r in read_jsonl(DATA / "recovery" / "control.jsonl")}
    known_pairs = {(r["word"], ps.nfc(r["khmer"])) for r in read_jsonl(DATA / "recovery" / "control.jsonl")} | \
                  {(r["word"], r["target"]) for r in read_jsonl(DATA / "recovery" / "recovered.jsonl")}
    held_targets = prot["held_targets"] | {ps.nfc(a) for s in ("dev", "eval") for x in
                                           e06.guard(E03 / f"{s}_references.jsonl").read_text(encoding="utf-8").splitlines()
                                           for a in json.loads(x)["accepted"]}
    support = Counter()
    for w in words:
        for c in guesses[w]:
            if len(c) < 3:
                continue
            for i in occ[w]:
                k = km[i].find(c)
                if k >= 0 and not DEPENDENT.match(km[i][k + len(c):]):
                    support[(w, c)] += 1
    rows, counts = [], Counter()
    for (w, c), n in sorted(support.items()):
        why = ("protected_family" if is_protected(w, prot) else "target_in_dev_eval_or_heldout" if c in held_targets
               else "exact_pair_already_present" if (w, c) in known_pairs else None)
        if why:
            counts[f"excluded:{why}"] += 1
            continue
        readings = cmu[w]
        scr = DG.compat_row(w, c, tuple(readings[0]), readings, km_lex, lat_lex)
        rec = {"word": w, "target": c, "source": "alt_parallel_cc_by_4.0", "sentences": n,
               "example_sentence_id": occ[w][0], "screen": {k: scr.get(k) for k in ("category", "level", "ops", "reasons")},
               "pronunciation": {"supplied": " ".join(readings[0])},
               "compound_extra_component": e03_data.compound_extra_component(w, c, km_lex, tuple(readings[0])),
               "possible_translation": ED.rule_similarity(w, c) < e03_data.TRANSLATION_SIM,
               "alternative_of_control_word": w in control}
        cat, rules = recovery_decision(rec)
        rec |= {"category": cat, "acceptance_rule": rules}
        counts[cat] += 1
        rows.append(rec)
    for cat in ("recovered", "uncertain", "error_or_translation"):
        write_jsonl(DATA / "attested" / f"alt_{'eligible' if cat == 'recovered' else cat}.jsonl",
                    [r for r in rows if r["category"] == cat])
    elig = [r for r in rows if r["category"] == "recovered"]
    sample = sorted(elig, key=lambda r: order("e08-alt-audit", r["word"] + r["target"]))[:40]
    out = {"tokens": len(words), "candidate_pairs": len(support), "counts": dict(counts),
           "eligible": {"pairs": len(elig), "words": len({r["word"] for r in elig}),
                        "new_words_not_in_control": len({r["word"] for r in elig} - control)},
           "audit_sample": [{k: r[k] for k in ("word", "target", "sentences", "acceptance_rule")} for r in sample]}
    write_json(REPORTS / "alt_attested_summary.json", out)
    return {k: v for k, v in out.items() if k != "audit_sample"}


def sources() -> dict:
    out = {"alt_parallel": alt_pilot(), "google_lexicon": google_pilot(), "khmerologie": khmerologie_pilot()}
    write_json(REPORTS / "source_pilots.json", out)
    return {k: {a: b for a, b in v.items() if not isinstance(b, list)} for k, v in out.items()}


# ------------------------------------------------------------------ synthetic input manifest (frozen)
ARPA_IPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
            "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d",
            "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
            "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z",
            "ZH": "ʒ"}


def ipa(arpa: str) -> str:
    """Broad IPA for a CMUdict reading (stress: ˈ primary, ˌ secondary; unstressed AH → ə).

    Examples:
        >>> ipa("K AA1 F IY0")
        'kˈɑfi'
    """
    out = []
    for p in arpa.split():
        base, stress = p.rstrip("012"), p[len(p.rstrip("012")):]
        sym = "ə" if base == "AH" and stress == "0" else ARPA_IPA[base]
        out.append(("ˈ" if stress == "1" else "ˌ" if stress == "2" else "") + sym)
    return "".join(out)


def web_counts(words: set[str]) -> Counter:
    """Lower-case occurrences of the candidate words as Latin tokens in the local FineWeb-2 Khmer shard (cached).

    A domain frequency proxy for "common everyday" (English words that Khmer web text actually uses)."""
    cache = DATA / "fineweb_latin_counts.json"
    if cache.exists():
        return Counter(json.loads(cache.read_text()))
    import pyarrow.parquet as pq
    c = Counter()
    pf = pq.ParquetFile(e06.guard(ROOT / "data/src/fineweb2_km/train_000_00000.parquet"))
    for b in pf.iter_batches(columns=["text"], batch_size=20000):
        for t in b.column(0).to_pylist():
            for tok in re.findall(r"[A-Za-z]+", t):
                tok = tok.lower()
                if tok in words:
                    c[tok] += 1
    write_json(cache, dict(c))
    return c


def clusters(phones: tuple[str, ...]) -> bool:
    """An initial or final run of ≥ 2 consonants (e.g. S T R…, …N K S)."""
    v = [p.rstrip("012") in VOWELS for p in phones]
    first = v.index(True) if True in v else len(v)
    last = len(v) - 1 - v[::-1].index(True) if True in v else -1
    return first >= 2 or len(v) - 1 - last >= 2


def stressed_vowel(phones: tuple[str, ...]) -> str | None:
    for p in phones:
        if p.endswith("1"):
            return p[:-1]
    return None


def is_abbreviation(word: str, phones: tuple[str, ...]) -> bool:
    """Letters read as an expansion or spelled out (amendment 1, before any output): no vowel letter, or a reading
    longer than the spelling + 2 phones.

    Examples:
        >>> is_abbreviation("blvd", ("B", "UH1", "L")), is_abbreviation("etc", tuple("EH2 T S EH1 T ER0 AH0".split()))
        (True, True)
        >>> is_abbreviation("coffee", ("K", "AA1", "F", "IY0"))
        False
    """
    return not re.search(r"[aeiouy]", word) or len(phones) > len(word) + 2


def select_inputs() -> dict:
    """The 200 synthetic inputs, before any AI Mode output: fresh, unprotected, unambiguous, one word per family."""
    cmu, _ = ps.cmudict()
    prot = protected()
    hist, _ = e06.historical_universe()
    hist_stems = e06.stem_set(hist)
    real = {r["word"] for f in ("recovered", "uncertain", "error_or_translation")
            for r in read_jsonl(DATA / "recovery" / f"{f}.jsonl")}
    real |= {r["word"] for f in ("eligible", "uncertain", "error_or_translation")
             for r in read_jsonl(DATA / "attested" / f"alt_{f}.jsonl")}
    romanised = set((ROOT / "models/model/romanised_forms.txt").read_text(encoding="utf-8").split())
    lower, names = e06.candidate_words()
    excluded, pool = Counter(), []
    for w, kind in sorted([(w, "lower") for w in lower] + [(w, "name") for w in names]):
        if w not in cmu or not 3 <= len(w) <= 14:
            excluded["no_cmudict_or_length"] += 1
        elif is_protected(w, prot):
            excluded["protected_family"] += 1
        elif not e06.is_fresh(w, hist_stems) or w in real:
            excluded["historical_family"] += 1
        elif w in romanised:
            excluded["romanised_khmer_form"] += 1
        elif is_abbreviation(w, cmu[w][0]):
            excluded["abbreviation_like"] += 1
        elif e06.is_ambiguous(cmu[w]):
            excluded["ambiguous_reading_quarantined"] += 1
        else:
            pool.append((w, kind))
    freq = web_counts({w for w, _ in pool})
    used_fam: set[str] = set()
    picked: dict[str, list] = {c: [] for c in CATEGORIES}
    vowel_seen = Counter()

    def take(cat: str, cands: list[str], key) -> None:
        for w in sorted(cands, key=key):
            if len(picked[cat]) >= CATEGORIES[cat]:
                return
            fam = min(rd.stems(w)) if rd.stems(w) else w
            if used_fam & rd.stems(w) or fam in used_fam:
                continue
            if cat == "vowels":
                v = stressed_vowel(cmu[w][0])
                if v is None or vowel_seen[v] >= -(-CATEGORIES[cat] // len(VOWELS)):
                    continue
                vowel_seen[v] += 1
            used_fam.update(rd.stems(w) or {w})
            picked[cat].append(w)

    lw = [w for w, k in pool if k == "lower"]
    nm = [w for w, k in pool if k == "name"]
    sh = lambda c: (lambda w: order(f"{SEED}:{c}", w))  # noqa: E731
    take("names", nm, sh("names"))
    take("irregular", [w for w in lw if e06.is_difficult(w, cmu[w][0])], sh("irregular"))
    take("common", [w for w in lw if freq[w] >= 5 and 3 <= len(w) <= 8], lambda w: (-freq[w], w))
    take("clusters", [w for w in lw if clusters(cmu[w][0])], sh("clusters"))
    take("vowels", lw, sh("vowels"))
    rows, n = [], 0
    for cat in CATEGORIES:
        for w in picked[cat]:
            n += 1
            ph = " ".join(cmu[w][0])
            rows.append({"id": f"e08s{n:03d}", "english": w, "category": cat, "pronunciation_arpabet": ph,
                         "pronunciation_ipa": ipa(ph), "pronunciation_source": "CMUdict (first reading; single reading "
                         "up to stress)", "selection": f"{cat}: " + ("FineWeb-2 km Latin-token count "
                         f"{freq[w]} (descending)" if cat == "common" else f"sha256({SEED}:{cat}:word) order"),
                         "fineweb_count": freq[w]})
    order_rows = sorted(rows, key=lambda r: order(f"{SEED}:batches", r["id"]))
    for k, r in enumerate(order_rows):
        r["batch"] = k // PER_BATCH + 1
    rows.sort(key=lambda r: r["id"])
    cover = Counter(stressed_vowel(tuple(r["pronunciation_arpabet"].split())) for r in rows)
    phones = Counter(p.rstrip("012") for r in rows for p in r["pronunciation_arpabet"].split())
    return {"rows": rows, "excluded": dict(excluded), "eligible_pool": len(pool),
            "per_category": {c: len(v) for c, v in picked.items()}, "stressed_vowel_coverage": dict(cover),
            "phoneme_coverage": {p: phones.get(p, 0) for p in ARPA_IPA}, "protection": prot["counts"]}


def inputs() -> dict:
    """Write the frozen manifest once; later runs verify its hash and membership rules instead of rebuilding."""
    frozen = REPORTS / "input_manifest_frozen.json"
    if MANIFEST.exists() and frozen.exists():
        f = json.loads(frozen.read_text())
        if e06.sha(MANIFEST) != f["sha256"]:
            raise RuntimeError("input_manifest.jsonl differs from its frozen hash")
        return {"frozen": True, **{k: v for k, v in f.items() if k != "excluded"}}
    sel = select_inputs()
    if len(sel["rows"]) != sum(CATEGORIES.values()):
        raise RuntimeError(f"only {len(sel['rows'])} inputs selected: {sel['per_category']}")
    write_jsonl(MANIFEST, sel["rows"])
    info = {"frozen_at": e06.now(), "sha256": e06.sha(MANIFEST), "n": len(sel["rows"]),
            **{k: v for k, v in sel.items() if k != "rows"}}
    write_json(frozen, info)
    return info


def load_manifest() -> list[dict]:
    rows = read_jsonl(MANIFEST)
    f = json.loads((REPORTS / "input_manifest_frozen.json").read_text())
    if e06.sha(MANIFEST) != f["sha256"]:
        raise RuntimeError("input_manifest.jsonl differs from its frozen hash")
    return rows


# ------------------------------------------------------------------ prompts
POLICY = """## Task
Write each English word below in Khmer script so that a Khmer reader would pronounce it like the given
**English** pronunciation. This is phonetic spelling (transliteration), not translation.

## Rules
1. Spell the sound, not the meaning. Never give a Khmer translation or a native Khmer word that means the same
   thing.
2. The pronunciation column (US English; ARPAbet with stress digits, and IPA) is the intended reading. Follow it,
   even for names: do not switch to the name's pronunciation in another language.
3. If an established Khmer loanword spelling exists and matches this pronunciation, you may use it.
4. Give one preferred spelling. Optionally give up to two alternative Khmer spellings **of the same
   pronunciation** (not a different pronunciation, not a translation).
5. If you are not sure, write `yes` in the `uncertain` column. You may then leave the spellings empty or give your
   best attempt.
6. Keep every id exactly as given: one row per id, in any order, and no extra rows.
7. Spelling columns contain Khmer script only: no Latin letters, digits, brackets, quotation marks or
   explanations. Put a short comment (optional, a few words) in `note` only.
8. Do not invent sources or confidence scores.

## Answer format
Return only this Markdown table, with no text before or after it:

| id | english | preferred_khmer | alternative_1 | alternative_2 | uncertain | note |
|---|---|---|---|---|---|---|
"""


def render_prompt(rows: list[dict], title: str) -> str:
    """One standalone prompt for a list of manifest rows.

    Examples:
        >>> p = render_prompt([{"id": "e08s001", "english": "coffee", "pronunciation_arpabet": "K AA1 F IY0",
        ...                     "pronunciation_ipa": "kˈɑfi"}], "test")
        >>> "| e08s001 | coffee | K AA1 F IY0 | /kˈɑfi/ |" in p, "not translation" in p
        (True, True)
    """
    body = "\n".join(f"| {r['id']} | {r['english']} | {r['pronunciation_arpabet']} | /{r['pronunciation_ipa']}/ |"
                     for r in rows)
    return (f"# Khmer phonetic spelling of English words ({title})\n\n{POLICY}\n## Words ({len(rows)})\n\n"
            "| id | english | pronunciation (ARPAbet) | pronunciation (IPA) |\n|---|---|---|---|\n" + body + "\n")


def prompts() -> dict:
    rows = load_manifest()
    PROMPTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for b in range(1, BATCHES + 1):
        sel = sorted([r for r in rows if r["batch"] == b], key=lambda r: r["id"])
        p = PROMPTS / f"batch_{b:02d}.md"
        p.write_text(render_prompt(sel, f"batch {b:02d} of {BATCHES}"), encoding="utf-8")
        out[p.name] = {"items": len(sel), "sha256": e06.sha(p)}
    RESPONSES.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "prompts_frozen.json", out)
    return out


# ------------------------------------------------------------------ importer
KHMER = re.compile(r"^[ក-៿᧠-᧿]+$")
YES, NO = {"yes", "y", "true", "1", "✓", "✔", "x", "uncertain"}, {"no", "n", "false", "0", "", "-", "—"}


def split_row(line: str) -> list[str] | None:
    """Cells of one table line (Markdown pipes or tabs); None for separators and non-table lines.

    Examples:
        >>> split_row("| e08s001 | coffee | កាហ្វេ | | | no | |")
        ['e08s001', 'coffee', 'កាហ្វេ', '', '', 'no', '']
        >>> split_row("e08s001\\tcoffee\\tកាហ្វេ"), split_row("|---|---|") is None
        (['e08s001', 'coffee', 'កាហ្វេ'], True)
    """
    s = line.strip()
    if "|" in s:
        s = s.strip("|")
        cells = [c.strip() for c in s.split("|")]
    elif "\t" in s:
        cells = [c.strip() for c in s.split("\t")]
    else:
        return None
    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return None
    return cells


def clean_cell(c: str) -> str:
    """Remove Markdown emphasis, code ticks and surrounding quotes."""
    c = c.strip()
    c = re.sub(r"^[*_`\"'“”‘’]+|[*_`\"'“”‘’]+$", "", c).strip()
    return c


def check_spelling(raw: str) -> tuple[str | None, str]:
    """`(nfc_text, status)`: `empty`, `ok`, `explanatory_text`, `non_khmer`, `too_long` or `malformed_orthography`.

    Examples:
        >>> check_spelling("កាហ្វេ"), check_spelling(""), check_spelling("កាហ្វេ (coffee)")[1]
        (('កាហ្វេ', 'ok'), (None, 'empty'), 'explanatory_text')
    """
    c = clean_cell(raw).replace("​", "")
    if not c or c in ("-", "—", "–", "N/A", "n/a", "none"):
        return None, "empty"
    if re.search(r"[A-Za-z0-9()\[\]/,;:=]", c) or " " in c:
        return None, "explanatory_text"
    t = ps.nfc(c)
    if not KHMER.match(t):
        return None, "non_khmer"
    if len(t) > MAX_CHARS:
        return None, "too_long"
    if e06.orthography(t) != "ok":
        return None, "malformed_orthography"
    return t, "ok"


def parse_response(text: str, batch_ids: set[str], manifest: dict[str, dict]) -> dict:
    """Records by id from one response; every problem is recorded, nothing is guessed."""
    found: dict[str, list[list[str]]] = defaultdict(list)
    stray = []
    for line in text.splitlines():
        cells = split_row(line)
        if not cells:
            continue
        cid = clean_cell(cells[0]).lower()
        if cid == "id":
            continue
        if ID_RE.match(cid):
            found[cid].append(cells)
        elif re.search(r"e08s\d", line):
            stray.append(line)
    recs, problems = {}, []
    for cid, rows in found.items():
        if cid not in manifest:
            problems.append({"id": cid, "problem": "unknown_id"})
            continue
        if cid not in batch_ids:
            problems.append({"id": cid, "problem": "id_from_another_batch"})
            continue
        if len({tuple(r) for r in rows}) > 1:
            problems.append({"id": cid, "problem": "duplicate_conflicting_rows", "raw": rows})
            continue
        recs[cid] = rows[0]
    missing = sorted(batch_ids - set(recs) - {p["id"] for p in problems})
    return {"records": recs, "problems": problems, "missing": missing, "unparsed_id_lines": stray}


def judge(cells: list[str], m: dict) -> dict:
    """One parsed row → an accepted or quarantined record (raw cells kept)."""
    rec = {"id": m["id"], "english": m["english"], "category": m["category"],
           "pronunciation_arpabet": m["pronunciation_arpabet"], "source": SOURCE, "raw_cells": cells}
    if len(cells) < 6:
        return rec | {"status": "quarantined", "reason": "too_few_columns"}
    if clean_cell(cells[1]).lower() != m["english"].lower():
        return rec | {"status": "quarantined", "reason": "english_mismatch"}
    unc = clean_cell(cells[5]).lower()
    if unc not in YES | NO:
        return rec | {"status": "quarantined", "reason": "uncertain_field_unreadable"}
    pref, pst = check_spelling(cells[2])
    alts = [check_spelling(c) for c in cells[3:5]]
    rec |= {"preferred": pref, "preferred_status": pst,
            "alternatives": [t for t, s in alts if s == "ok" and t != pref],
            "alternative_statuses": [s for _, s in alts], "uncertain": unc in YES,
            "note": clean_cell(cells[6]) if len(cells) > 6 else ""}
    if rec["uncertain"]:
        return rec | {"status": "quarantined", "reason": "marked_uncertain"}
    if pst != "ok":
        return rec | {"status": "quarantined", "reason": f"preferred_{pst}"}
    return rec | {"status": "accepted"}


def import_responses(directory: Path) -> dict:
    """Import every `batch_NN.*` / `regen_NN.*` response file in `directory`.

    Raw files are copied to `data/e08/synthetic/raw/`. Accepted records are never regenerated, and a later file
    cannot overwrite an accepted record. Writes `accepted.jsonl`, `quarantine.jsonl` and `import_summary.json`
    (local), `reports/import_summary.json` (counts only), `reports/needs_regeneration.md`, and regeneration prompts.
    """
    manifest = {r["id"]: r for r in load_manifest()}
    prot = protected()
    bad = [r["english"] for r in manifest.values() if is_protected(r["english"], prot)]
    if bad:
        raise RuntimeError(f"manifest words now overlap protected families: {bad[:5]}")
    files = sorted(p for p in Path(directory).glob("*") if re.match(r"(batch|regen)_\d\d", p.name) and p.is_file())
    (IMPORTED / "raw").mkdir(parents=True, exist_ok=True)
    accepted: dict[str, dict] = {}
    quarantine: dict[str, dict] = {}
    per_file = {}
    covered: set[str] = set()                    # ids of batches that have a non-empty response file
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            per_file[p.name] = {"skipped": "empty file (not filled in yet)"}
            continue
        shutil.copyfile(p, IMPORTED / "raw" / p.name)
        kind, num = p.name[:5], int(p.name[6:8])
        if kind == "batch":
            ids = {i for i, r in manifest.items() if r["batch"] == num}
        else:
            reg = PROMPTS / f"regen_{num:02d}.ids.json"
            ids = set(json.loads(reg.read_text())) if reg.exists() else set()
        covered |= ids
        res = parse_response(text, ids, manifest)
        stats = Counter()
        for cid, cells in res["records"].items():
            if cid in accepted:
                stats["already_accepted_kept"] += 1
                continue
            j = judge(cells, manifest[cid]) | {"file": p.name, "file_sha256": e06.sha(p)}
            if j["status"] == "accepted":
                accepted[cid] = j
                quarantine.pop(cid, None)
            else:
                quarantine[cid] = j
            stats[j["status"] if j["status"] == "accepted" else j["reason"]] += 1
        for pr in res["problems"]:
            stats[pr["problem"]] += 1
            if pr["id"] in manifest and pr["id"] not in accepted:
                quarantine[pr["id"]] = {"id": pr["id"], "english": manifest[pr["id"]]["english"], "source": SOURCE,
                                        "status": "quarantined", "reason": pr["problem"], "file": p.name}
        per_file[p.name] = {"batch_items": len(ids), "rows_found": len(res["records"]), "missing": len(res["missing"]),
                            "unparsed_id_lines": len(res["unparsed_id_lines"]), **stats}
    missing = sorted(covered - set(accepted) - set(quarantine))
    not_provided = sorted(set(manifest) - covered - set(accepted) - set(quarantine))
    regenerate = sorted([i for i in quarantine if quarantine[i]["reason"] != "marked_uncertain"] + missing)
    write_jsonl(IMPORTED / "accepted.jsonl", [accepted[i] for i in sorted(accepted)])
    write_jsonl(IMPORTED / "quarantine.jsonl", [quarantine[i] for i in sorted(quarantine)])
    summary = {"imported_at": e06.now(), "files": per_file, "manifest_items": len(manifest),
               "accepted": len(accepted), "accepted_alternatives": sum(len(r["alternatives"]) for r in accepted.values()),
               "quarantined": dict(Counter(q["reason"] for q in quarantine.values())), "missing": len(missing),
               "needs_regeneration": len(regenerate), "source": SOURCE,
               "batches_not_provided": sorted({manifest[i]["batch"] for i in not_provided}),
               "accepted_per_category": dict(Counter(r["category"] for r in accepted.values())),
               "state": state_name(len(covered), len(accepted), regenerate, not_provided)}
    write_json(IMPORTED / "import_summary.json", summary)
    write_json(REPORTS / "import_summary.json", summary)
    regen_prompts(regenerate, manifest, quarantine)
    set_state(summary["state"], summary)
    return summary


def state_name(n_covered: int, n_accepted: int, regenerate: list[str], not_provided: list[str] = ()) -> str:
    if n_covered == 0 or not_provided:
        return "awaiting_ai_mode_responses"
    return "awaiting_regeneration_responses" if regenerate else "imported_ready_to_freeze"


def regen_prompts(ids: list[str], manifest: dict, quarantine: dict) -> None:
    """Replacement prompts for missing or malformed items only (accepted and uncertain items are not asked again)."""
    for old in PROMPTS.glob("regen_*"):
        old.unlink()
    lines = ["# Items needing regeneration", "", "| id | english | reason |", "|---|---|---|"]
    lines += [f"| {i} | {manifest[i]['english']} | {quarantine.get(i, {}).get('reason', 'missing')} |" for i in ids]
    (REPORTS / "needs_regeneration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k in range(0, len(ids), PER_BATCH):
        chunk = ids[k:k + PER_BATCH]
        n = k // PER_BATCH + 1
        (PROMPTS / f"regen_{n:02d}.md").write_text(
            render_prompt([manifest[i] for i in chunk], f"replacement batch {n:02d}"), encoding="utf-8")
        write_json(PROMPTS / f"regen_{n:02d}.ids.json", chunk)


# ------------------------------------------------------------------ training arms (data only; no training here)
TRAIN = DATA / "training"
ARM_COLS = ("word", "khmer", "provenance", "phones", "part")


def arm_rows(arm: str) -> list[dict]:
    """Rows of one arm: C = control; R = C + recovered + ALT-attested; RS = R + accepted synthetic.

    Train-val rows (E03 `train_val`) stay in every arm unchanged, for the loss tie-break only.
    """
    control = read_jsonl(DATA / "recovery" / "control.jsonl")
    rows = [{"word": r["word"], "khmer": r["khmer"], "provenance": "control", "phones": r["phones"],
             "part": r["part"]} for r in control]
    if arm in ("R", "RS"):
        rows += [{"word": r["word"], "khmer": r["target"], "provenance": "recovered",
                  "phones": (r.get("pronunciation") or {}).get("supplied", ""), "part": "train"}
                 for r in read_jsonl(DATA / "recovery" / "recovered.jsonl")]
        rows += [{"word": r["word"], "khmer": r["target"], "provenance": "attested_alt",
                  "phones": r["pronunciation"]["supplied"], "part": "train"}
                 for r in read_jsonl(DATA / "attested" / "alt_eligible.jsonl")]
    if arm == "RS":
        for r in read_jsonl(IMPORTED / "accepted.jsonl"):
            for k in [r["preferred"], *r["alternatives"]]:
                rows.append({"word": r["english"], "khmer": k, "provenance": SOURCE,
                             "phones": r["pronunciation_arpabet"], "part": "train"})
    tv = {r["word"] for r in control if r["part"] == "train_val"}
    tv_stems = {s for w in tv for s in rd.stems(w)}
    seen, out = set(), []
    for r in rows:                                # exact pairs deduplicated; the first provenance is kept
        if r["provenance"] != "control" and (r["word"] in tv or rd.related(r["word"], tv_stems)):
            continue                              # keep E03 train-val families unseen (diagnostic set)
        if (r["word"], r["khmer"]) not in seen:
            seen.add((r["word"], r["khmer"]))
            out.append(r)
    return out


def build_arms(arms=("C", "R")) -> dict:
    """Write `data/e08/training/arm_<A>.tsv` and re-check protection, dev/eval overlap and exact pairs."""
    prot = protected()
    held = {json.loads(x)["word"] for s in ("dev", "eval")
            for x in e06.guard(E03 / f"{s}_references.jsonl").read_text(encoding="utf-8").splitlines()}
    info = {}
    for a in arms:
        rows = arm_rows(a)
        new = [r for r in rows if r["provenance"] != "control"]
        bad = [r["word"] for r in new if is_protected(r["word"], prot) or r["word"] in held]
        if bad:
            raise RuntimeError(f"arm {a}: protected words {bad[:5]}")
        p = TRAIN / f"arm_{a}.tsv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\t".join(ARM_COLS) + "\n" + "".join("\t".join(r[c] for c in ARM_COLS) + "\n" for r in rows),
                     encoding="utf-8")
        opt = [r for r in rows if r["part"] == "train"]
        info[a] = {"path": str(p.relative_to(ROOT)), "sha256": e06.sha(p), "rows": len(rows),
                   "optimisation_rows": len(opt), "optimisation_words": len({r["word"] for r in opt}),
                   "by_provenance": dict(Counter(r["provenance"] for r in rows)),
                   "words_with_alternatives": sum(n > 1 for n in Counter(r["word"] for r in opt).values())}
    write_json(REPORTS / "training_arms.json", json.loads((REPORTS / "training_arms.json").read_text()) | info
               if (REPORTS / "training_arms.json").exists() else info)
    return info


def freeze() -> dict:
    """After the import: build arm RS and freeze every arm's hash (the precondition for the E08 training plan)."""
    st = json.loads((EXP / "state.json").read_text())["state"]
    if st == "awaiting_ai_mode_responses":
        raise RuntimeError("no AI Mode responses imported yet")
    info = build_arms(("C", "R", "RS"))
    summ = json.loads((IMPORTED / "import_summary.json").read_text())
    out = {"frozen_at": e06.now(), "arms": info, "synthetic_import_state": st,
           "synthetic_quarantined_left": summ["quarantined"], "synthetic_missing_left": summ["missing"],
           "conditions": {"provenance_records": True, "protected_overlap": "none (checked at build)",
                          "structural_import": st, "uncertain_and_malformed": "quarantined",
                          "policy": "DATA_POLICY.md", "selection_and_filtering": "scripts/e08.py"}}
    write_json(REPORTS / "data_frozen.json", out)
    set_state("data_frozen_ready_for_registration", {"arms": {a: v["sha256"] for a, v in info.items()}})
    return out


# ------------------------------------------------------------------ state
def set_state(name: str, extra: dict | None = None) -> None:
    write_json(EXP / "state.json", {"state": name, "updated": e06.now(), **({"detail": extra} if extra else {})})


def prepare() -> dict:
    out = {}
    if not (REPORTS / "recovery_summary.json").exists():
        out["recover"] = recover()
    if not (REPORTS / "source_pilots.json").exists():
        out["sources"] = sources()
    if not (REPORTS / "alt_attested_summary.json").exists():
        out["attested"] = alt_attested()
    out["inputs"] = inputs()
    out["prompts"] = prompts()
    out["arms"] = build_arms(("C", "R"))
    have = [p for p in RESPONSES.glob("*") if re.match(r"(batch|regen)_\d\d", p.name)]
    if have:
        out["import"] = import_responses(RESPONSES)
    else:
        set_state("awaiting_ai_mode_responses", {"save_responses_to": str(RESPONSES.relative_to(ROOT)),
                                                 "files": [f"batch_{b:02d}.md" for b in range(1, BATCHES + 1)]})
        out["state"] = "awaiting_ai_mode_responses"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["prepare", "import", "status", "recover", "sources", "attested", "inputs", "prompts", "freeze"])
    ap.add_argument("--responses", type=Path, default=RESPONSES)
    a = ap.parse_args()
    if a.step == "import":
        r = import_responses(a.responses)
    elif a.step == "status":
        r = json.loads((EXP / "state.json").read_text())
    else:
        r = {"prepare": prepare, "recover": recover, "sources": sources, "attested": alt_attested, "inputs": inputs, "prompts": prompts,
             "freeze": freeze}[a.step]()
    print(json.dumps(r, ensure_ascii=False, default=str)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
