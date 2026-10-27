#!/usr/bin/env python3
"""E02 diagnostics (2026-10-27): pronunciation compatibility of the training targets, and phone sensitivity.

    .venv/bin/python scripts/e02_diagnostics.py compat        # §2: training-pair audit (no model)
    .venv/bin/python scripts/e02_diagnostics.py freeze        # §3: freeze the sensitivity sample and settings
    .venv/bin/python scripts/e02_diagnostics.py sensitivity   # §3: run the frozen sample through saved checkpoints

Outputs go to `experiments/E02-diagnostics-2026-10-27/` (`OUT`). Diagnostics only: no training, no new model, no
download. Reads only the inputs in `INPUTS` (training rows, mined files, Google lexicon, CMUdict, the English word
list, dev references, saved checkpoints); every read goes through `guarded`, which refuses the final references,
the E02 result files, benchmark panels and the E01 reserve.

**Shared representation (compat).** English ARPAbet and Khmer forms are compared on **consonants only**, mapped to
broad classes from each symbol's IPA value (`EN_CONS`, `KM_LETTER`, `GOOGLE_CONS`). Khmer targets are read in two
ways, kept apart:
- `google_exact`: the target's own entry in the Google lexicon (the preferred track);
- `orthographic`: consonants read from the Khmer spelling by fixed rules (`km_skeleton`), whose agreement with the
  Google transcriptions of 62k Khmer headwords is measured first (`validate_orthography`). It is a reading of the
  written consonants, not a pronunciation.
Known loan adaptations (`ADAPT`, e.g. English f → Khmer ph) count as compatible; any other substitution, insertion
or deletion is "unexplained". Vowels are not compared. Categories (`compatible_evidence`, `suspected_conflict`,
`unknown`) are audit flags, not linguistic labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import phonetic_source as ps  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "experiments" / "E02-diagnostics-2026-10-27"
SEED = 20261027
SAMPLE_MAX = 200
INPUTS = {
    "train": DATA / "e02" / "train.tsv",
    "mined_kmwiki": DATA / "mined_kmwiki.tsv",
    "mined_fineweb": DATA / "mined_fineweb.tsv",
    "lexicon": DATA / "lexicon" / "km_lexicon.tsv",
    "cmudict": ps.CMUDICT,
    "en_words": DATA / "en_common_words.txt",
    "dev_references": DATA / "e02" / "dev_references.jsonl",
}
CHECKPOINTS = ["phones_s0", "phones_s1", "phones_s2", "letters_s0", "letters_s1", "letters_s2", "phones_p2g_s0"]
FORBIDDEN = ("data/e02/references.jsonl", "data/e02/results", "data/benchmark", "e2/reserve", "reserve.json",
             "experiments/E02-english-phonetic/results.json")


def guarded(path: Path | str) -> Path:
    """Refuse a path the diagnostics must never read (final references, results, benchmark panels, reserve).

    Raises:
        PermissionError: For a forbidden path.

    Examples:
        >>> guarded(ROOT / "data/e02/dev_references.jsonl").name
        'dev_references.jsonl'
    """
    p = Path(path).resolve().as_posix()
    if any(f in p for f in FORBIDDEN):
        raise PermissionError(f"diagnostics may not read {p}")
    return Path(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """A headed TSV file, through the guard."""
    with open(guarded(path), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def sha(path: Path) -> str:
    """sha256 of a file (through the guard)."""
    return hashlib.sha256(guarded(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- shared consonant classes
# Broad classes, from the IPA value of each symbol. Voicing and aspiration are merged inside a class.
LAB, COR, VEL, PAL, LABF, DENT, SIB, POST, GLOT = "LAB", "COR", "VEL", "PAL", "LABF", "DENT", "SIB", "POST", "GLOT"
NM, NN, NP, NG, L, R, W, J = "NM", "NN", "NP", "NG", "L", "R", "W", "J"
EN_CONS = {"P": LAB, "B": LAB, "T": COR, "D": COR, "K": VEL, "G": VEL, "CH": PAL, "JH": PAL, "F": LABF, "V": LABF,
           "TH": DENT, "DH": DENT, "S": SIB, "Z": SIB, "SH": POST, "ZH": POST, "HH": GLOT, "M": NM, "N": NN,
           "NG": NG, "L": L, "R": R, "W": W, "Y": J}
# Known English → Khmer loan adaptations (accepted as compatible, each with its reason).
ADAPT = {
    "F": {LAB: "f→ph (older loans write ផ/ភ for f)"},
    "V": {W: "v→w (Khmer វ is a labial approximant)", LAB: "v→ph/b"},
    "TH": {COR: "θ→t/th", SIB: "θ→s"},
    "DH": {COR: "ð→d/t"},
    "SH": {SIB: "ʃ→s", PAL: "ʃ→c/ch"},
    "ZH": {SIB: "ʒ→s/z", PAL: "ʒ→c"},
    "JH": {SIB: "dʒ→s/z"},
    "N": {NP: "n→ɲ before a front glide"},
    "NG": {NN: "ŋ→n"},
}
# Google lexicon consonant symbols (IPA): ɓ/ɗ implosives, c palatal stop, j palatal glide, w labial approximant.
GOOGLE_CONS = {"p": LAB, "ph": LAB, "ɓ": LAB, "t": COR, "th": COR, "ɗ": COR, "k": VEL, "kh": VEL, "g": VEL,
               "c": PAL, "ch": PAL, "f": LABF, "s": SIB, "z": SIB, "h": GLOT, "m": NM, "n": NN, "ɲ": NP, "ŋ": NG,
               "l": L, "r": R, "w": W, "j": J}
GOOGLE_SILENT = {"ʔ", "."}
# Khmer consonant letters → class (អ carries a vowel: silent here, like Google's ʔ).
KM_LETTER = {**{c: VEL for c in "កខគឃ"}, "ង": NG, **{c: PAL for c in "ចឆជឈ"}, "ញ": NP,
             **{c: COR for c in "ដឋឌឍតថទធ"}, "ណ": NN, "ន": NN, **{c: LAB for c in "បផពភ"}, "ម": NM, "យ": J,
             "រ": R, "ល": L, "ឡ": L, "វ": W, "ស": SIB, "ឝ": SIB, "ឞ": SIB, "ហ": GLOT, "អ": None}
# ហ + subscript: a silent series marker in loans (ហ្វ = f, ហ្គ = g, ហ្ស = z ...).
HO_SERIES = {"វ": LABF, "គ": VEL, "ស": SIB, "ល": L, "ម": NM, "ន": NN, "យ": J, "រ": R, "ង": NG, "ញ": NP, "ប": LAB}
COENG, TOAN, ROBAT, NIKAHIT, REAHMUK = "្", "៍", "៌", "ំ", "ះ"
INDEP = {"ឫ": R, "ឬ": R, "ឭ": L, "ឮ": L}


@dataclass(frozen=True)
class Seg:
    """One consonant position: allowed classes, whether it may be absent, and where it came from."""

    classes: frozenset
    optional: bool = False
    label: str = ""
    why_optional: str = ""


def en_skeleton(phones: tuple[str, ...] | list[str]) -> list[Seg]:
    """English consonant positions from ARPAbet (stress digits allowed).

    ER adds an optional R (the written r of -er); HH is optional; a word-final consonant after another consonant
    is optional (final cluster reduction).

    Examples:
        >>> [s.label for s in en_skeleton("B UW1 S T ER0".split())]
        ['B', 'S', 'T', 'ER:R']
    """
    out: list[Seg] = []
    base = [p.rstrip("012") for p in phones]
    for i, p in enumerate(base):
        if p == "ER":
            out.append(Seg(frozenset({R}), True, "ER:R", "r of -er"))
        elif p in EN_CONS:
            cls = {EN_CONS[p], *ADAPT.get(p, {})}
            opt, why = (True, "h may drop") if p == "HH" else (True, "coda r") if p == "R" and (
                i + 1 == len(base) or base[i + 1] in EN_CONS) else (False, "")
            out.append(Seg(frozenset(cls), opt, p, why))
    if len(base) >= 2 and base[-1] in EN_CONS and base[-2] in EN_CONS and out and not out[-1].optional:
        last = out[-1]
        out[-1] = Seg(last.classes, True, last.label, "final cluster reduction")
    return out


def google_skeleton(phones: str) -> list[Seg]:
    """Consonant positions of a Google-lexicon transcription ("ʔ aa . r o n").

    A j or w that closes a syllable after a vowel (before "." or at the end) is the glide of a diphthong
    (ៃ = "a j", ៅ = "a w"), so it is optional.

    Examples:
        >>> [sorted(s.classes)[0] for s in google_skeleton("ʔ aa . r o n")]
        ['R', 'NN']
        >>> [s.optional for s in google_skeleton("t a j")]
        [False, True]
    """
    toks = phones.split()
    out = []
    for k, t in enumerate(toks):
        if t in GOOGLE_CONS:
            glide = (t in ("j", "w") and k > 0 and toks[k - 1] not in GOOGLE_CONS and toks[k - 1] not in GOOGLE_SILENT
                     and (k + 1 == len(toks) or toks[k + 1] == "."))
            out.append(Seg(frozenset({GOOGLE_CONS[t]}), glide, t, "diphthong glide" if glide else ""))
    return out


def km_skeleton(text: str) -> list[Seg] | None:
    """Consonants read from a Khmer spelling by fixed orthographic rules; None if it cannot be read.

    Rules: subscripts are pronounced; ហ + subscript is the silent series marker of loans (ហ្វ → f); ៍ silences the
    consonant before it; ំ adds m, ះ adds h, ៌ adds r; អ is silent; the word-final ស may be read h and the
    word-final រ may be silent; independent vowels ឫ ឬ ឭ ឮ add r or l.

    Examples:
        >>> [s.label for s in km_skeleton("ហ្វេសប៊ុក")]
        ['ហ្វ', 'ស', 'ប', 'ក']
        >>> [s.label for s in km_skeleton("ឡៃវ៍")]
        ['ឡ']
    """
    text = unicodedata.normalize("NFC", text)
    segs: list[Seg] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in KM_LETTER:
            sub = text[i + 2] if i + 2 < n and text[i + 1] == COENG else ""
            if ch == "ហ" and sub in HO_SERIES:
                segs.append(Seg(frozenset({HO_SERIES[sub]}), False, ch + COENG + sub))
                i += 3
                continue
            if KM_LETTER[ch] is not None:
                segs.append(coda_seg(text, i) or Seg(frozenset({KM_LETTER[ch]}), False, ch))
            i += 1
        elif ch == COENG:
            if i + 1 >= n or text[i + 1] not in KM_LETTER:
                return None
            c = text[i + 1]
            if KM_LETTER[c] is not None:
                segs.append(Seg(frozenset({KM_LETTER[c]}), False, COENG + c))
            i += 2
        elif ch == TOAN:
            if segs:
                segs.pop()
            i += 1
        elif ch == NIKAHIT:
            if text[i + 1:i + 2] != "ង":                  # ាំង: ំ belongs to the vowel before ង
                segs.append(Seg(frozenset({NM, NG}), False, ch))
            i += 1
        elif ch == REAHMUK:
            segs.append(Seg(frozenset({GLOT}), True, ch, "ះ may be unreleased"))
            i += 1
        elif ch == ROBAT:
            segs.append(Seg(frozenset({R}), True, ch, "robat"))
            i += 1
        elif ch in INDEP:
            segs.append(Seg(frozenset({INDEP[ch]}), False, ch))
            i += 1
        elif "ឣ" <= ch <= "ឳ" or "ា" <= ch <= "ៅ" or "ៈ" <= ch <= "៓" or ch == "៝":
            i += 1                                        # vowels and other signs: no consonant
        else:
            return None
    return segs


VOWEL_SIGNS = {chr(c) for c in range(0x17B6, 0x17C6)}


def is_coda(text: str, i: int) -> bool:
    """Whether the base consonant at `i` closes a syllable: word-final, or right after a vowel sign, and not
    followed by a vowel sign, a subscript or ំ/ះ/៍ (series and shortening signs are skipped)."""
    j = i + 1
    while j < len(text) and text[j] in "៉៊់":
        j += 1
    if j < len(text) and (text[j] in VOWEL_SIGNS or text[j] in (COENG, NIKAHIT, REAHMUK, TOAN)):
        return False
    return j == len(text) or (i > 0 and text[i - 1] in VOWEL_SIGNS)


def coda_seg(text: str, i: int) -> Seg | None:
    """Khmer coda readings: a coda ស may be read h, a coda រ may be silent; None otherwise.

    Examples:
        >>> sorted(coda_seg("ប៉ារីស", 5).classes)
        ['GLOT', 'SIB']
        >>> coda_seg("សារ", 0) is None
        True
    """
    ch = text[i]
    if ch not in ("ស", "រ") or not is_coda(text, i):
        return None
    if ch == "ស":
        return Seg(frozenset({SIB, GLOT}), False, ch)
    return Seg(frozenset({R}), True, ch, "coda r silent")


def align(a: list[Seg], b: list[Seg], adapt_labels: bool = True) -> tuple[int, list[tuple[str, str, str]]]:
    """Minimum count of unexplained operations aligning two consonant skeletons, with the operations.

    A position pair matches if their classes intersect; an optional position may be left out at no cost. Each
    operation is `(kind, a_label, b_label)` with kind "same", "adapt:<reason>", "free:<reason>", "sub", "del_a",
    "del_b". `adapt` reasons are looked up for English labels in `ADAPT`.

    Examples:
        >>> align(en_skeleton("F AA1 R".split()), km_skeleton("ហ្វា"))[0]
        0
    """
    na, nb = len(a), len(b)
    INF = 10 ** 9
    D = [[INF] * (nb + 1) for _ in range(na + 1)]
    D[0][0] = 0
    for i in range(na + 1):
        for j in range(nb + 1):
            if i < na:
                D[i + 1][j] = min(D[i + 1][j], D[i][j] + (0 if a[i].optional else 1))
            if j < nb:
                D[i][j + 1] = min(D[i][j + 1], D[i][j] + (0 if b[j].optional else 1))
            if i < na and j < nb:
                D[i + 1][j + 1] = min(D[i + 1][j + 1], D[i][j] + (0 if a[i].classes & b[j].classes else 1))
    ops: list[tuple[str, str, str]] = []
    i, j = na, nb
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (0 if a[i - 1].classes & b[j - 1].classes else 1):
            x, y = a[i - 1], b[j - 1]
            if not x.classes & y.classes:
                kind = "sub"
            else:
                base = EN_CONS.get(x.label)
                inter = x.classes & y.classes
                if adapt_labels and base and base not in inter:
                    kind = "adapt:" + ADAPT.get(x.label, {}).get(sorted(inter)[0], "class")
                else:
                    kind = "same"
            ops.append((kind, x.label, y.label))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + (0 if a[i - 1].optional else 1):
            ops.append(("free:" + a[i - 1].why_optional if a[i - 1].optional else "del_a", a[i - 1].label, ""))
            i -= 1
        else:
            ops.append(("free:" + b[j - 1].why_optional if b[j - 1].optional else "del_b", "", b[j - 1].label))
            j -= 1
    return D[na][nb], ops[::-1]


def load_lexicon() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Google lexicon: `{khmer headword: [transcriptions]}` and `{latin headword (normalised): [transcriptions]}`."""
    km, lat = defaultdict(list), defaultdict(list)
    for line in guarded(INPUTS["lexicon"]).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        w, p = line.split("\t")[:2]
        w = unicodedata.normalize("NFC", w)
        if ps.is_khmer_word(w):
            km[w].append(p)
        else:
            nw, st = ps.normalize_word(w)
            if st == "ok":
                lat[nw].append(p)
    return dict(km), dict(lat)


def lexicon_half(word: str) -> str:
    """"calibration" or "heldout": a fixed hash split of the Khmer headwords. The reading rules were adjusted on
    the calibration half only; the held-out half measures them."""
    return "calibration" if hashlib.sha256(word.encode()).digest()[0] % 2 == 0 else "heldout"


def validate_orthography(km_lex: dict[str, list[str]]) -> dict:
    """Per lexicon half: how often the orthographic consonant reading equals the Google transcription's (cost 0)."""
    return {half: _validate({w: p for w, p in km_lex.items() if lexicon_half(w) == half})
            for half in ("calibration", "heldout")}


def _validate(km_lex: dict[str, list[str]]) -> dict:
    """Agreement overall and for spellings with loan markers (ហ្វ ហ្គ ហ្ស, ៊ ៉, ៍, ័); any transcription may match."""
    loan = ("ហ្វ", "ហ្គ", "ហ្ស", "៉", "៊", "៍", "័")
    tot, ok, unread = Counter(), Counter(), Counter()
    errs: Counter = Counter()
    for w, prons in km_lex.items():
        g = "loan_marked" if any(m in w for m in loan) else "other"
        sk = km_skeleton(w)
        tot[g] += 1
        if sk is None:
            unread[g] += 1
            continue
        best = min((align(sk, google_skeleton(p), adapt_labels=False) for p in prons), key=lambda x: x[0])
        if best[0] == 0:
            ok[g] += 1
        else:
            errs.update(f"{k}:{x}>{y}" for k, x, y in best[1] if k in ("sub", "del_a", "del_b"))
    out = {g: {"headwords": tot[g], "unreadable": unread[g], "consonants_equal": ok[g],
               "rate": ok[g] / max(1, tot[g] - unread[g])} for g in tot}
    out["all"] = {"headwords": sum(tot.values()), "unreadable": sum(unread.values()),
                  "consonants_equal": sum(ok.values()),
                  "rate": sum(ok.values()) / max(1, sum(tot.values()) - sum(unread.values()))}
    out["top_disagreements"] = errs.most_common(25)
    return out


def seg_str(segs: list[Seg] | None) -> str:
    """A readable skeleton: labels joined by spaces (optional ones in brackets)."""
    if segs is None:
        return ""
    return " ".join(f"({s.label})" if s.optional else s.label for s in segs)


def level(cost: int | None) -> str:
    """Discrepancy level of an unexplained-operation count."""
    return "na" if cost is None else str(cost) if cost < 3 else "3+"


def compat_row(word: str, target: str, supplied: tuple[str, ...], readings: tuple[tuple[str, ...], ...],
               km_lex: dict[str, list[str]], lat_lex: dict[str, list[str]]) -> dict:
    """The evidence and audit category of one training pair (see the module docstring)."""
    exact = km_lex.get(target)
    en = en_skeleton(supplied)
    rec: dict = {"khmer_lookup": {"provider": "google_lexicon_exact", "status": "hit" if exact else "miss",
                                  "readings": exact or []}}
    km_reads = [("google_exact", google_skeleton(p)) for p in exact] if exact else []
    ortho = km_skeleton(target)
    if not km_reads and ortho is not None:
        km_reads = [("orthographic", ortho)]
    rec["evidence_track"] = km_reads[0][0] if km_reads else "none"
    rec["target_consonants"] = seg_str(ortho)
    if not km_reads or not en and not any(r for _t, r in km_reads):
        rec.update(category="unknown", reasons=["no Khmer reading (exact lookup miss, spelling unreadable)"
                                               if not km_reads else "no consonants on either side"],
                   cost_supplied=None, level="na")
        return rec
    cost, ops = min((align(en, r) for _t, r in km_reads), key=lambda x: x[0])
    rec["cost_supplied"], rec["level"] = cost, level(cost)
    rec["ops"] = [list(o) for o in ops if o[0] != "same"]
    alt = [(align(en_skeleton(r), k)[0], " ".join(r)) for r in readings[1:] for _t, k in km_reads]
    rec["cost_best_alternative"] = min(alt)[0] if alt else None
    glat = lat_lex.get(word, [])
    rec["google_latin_readings"] = glat
    if glat:
        rec["google_latin_vs_supplied"] = min(align(en, google_skeleton(p))[0] for p in glat)
        rec["target_vs_google_latin"] = min(align(km_reads[0][1], google_skeleton(p), adapt_labels=False)[0]
                                            for p in glat)
    reasons = []
    if cost == 0:
        rec["category"] = "compatible_evidence"
        used = sorted({o[0] for o in ops if o[0].startswith(("adapt:", "free:"))})
        reasons = ["consonants align with the supplied reading"] + used
    else:
        rec["category"] = "suspected_conflict"
        if alt and min(alt)[0] == 0:
            reasons.append("fits another CMUdict reading, not the supplied first one: " + min(alt)[1])
        if glat and rec["target_vs_google_latin"] == 0 and rec["google_latin_vs_supplied"] > 0:
            reasons.append("fits the Google lexicon's Khmer reading of the word, which departs from the English one")
        if ops and ops[0][0] == "del_b":
            reasons.append("extra Khmer consonant at the start (possible mining alignment artefact)")
        reasons.append(f"{cost} unexplained consonant operation(s): "
                       + ", ".join(f"{k} {x or '∅'}→{y or '∅'}" for k, x, y in ops if k in ("sub", "del_a", "del_b")))
    rec["reasons"] = reasons + ["vowels not assessed"]
    return rec


def compat() -> dict:
    """§2: audit every supervised training pair; writes the table, summary and stratified sample."""
    t0 = time.time()
    rows = read_tsv(INPUTS["train"])
    cmu, _ = ps.cmudict(INPUTS["cmudict"])
    km_lex, lat_lex = load_lexicon()
    en_words = set(guarded(INPUTS["en_words"]).read_text(encoding="utf-8").split())
    prov: dict[tuple[str, str], dict] = defaultdict(lambda: {"files": set(), "count": 0, "max_similarity": 0.0})
    for name in ("mined_kmwiki", "mined_fineweb"):
        for r in read_tsv(INPUTS[name]):
            p = prov[(r["latin"], ps.nfc(r["khmer"]))]
            p["files"].add(name.split("_")[1])
            p["count"] += int(r["count"])
            p["max_similarity"] = max(p["max_similarity"], float(r["similarity"]))
    targets_per_word = Counter(r["word"] for r in rows)
    from english_phonetic_data import rule_similarity
    table = []
    for n, r in enumerate(rows):
        w, km = r["word"], r["khmer"]
        readings = cmu.get(w, ())
        supplied = tuple(r["phones"].split())
        p = prov.get((w, km), {"files": set(), "count": 0, "max_similarity": 0.0})
        ambiguous = len({tuple(x.rstrip("012") for x in rd) for rd in readings}) > 1
        rec = {"row": n, "id": f"{w}|{km}", "word": w, "target": km, "part": r["part"],
               "source_files": "+".join(sorted(p["files"])) or "unknown", "mined_count": p["count"],
               "mined_max_similarity": p["max_similarity"],
               "rule_similarity": round(rule_similarity(w, km), 3),
               "word_class": "common_english_word" if w in en_words else "other_or_name",
               "letters": len(w), "length_band": "<=4" if len(w) <= 4 else ">=9" if len(w) >= 9 else "5-8",
               "targets_for_word": targets_per_word[w],
               "pronunciation": {"provider": "cmudict", "supplied": " ".join(supplied),
                                 "alternatives": [" ".join(x) for x in readings[1:]], "ambiguous": ambiguous,
                                 "supplied_is_default": tuple(readings[0]) == supplied if readings else False}}
        rec.update(compat_row(w, km, supplied, readings, km_lex, lat_lex))
        table.append(rec)
    # several targets for one word: do their consonant readings agree?
    by_word = defaultdict(list)
    for rec in table:
        by_word[rec["word"]].append(rec)
    for w, recs in by_word.items():
        sks = [km_skeleton(x["target"]) for x in recs]
        disagree = any(a is not None and b is not None and align(a, b, adapt_labels=False)[0] > 0
                       for i, a in enumerate(sks) for b in sks[i + 1:])
        for x in recs:
            x["word_targets_disagree_on_consonants"] = disagree if len(recs) > 1 else None
    validation = validate_orthography(km_lex)
    summary = summarize(table, validation, lat_lex, cmu)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    write_compat(table, summary)
    return summary


def summarize(table: list[dict], validation: dict, lat_lex: dict, cmu: dict) -> dict:
    """Coverage, ambiguity and discrepancy distributions."""
    n = len(table)
    cat = Counter(r["category"] for r in table)

    def dist(key):
        d = defaultdict(Counter)
        for r in table:
            d[str(r[key])][r["category"]] += 1
            d[str(r[key])]["level_" + r["level"]] += 1
        return {k: dict(v) for k, v in sorted(d.items())}

    reasons = Counter()
    for r in table:
        if r["category"] == "suspected_conflict":
            for x in r["reasons"]:
                reasons[x.split(":")[0] if "unexplained" in x else x] += 1
    subs = Counter(f"{k} {x or '∅'}→{y or '∅'}" for r in table for k, x, y in r.get("ops", [])
                   if k in ("sub", "del_a", "del_b"))
    adapts = Counter(k for r in table for k, _x, _y in r.get("ops", []) if k.startswith(("adapt:", "free:")))
    multi = [r for r in table if r["targets_for_word"] > 1]
    glat = [r for r in table if r.get("google_latin_readings")]
    return {
        "rows": n, "words": len({r["word"] for r in table}),
        "coverage": {
            "cmudict_supplied": n,
            "supplied_equals_dictionary_default": sum(r["pronunciation"]["supplied_is_default"] for r in table),
            "cmudict_ambiguous_beyond_stress": sum(r["pronunciation"]["ambiguous"] for r in table),
            "google_exact_target_hits": sum(r["khmer_lookup"]["status"] == "hit" for r in table),
            "orthographic_readable": sum(r["evidence_track"] == "orthographic" for r in table),
            "google_latin_reading_of_word": len(glat),
            "assessable_any_track": n - cat["unknown"],
        },
        "orthography_validation": validation,
        "categories": dict(cat),
        "levels": dict(Counter(r["level"] for r in table)),
        "by_source": dist("source_files"), "by_length": dist("length_band"), "by_word_class": dist("word_class"),
        "by_cmudict_ambiguity": {str(k): dict(v) for k, v in _group(table, lambda r: r["pronunciation"]["ambiguous"]).items()},
        "suspected_conflict_reasons": dict(reasons.most_common()),
        "unexplained_operations_top": subs.most_common(30),
        "adaptations_used": dict(adapts.most_common()),
        "multi_target": {"words_with_several_targets": len({r["word"] for r in multi}), "rows": len(multi),
                         "words_whose_targets_disagree_on_consonants":
                             len({r["word"] for r in multi if r["word_targets_disagree_on_consonants"]})},
        "google_latin_track": {
            "rows": len(glat),
            "google_reading_departs_from_supplied_english": sum(r["google_latin_vs_supplied"] > 0 for r in glat),
            "target_fits_google_reading": sum(r["target_vs_google_latin"] == 0 for r in glat),
            "target_fits_google_but_not_english": sum(r["target_vs_google_latin"] == 0 and r["cost_supplied"]
                                                      for r in glat if r["cost_supplied"] is not None)},
    }


def _group(table, key):
    d = defaultdict(Counter)
    for r in table:
        d[key(r)][r["category"]] += 1
    return d


def stratified_sample(table: list[dict], k: int = SAMPLE_MAX, seed: int = SEED) -> list[dict]:
    """At most `k` rows, spread over (category, level) strata plus ambiguous / several-target / Google-reading
    strata; deterministic for a seed (each stratum shuffled by its own string-seeded RNG)."""
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in table:
        strata[f"{r['category']}|level={r['level']}"].append(r)
        if r["pronunciation"]["ambiguous"]:
            strata["cmudict_ambiguous"].append(r)
        if r.get("word_targets_disagree_on_consonants"):
            strata["several_targets_disagree"].append(r)
        if r.get("google_latin_readings"):
            strata["google_latin_reading"].append(r)
    names = sorted(strata)
    for s in names:
        random.Random(f"{seed}:{s}").shuffle(strata[s])
    picked, seen, i = [], set(), 0
    while len(picked) < k and any(strata[s] for s in names):
        s = names[i % len(names)]
        while strata[s]:
            r = strata[s].pop()
            if r["row"] not in seen:
                seen.add(r["row"])
                picked.append({**r, "stratum": s})
                break
        i += 1
    return sorted(picked, key=lambda r: (r["stratum"], r["row"]))


def write_compat(table: list[dict], summary: dict) -> None:
    """Write `compat/` outputs."""
    d = OUT / "compat"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "training_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in table:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cols = ["stratum", "row", "word", "target", "source_files", "word_class", "pronunciation.supplied",
            "pronunciation.alternatives", "target_consonants", "evidence_track", "category", "level", "reasons",
            "google_latin_readings", "targets_for_word", "rule_similarity", "mined_count"]
    with open(d / "sample.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in stratified_sample(table):
            row = []
            for c in cols:
                v = r[c.split(".")[0]][c.split(".")[1]] if "." in c else r.get(c, "")
                row.append(" | ".join(v) if isinstance(v, list) else v)
            w.writerow(row)


# ---------------------------------------------------------------- §3 sensitivity
SENS = {"beam": 4, "top_k": 4, "seed": SEED, "donor_rule": "another dev word with a dictionary reading, "
        "phone count within ±1 and a different stress-free reading; nearest count first, ties by the item's RNG",
        "conditions": ["default", "no_pron", "donor", "alternative"]}


def load_dev() -> list:
    """Dev items (the only references read here)."""
    from khmer_translit import reference_data as rd
    items = rd.load_reference_items(guarded(INPUTS["dev_references"]))
    assert all(i.split == "dev" for i in items)
    return items


def build_sample(items: list, cmu: dict) -> list[dict]:
    """The frozen sensitivity sample: per dev word with a dictionary reading, its conditions and donor."""
    have = [i for i in items if cmu.get(i.word)]
    out = []
    for it in have:
        default = cmu[it.word][0]
        strip = lambda ph: tuple(p.rstrip("012") for p in ph)
        rng = random.Random(f"{SEED}:donor:{it.id}")
        cands = [o for o in have if o.word != it.word and strip(cmu[o.word][0]) != strip(default)
                 and abs(len(cmu[o.word][0]) - len(default)) <= 1]
        rng.shuffle(cands)
        cands.sort(key=lambda o: abs(len(cmu[o.word][0]) - len(default)))
        conds = [{"condition": "default", "phones": " ".join(default)},
                 {"condition": "no_pron", "phones": None}]
        if cands:
            conds.append({"condition": "donor", "phones": " ".join(cmu[cands[0].word][0]), "donor_id": cands[0].id})
        else:
            conds.append({"condition": "donor", "skipped": "no donor within ±1 phone"})
        for k, alt in enumerate(cmu[it.word][1:], 1):
            conds.append({"condition": "alternative", "phones": " ".join(alt), "alt_index": k,
                          "stress_only": strip(alt) == strip(default)})
        out.append({"id": it.id, "word": it.word, "accepted": list(it.accepted), "conditions": conds})
    return out


def freeze() -> dict:
    """Write the frozen sample and settings (`sensitivity/frozen.json`) before any model is run."""
    cmu, cmu_sha = ps.cmudict(INPUTS["cmudict"])
    items = load_dev()
    sample = build_sample(items, cmu)
    models = {m: sha(ROOT / "models" / "e02" / m / "model.npz") for m in CHECKPOINTS}
    frozen = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": SENS, "checkpoints": models,
              "cmudict_sha256": cmu_sha, "dev_references_sha256": sha(INPUTS["dev_references"]),
              "dev_items": len(items), "items_with_dictionary_reading": len(sample), "sample": sample}
    d = OUT / "sensitivity"
    d.mkdir(parents=True, exist_ok=True)
    body = json.dumps(frozen, ensure_ascii=False, indent=1) + "\n"
    (d / "frozen.json").write_text(body, encoding="utf-8")
    (d / "frozen.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n", encoding="utf-8")
    return {k: v for k, v in frozen.items() if k != "sample"}


def condition_ids(tl, word: str, cond: dict) -> list[int]:
    """Source ids of one condition through the real encoder (`phonetic_source.encode_source`)."""
    if tl.mode == "letters":
        return ps.encode_source(word, None, "letters", tl.vocab)
    if cond["condition"] == "no_pron":
        pron = ps.get_pronunciation(word, path=tl.cmudict_path)
        return ps.encode_source(word, pron, "phones", tl.vocab, drop_phones=True)
    pron = ps.PronunciationResult(word, tuple(cond["phones"].split()), "diagnostic", "frozen sample")
    return ps.encode_source(word, pron, "phones", tl.vocab)


def check_encoding(tl, word: str, cond: dict, ids: list[int]) -> bool:
    """The letters stay `<task:en>` + word; the phone part is exactly the condition's phones (or `<no_pron>`)."""
    toks = [tl.vocab.tokens[i] for i in ids]
    if toks[:len(word) + 1] != ["<task:en>", *word]:
        return False
    rest = toks[len(word) + 1:]
    if tl.mode == "letters":
        return rest == []
    if cond["condition"] == "no_pron":
        return rest == ["<sep>", "<no_pron>"]
    return rest == ["<sep>", *("en:" + p for p in cond["phones"].split())]


def score_text(tl, ids: list[int], text: str) -> tuple[float, int]:
    """Teacher-forced total log-probability of a spelling (chars + EOS) and its token count."""
    m = tl.model
    memory, pad = m.encode_ids([ids])
    toks = [ps.BOS] + [m.tgt_vocab.index(c) + 3 for c in text] + [ps.EOS]
    total = 0.0
    for t in range(1, len(toks)):
        total += float(m.decode_step(memory, pad, np.array([toks[:t]]))[0][toks[t]])
    return total, len(toks) - 1


def sensitivity() -> dict:
    """§3: run the frozen sample through each checkpoint; per-item rows and aggregates."""
    from khmer_translit.english import Transliterator
    d = OUT / "sensitivity"
    body = (d / "frozen.json").read_text(encoding="utf-8")
    if hashlib.sha256(body.encode()).hexdigest() != (d / "frozen.sha256").read_text().strip():
        raise RuntimeError("frozen.json changed after freezing")
    frozen = json.loads(body)
    t0 = time.time()
    rows, checks = [], {}
    for m in frozen["checkpoints"]:
        path = ROOT / "models" / "e02" / m / "model.npz"
        tl = Transliterator.from_pretrained(guarded(path), cmudict_path=INPUTS["cmudict"],
                                            expected_sha256=frozen["checkpoints"][m])
        enc_ok, tf_gap, repeat_same = True, 0.0, True
        for it in frozen["sample"]:
            conds = [c for c in it["conditions"] if "skipped" not in c]
            ids = [condition_ids(tl, it["word"], c) for c in conds]
            enc_ok &= all(check_encoding(tl, it["word"], c, x) for c, x in zip(conds, ids))
            memory, pad = tl.model.encode_ids(ids)
            beams = tl.model.beam_search(memory, pad, frozen["settings"]["beam"])
            if tl.mode == "letters":                      # control: same spelling → same ids, same output
                again = tl.model.beam_search(*tl.model.encode_ids(ids), frozen["settings"]["beam"])
                repeat_same &= beams == again and all(x == ids[0] for x in ids)
            ref_text, ref_score = beams[0][0]
            base_tf, ntok = score_text(tl, ids[0], ref_text)
            tf_gap = max(tf_gap, abs(base_tf - ref_score))
            base_top = [t for t, _s in beams[0]][:frozen["settings"]["top_k"]]
            for c, x, nb in zip(conds, ids, beams):
                top = [t for t, _s in nb][:frozen["settings"]["top_k"]]
                tf, _ = score_text(tl, x, ref_text)
                rows.append({
                    "checkpoint": m, "mode": tl.mode, "id": it["id"], "word": it["word"],
                    "condition": c["condition"], "phones": c.get("phones"), "donor_id": c.get("donor_id"),
                    "alt_index": c.get("alt_index"), "stress_only": c.get("stress_only"), "source_ids": x,
                    "top1": top[0], "topk": top, "scores": [round(s, 4) for _t, s in nb][:len(top)],
                    "top1_changed": top[0] != base_top[0],
                    "topk_overlap": len(set(top) & set(base_top)) / max(len(set(top) | set(base_top)), 1),
                    "default_top1_rank": top.index(ref_text) + 1 if ref_text in top else None,
                    "default_top1_norm_logprob": tf / ntok, "delta_norm_logprob": (tf - base_tf) / ntok,
                    "dev_agree": ps.nfc(top[0]) in {ps.nfc(a) for a in it["accepted"]}})
        checks[m] = {"encoding_as_intended": enc_ok, "teacher_forced_vs_beam_max_gap": tf_gap,
                     **({"letters_control_identical": repeat_same} if tl.mode == "letters" else {})}
    agg = aggregate(rows)
    skipped = sum("skipped" in c for it in frozen["sample"] for c in it["conditions"])
    result = {"frozen_sha256": (d / "frozen.sha256").read_text().strip(), "elapsed_s": round(time.time() - t0, 1),
              "checks": checks, "skipped_conditions": skipped, "aggregate": agg}
    with open(d / "per_item.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (d / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def aggregate(rows: list[dict]) -> dict:
    """Per checkpoint × condition: counts, top-1 change rate, overlap, rank and score changes, dev agreement."""
    g = defaultdict(list)
    for r in rows:
        g[(r["checkpoint"], r["condition"] if r["condition"] != "alternative" else
           "alternative_stress_only" if r["stress_only"] else "alternative_segmental")].append(r)
    out = {}
    for (m, c), rs in sorted(g.items()):
        dl = [r["delta_norm_logprob"] for r in rs]
        out.setdefault(m, {})[c] = {
            "n": len(rs), "top1_changed": sum(r["top1_changed"] for r in rs),
            "top1_change_rate": sum(r["top1_changed"] for r in rs) / len(rs),
            "mean_topk_overlap": float(np.mean([r["topk_overlap"] for r in rs])),
            "default_top1_still_in_topk": sum(r["default_top1_rank"] is not None for r in rs),
            "default_top1_rank_changed": sum(r["default_top1_rank"] != 1 for r in rs),
            "mean_delta_norm_logprob": float(np.mean(dl)), "median_delta_norm_logprob": float(np.median(dl)),
            "dev_agree": sum(r["dev_agree"] for r in rs)}
    return out


def main() -> int:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["compat", "freeze", "sensitivity"])
    a = ap.parse_args()
    res = {"compat": compat, "freeze": freeze, "sensitivity": sensitivity}[a.step]()
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
