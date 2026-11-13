#!/usr/bin/env python3
"""Prepare the training data of both models from raw inputs, and check it against the published files.

    python pipeline/prepare.py --out runs/data            # -> runs/data/small/, runs/data/byt5/
    python pipeline/prepare.py --out runs/data --check    # and compare byte by byte with training-data/

Downloads (once, into `<out>/src/`, e.g. `runs/data/src/`, checked by sha256): the CMU Pronouncing Dictionary and Google's Khmer lexicon.

**Small model** (`training-data/raw/mined_*.tsv`: every bracket pair mined from Khmer Wikipedia and FineWeb-2, with
how often it was seen and how similar the Khmer is to a rule spelling of the Latin word):
1. Held out: the words of every test set (`test-sets/small_*.jsonl`) and of the earlier English panels
   (`training-data/raw/historical_panels.tsv`), with their suffix stems, and every accepted test spelling.
2. Per Latin word, the noise filter keeps a Khmer form seen at least twice or once with similarity >= 0.7, drops a
   form that is a strict suffix of another kept form, and keeps at most 3, most frequent first.
3. A mined pair is kept if the Latin word is plain a-z, not flagged as romanised Khmer, the Khmer is one word of at
   most 40 characters, the form passed the noise filter, the word has an English reading in the CMU dictionary,
   and it is not related to a held-out word or target. The author's example spellings need a plain word, one Khmer
   word and no relation to a held-out word or target (in the release, every example was removed by these rules).
4. About 3% of the training word families are held back (`train_val`) to measure loss only.
5. The Khmer characters and pronunciation symbols come from the training pairs and from Google's lexicon
   (headword -> pronunciation, held-out spellings removed).

**ByT5 adapter** (`training-data/raw/byt5_pairs.tsv`): pairs already screened per provenance: `control` and
`recovered` web-mined pairs (an automatic consonant screen against the English pronunciation, and documented
recovery rules for pairs it rejected wrongly) and `attested_alt` pairs from the ALT corpus. Those screens are not
re-run here. The training rows are the `train` part, checked to share no word with the development or evaluation
words.

The released data also removed rows related to a sealed reserve of English test words, kept unpublished; those
rows (189 mined rows, 1 example) are absent from the raw inputs, so this reproduces the release exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from khmer_translit import phonetic_source as ps

ROOT = Path(__file__).resolve().parents[1]
RAW, TESTS, PUBLISHED = ROOT / "training-data" / "raw", ROOT / "test-sets", ROOT / "training-data"
SALT = "e02-english-phonetic"
SUFFIXES = ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "ly", "s")
DOWNLOADS = {
    "cmudict.dict": ("https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/"
                     "cmudict.dict", "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"),
    "km_lexicon.tsv": ("https://raw.githubusercontent.com/google/language-resources/"
                       "bc72dd6527aee6963910b9afc69a57f7b9aacc1d/km/data/lexicon.tsv",
                       "b3f2e700d07d6c01a4c8cac5aa2f13b9a66885668ac482cc0c513f73c56495e5"),
}


def fetch(name: str, folder: Path) -> Path:
    """Download a pinned file once and check its sha256."""
    url, sha = DOWNLOADS[name]
    p = folder / name
    if not p.exists():
        folder.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, p)
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        raise SystemExit(f"{p}: unexpected sha256 (delete it to download again)")
    return p


def tsv(path: Path) -> list[list[str]]:
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:] if l]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stems(word: str) -> set[str]:
    """The word and its simple suffix-stripped stems: "sharing" -> {"sharing", "shar", "share"}."""
    out = {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            out |= {base, base + "e", base + "y"} if suf in ("ies", "ied") else {base, base + "e"}
            if len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    return {s for s in out if len(s) >= 3} | {word}


def families(words) -> dict[str, str]:
    """Lexical families: words whose stems meet are connected; each family is named by its first word."""
    words = sorted(set(words))
    parent = {w: w for w in words}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w
    by_stem: dict[str, str] = {}
    for w in words:
        for s in stems(w):
            if s in by_stem:
                a, b = find(w), find(by_stem[s])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                by_stem[s] = w
    return {w: find(w) for w in words}


def bucket(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest(), 16) % 100


def keep_forms(forms: dict[str, tuple[int, float]]) -> list[str]:
    """The noise filter for one Latin word's mined Khmer forms (see the module docstring, step 2)."""
    ok = {k: v for k, v in forms.items() if v[0] >= 2 or v[1] >= 0.7}
    ok = {k: v for k, v in ok.items() if not any(o != k and o.endswith(k) for o in ok)}
    return sorted(ok, key=lambda k: (-ok[k][0], k))[:3]


def small(src: Path) -> dict:
    """The small model's training rows, development items and vocabulary."""
    items = [it for f in ("small_dev.jsonl", "small_final.jsonl", "small_proxy.jsonl") for it in jsonl(TESTS / f)]
    hist = {w for _p, w in tsv(RAW / "historical_panels.tsv")}
    held_stems = {s for w in {it["word"] for it in items} | hist for s in stems(w)}
    held_targets = {a for it in items for a in it["accepted"]}
    related = lambda w: bool(stems(w) & held_stems)  # noqa: E731
    cmu, _sha = ps.cmudict(fetch("cmudict.dict", src))
    forms: dict[str, dict[str, list]] = defaultdict(dict)
    for name in ("mined_kmwiki.tsv", "mined_fineweb.tsv"):
        for r in tsv(RAW / name):
            lat, km, cls, n, sim = r[:5]
            km = ps.nfc(km)
            o = forms[lat].setdefault(km, [0, 0.0, set()])
            o[0] += int(n)
            o[1] = max(o[1], float(sim))
            o[2].add(cls)
    train = []
    for lat in sorted(forms):
        w, st = ps.normalize_word(lat)
        kept = keep_forms({k: (v[0], v[1]) for k, v in forms[lat].items()})
        for km in sorted(forms[lat]):
            if (st == "ok" and w == lat and "khmer_like" not in forms[lat][km][2] and ps.is_khmer_word(km)
                    and len(km) <= ps.MAX_TGT and km in kept and w in cmu and not related(w) and km not in held_targets):
                train.append((w, km, "mined_web"))
    for r in tsv(RAW / "author_examples.tsv"):
        w, st = ps.normalize_word(r[0])
        km = ps.nfc(r[1])
        if st == "ok" and ps.is_khmer_word(km) and not related(w) and km not in held_targets:
            train.append((w, km, "owner_examples"))
    train = sorted(set(train))
    fam = families(w for w, *_ in train)
    rows = [(w, km, "train_val" if bucket(fam[w], SALT + ":tv") < 3 else "train") for w, km, _s in train]
    # vocabulary: Khmer characters and pronunciation symbols, from training material only
    p2g = set()
    for line in fetch("km_lexicon.tsv", src).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        head, phones = line.split("\t")[:2]
        km = ps.nfc(head.strip())
        if (ps.is_khmer_word(km) and phones.strip() and len(km) <= ps.MAX_TGT
                and len(phones.split()) + 1 <= ps.MAX_SRC and km not in held_targets):
            p2g.add((km, " ".join(phones.split())))
    km_phones = sorted({p for _k, ph in p2g for p in ph.split()})
    tgt = "".join(sorted({c for _w, km, _p in rows for c in km} | {c for km, _ph in p2g for c in km}))
    dev = [{k: it[k] for k in ("id", "word", "family", "accepted", "preferred", "flags")} for it in jsonl(TESTS / "small_dev.jsonl")]
    return {"rows": rows, "dev": dev, "vocab": {"src_tokens": ps.SourceVocab(km_phones).tokens, "km_phones": km_phones,
                                               "tgt_chars": tgt}}


def byt5() -> list[tuple[str, str, str]]:
    rows = [dict(zip(["word", "khmer", "provenance", "phones", "part"], r)) for r in tsv(RAW / "byt5_pairs.tsv")]
    held = {d["word"] for d in jsonl(PUBLISHED / "byt5" / "dev.jsonl")} | {d["word"] for d in jsonl(TESTS / "byt5_eval.jsonl")}
    train = [r for r in rows if r["part"] == "train"]
    overlap = sorted({r["word"] for r in train} & held)
    if overlap:
        raise SystemExit(f"training words overlap development/evaluation words: {overlap[:5]}")
    source = {"control": "web", "recovered": "web", "attested_alt": "alt"}
    return [(r["word"], r["khmer"], source[r["provenance"]]) for r in train]


def write(out: Path) -> dict[str, Path]:
    s = small(out / "src")
    (out / "small").mkdir(parents=True, exist_ok=True)
    (out / "byt5").mkdir(parents=True, exist_ok=True)
    (out / "small" / "train.tsv").write_text("word\tkhmer\tpart\n" + "".join(f"{w}\t{k}\t{p}\n" for w, k, p in s["rows"]),
                                             encoding="utf-8")
    (out / "small" / "vocab.json").write_text(json.dumps(s["vocab"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "small" / "dev.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in s["dev"]),
                                             encoding="utf-8")
    (out / "byt5" / "train.tsv").write_text("word\tkhmer\tsource\n" + "".join(f"{w}\t{k}\t{x}\n" for w, k, x in byt5()),
                                            encoding="utf-8")
    (out / "byt5" / "dev.jsonl").write_bytes((PUBLISHED / "byt5" / "dev.jsonl").read_bytes())
    print(json.dumps({"small_train": sum(p == "train" for *_x, p in s["rows"]),
                      "small_train_val": sum(p == "train_val" for *_x, p in s["rows"]),
                      "byt5_train": len(byt5())}))
    return {f"{m}/{f}": out / m / f for m, fs in (("small", ("train.tsv", "vocab.json", "dev.jsonl")),
                                                    ("byt5", ("train.tsv",))) for f in fs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="compare with training-data/")
    a = ap.parse_args()
    files = write(a.out)
    if not a.check:
        return 0
    ok = True
    for rel, p in files.items():
        same = p.read_bytes() == (PUBLISHED / rel).read_bytes()
        print(f"{'identical' if same else 'DIFFERENT'}  {rel}")
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
