#!/usr/bin/env python3
"""Build the Latin → Khmer transliteration data for the learned transliterator (V02-N1).

Run from the repository root (needs the sources fetched into `data/src/`, see below):

    .venv/bin/python scripts/translit_data.py     # writes data/pairs.tsv, names_test.tsv, summary.json

**Sources** (all free; licence status in `data/SOURCES.md`):
- `chhunneng/.../eng_khm_data.txt`: ~28.6k "Khmer: romanisation" word pairs (repository MIT; the original
  provenance of the pairs is not documented). Lower-case rows are romanised Khmer; capitalised rows are English
  country names and are dropped (English words go through the loanword list instead).
- `chhunneng/khmer-english-transliteration/*.txt`: 325 "Khmer=variant,variant" rows of informal chat spellings.
- `wikidata/persons_kh.csv`: Cambodian people (P27 = Q424) with English and Khmer labels (CC0). **Test only.**

**Split:** romanised-Khmer pairs are grouped by their Khmer word, so every romanisation of one Khmer word falls in
the same split: sha1(khmer) mod 100 < 8 → test, < 14 → val, else train. The Wikidata names are a separate,
out-of-domain test of real personal names; name words that occur in the training pairs are kept but counted
(`seen_in_train`), because the realistic question is "does the model read real names", not "unseen words only".
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "src"
OUT = ROOT / "data"
KHMER_WORD = re.compile(r"^[ក-៝​]+$")
LATIN_WORD = re.compile(r"^[a-z]+$")


def split_of(khmer: str) -> str:
    """Deterministic split for one Khmer word (grouped: all its romanisations share it).

    Args:
        khmer: The Khmer target word.

    Returns:
        "test" (8%), "val" (6%) or "train".

    Examples:
        >>> split_of("សុភា") in {"train", "val", "test"}
        True
    """
    bucket = int(hashlib.sha1(khmer.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 8 else "val" if bucket < 14 else "train"


def clean_latin(text: str) -> str:
    """Lower-case a romanisation and drop apostrophes and spaces ("bong kern" → "bongkern").

    Args:
        text: Raw romanisation.

    Returns:
        The cleaned form; may be non-letters-only (the caller filters with `LATIN_WORD`).

    Examples:
        >>> clean_latin("hochl'aonh"), clean_latin("bong kern")
        ('hochlaonh', 'bongkern')
    """
    return re.sub(r"['\s]", "", text.strip().lower())


def read_pairs() -> tuple[list[dict], Counter]:
    """Read and clean the two Chhunneng files into romanised-Khmer pairs.

    Returns:
        `(pairs, dropped)`: unique `{latin, khmer, source}` dicts, and a counter of dropped rows by reason
        (`english_name`, `not_letters`, `khmer_chars`).

    Raises:
        FileNotFoundError: If the sources were not fetched.
    """
    raw = SRC / "chhunneng" / "khmer_text_transliteration" / "data" / "raw" / "eng_khm_data.txt"
    rows: list[tuple[str, str, str]] = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        khmer, latin = (part.strip() for part in line.split(":", 1))
        rows.append((khmer, latin, "chh_dict"))
    for path in sorted((SRC / "chhunneng" / "khmer-english-transliteration").glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                khmer, variants = line.split("=", 1)
                rows += [(khmer.strip(), v, "chh_informal") for v in variants.split(",") if v.strip()]
    pairs, seen, dropped = [], set(), Counter()
    for khmer, latin, source in rows:
        if latin[:1].isupper():
            dropped["english_name"] += 1
            continue
        latin = clean_latin(latin)
        khmer = khmer.replace("​", "")
        if not LATIN_WORD.match(latin):
            dropped["not_letters"] += 1
            continue
        if not KHMER_WORD.match(khmer) or "ៗ" in khmer:
            dropped["khmer_chars"] += 1
            continue
        if (latin, khmer) not in seen:
            seen.add((latin, khmer))
            pairs.append({"latin": latin, "khmer": khmer, "source": source})
    return pairs, dropped


def read_names(train_latin: set[str]) -> list[dict]:
    """Read the Wikidata person names whose English and Khmer labels have the same number of words.

    Labels with titles or joined forms (different word counts) are skipped: their word alignment is unknown.

    Args:
        train_latin: Latin words present in the training split (to flag `seen_in_train`).

    Returns:
        `{qid, latin, khmer, seen_in_train}` dicts; `latin` is lower-case words joined by spaces, accents removed.
    """
    import unicodedata
    names = []
    with open(SRC / "wikidata" / "persons_kh.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            latin = unicodedata.normalize("NFKD", row["en"]).encode("ascii", "ignore").decode().lower()
            latin_words = re.sub(r"[-']", "", latin).split()
            khmer_words = row["km"].replace("​", "").split()
            if (len(latin_words) != len(khmer_words) or not all(LATIN_WORD.match(w) for w in latin_words)
                    or not all(KHMER_WORD.match(w) for w in khmer_words)):
                continue
            names.append({"qid": row["p"].rsplit("/", 1)[-1], "latin": " ".join(latin_words),
                          "khmer": " ".join(khmer_words),
                          "seen_in_train": sum(w in train_latin for w in latin_words)})
    return names


def main() -> int:
    """Write `pairs.tsv`, `names_test.tsv` and `summary.json` under `data/`.

    Returns:
        0 on success.
    """
    pairs, dropped = read_pairs()
    for p in pairs:
        p["split"] = split_of(p["khmer"])
    train_latin = {p["latin"] for p in pairs if p["split"] == "train"}
    names = read_names(train_latin)
    with open(OUT / "pairs.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tkhmer\tsource\tsplit\n")
        fh.writelines(f"{p['latin']}\t{p['khmer']}\t{p['source']}\t{p['split']}\n" for p in pairs)
    with open(OUT / "names_test.tsv", "w", encoding="utf-8") as fh:
        fh.write("qid\tlatin\tkhmer\tseen_in_train\n")
        fh.writelines(f"{n['qid']}\t{n['latin']}\t{n['khmer']}\t{n['seen_in_train']}\n" for n in names)
    by_split = Counter(p["split"] for p in pairs)
    ambiguous = Counter(Counter(p["latin"] for p in pairs).values())
    summary = {"pairs": len(pairs), "by_split": dict(by_split), "by_source": dict(Counter(p["source"] for p in pairs)),
               "dropped": dict(dropped), "khmer_words": len({p["khmer"] for p in pairs}),
               "latin_forms": len({p["latin"] for p in pairs}),
               "latin_forms_by_n_khmer": {str(k): v for k, v in sorted(ambiguous.items())},
               "names_test": len(names),
               "name_words_seen_in_train": sum(n["seen_in_train"] for n in names),
               "name_words": sum(len(n["latin"].split()) for n in names)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
