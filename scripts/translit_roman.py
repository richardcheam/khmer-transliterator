#!/usr/bin/env python3
"""Clean romanised-Khmer training pairs from the Google Khmer lexicon (D170), replacing the Chhunneng pairs.

Run from the repository root:

    .venv/bin/python scripts/translit_roman.py tune     # agreement report -> datasets/build/translit_roman_tune.json
    .venv/bin/python scripts/translit_roman.py build    # -> data/roman/pairs.tsv (same columns as pairs.tsv)

**Source:** google/language-resources `km` lexicon (CC BY 4.0, © Google): the Khmer-script headwords only.
Each word is romanised by the rules in `khmer_translit.romanise` (spelling and series based, the common Cambodian
typing). The canonical form and one variant are written, so the model sees some typing variety. Nothing from the
Chhunneng repository goes into the file. Its **test** targets are left out, so the romanised-Khmer test stays a fair
measure of real typing.

**Tuning evidence** (`tune`):
- Wikidata Cambodian names, dev split only (CC0, already used for choices);
- for information only, agreement with real typings on the Chhunneng test split. This is never used to change
  rules or data.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit.romanise import romanise, romanise_variants  # noqa: E402

LEX = ROOT / "data/lexicon/km_lexicon.tsv"
D = ROOT / "data/roman"
VARIANTS_PER_WORD = 2


def lexicon_words() -> list[str]:
    """Khmer-script headwords of the Google lexicon, sorted, without duplicates.

    Returns:
        Words made only of Khmer-block characters.
    """
    out = set()
    for line in LEX.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        k = line.split("\t")[0]
        if k and all("ក" <= c <= "៿" for c in k):
            out.add(k)
    return sorted(out)


def test_targets() -> set[str]:
    """Khmer targets of the romanised-Khmer test split, kept out of the generated pairs (fair evaluation)."""
    return {l.split("\t")[1] for l in (ROOT / "data/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]
            if l.split("\t")[3] == "test"}


def main() -> int:
    """CLI entry point (`tune`, `build`); see the module docstring.

    Returns:
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "build"])
    mode = ap.parse_args().mode
    D.mkdir(parents=True, exist_ok=True)
    if mode == "tune":
        rows = [l.split("\t") for l in (ROOT / "data/names_dev.tsv").read_text(encoding="utf-8").splitlines()[1:]]
        names = [(a, b) for _q, l, k, _s in rows for a, b in zip(l.split(), k.split()) if len(l.split()) == len(k.split())]
        test = [l.split("\t") for l in (ROOT / "data/pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]
                if l.split("\t")[3] == "test"]
        out = {"names_dev_words": len(names),
               "names_dev_canonical": sum(romanise(k) == l for l, k in names),
               "names_dev_in_top3": sum(l in romanise_variants(k, 3) for l, k in names),
               "real_typing_test_words": len(test),
               "real_typing_in_top3": sum(r[0] in romanise_variants(r[1], 3) for r in test),
               "real_typing_similarity": round(sum(difflib.SequenceMatcher(None, romanise(r[1]), r[0]).ratio()
                                                   for r in test) / len(test), 3)}
        (ROOT / "datasets/build/translit_roman_tune.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out))
        return 0
    held = test_targets()
    lines = ["latin\tkhmer\tsource\tsplit"]
    words = [k for k in lexicon_words() if k not in held]
    for k in words:
        split = "val" if int(hashlib.sha1(k.encode()).hexdigest(), 16) % 20 == 0 else "train"
        for form in romanise_variants(k, VARIANTS_PER_WORD):
            if 0 < len(form) <= 40:
                lines.append(f"{form}\t{k}\tgoogle_lexicon_rules\t{split}")
    (D / "pairs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines) - 1, "pairs from", len(words), "lexicon words (", len(held), "test targets left out )")
    return 0


if __name__ == "__main__":
    sys.exit(main())
