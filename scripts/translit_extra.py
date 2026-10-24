#!/usr/bin/env python3
"""Build the extra training pairs for transliterator arm B from the mined pairs (after D135).

Run from the repository root after `scripts/translit_mine.py` (both sources):

    .venv/bin/python scripts/translit_extra.py    # -> data/extra_train.tsv, reports/translit_extra.json

**Noise filter** (mined pairs come from automatic bracket alignment and cutting):
- keep a pair seen ≥ 2 times (documents), or once with similarity ≥ 0.7;
- drop a Khmer form that is a strict suffix of another kept form of the same Latin word (a cut too far: dara →
  រា next to ដារា);
- keep at most 3 Khmer forms per Latin word, most frequent first.

**Leakage exclusions** (so every evaluation stays unseen by the model):
- Latin words in the loanword list (`configs/latin_khmer.tsv` in the speech repo, `khmer_translit.loanwords`: the English-loans panel);
- pairs whose Khmer form is a target of the romanised-Khmer **test** split (`pairs.tsv`);
- entity-level exclusion of dev / test2 names was done by the miner. Word-level overlap with test2 names is counted,
  not removed: shared name words (sok, chan) are what generalisation to new people relies on.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khmer_translit.loanwords import LIST_TSV as LOANWORDS  # noqa: E402

T = ROOT / "data"


def keep_forms(forms: dict[str, tuple[int, float]]) -> list[str]:
    """Apply the noise filter to the Khmer forms of one Latin word.

    Args:
        forms: Khmer form → (document count, similarity).

    Returns:
        Kept forms, most frequent first (≤ 3).

    Examples:
        >>> keep_forms({"ដារា": (5, 0.8), "រា": (6, 0.6), "ដារ៉ា": (7, 0.9)})
        ['ដារ៉ា', 'ដារា']
        >>> keep_forms({"សុខា": (1, 0.6)})
        []
    """
    ok = {k: v for k, v in forms.items() if v[0] >= 2 or v[1] >= 0.7}
    ok = {k: v for k, v in ok.items() if not any(o != k and o.endswith(k) for o in ok)}
    return sorted(ok, key=lambda k: -ok[k][0])[:3]


def main() -> int:
    """Merge both mined files, filter, exclude, and write `extra_train.tsv` plus a summary.

    Returns:
        0 on success.
    """
    forms: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    for name in ("mined_kmwiki.tsv", "mined_fineweb.tsv"):
        for r in (T / name).read_text(encoding="utf-8").splitlines()[1:]:
            lat, km, _cls, n, sim = r.split("\t")[:5]
            old = forms[lat].get(km, (0, 0.0))
            forms[lat][km] = (old[0] + int(n), max(old[1], float(sim)))
    listed = {line.split("\t")[0] for line in LOANWORDS.read_text(encoding="utf-8").splitlines()
              if line and not line.startswith("#")}
    pairs = [r.split("\t") for r in (T / "pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    test_khmer = {r[1] for r in pairs if r[3] == "test"}
    test2_words = {w for r in (T / "names_test2.tsv").read_text(encoding="utf-8").splitlines()[1:] for w in r.split("\t")[1].split()}
    rows, excluded = [], {"listed_latin": 0, "test_khmer": 0}
    for lat in sorted(forms):
        if lat in listed:
            excluded["listed_latin"] += 1
            continue
        for km in keep_forms(forms[lat]):
            if km in test_khmer:
                excluded["test_khmer"] += 1
                continue
            rows.append((lat, km, forms[lat][km][0]))
    with open(T / "extra_train.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tkhmer\tcount\n")
        fh.writelines(f"{a}\t{b}\t{c}\n" for a, b, c in rows)
    summary = {"pairs": len(rows), "latin_forms": len({r[0] for r in rows}), "excluded": excluded,
               "latin_forms_also_in_test2_names": len({r[0] for r in rows} & test2_words),
               "test2_name_words": len(test2_words)}
    (ROOT / "reports/translit_extra.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
