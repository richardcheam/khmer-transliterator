#!/usr/bin/env python3
"""Try the clean unified transliterator next to the served one (informal owner look, outside the D173 protocol).

Run from the repository root:

    .venv/bin/python scripts/translit_try.py sophea knhom nham bay safety thunder ronaldo
    .venv/bin/python scripts/translit_try.py            # interactive: type words, empty line to quit

**Served:** the served fallback router as if the words were not in the loanword list (`translit.guess_many`):
English-dictionary words → arm E, f/q/w/x/z leftovers → rules, others → arm B, with the served re-ranking
resources.

**Clean:** the D173 auxiliary model chosen by spelling validation alone (`CLEAN`, aux seed 1: best mean
validation 0.4019). It is reranked with the Google lexicon only. The loanword list is not consulted by either
column. The model is not served.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CLEAN = ROOT / "models/unified/aux_s1/translit.npz"


def compare(words: list[str]) -> list[tuple[str, str, str, str]]:
    """`(word, served spelling, served route, clean spelling)` for each word.

    Args:
        words: Latin words (any case).

    Returns:
        One row per input word, in order.
    """
    from khmer_translit import translit
    from khmer_translit.lexicon import keys as _lexicon_keys
    served = translit.guess_many(words)
    model = translit.load(CLEAN)
    google = frozenset(_lexicon_keys())
    clean = [translit.pick(nb, google) for nb in model.nbest_batch([w.lower() for w in words], 4)]
    return [(w, s[0], s[1], c) for w, s, c in zip(words, served, clean)]


def show(rows: list[tuple[str, str, str, str]]) -> None:
    """Print rows as a table; `=` marks the words where the two agree."""
    print(f"{'word':16} {'served (route)':30} clean")
    for w, s, r, c in rows:
        print(f"{w:16} {s + ' (' + r + ')':30} {c}{'   =' if s == c else ''}")


def main() -> int:
    """CLI: compare the given words, or read words interactively.

    Returns:
        0.
    """
    words = sys.argv[1:]
    if words:
        show(compare(words))
        return 0
    while True:
        try:
            line = input("words> ").strip()
        except EOFError:
            break
        if not line:
            break
        show(compare(line.split()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
