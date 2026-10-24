"""Google's Khmer pronunciation lexicon: reading it and splitting text into its words.

Lexicon: `google/language-resources` `km/data/lexicon.tsv` at commit `bc72dd6527aee6963910b9afc69a57f7b9aacc1d`
(Copyright 2018 Google Inc., **CC BY 4.0**), kept at `data/lexicon/km_lexicon.tsv` with its `ATTRIBUTION.md`.

The transliterator uses the spellings to re-rank candidate spellings (`translit.pick`) and to build clean training
pairs; the speech front-end uses them to find words inside unspaced Khmer text. Standard library only.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

KHMER = re.compile(r"[ក-៿]")
MAX_WORD = 24
LEXICON_TSV = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "km_lexicon.tsv"


@lru_cache(maxsize=1)
def keys(path: Path = LEXICON_TSV) -> frozenset[str]:
    """The lexicon's Khmer spellings, read once.

    Args:
        path: The lexicon TSV (default `data/lexicon/km_lexicon.tsv`).

    Returns:
        The spellings; empty if the file is absent (re-ranking and word splitting then do nothing).

    Example:
        `"ខ្មែរ" in keys()` → True (62,175 spellings in the pinned lexicon).
    """
    if not path.exists():
        return frozenset()
    return frozenset(read_lexicon(path.read_text(encoding="utf-8").splitlines()))


def read_lexicon(lines: Iterable[str]) -> dict[str, list[str]]:
    """Map each Khmer spelling to its phonemic transcriptions, from lexicon TSV lines.

    Args:
        lines: The file's lines, each `spelling<TAB>phonemes[<TAB>label]`. Blank lines, `#` comments and entries
            whose spelling has no Khmer character are skipped.

    Returns:
        `{spelling: [phonemes, ...]}`, every pronunciation per spelling in file order. Pure.

    Examples:
        >>> read_lexicon(["# c", "ក\\tk ɑɑ", "ab\\tʔ a p", "ក\\tk a"])
        {'ក': ['k ɑɑ', 'k a']}
    """
    out: dict[str, list[str]] = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and KHMER.search(parts[0]):
            out.setdefault(parts[0], []).append(parts[1])
    return out


def segment(phrase: str, lexicon: Mapping[str, Any], max_word: int = MAX_WORD) -> list[tuple[str, bool]]:
    """Split one phrase into lexicon words by greedy longest match; characters outside any word merge into runs.

    A heuristic, not a validated Khmer word segmenter.

    Args:
        phrase: One text span without spaces.
        lexicon: Any mapping whose keys are spellings (e.g. `read_lexicon(...)`).
        max_word: The longest match tried, in code points (default 24).

    Returns:
        `[(piece, in_lexicon), ...]` covering `phrase` in order. Pure.

    Examples:
        >>> segment("កខគX", {"កខ": 1, "គ": 1})
        [('កខ', True), ('គ', True), ('X', False)]
    """
    out: list[tuple[str, bool]] = []
    i = 0
    while i < len(phrase):
        for j in range(min(len(phrase), i + max_word), i, -1):
            if phrase[i:j] in lexicon:
                out.append((phrase[i:j], True))
                i = j
                break
        else:
            if out and not out[-1][1]:
                out[-1] = (out[-1][0] + phrase[i], False)
            else:
                out.append((phrase[i], False))
            i += 1
    return out
