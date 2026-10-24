"""Where the transliterator finds the loanword list, which the speech repository owns.

The list (`configs/latin_khmer.tsv` in `pocket-tts-khmer`: Latin word → Khmer spelling, owner-verified where
marked) is looked up by the speech front-end before the transliterator runs. Training and evaluation here read it
to build the frozen list split, to add the list's training words, and to keep test words out of training data.

The path is `KHMER_LOANWORD_LIST` if that environment variable is set, else the sibling checkout
`../pocket-tts-khmer/configs/latin_khmer.tsv`. Standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIST_TSV = Path(os.environ.get("KHMER_LOANWORD_LIST", ROOT.parent / "pocket-tts-khmer" / "configs" / "latin_khmer.tsv"))


def read(path: Path = LIST_TSV) -> dict[str, str]:
    """The loanword list as `{latin_word_lowercase: khmer_spelling}` (the same parsing as the speech front-end).

    Args:
        path: The TSV (`word<TAB>khmer[<TAB>...]`; `#` comments and the header skipped); default `LIST_TSV`.

    Returns:
        The mapping; `{}` if the file is absent.

    Example:
        `read()["ipad"]` → `'អាយផេត'`.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.split("\t")[:2] == ["word", "khmer"]:   # header only:
            continue                                                                            # "word" is also an entry
        word, khmer = line.split("\t")[:2]
        out[word.strip().lower()] = khmer.strip()
    return out
