#!/usr/bin/env python3
"""Mine Latin ↔ Khmer name pairs from the Khmer Wikipedia dump (bracketed forms; after D135).

Run from the repository root (dump in `data/src/kmwiki/`, see `data/SOURCES.md`):

    .venv/bin/python scripts/translit_mine.py                     # -> data/mined_kmwiki.tsv
    .venv/bin/python scripts/translit_mine.py --source fineweb    # -> data/mined_fineweb.tsv

Sources: the Khmer Wikipedia articles dump (CC BY-SA) or the FineWeb-2 `khm_Khmr` train shard (ODC-By, Common
Crawl terms). Summaries go to `datasets/build/translit_mine_<source>.json`.

**Extraction.** Khmer writers often give the Latin form in brackets: "ចូ ប៉េពីតូន (Joe Pepitone)", and less often the
reverse, "Sok Hour (ហុង សុហួរ)". Wiki links and bold/italic marks are stripped; for each bracket of 1–4 capitalised
Latin words, the same number of space-separated Khmer tokens directly before it (or inside it, for the reverse
form) is taken. Word pairs are aligned by position.

**Filtering.** Khmer text has few spaces, so a token often carries extra words in front ("និងឧបទ្វីបខេមឆាតកា"). For
each Latin word the Khmer token is cut at the syllable start (`syllable_starts`) whose remainder sounds most like the
Latin word under the rule sound-out (`latin_g2p`) or the learned transliterator (`best_cut`), and the pair is kept
if that similarity is ≥ `THRESHOLD` (0.5, set by inspecting a 40-pair sample). This removes translations and acronyms (blueprint → សង់,
OPEC → ...ប្រេងកាត) but also loses some good pairs whose spelling the English rules predict badly.

**Classes.** `khmer_like` if the Latin word is a romanised-Khmer form seen in training or the Khmer word is a known
Khmer word (lexicon or training target); otherwise `foreign`. A heuristic: Khmer names written in Latin can fall in
either class.

**Leakage guard.** Bracket pairs whose full Latin form equals a name in `names_dev.tsv` or `names_test2.tsv` are
dropped (entity level), so those sets stay unseen.
"""
from __future__ import annotations

import bz2
import json
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import iterparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from khmer_translit import latin_g2p, translit  # noqa: E402
from khmer_translit.latin_g2p import similarity  # noqa: E402

DUMP = ROOT / "data/src/kmwiki/kmwiki-latest-pages-articles.xml.bz2"
FINEWEB = ROOT / "data/src/fineweb2_km/train_000_00000.parquet"
THRESHOLD = 0.5
LAT = re.compile(r"\(\s*([A-Z][A-Za-z'\-]*(?: [A-Z][A-Za-z'\-]*){0,3})\s*[,;)]")
KMB = re.compile(r"\(\s*([ក-៝​ ]{1,60}?)\s*\)")
KMTOK = re.compile(r"[ក-៝]+$")


def strip_markup(text: str) -> str:
    """Remove wiki links (keeping the shown text), bold/italic quotes and zero-width spaces.

    Args:
        text: Raw wikitext.

    Returns:
        Plain-ish text.

    Examples:
        >>> strip_markup("[[ភ្នំពេញ|ក្រុងភ្នំពេញ]] '''Phnom'''")
        'ក្រុងភ្នំពេញ Phnom'
    """
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    return re.sub(r"'{2,}", "", text).replace("​", "")


def bracket_pairs(text: str) -> list[tuple[str, str]]:
    """`(khmer, latin)` phrase pairs from "Khmer (Latin)" and "Latin (Khmer)" brackets with equal word counts.

    Args:
        text: Text after `strip_markup`.

    Returns:
        The pairs, in text order.

    Examples:
        >>> bracket_pairs("លោក ចូ ប៉េពីតូន (Joe Pepitone) និង Sok Hour (ហុង សុហួរ)")
        [('ចូ ប៉េពីតូន', 'Joe Pepitone'), ('ហុង សុហួរ', 'Sok Hour')]
    """
    out = []
    for m in LAT.finditer(text):
        lat = m.group(1)
        k = len(lat.split())
        toks: list[str] = []
        for w in reversed(text[max(0, m.start() - 120):m.start()].rstrip().split(" ")):
            if KMTOK.match(w) and len(toks) < k:
                toks.insert(0, w)
            else:
                break
        if len(toks) == k:
            out.append((" ".join(toks), lat))
    for m in KMB.finditer(text):
        km = m.group(1).strip()
        k = len(km.split())
        before = text[max(0, m.start() - 80):m.start()].rstrip().split(" ")[-k:]
        if len(before) == k and all(re.fullmatch(r"[A-Z][A-Za-z'\-]*", w) for w in before):
            out.append((km, " ".join(before)))
    out.sort(key=lambda p: text.find(p[1]))
    return out


def syllable_starts(word: str) -> list[int]:
    """Indices where a Khmer syllable can start: a consonant or independent vowel not preceded by coeng (្).

    Args:
        word: Khmer word.

    Returns:
        Candidate cut positions, always including 0.

    Examples:
        >>> syllable_starts("និងប៉ា")
        [0, 2, 3]
    """
    return [i for i, c in enumerate(word)
            if i == 0 or ("ក" <= c <= "ឳ" and word[i - 1] != "្")]


_MODEL_GUESS: dict[str, str] = {}


def best_cut(latin: str, khmer: str, use_model: bool = True) -> tuple[str, float]:
    """The suffix of `khmer` (cut at a syllable start) that sounds most like `latin`.

    "Sounds like" = the higher similarity to two readings of `latin`: the English rule sound-out (`latin_g2p`) and,
    with `use_model`, the learned transliterator's greedy reading (romanised-Khmer conventions: sophea → សុភា).
    Using only the rules would drop exactly the romanised-Khmer pairs this mining is for.

    Args:
        latin: Lower-case Latin word.
        khmer: Khmer token, possibly with extra words glued in front.
        use_model: Also compare with the transliterator reading (default True; cached per word).

    Returns:
        `(suffix, similarity)`.

    Examples:
        >>> best_cut("colin", "កូលីន", use_model=False)
        ('កូលីន', 0.8)
    """
    guesses = [latin_g2p.transliterate(latin)]
    if use_model:
        if latin not in _MODEL_GUESS:
            _MODEL_GUESS[latin] = translit._resources()[0].nbest(latin, beam=1)[0][0]
        guesses.append(_MODEL_GUESS[latin])
    return max(((khmer[i:], max(similarity(g, khmer[i:]) for g in guesses)) for i in syllable_starts(khmer)),
               key=lambda x: x[1])


def texts(source: str):
    """Yield the documents of one source as plain text.

    Args:
        source: "kmwiki" (wikitext pages, markup stripped) or "fineweb" (web documents, read in batches).

    Yields:
        One document string at a time.
    """
    if source == "kmwiki":
        with bz2.open(DUMP) as fh:
            for _ev, el in iterparse(fh):
                if el.tag.endswith("}text") and el.text:
                    yield strip_markup(el.text)
                el.clear()
        return
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(FINEWEB).iter_batches(columns=["text"], batch_size=2000):
        for doc in batch.column(0).to_pylist():
            if "(" in doc:
                yield doc.replace("\u200b", "")


def main() -> int:
    """Extract, filter, classify and write the mined pairs and a summary for `--source`.

    Returns:
        0 on success.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["kmwiki", "fineweb"], default="kmwiki")
    source = ap.parse_args().source
    held = set()
    for name in ("names_dev.tsv", "names_test2.tsv"):
        held |= {r.split("\t")[1] for r in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines()[1:]}
    phrases, pages = Counter(), 0
    for doc in texts(source):
        phrases.update(set(bracket_pairs(doc)))   # once per document: boilerplate repeats count once per page
        pages += 1
    _model, _english, romanised, known = translit._resources()
    pairs: dict[tuple[str, str], dict] = {}
    dropped = Counter()
    for (km, lat), n in phrases.items():
        if " ".join(lat.lower().replace("'", "").split()) in held:
            dropped["held_out_entity"] += n
            continue
        for kw, lw in zip(km.split(), lat.split()):
            if lw.isupper():   # acronyms (OPEC, BBC) are spelled by letters elsewhere
                dropped["acronym"] += 1
                continue
            lw = re.sub(r"['\-]", "", lw).lower()
            if not re.fullmatch("[a-z]{2,}", lw):
                dropped["not_word"] += 1
                continue
            cut, sim = best_cut(lw, kw)
            if sim < THRESHOLD:
                dropped["low_similarity"] += 1
                continue
            rec = pairs.setdefault((lw, cut), {"count": 0, "sim": sim, "phrase": f"{km} ({lat})"})
            rec["count"] += n
    rows = []
    for (lw, kw), rec in sorted(pairs.items()):
        cls = "khmer_like" if lw in romanised or kw in known else "foreign"
        rows.append((lw, kw, cls, rec["count"], round(rec["sim"], 2), rec["phrase"]))
    with open(ROOT / f"data/mined_{source}.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tkhmer\tclass\tcount\tsimilarity\tsource_phrase\n")
        fh.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    summary = {"pages": pages, "bracket_phrases": len(phrases), "word_pairs_kept": len(rows),
               "by_class": dict(Counter(r[2] for r in rows)), "latin_forms": len({r[0] for r in rows}),
               "dropped": dict(dropped), "threshold": THRESHOLD,
               "file": (DUMP if source == "kmwiki" else FINEWEB).name}
    (ROOT / f"datasets/build/translit_mine_{source}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
