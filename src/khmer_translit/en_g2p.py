"""Sound-first English → Khmer spelling: CMUdict pronunciation → Khmer by rules (prototype, D144 step B; not served).

English spelling does not follow its sound (one, time, -tion), so letter-based guessing fails on English words
(D142, D143). This module spells by **pronunciation**: the CMU Pronouncing Dictionary gives the phonemes (ARPAbet),
and rules turn each syllable into Khmer, following the conventions in the owner's Gemini guides and Khmer loan
spelling:

1. **Syllables:** phonemes are split with the maximal-onset principle (`syllabify`): consonants between vowels go
   to the next syllable as long as they form a legal English onset (`ONSETS`).
2. **Onset consonant** (`ONSET`): each has an a-series and an o-series Khmer form (ថ / ធ, ស / ស៊, ណ / ន, ម៉ / ម).
   Khmer vowel signs sound different after each series, so the series is chosen **by the vowel** (`VOWELS`): e.g.
   /iː/ needs o-series + ី (ធី "thii"), /aː/ needs a-series + ា (ថា "thaa"). A second onset consonant is written
   subscript (ស្ត, ព្រ, ក្ល).
3. **Vowel** (`VOWELS`): AA/AO → ា, AE/EH → ែ, EY → េ, IH → ិ, IY → ី, OW → ូ (a-series), UW → ូ (o-series),
   UH → ុ, AW → ោ, AY → ាយ (ាញ / ាម before N / M: line, time), OY → យ (voice វ៉យ), ER → ឺ (its R starts the next
   syllable before a vowel); unstressed AH0 → ិ inside a word, ា at its end; stressed /ʌ/ in a closed syllable
   adds ់ (one → វ៉ាន់).
4. **Coda** (`_coda`): one final consonant (ក ង ន ម ត ប ល ច ...); a post-vowel R is silent; "-st" → ស្ត; at the
   word end a trailing S / Z / TH after another consonant, or alone after a long vowel, → ស៍ (Khmer loan
   convention, news → ញូវស៍; see D144 A on whether the model speaks it), alone after a short vowel → ស (miss);
   other extra coda consonants are dropped.
5. **Suffix:** SH AH0 N ("-tion") → សិន.

Words missing from CMUdict return None (the transliterator keeps them). Approximate by design: listed spellings
always win, and the owner's ear decides.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

CMUDICT = Path(__file__).resolve().parents[2] / "data" / "src" / "cmudict" / "cmudict.dict"
VOWEL_PHONES = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ONSETS = {("S", "T"), ("S", "K"), ("S", "P"), ("S", "M"), ("S", "N"), ("S", "L"), ("S", "W"),
          ("S", "T", "R"), ("S", "K", "R"), ("S", "P", "R"), ("S", "P", "L"), ("S", "K", "W")} | {
          (c, "R") for c in ("B", "D", "F", "G", "K", "P", "T", "TH", "SH")} | {
          (c, "L") for c in ("B", "F", "G", "K", "P", "S")} | {(c, "W") for c in ("D", "K", "T", "S", "G")} | {
          (c, "Y") for c in ("B", "F", "HH", "K", "M", "P", "V")}
ONSET = {  # phoneme -> (a-series, o-series)
    "B": ("ប", "ប៊"), "CH": ("ឆ", "ឈ"), "D": ("ដ", "ឌ"), "DH": ("ដ", "ឌ"), "F": ("ហ្វ", "ហ្វ៊"), "G": ("ហ្គ", "ហ្គ៊"),
    "HH": ("ហ", "ហ៊"), "JH": ("ច", "ជ"), "K": ("ខ", "ឃ"), "L": ("ឡ", "ល"), "M": ("ម៉", "ម"), "N": ("ណ", "ន"),
    "NG": ("ង៉", "ង"), "P": ("ផ", "ភ"), "R": ("រ៉", "រ"), "S": ("ស", "ស៊"), "SH": ("ស", "ស៊"), "T": ("ថ", "ធ"),
    "TH": ("ថ", "ធ"), "V": ("វ៉", "វ"), "W": ("វ៉", "វ"), "Y": ("យ៉", "យ"), "Z": ("ហ្ស", "ហ្ស៊"), "ZH": ("ហ្ស", "ហ្ស៊"),
    "": ("អ", "អ៊")}
SUBSCRIPT = {"R": "្រ", "L": "្ល", "W": "្វ", "Y": "្យ", "M": "្ម", "N": "្ន", "T": "្ត", "K": "្ក", "P": "្ព"}
CLUSTER_BASE = {"T": "ត", "K": "ក", "P": "ព", "B": "ប", "D": "ដ", "G": "គ", "F": "ហ្វ", "S": "ស", "TH": "ថ", "SH": "ស"}
VOWELS = {  # vowel -> (series, sign); series "a" or "o"
    "AA": ("a", "ា"), "AO": ("a", "ា"), "AE": ("a", "ែ"), "EH": ("a", "ែ"), "EY": ("a", "េ"), "IH": ("o", "ិ"),
    "IY": ("o", "ី"), "OW": ("a", "ូ"), "UW": ("o", "ូ"), "UH": ("o", "ុ"), "AW": ("a", "ោ"), "AY": ("a", "ាយ"),
    "OY": ("a", "យ"), "ER": ("a", "ឺ"), "AH": ("a", "ា")}
LONG = {"AY", "OY", "AW", "OW", "UW", "IY", "EY", "AA", "AO"}   # a single final S after these is written ស៍
CODA = {"B": "ប", "CH": "ច", "D": "ដ", "DH": "ដ", "F": "ហ្វ", "G": "គ", "JH": "ជ", "K": "ក", "L": "ល", "M": "ម",
        "N": "ន", "NG": "ង", "P": "ប", "S": "ស៍", "SH": "ស", "T": "ត", "TH": "ស៍", "V": "វ", "Z": "ស៍", "ZH": "ស",
        "R": "", "HH": "", "W": "", "Y": "យ"}


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[str]]:
    """CMUdict: word → first listed pronunciation (ARPAbet phones with stress digits).

    Returns:
        The dictionary (126k words).
    """
    out: dict[str, list[str]] = {}
    for line in CMUDICT.read_text(encoding="utf-8").splitlines():
        if line[:1].isalpha():
            head, *phones = line.split("#")[0].split()
            out.setdefault(head.split("(")[0], phones)
    return out


def phones(word: str) -> list[str] | None:
    """CMUdict phones of a word, or None if unknown.

    Args:
        word: Any case.

    Returns:
        Phones such as `['T', 'AY1', 'M']`.

    Examples:
        >>> phones("time")
        ['T', 'AY1', 'M']
    """
    return _cmu().get(word.lower())


def syllabify(ph: list[str]) -> list[tuple[list[str], str, list[str]]]:
    """Split phones into `(onset, vowel, coda)` syllables by the maximal-onset principle.

    Args:
        ph: ARPAbet phones (vowels carry a stress digit).

    Returns:
        Syllables; `vowel` keeps its stress digit.

    Examples:
        >>> syllabify(["S", "T", "EY1", "D", "IY0", "AH0", "M"])
        [(['S', 'T'], 'EY1', []), (['D'], 'IY0', []), ([], 'AH0', ['M'])]
    """
    idx = [i for i, p in enumerate(ph) if p[:2] in VOWEL_PHONES]
    if not idx:
        return []
    out = []
    start = 0
    for k, vi in enumerate(idx):
        onset = ph[start:vi]
        if k + 1 < len(idx):
            mid = ph[vi + 1:idx[k + 1]]
            split = len(mid)
            for j in range(len(mid) + 1):   # longest legal onset for the next syllable
                tail = tuple(mid[j:])
                if len(tail) <= 1 or tail in ONSETS:
                    split = j
                    break
            coda, start = mid[:split], vi + 1 + split
        else:
            coda = ph[vi + 1:]
        out.append((onset, ph[vi], coda))
    return out


def _onset(onset: list[str], series: str) -> str:
    """Khmer for an onset (0–3 consonants) with the series the vowel needs.

    Args:
        onset: Onset phones.
        series: "a" or "o".

    Returns:
        Khmer consonant cluster (អ / អ៊ for an empty onset).
    """
    k = 0 if series == "a" else 1
    if not onset:
        return ONSET[""][k]
    if len(onset) == 1:
        return ONSET[onset[0]][k]
    base = CLUSTER_BASE.get(onset[0], ONSET[onset[0]][0])
    subs = "".join(SUBSCRIPT.get(p, "") for p in onset[1:])
    return base + subs + ("៊" if series == "o" and base in ("ស", "ប", "ហ្វ") else "")


def _coda(coda: list[str], final: bool = True, vowel: str = "") -> str:
    """Khmer for a coda: one final consonant (a post-vowel R / HH / W is silent and skipped), "-st" → ស្ត, and at
    the end of the word a trailing S / Z / TH → ស៍ (after another consonant, or alone after a long vowel; a lone S
    after a short vowel is plain ស: miss → មិស).

    Args:
        coda: Coda phones.
        final: Whether this is the word's last syllable (ស៍ only there).
        vowel: The syllable's vowel (no stress digit), for the long-vowel rule.

    Returns:
        Khmer final(s), possibly empty.

    Examples:
        >>> _coda(["R", "K"]), _coda(["S"], vowel="IH"), _coda(["S"], vowel="OY"), _coda(["S"], final=False)
        ('ក', 'ស', 'ស៍', 'ស')
    """
    coda = [p for p in coda if p not in ("R", "HH", "W")] if coda[:1] in (["R"], ["HH"], ["W"]) else coda
    if not coda:
        return ""
    if coda[-2:] == ["S", "T"]:
        return "ស្ត"
    sib = ("S", "Z", "TH")
    if final and len(coda) > 1 and coda[-1] in sib and coda[0] not in sib:
        return CODA.get(coda[0], "") + "ស៍"
    if final and len(coda) == 1 and coda[0] in sib:
        return "ស៍" if vowel in LONG else "ស"
    return CODA.get(coda[0], "").replace("ស៍", "ស")


def transliterate(word: str) -> str | None:
    """Khmer spelling of an English word from its CMUdict pronunciation, or None if the word is not in CMUdict.

    Args:
        word: An English word (any case).

    Returns:
        Khmer spelling, or None.

    Examples:
        >>> transliterate("time"), transliterate("news"), transliterate("station"), transliterate("one")
        ('ថាម', 'ញូស៍', 'ស្តេសិន', 'វ៉ាន់')
        >>> transliterate("zzzqq") is None
        True
    """
    ph = phones(word)
    if not ph:
        return None
    syls = syllabify(ph)
    out = []
    for n, (onset, vowel, coda) in enumerate(syls):
        v, stress = vowel[:2], vowel[2:]
        nxt = syls[n + 1] if n + 1 < len(syls) else None
        last = nxt is None
        if v == "ER" and nxt is not None and not nxt[0]:   # ER before a vowel: its R starts the next syllable
            syls[n + 1] = (["R"], nxt[1], nxt[2])
        if onset == ["SH"] and v == "AH" and stress == "0" and (coda[:1] == ["N"] or (nxt and nxt[0][:1] == ["N"])):
            out.append("សិន" + _coda(coda[1:], last))   # -tion (also before a vowel: -tional → សិនណល)
            continue
        if onset == ["N"] and v == "UW" and n == 0:   # new, news: ញូ (Khmer loan convention)
            out.append("ញូ" + _coda(coda, last, v))
            continue
        series, sign = VOWELS[v]
        if v == "AH" and stress == "0":
            sign = "ា" if last and not coda else "ិ"
            series = "a"
        if v == "AY" and coda[:1] in (["N"], ["M"]):   # line → ឡាញ, time → ថាម (owner-verified forms)
            sign, coda = ("ាញ" if coda[0] == "N" else "ាម"), coda[1:]
        tail = _coda(coda, last, v)
        if v == "AH" and stress in ("1", "2") and tail and coda[0] not in ("NG", "Y"):
            tail = tail[:1] + "់" + tail[1:]   # short stressed /ʌ/ closed: one → វ៉ាន់
        out.append(_onset(onset, series) + sign + tail)
    return "".join(out)
