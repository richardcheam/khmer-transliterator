"""Rule-based Latin → Khmer transliteration for words missing from the loanword list (D131).

The model cannot read Latin letters (it babbles). A word that is not listed, not an all-caps abbreviation and not a
run-together of listed words is sounded out with these rules instead. The owner chose this over spelling the
letters (2026-10-12). The result is a guess: the word is still logged as unknown so its right spelling can be added
to `configs/latin_khmer.tsv`.

**Method:**
1. Lower-case, then split into letter units (longest first: "sh", "ch", "th", "ph", "ng", "ee", "oo", "ai", ...).
2. Group into syllables: onset consonants, one vowel unit, then coda consonants. A single consonant between vowels
   starts the next syllable; of two or more, the first closes the current syllable.
3. Each onset consonant has an a-series and an o-series form (ណ / ន, ឡ / ល, ម៉ / ម, ប / ប៊ ...). Khmer vowel signs
   sound different after each series: ណេ is "nay" but នេ "nee", and លី is "lee" but ឡី "lei". The series is
   chosen per vowel (`O_SERIES_VOWELS`: i, y, ee, ea, ie, u, oo take o-series). A second onset consonant is
   written subscript (ស្ក, ប្រ).
4. Vowels:
   - an open syllable takes the long vowel (ា េ ី ូ);
   - a closed syllable takes the short vowel (ា ែ ិ ុ ុ);
   - a vowel with no onset gets អ.
5. Codas map to final consonants (ក ង ន ម ត ប ស ល ច). A final "r" or "h" after a vowel is silent. A final "e"
   after one consonant is silent and gives the vowel before it its English long value (`MAGIC`: cake, line,
   phone). "c" before e, i or y is read "s". Doubled consonants are read once.

Calibrated on the single-word entries of `configs/latin_khmer.tsv` (archived `scripts/latin_g2p_eval.py`). These rules are
approximate; listed spellings always win.
"""
from __future__ import annotations

import re

UNITS = ("sch", "tch", "sh", "ch", "th", "ph", "kh", "ng", "ck", "qu", "gh", "wh",
         "ee", "ea", "oo", "ou", "ow", "ai", "ay", "au", "aw", "oa", "oi", "oy", "ie", "ei", "ey")
VOWEL_UNITS = {"a", "e", "i", "o", "u", "y", "ee", "ea", "oo", "ou", "ow", "ai", "ay", "au", "aw", "oa", "oi", "oy",
               "ie", "ei", "ey"}
ONSET = {  # unit -> (a-series form, o-series form)
    "b": ("ប", "ប៊"), "c": ("ក", "គ"), "d": ("ដ", "ឌ"), "f": ("ហ្វ", "ហ្វ៊"), "g": ("ហ្គ", "ហ្គ៊"), "h": ("ហ", "ហ៊"),
    "j": ("ជ", "ជ"), "k": ("ខ", "ឃ"), "l": ("ឡ", "ល"), "m": ("ម៉", "ម"), "n": ("ណ", "ន"), "p": ("ផ", "ភ"),
    "q": ("ខ", "ឃ"), "r": ("រ៉", "រ"), "s": ("ស", "ស៊"), "t": ("ត", "ទ"), "v": ("វ៉", "វ"), "w": ("វ៉", "វ"),
    "x": ("ស", "ស៊"), "z": ("ហ្ស", "ហ្ស៊"), "y": ("យ៉", "យ"), "sh": ("ស", "ស៊"), "ch": ("ឆ", "ឈ"), "th": ("ថ", "ធ"),
    "ph": ("ហ្វ", "ហ្វ៊"), "kh": ("ខ", "ឃ"), "ng": ("ង៉", "ង"), "ck": ("ក", "គ"), "qu": ("ខ្វ", "ឃ្វ"),
    "gh": ("ហ្គ", "ហ្គ៊"), "wh": ("វ៉", "វ"), "sch": ("ស្ក", "ស្គ"), "tch": ("ឆ", "ឈ"), "": ("អ", "អ៊")}
MAGIC = {"a": "េ", "i": "ាយ", "o": "ូ", "u": "ូ", "e": "ី", "y": "ាយ"}   # English vowel before consonant + silent e
O_SERIES_VOWELS = {"i", "y", "ee", "ea", "ie", "u", "oo"}   # these sound English-like after o-series consonants
SUBSCRIPT = {"r": "្រ", "l": "្ល", "w": "្វ", "m": "្ម", "n": "្ន", "k": "្ក", "t": "្ត", "p": "្ព", "y": "្យ",
             "c": "្ក", "h": "្ហ"}
CODA = {"b": "ប", "c": "ក", "ck": "ក", "d": "ដ", "f": "ហ្វ", "g": "គ", "k": "ក", "l": "ល", "m": "ម", "n": "ន",
        "ng": "ង", "p": "ប", "s": "ស", "t": "ត", "x": "ក្ស", "z": "ស", "ch": "ច", "tch": "ច", "sh": "ស", "th": "ធ",
        "ph": "ហ្វ", "v": "វ", "j": "ជ", "gh": "", "r": "", "h": "", "w": "", "y": "យ", "q": "ក"}
OPEN = {"a": "ា", "e": "េ", "i": "ី", "o": "ូ", "u": "ូ", "y": "ី", "ee": "ី", "ea": "ី", "oo": "ូ", "ou": "ូ",
        "ow": "ៅ", "ai": "ៃ", "ay": "េ", "au": "ៅ", "aw": "", "oa": "ូ", "oi": "យ", "oy": "យ", "ie": "ី", "ei": "េ",
        "ey": "េ"}
CLOSED = {"a": "ា", "e": "ែ", "i": "ិ", "o": "ុ", "u": "ុ", "y": "ិ", "ee": "ី", "ea": "ី", "oo": "ូ", "ou": "ៅ",
          "ow": "ៅ", "ai": "ៃ", "ay": "េ", "au": "ៅ", "aw": "", "oa": "ូ", "oi": "យ", "oy": "យ", "ie": "ី", "ei": "េ",
          "ey": "េ"}


def units(word: str) -> list[str]:
    """Split a lower-case Latin word into letter units (digraphs and vowel pairs first).

    Args:
        word: Lower-case ASCII letters only.

    Returns:
        The units in order.

    Example:
        >>> units("phoneshop")
        ['ph', 'o', 'n', 'e', 'sh', 'o', 'p']
    """
    word = re.sub(r"([b-df-hj-np-tv-z])\1", r"\1", word)      # doubled consonants: coffee -> cofee
    out, i = [], 0
    while i < len(word):
        for u in sorted(UNITS, key=len, reverse=True):
            if word.startswith(u, i) and not (u in ("ay", "oy", "ey", "aw", "ow") and i + 2 < len(word)
                                               and word[i + 2] in "aeiou"):   # toyota: o + y + o
                out.append(u)
                i += len(u)
                break
        else:
            out.append(word[i])
            i += 1
    return out


def is_vowel(units_: list[str], k: int) -> bool:
    """Whether unit `k` is a vowel. "y" is a vowel unless it starts the word or comes before a vowel.

    Args:
        units_: Units from `units`.
        k: Index into `units_`.

    Returns:
        True for a vowel unit. Example: `is_vowel(units("sokly"), 4)` → `True`.
    """
    u = units_[k]
    if u == "y":
        nxt = k + 1 < len(units_) and units_[k + 1] in VOWEL_UNITS - {"y"}
        return k > 0 and not nxt
    return u in VOWEL_UNITS


def syllables(word: str) -> list[tuple[list[str], str, list[str]]]:
    """Group a word's units into `(onset, vowel, coda)` syllables (see the module docstring, step 2).

    A silent final "e" after a consonant is dropped first. Consonants before the first vowel form its onset;
    consonants after the last vowel form its coda.

    Args:
        word: Lower-case ASCII letters only.

    Returns:
        Syllables; a word without vowels gives one syllable with an empty vowel.

    Example:
        >>> syllables("sokly")
        [(['s'], 'o', ['k']), (['l'], 'y', [])]
    """
    us = units(word)
    if len(us) > 2 and us[-1] == "e" and not is_vowel(us, len(us) - 2):
        us = us[:-1] + ["-e"]                             # silent e: marks the previous vowel long
    magic = us[-1] == "-e"
    us = [u for u in us if u != "-e"]
    vowels = [k for k in range(len(us)) if is_vowel(us, k)]
    if not vowels:
        return [(us, "", [])]
    out = []
    start = 0
    for n, v in enumerate(vowels):
        nxt = vowels[n + 1] if n + 1 < len(vowels) else None
        cons = us[v + 1:nxt] if nxt is not None else us[v + 1:]
        if nxt is None:
            coda, carry = cons, []
        elif len(cons) <= 1:
            coda, carry = [], cons
        else:
            coda, carry = cons[:1], cons[1:]
        out.append((us[start:v], us[v], coda))
        start = v + 1 + len(coda)
    if magic and len(out[-1][2]) == 1:
        onset, vowel, coda = out[-1]
        out[-1] = (onset, vowel + "_", coda)              # phone: o is long before the coda
    return out


def onset_khmer(onset: list[str], o_series: bool = False) -> str:
    """Khmer for an onset cluster: the first consonant in the chosen series, the rest subscript where possible.

    Args:
        onset: Consonant units (empty: អ).
        o_series: Use the o-series form (for vowels in `O_SERIES_VOWELS`), else the a-series form.

    Returns:
        Khmer consonant cluster. Example: `onset_khmer(["p", "r"])` → `'ផ្រ'`.
    """
    first = onset[0] if onset else ""
    head = ONSET.get(first, ONSET[""])[1 if o_series else 0]
    for u in onset[1:]:
        if u in SUBSCRIPT:
            head = head.replace("៉", "").replace("៊", "") + SUBSCRIPT[u]
        else:
            head += ONSET.get(u, ONSET[""])[0]
    return head


def transliterate(word: str) -> str:
    """Sound out one Latin word in Khmer script (see the module docstring).

    Args:
        word: A Latin word (letters, optionally with ' or -); case is ignored.

    Returns:
        The Khmer guess; "" if the word has no letters.

    Examples:
        >>> transliterate("neo"), transliterate("sokly"), transliterate("phone")
        ('ណេអូ', 'សុកលី', 'ហ្វូន')
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    parts = []
    for onset, vowel, coda in syllables(w):
        if not vowel:
            return "".join(ONSET.get(u, ONSET[""])[0] for u in onset)
        long_ = vowel.endswith("_")
        vowel = vowel.rstrip("_")
        table = CLOSED if coda and not long_ else OPEN
        sign = MAGIC.get(vowel, table[vowel]) if long_ else table[vowel]
        if onset and onset[0] == "c" and vowel[0] in "eiy":
            onset = ["s"] + onset[1:]             # soft c: city
        if long_ and coda == ["c"]:
            coda = ["s"]                          # soft c before silent e: ice
        if not sign:                      # "aw": អ + ៅ is closest
            sign = "ៅ"
        tail = "".join(CODA.get(u, "") for u in coda)
        parts.append(onset_khmer(onset, vowel in O_SERIES_VOWELS and not long_) + sign + tail)
    return "".join(parts)


def similarity(a: str, b: str) -> float:
    """Character similarity of two strings: 1 − Levenshtein distance / the longer length.

    Args:
        a: First string.
        b: Second string.

    Returns:
        A value in [0, 1]; 1.0 for two empty strings. Pure.

    Examples:
        >>> similarity("ណេអូ", "ណេអូ"), similarity("ab", "")
        (1.0, 0.0)
    """
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))
